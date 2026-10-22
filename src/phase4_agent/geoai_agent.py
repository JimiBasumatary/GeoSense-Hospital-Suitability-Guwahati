
"""GeoSense Phase IV Exercise 3: LangGraph agent using the MCP server."""

import asyncio
import json
from datetime import datetime, timezone
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.phase4_agent.agent_result_store import publish_result

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8002/mcp/")
MODEL_NAME = os.getenv("AGENT_MODEL", "google_genai:gemini-3.5-flash-lite")
model = init_chat_model(
    MODEL_NAME,
    timeout=180000,
    max_retries=0,
)
DEMO_LOCATION = {"latitude": 26.1445, "longitude": 91.7362}


class AgentState(TypedDict, total=False):
    question: str
    location: dict
    selected_tools: list
    tool_results: list
    answer: str
    error: str
    history: list
    token_usage: dict


async def get_tools():
    """Discover tools from the running MCP server."""
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": tool.name, "description": tool.description or ""}
                for tool in result.tools
            ]


async def call_mcp_tool(name, arguments):
    """Call one tool through the MCP protocol."""
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            blocks = []
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        blocks.append(json.loads(item.text))
                    except (json.JSONDecodeError, TypeError):
                        blocks.append(item.text)
            return {"name": name, "is_error": result.is_error, "result": blocks}


def extract_token_usage(response):
    """Extract token counts when exposed by the model response."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        return {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def select_tools(state: AgentState):
    """Ask Gemini which tools are relevant to the question."""
    tools = asyncio.run(get_tools())
    prompt = f"""
You are selecting tools for GeoSense, a geospatial analysis agent.
Question: {state['question']}
Available tools:
{json.dumps(tools, indent=2)}

Return ONLY JSON with this structure:
{{"tools":["tool_name"],"latitude":26.1445,"longitude":91.7362}}

Select only relevant tools from the available list.
Use coordinates explicitly stated in the question when available.
Otherwise use the demo coordinates 26.1445, 91.7362.
"""
    response = model.invoke(prompt)
    token_usage = extract_token_usage(response)
    text = response.content
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )
    text = str(text).strip().removeprefix("```json").removesuffix("```").strip()
    choice = json.loads(text)
    allowed = {tool["name"] for tool in tools}
    selected = [name for name in choice.get("tools", []) if name in allowed]
    question_lower = state["question"].lower()
    prior_location = state.get("location")
    follow_up_terms = (
        "previous", "same location", "that location",
        "previous assessment", "there", "this location"
    )
    has_new_coordinates = (
        "latitude" in question_lower and "longitude" in question_lower
    )

    if (
        prior_location
        and any(term in question_lower for term in follow_up_terms)
        and not has_new_coordinates
    ):
        location = prior_location
    else:
        location = {
            "latitude": float(choice.get("latitude", DEMO_LOCATION["latitude"])),
            "longitude": float(choice.get("longitude", DEMO_LOCATION["longitude"])),
        }

    return {
        "selected_tools": selected,
        "location": location,
        "token_usage": {"tool_selection": token_usage},
    }


def execute_tools(state: AgentState):
    """Execute selected MCP tools with location-appropriate arguments."""
    lat = state["location"]["latitude"]
    lon = state["location"]["longitude"]
    results = []

    for name in state["selected_tools"]:
        if name in ("mcp_score_location", "mcp_classify_imagery",
                    "mcp_streetview_scene", "mcp_get_climate_risk",
                    "demographics"):
            arguments = {"latitude": lat, "longitude": lon}
        elif name == "mcp_lidar_profile":
            arguments = {
                "min_lat": lat - 0.0015, "min_lon": lon - 0.0015,
                "max_lat": lat + 0.0015, "max_lon": lon + 0.0015,
            }
        elif name == "mcp_find_similar_sites":
            arguments = {"latitude": lat, "longitude": lon, "radius_km": 5.0}
        else:
            continue

        try:
            results.append(asyncio.run(call_mcp_tool(name, arguments)))
        except Exception as exc:
            results.append({
                "name": name,
                "is_error": True,
                "result": [f"Tool call failed: {type(exc).__name__}: {exc}"],
            })

    return {"tool_results": results}


def synthesize_answer(state: AgentState):
    """Generate an answer grounded in tool data and conversation history."""
    history = state.get("history", [])
    prompt = f"""
Answer the user's geospatial question using only the tool results below.
Explain unavailable data honestly. Do not invent measurements or claim that
a tool succeeded when it returned an error. Mention important limitations.

Previous conversation (for context only):
{json.dumps(history[-6:], default=str)}

Current question: {state["question"]}
Coordinates: {json.dumps(state.get("location", {}))}
Tool results: {json.dumps(state.get("tool_results", []), default=str)}

Use previous turns to understand follow-up references, but treat the current
question and current tool results as authoritative. If evidence is missing,
say so. Give a clear answer and distinguish measured results from limitations.
"""
    response = model.invoke(prompt)
    token_usage = extract_token_usage(response)
    text = response.content
    if isinstance(text, list):
        text = "\n".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )
    answer = str(text)
    updated_history = list(history)
    updated_history.append({"question": state["question"], "answer": answer})
    existing_usage = state.get("token_usage", {})
    combined_usage = dict(existing_usage)
    combined_usage["synthesis"] = token_usage
    return {
        "answer": answer,
        "history": updated_history[-12:],
        "token_usage": combined_usage,
    }

graph_builder = StateGraph(AgentState)
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("execute_tools", execute_tools)
graph_builder.add_node("synthesize_answer", synthesize_answer)
graph_builder.add_edge(START, "select_tools")
graph_builder.add_edge("select_tools", "execute_tools")
graph_builder.add_edge("execute_tools", "synthesize_answer")
graph_builder.add_edge("synthesize_answer", END)
checkpointer = MemorySaver()
agent = graph_builder.compile(checkpointer=checkpointer)


def save_transcript(question: str, thread_id: str, result: dict):
    """Save one agent run as a JSON transcript."""
    log_dir = ROOT / "outputs" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    transcript = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "question": question,
        "location": result.get("location"),
        "selected_tools": result.get("selected_tools", []),
        "tool_results": result.get("tool_results", []),
        "answer": result.get("answer", ""),
        "error": result.get("error"),
        "token_usage": result.get("token_usage"),
    }

    filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"transcript_{filename}.json"
    path.write_text(json.dumps(transcript, indent=2, default=str), encoding="utf-8")
    return str(path)


def run_agent(question: str, thread_id: str = "default"):
    """Run the agent and log both successful and failed attempts."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke({"question": question}, config=config)
    except Exception as exc:
        error_message = (
            f"Agent execution failed ({type(exc).__name__}). "
            "The language model or another agent component was unavailable. "
            "No successful analysis can be reported; please retry later."
        )
        result = {
            "question": question,
            "location": DEMO_LOCATION,
            "selected_tools": [],
            "tool_results": [],
            "answer": error_message,
            "error": f"{type(exc).__name__}: {exc}",
            "token_usage": {},
        }
        print(f"GeoAI agent error: {type(exc).__name__}: {exc}")

    try:
        publish_result(question, thread_id, result)
    except Exception as exc:
        print(f"Latest-result publishing warning: {type(exc).__name__}: {exc}")

    try:
        transcript_path = save_transcript(question, thread_id, result)
        print(f"Transcript saved: {transcript_path}")
    except Exception as exc:
        print(f"Transcript logging warning: {type(exc).__name__}: {exc}")

    return result


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "Is this location suitable for a hospital?"
    try:
        result = run_agent(question)
        print("\nQUESTION:", question)
        print("\nLOCATION:", json.dumps(result.get("location"), indent=2))
        print("\nTOOLS SELECTED:", result.get("selected_tools"))
        print("\nTOOL RESULTS:", json.dumps(result.get("tool_results"), indent=2))
        print("\nANSWER:\n", result.get("answer"))
    except Exception as exc:
        print(f"Agent failed: {type(exc).__name__}: {exc}")
        raise
