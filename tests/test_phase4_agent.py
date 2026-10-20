import json
from pathlib import Path
from types import SimpleNamespace

import src.phase4_agent.geoai_agent as agent_module


def test_extract_token_usage():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 35,
            "total_tokens": 155,
        }
    )

    result = agent_module.extract_token_usage(response)

    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 35
    assert result["total_tokens"] == 155


def test_extract_token_usage_without_metadata():
    response = SimpleNamespace(content="Test answer")

    assert agent_module.extract_token_usage(response) == {}


def test_save_transcript_with_token_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "ROOT", tmp_path)

    sample = {
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "selected_tools": ["mcp_score_location"],
        "tool_results": [],
        "answer": "Token logging test.",
        "token_usage": {
            "tool_selection": {
                "input_tokens": 120,
                "output_tokens": 35,
                "total_tokens": 155,
            },
            "synthesis": {
                "input_tokens": 200,
                "output_tokens": 50,
                "total_tokens": 250,
            },
        },
    }

    saved = agent_module.save_transcript(
        "Token persistence test", "pytest_thread", sample
    )

    path = Path(saved)
    assert path.exists()

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["question"] == "Token persistence test"
    assert loaded["thread_id"] == "pytest_thread"
    assert loaded["token_usage"]["tool_selection"]["total_tokens"] == 155
    assert loaded["token_usage"]["synthesis"]["total_tokens"] == 250
from types import SimpleNamespace
import json


def test_select_tools_filters_unknown_tools(monkeypatch):
    monkeypatch.setattr(
        agent_module.asyncio,
        "run",
        lambda coroutine: (
            coroutine.close() or [
                {"name": "mcp_score_location"},
                {"name": "mcp_classify_imagery"},
            ]
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "model",
        SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content=json.dumps({
                    "tools": ["mcp_score_location", "fake_tool"],
                    "latitude": 26.15,
                    "longitude": 91.74,
                }),
                usage_metadata={},
            )
        ),
    )

    result = agent_module.select_tools({"question": "Score this location"})

    assert result["selected_tools"] == ["mcp_score_location"]
    assert result["location"] == {"latitude": 26.15, "longitude": 91.74}


def test_select_tools_retains_previous_location(monkeypatch):
    monkeypatch.setattr(
        agent_module.asyncio,
        "run",
        lambda coroutine: (
            coroutine.close() or [{"name": "mcp_score_location"}]
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "model",
        SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content=json.dumps({
                    "tools": ["mcp_score_location"],
                    "latitude": 10.0,
                    "longitude": 20.0,
                }),
                usage_metadata={},
            )
        ),
    )

    previous = {"latitude": 26.1445, "longitude": 91.7362}
    result = agent_module.select_tools({
        "question": "Summarize the previous assessment at the same location",
        "location": previous,
    })

    assert result["location"] == previous
def test_execute_tools_captures_tool_failure(monkeypatch):
    async def failing_tool(name, arguments):
        raise RuntimeError("Mock MCP failure")

    monkeypatch.setattr(agent_module, "call_mcp_tool", failing_tool)

    result = agent_module.execute_tools({
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "selected_tools": ["mcp_score_location"],
    })

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["is_error"] is True
    assert "Mock MCP failure" in result["tool_results"][0]["result"][0]
from types import SimpleNamespace
import json


def test_select_tools_filters_unknown_tools(monkeypatch):
    monkeypatch.setattr(
        agent_module.asyncio,
        "run",
        lambda coroutine: (
            coroutine.close() or [
                {"name": "mcp_score_location"},
                {"name": "mcp_classify_imagery"},
            ]
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "model",
        SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
            content=json.dumps({
                "tools": ["mcp_score_location", "fake_tool"],
                "latitude": 26.15,
                "longitude": 91.74,
            }),
            usage_metadata={},
        )),
    )

    result = agent_module.select_tools({"question": "Score this location"})

    assert result["selected_tools"] == ["mcp_score_location"]
    assert result["location"] == {"latitude": 26.15, "longitude": 91.74}


def test_select_tools_retains_previous_location(monkeypatch):
    monkeypatch.setattr(
        agent_module.asyncio,
        "run",
        lambda coroutine: (
            coroutine.close() or [{"name": "mcp_score_location"}]
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "model",
        SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
            content=json.dumps({
                "tools": ["mcp_score_location"],
                "latitude": 10.0,
                "longitude": 20.0,
            }),
            usage_metadata={},
        )),
    )

    previous = {"latitude": 26.1445, "longitude": 91.7362}
    result = agent_module.select_tools({
        "question": "Summarize the previous assessment at the same location",
        "location": previous,
    })

    assert result["location"] == previous


def test_execute_tools_captures_tool_failure(monkeypatch):
    async def failing_tool(name, arguments):
        raise RuntimeError("Mock MCP failure")

    monkeypatch.setattr(agent_module, "call_mcp_tool", failing_tool)

    result = agent_module.execute_tools({
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "selected_tools": ["mcp_score_location"],
    })

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["is_error"] is True
    assert "Mock MCP failure" in result["tool_results"][0]["result"][0]
from types import SimpleNamespace


def test_synthesize_answer_updates_history_and_token_usage(monkeypatch):
    monkeypatch.setattr(
        agent_module,
        "model",
        SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content="The location has high suitability.",
                usage_metadata={
                    "input_tokens": 200,
                    "output_tokens": 50,
                    "total_tokens": 250,
                },
            )
        ),
    )

    previous_history = [
        {
            "question": "Assess the location",
            "answer": "The initial assessment was completed.",
        }
    ]

    result = agent_module.synthesize_answer({
        "question": "Summarize the assessment",
        "location": {"latitude": 26.1445, "longitude": 91.7362},
        "tool_results": [],
        "history": previous_history,
        "token_usage": {
            "tool_selection": {
                "input_tokens": 120,
                "output_tokens": 35,
                "total_tokens": 155,
            }
        },
    })

    assert result["answer"] == "The location has high suitability."
    assert len(result["history"]) == 2
    assert result["history"][-1]["question"] == "Summarize the assessment"
    assert result["history"][-1]["answer"] == result["answer"]
    assert result["token_usage"]["tool_selection"]["total_tokens"] == 155
    assert result["token_usage"]["synthesis"]["total_tokens"] == 250
