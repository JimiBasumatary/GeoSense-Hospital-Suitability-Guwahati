import uuid
import json
import streamlit as st
import streamlit.components.v1 as components

from src.phase4_agent.geoai_agent import run_agent
from src.phase4_agent.agent_result_store import get_latest_result

st.set_page_config(
    page_title="GeoSense GeoAI Agent",
    page_icon="🌍",
    layout="wide",
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

st.title("GeoSense GeoAI Agent")
st.caption("Phase IV | LangGraph Agent + MCP Tools + Cesium Digital Twin")

with st.sidebar:
    st.subheader("System status")
    st.write("MCP API: http://127.0.0.1:8002")
    st.write("Digital twin: http://127.0.0.1:5173")
    st.write("Conversation thread:")
    st.code(st.session_state.thread_id)

    if st.button("Clear displayed conversation"):
        st.session_state.history = []
        st.rerun()

agent_tab, twin_tab, result_tab = st.tabs(
    ["Ask GeoSense", "Digital Twin", "Latest Published Result"]
)

with agent_tab:
    st.subheader("Ask a geographic question")
    st.write(
        "Ask about location suitability, imagery, demographics, "
        "climate risk, or similar sites."
    )

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item.get("tools"):
                st.caption("Selected tools")
                st.write(item["tools"])
            if item.get("tool_results"):
                with st.expander("Tool results"):
                    st.json(item["tool_results"])
            if item.get("token_usage") is not None:
                st.caption("Token usage")
                st.json(item["token_usage"])

    question = st.chat_input("Enter your geographic question...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        try:
            with st.spinner("Running GeoSense agent..."):
                result = run_agent(
                    question,
                    thread_id=st.session_state.thread_id,
                )

            answer = result.get("answer", "No answer was returned.")
            tools = result.get("selected_tools", [])
            tool_results = result.get("tool_results", [])
            token_usage = result.get("token_usage")

            st.session_state.history.append({
                "question": question,
                "answer": str(answer),
                "tools": tools,
                "tool_results": tool_results,
                "token_usage": token_usage,
            })

            st.rerun()

        except Exception as exc:
            st.error(
                "The agent could not complete this request. "
                "Check the terminal for details."
            )
            st.code(f"{type(exc).__name__}: {exc}")
            if "429" in str(exc) or "quota" in str(exc).lower():
                st.warning(
                    "Gemini API quota appears exhausted. "
                    "Wait for quota reset or check your API limits."
                )

with twin_tab:
    st.subheader("Guwahati Digital Twin")
    st.write(
        "The twin polls the MCP server for the latest published agent result."
    )
    components.iframe(
        "http://127.0.0.1:5173/",
        height=650,
        scrolling=True,
    )
    st.link_button(
        "Open Digital Twin in a separate tab",
        "http://127.0.0.1:5173/",
    )

with result_tab:
    st.subheader("Latest published agent result")
    try:
        latest = get_latest_result()
        if latest is None:
            st.info("No agent result has been published yet.")
        else:
            st.json(latest)
            st.caption(
                "This displays the stored result; it may be from an earlier run."
            )
            if st.button("Refresh latest result"):
                st.rerun()
    except Exception as exc:
        st.error(f"Could not read the latest result: {exc}")

st.divider()
st.caption(
    "GeoSense Phase IV prototype. Model calls run on the Python side; "
    "API credentials are not placed in the browser."
)
