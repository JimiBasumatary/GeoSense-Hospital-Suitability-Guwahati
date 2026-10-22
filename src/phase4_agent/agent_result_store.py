import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "outputs" / "agent_logs" / "latest_result.json"


def publish_result(question: str, thread_id: str, result: dict) -> dict:
    """Atomically publish the latest real agent result."""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "result_id": str(uuid.uuid4()),
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

    temp_path = RESULT_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temp_path, RESULT_PATH)
    return payload


def get_latest_result() -> dict | None:
    """Return the latest published result, or None if none exists."""
    if not RESULT_PATH.exists():
        return None
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
