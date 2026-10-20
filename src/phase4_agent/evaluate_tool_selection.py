"""Compare three prompt styles for GeoSense MCP tool selection.

Selection only: this script never executes an MCP tool.
Run from the project root with the Phase IV environment active.
"""
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.phase4_agent import geoai_agent as agent_module


ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = [
    {
        "id": "flood",
        "question": "Assess the flood and climate risk at latitude 26.1445, longitude 91.7362.",
        "expected": ["mcp_get_climate_risk"],
    },
    {
        "id": "building_height",
        "question": "Estimate building height around latitude 26.1445, longitude 91.7362.",
        "expected": ["mcp_lidar_profile"],
    },
    {
        "id": "walkability",
        "question": "Assess pedestrian walkability at latitude 26.1445, longitude 91.7362.",
        "expected": ["mcp_streetview_scene"],
    },
    {
        "id": "land_cover",
        "question": "Classify satellite imagery and vegetation around latitude 26.1445, longitude 91.7362.",
        "expected": ["mcp_classify_imagery"],
    },
    {
        "id": "site_suitability",
        "question": "Score the suitability of latitude 26.1445, longitude 91.7362 for a new facility.",
        "expected": ["mcp_score_location"],
    },
    {
        "id": "similar_sites",
        "question": "Find sites similar to latitude 26.1445, longitude 91.7362.",
        "expected": ["mcp_find_similar_sites"],
    },
]

PROMPT_STYLES = ("vague", "manual", "described")


def response_text(response):
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def selected_names(text, available_names):
    """Find exact available tool names mentioned in a model response."""
    return [name for name in available_names if name in text]


def build_prompt(style, question, tools):
    catalogue = json.dumps(tools, indent=2)
    names = [tool["name"] for tool in tools]

    if style == "vague":
        return (
            f"Question: {question}\n"
            f"Available tools:\n{catalogue}\n"
            "Which of these tools would you use? Explain briefly."
        )

    if style == "manual":
        return (
            f"Question: {question}\n"
            f"Available tool names: {', '.join(names)}\n"
            "Return only a comma-separated list of the exact tool names needed. "
            "Do not include tools that are not needed."
        )

    descriptions = "\n".join(
        f"- {tool['name']}: {tool.get('description', '')}"
        for tool in tools
    )
    return (
        "You select MCP tools for a geospatial analysis agent.\n"
        f"Question: {question}\n"
        f"Tool catalogue:\n{descriptions}\n"
        "Choose only the tools required to answer the question. "
        "Return only valid JSON in this format: "
        '{"tools": ["exact_tool_name"]}'
    )


def evaluate():
    tools = asyncio.run(agent_module.get_tools())
    available_names = [tool["name"] for tool in tools]
    results = []

    print(f"Discovered {len(tools)} tools.")
    print("Selection-only evaluation: no MCP tools will be executed.")
    print("This run makes up to 18 model calls; API quota may limit it.")

    for scenario in SCENARIOS:
        for style in PROMPT_STYLES:
            print(f"Evaluating {scenario['id']} / {style} ...")
            row = {
                "scenario": scenario["id"],
                "prompt_style": style,
                "question": scenario["question"],
                "expected_tools": scenario["expected"],
                "selected_tools": [],
                "status": "ERROR",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": None,
                "raw_response": "",
            }

            try:
                response = agent_module.model.invoke(
                    build_prompt(style, scenario["question"], tools)
                )
                text = response_text(response).strip()
                usage = agent_module.extract_token_usage(response)

                row["raw_response"] = text
                row["input_tokens"] = usage.get("input_tokens")
                row["output_tokens"] = usage.get("output_tokens")
                row["total_tokens"] = usage.get("total_tokens")
                row["selected_tools"] = selected_names(text, available_names)

                expected = set(scenario["expected"])
                selected = set(row["selected_tools"])

                if expected == selected:
                    row["status"] = "MATCH"
                elif expected.issubset(selected):
                    row["status"] = "EXTRA"
                else:
                    row["status"] = "MISSED"

            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  Model call failed: {row['error']}")

            results.append(row)
            print(
                f"  Expected={row['expected_tools']} "
                f"Selected={row['selected_tools']} Status={row['status']}"
            )

    output_dir = ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"tool_selection_{stamp}.json"
    csv_path = output_dir / f"tool_selection_{stamp}.csv"

    json_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available_tools": tools,
                "method": "Selection only; no MCP tool execution",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    columns = [
        "scenario", "prompt_style", "question", "expected_tools",
        "selected_tools", "status", "input_tokens", "output_tokens",
        "total_tokens", "error", "raw_response",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            row = row.copy()
            row["expected_tools"] = json.dumps(row["expected_tools"])
            row["selected_tools"] = json.dumps(row["selected_tools"])
            writer.writerow(row)

    print("\nSUMMARY")
    for style in PROMPT_STYLES:
        subset = [r for r in results if r["prompt_style"] == style]
        counts = {
            status: sum(r["status"] == status for r in subset)
            for status in ("MATCH", "EXTRA", "MISSED", "ERROR")
        }
        print(f"{style}: {counts}")

    print(f"\nJSON results: {json_path}")
    print(f"CSV results:  {csv_path}")


if __name__ == "__main__":
    evaluate()
