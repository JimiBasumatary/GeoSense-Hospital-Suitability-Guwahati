
import asyncio
import json

from mcp_server import mcp

EXPECTED_TOOLS = {
    "mcp_score_location",
    "mcp_classify_imagery",
    "mcp_lidar_profile",
    "mcp_streetview_scene",
    "mcp_get_climate_risk",
    "demographics",
    "mcp_find_similar_sites",
}

TEST_ARGUMENTS = {
    "mcp_score_location": {
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "mcp_classify_imagery": {
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "mcp_lidar_profile": {
        "min_lat": 26.10,
        "min_lon": 91.70,
        "max_lat": 26.20,
        "max_lon": 91.80,
    },
    "mcp_streetview_scene": {
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "mcp_get_climate_risk": {
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "demographics": {
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "mcp_find_similar_sites": {
        "latitude": 26.1445,
        "longitude": 91.7362,
        "radius_km": 5.0,
        "top_k": 3,
    },
}


async def main():
    tools = await mcp.list_tools()
    actual_names = {tool.name for tool in tools}

    print("\n=== MCP TOOL REGISTRATION ===")
    for name in sorted(actual_names):
        print("REGISTERED:", name)

    missing = EXPECTED_TOOLS - actual_names
    unexpected = actual_names - EXPECTED_TOOLS

    print("\nMissing tools:", sorted(missing))
    print("Unexpected tools:", sorted(unexpected))

    if missing or unexpected:
        raise RuntimeError("MCP tool registration does not match expectations.")

    print("\n=== MCP TOOL EXECUTION TESTS ===")
    failed = []

    for name in sorted(EXPECTED_TOOLS):
        print(f"\n--- {name} ---")

        try:
            result = await mcp.call_tool(
                name,
                TEST_ARGUMENTS[name],
            )

            is_error = getattr(result, "is_error", False)
            print("is_error:", is_error)

            structured = getattr(result, "structured_content", None)

            if structured is not None:
                print(json.dumps(structured, indent=2, default=str))
            else:
                for item in getattr(result, "content", []):
                    if hasattr(item, "text"):
                        print(item.text)
                    else:
                        print(str(item))

            if is_error:
                failed.append(name)

        except Exception as exc:
            failed.append(name)
            print(f"FAILED: {type(exc).__name__}: {exc}")

    print("\n=== TEST SUMMARY ===")
    print("Total tools:", len(EXPECTED_TOOLS))
    print("Failed calls:", len(failed))
    print("Failed tools:", failed)

    if failed:
        raise RuntimeError("One or more MCP tool calls failed.")


if __name__ == "__main__":
    asyncio.run(main())
