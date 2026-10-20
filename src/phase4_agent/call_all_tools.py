
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8002/mcp/"

LOCATION = {"latitude": 26.1445, "longitude": 91.7362}

TOOL_ARGUMENTS = {
    "mcp_score_location": LOCATION,
    "mcp_classify_imagery": LOCATION,
    "mcp_lidar_profile": {
        "min_lat": 26.10,
        "min_lon": 91.70,
        "max_lat": 26.20,
        "max_lon": 91.80,
    },
    "mcp_streetview_scene": LOCATION,
    "mcp_get_climate_risk": LOCATION,
    "demographics": LOCATION,
    "mcp_find_similar_sites": {
        **LOCATION,
        "radius_km": 5.0,
        "top_k": 3,
    },
}


async def main():
    failures = []

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            response = await session.list_tools()
            registered = {tool.name for tool in response.tools}

            print("Registered MCP tools:", len(registered))

            for name, arguments in TOOL_ARGUMENTS.items():
                print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

                if name not in registered:
                    print("FAIL: Tool is not registered.")
                    failures.append(name)
                    continue

                try:
                    result = await session.call_tool(
                        name, arguments=arguments
                    )

                    print("MCP error:", result.is_error)

                    for item in result.content:
                        if hasattr(item, "text"):
                            try:
                                print(json.dumps(
                                    json.loads(item.text),
                                    indent=2,
                                    default=str,
                                ))
                            except (json.JSONDecodeError, TypeError):
                                print(item.text)

                    if result.is_error:
                        failures.append(name)

                except Exception as exc:
                    print(f"FAIL: {type(exc).__name__}: {exc}")
                    failures.append(name)

    print("\n=== SUMMARY ===")
    print("Tools attempted:", len(TOOL_ARGUMENTS))
    print("Failed calls:", len(failures))
    print("Failures:", failures)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
