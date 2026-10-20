
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8002/mcp/"


async def main():
    async with streamable_http_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            print("MCP connection initialized.")
            print("Server:", initialized.server_info.name)

            response = await session.list_tools()
            print(f"\nDiscovered {len(response.tools)} tools:")

            for tool in response.tools:
                print(f"- {tool.name}: {tool.description}")

            print("\nCalling mcp_find_similar_sites...")
            result = await session.call_tool(
                "mcp_find_similar_sites",
                arguments={
                    "latitude": 26.1445,
                    "longitude": 91.7362,
                    "radius_km": 5.0,
                    "top_k": 3,
                },
            )

            print("MCP error:", result.is_error)

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
