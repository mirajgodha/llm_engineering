import asyncio
import sys
import traceback
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client


def print_items(name: str, result: any) -> None:
    """Print items with formatting.

    Args:
        name: Category name (tools/resources/prompts)
        result: Result object containing items list
    """
    print(f"\nAvailable {name}:")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")


async def main(server_url: str, article_url: str = None):
    """Connect to the MCP server, list its capabilities, and optionally call a tool.

    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
        article_url: (Optional) Wikipedia URL to fetch an article
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("Connected to MCP server at", server_url)
                print_items("tools", await session.list_tools())
                print_items("resources", await session.list_resources())
                print_items("prompts", await session.list_prompts())

                if article_url:
                    print("\nCalling read_wikipedia_article tool...")
                    try:
                        # Use the documented call_tool method to invoke the tool
                        response = await session.call_tool(
                            "read_wikipedia_article", arguments={"url": article_url}
                        )
                        print("\n=== Wikipedia Article Markdown Content ===\n")
                        print(response)
                    except Exception as tool_exc:
                        print("Error calling read_wikipedia_article tool:")
                        traceback.print_exception(
                            type(tool_exc), tool_exc, tool_exc.__traceback__
                        )
    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run -- client.py <server_url> [<wikipedia_article_url>]\n"
            "Example: uv run -- client.py http://localhost:8000/sse https://en.wikipedia.org/wiki/Python_(programming_language)"
        )
        sys.exit(1)

    server_url = sys.argv[1]
    article_url = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(server_url, article_url))
