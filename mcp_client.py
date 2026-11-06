import asyncio
import os

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _sync_proxy(remote_tool):
    async def ainvoke_remote(**kwargs):
        return await remote_tool.ainvoke(kwargs)

    def invoke_remote(**kwargs):
        # Each adapter call opens and closes its own remote MCP session.
        return asyncio.run(remote_tool.ainvoke(kwargs))

    return StructuredTool(
        name=remote_tool.name,
        description=remote_tool.description,
        args_schema=remote_tool.args_schema,
        func=invoke_remote,
        coroutine=ainvoke_remote,
    )


async def load_mcp_tools():
    """Discover tools over HTTP; never import or execute Kubernetes SDK locally."""
    url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8002/mcp")
    token = os.environ.get("MCP_AUTH_TOKEN")
    connection = {"transport": "streamable_http", "url": url}
    if token:
        connection["headers"] = {"Authorization": f"Bearer {token}"}
    client = MultiServerMCPClient({"kubernetes": connection})
    try:
        tools = await client.get_tools()
    except Exception as exc:
        raise RuntimeError("Unable to connect to the Kubernetes MCP server. Check MCP_SERVER_URL, authentication, and that the server is running.") from exc
    if not tools:
        raise RuntimeError("The Kubernetes MCP server did not expose any tools.")
    return [_sync_proxy(tool) for tool in tools]
