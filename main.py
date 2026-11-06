import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from graph import build_graph
from mcp_client import load_mcp_tools


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Add OPENAI_API_KEY to the .env file before running this script.")

    llm = ChatOpenAI(model="gpt-5.6-luna", use_responses_api=True)
    tools = asyncio.run(load_mcp_tools())
    messages = [
        SystemMessage(
            content="list out all the the namespaces"
        ),
    ]
    graph = build_graph(llm, tools)
    try:
        result = graph.invoke({"messages": messages}, config={"recursion_limit": 21})
    except GraphRecursionError:
        raise SystemExit("Stopped at the graph step limit. Try a more specific instruction.")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
