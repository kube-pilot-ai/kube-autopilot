from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_graph(llm, tools, checkpointer=None):
    """Build the LLM -> tools -> LLM workflow using shared message state."""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_llm(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def call_tools(state: MessagesState):
        for tool_call in state["messages"][-1].tool_calls:
            print(f"Calling tool: {tool_call['name']}")
        return tool_node.invoke(state)

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    return builder.compile(checkpointer=checkpointer)
