"""Summarize workflow events for the chat activity timeline."""
from contextlib import closing

from langchain_core.messages import AIMessage, ToolMessage


STAGE_LABELS = {
    "supervisor": "Supervisor is reviewing the request",
    "route_request": "Supervisor is reviewing the request",
    "llm": "Cluster Information Agent is inspecting the evidence",
    "tools": "Running the requested Kubernetes MCP tools",
    "retrieve": "Searching the troubleshooting guides",
    "troubleshooting_agent": "Troubleshooting Agent is preparing a recommendation",
    "prepare_remediation": "Preparing the Kubernetes change preview",
    "execute_remediation": "Sending the approved change to Kubernetes",
}


def read_activity(conn, conversation_id):
    return conn.execute(
        "SELECT stage, detail, created_at FROM chat_activity WHERE conversation_id = %s ORDER BY id",
        (conversation_id,),
    ).fetchall()


def _record(conn, conversation_id, stage, detail=""):
    conn.execute(
        "INSERT INTO chat_activity (conversation_id, stage, detail) VALUES (%s, %s, %s)",
        (conversation_id, stage, detail),
    )


def _summarize(node, update):
    if node == "__interrupt__":
        return "Waiting for your approval", "Review the change card to approve or decline."
    if not isinstance(update, dict):
        return None

    if node in {"supervisor", "route_request"}:
        target = {
            "cluster_information": "Cluster Information Agent",
            "troubleshooting": "Troubleshooting Agent",
            "finish": "Finish investigation",
        }.get(update.get("supervisor_next"))
        if target:
            return "Supervisor decision", target
    if node == "llm":
        names = [call["name"] for message in update.get("messages", [])
                 if isinstance(message, AIMessage) for call in message.tool_calls]
        if names:
            return "MCP tools requested", ", ".join(names)
    if node == "tools":
        names = [message.name for message in update.get("messages", [])
                 if isinstance(message, ToolMessage) and message.name]
        if names:
            return "MCP tools returned results", ", ".join(names)
    if node == "retrieve":
        matches = update.get("rag_matches", [])
        sections = [
            " > ".join(part for part in (metadata.get("source"), metadata.get("section_title")) if part)
            for match in matches if (metadata := match.get("metadata"))
        ]
        return f"Retrieved {len(matches)} reference chunks", "\n".join(filter(None, sections))
    if node == "prepare_remediation":
        pending = update.get("pending_remediation")
        if pending:
            change = pending["change"]
            return "Change preview ready", (
                f'{change.get("operation", "patch").capitalize()} Deployment '
                f'{change["namespace"]}/{change["deployment"]}'
            )
        return "Preview could not be prepared", ""
    if node == "approve_remediation":
        return ("Change approved" if update.get("approved_plan_id") else "Change declined"), ""
    if node == "execute_remediation":
        return "Kubernetes change result", update.get("remediation_outcome", "")
    return None


def run_with_activity(graph, graph_input, config, conn, conversation_id):
    try:
        with closing(graph.stream(graph_input, config=config, durability="sync",
                                  stream_mode=["tasks", "updates"])) as stream:
            for mode, event in stream:
                if mode == "tasks":
                    label = STAGE_LABELS.get(event.get("name"))
                    if label and "input" in event:
                        _record(conn, conversation_id, label)
                    continue
                for node, update in event.items():
                    if entry := _summarize(node, update):
                        _record(conn, conversation_id, *entry)

        snapshot = graph.get_state(config)
        if not snapshot.next:
            _record(conn, conversation_id, "Request finished")
        return snapshot.values
    except Exception:
        _record(conn, conversation_id, "Request interrupted", "See the chat for the latest status.")
        raise
