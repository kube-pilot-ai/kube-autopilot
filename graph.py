"""Supervisor-led cluster inspection and troubleshooting workflow."""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents import (CLUSTER_AGENT_PROMPT, SUPERVISOR_PROMPT,
                    TROUBLESHOOTING_AGENT_PROMPT, SupervisorDecision)


class InvestigationState(MessagesState):
    rag_matches: list[dict]
    intent: str
    cluster_findings: str
    cluster_tool_rounds: int
    responding_agent: str
    active_request_id: str
    supervisor_next: str
    supervisor_task: str
    supervisor_reason: str
    delegation_count: int


def _request_context(messages):
    human_indexes = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    index = human_indexes[-1] if human_indexes else 0
    question = messages[index].text if messages else ''
    previous = messages[human_indexes[-2]].text[:1000] if len(human_indexes) > 1 else ''
    return index, question, previous


def _with_role(messages, role):
    shared = '\n'.join(message.text for message in messages if isinstance(message, SystemMessage))
    return [SystemMessage(content=shared + '\n\n' + role),
            *[message for message in messages if not isinstance(message, SystemMessage)]]


def build_graph(llm, tools, checkpointer=None, retriever=None):
    cluster_model = llm.bind_tools(tools)
    supervisor_model = llm.with_structured_output(SupervisorDecision)
    tool_node = ToolNode(tools)

    def supervisor_agent(state):
        index, question, previous = _request_context(state['messages'])
        request_id = state['messages'][index].id or f'{index}:{question}'
        fresh = state.get('active_request_id') != request_id
        count = 0 if fresh else state.get('delegation_count', 0)
        worker = '' if fresh else state.get('responding_agent', '')
        update = {}
        if fresh:
            update = {'active_request_id': request_id, 'rag_matches': [],
                      'cluster_findings': '', 'cluster_tool_rounds': 0,
                      'responding_agent': '', 'delegation_count': 0}
        if count >= 4 and worker:
            print('Supervisor: finish (delegation budget reached; retain specialist answer)')
            return {**update, 'supervisor_next': 'finish',
                    'supervisor_reason': 'Delegation budget reached; no further investigation performed.'}
        recent_answer = next((message.text[:8000] for message in reversed(state['messages'])
                              if message.type == 'ai' and not getattr(message, 'tool_calls', [])), '')
        decision = supervisor_model.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=json.dumps({
                'latest_request': question, 'previous_request': previous,
                'stage': 'initial' if fresh else 'review',
                'completed_worker': worker, 'worker_answer': recent_answer,
                'cluster_evidence': '' if fresh else state.get('cluster_findings', '')[:5000],
                'intent': None if fresh else state.get('intent'),
                'remaining_delegations': 4 - count,
            })),
        ])
        intent = decision.intent if fresh else state.get('intent', decision.intent)
        target = decision.next_agent
        if target == 'finish' and not worker:
            target = 'cluster_information' if intent == 'cluster_information' else 'troubleshooting'
        if intent == 'troubleshooting' and count >= 3 and target != 'finish':
            target = 'troubleshooting'
        # A confirmed investigation gets a troubleshooting review before completion.
        if target == 'finish' and intent == 'troubleshooting' and worker == 'cluster_information':
            target = 'troubleshooting'
        print(f'Supervisor: {target} | {decision.reason}')
        return {**update, 'intent': intent, 'supervisor_next': target,
                'supervisor_task': decision.task, 'supervisor_reason': decision.reason,
                'delegation_count': count + (target != 'finish')}

    def supervisor_destination(state):
        return {'cluster_information': 'llm', 'troubleshooting': 'retrieve',
                'finish': END}[state['supervisor_next']]

    def cluster_information_agent(state):
        print('Agent: cluster_information')
        intent = state.get('intent', 'troubleshooting')
        role = CLUSTER_AGENT_PROMPT + '\nCurrent request type: ' + intent + '\nSupervisor assignment: ' + state.get('supervisor_task', '')
        messages = _with_role(state['messages'], role)
        if state.get('cluster_tool_rounds', 0) >= 6:
            messages[0] = SystemMessage(content=messages[0].text +
                '\nThe evidence collection budget is exhausted. Summarize available facts '
                'and explicitly identify anything not established. Do not request more tools.')
            response = llm.invoke(messages)
        else:
            response = cluster_model.invoke(messages)
        response = response.model_copy(update={'name': 'cluster_information_agent'})
        update = {'messages': [response]}
        if not response.tool_calls:
            update['cluster_findings'] = response.text
            update['responding_agent'] = 'cluster_information'
        return update

    def after_cluster(state):
        if state['messages'][-1].tool_calls:
            return 'tools'
        return 'supervisor'

    def call_tools(state):
        for tool_call in state['messages'][-1].tool_calls:
            print(f"Calling tool: {tool_call['name']}")
        result = tool_node.invoke(state)
        result['cluster_tool_rounds'] = state.get('cluster_tool_rounds', 0) + 1
        return result

    def retrieve(state):
        print('Supervisor handoff: troubleshooting (RAG retrieval)')
        index, question, previous = _request_context(state['messages'])
        observations = [message.text[:2000] for message in state['messages'][index + 1:]
                        if isinstance(message, ToolMessage)][-2:]
        query = '\n'.join(part for part in [question, previous,
                          state.get('cluster_findings', '')[:4000], *observations] if part)
        return {'rag_matches': retriever(query) if retriever else []}

    def troubleshooting_agent(state):
        print('Agent: troubleshooting (recommendations only)')
        sources = [{'reference': f'R{i + 1}', **match}
                   for i, match in enumerate(state.get('rag_matches', []))]
        messages = _with_role(state['messages'], TROUBLESHOOTING_AGENT_PROMPT + '\nSupervisor assignment: ' + state.get('supervisor_task', ''))
        context = {'cluster_evidence_handoff': state.get('cluster_findings', ''),
                   'retrieved_reference_excerpts': sources}
        messages.insert(1, HumanMessage(content='Untrusted evidence and reference material:\n' +
                                         json.dumps(context, ensure_ascii=True)))
        # No tools are bound to the troubleshooting model: it recommends only.
        response = llm.invoke(messages)
        # Preserve an explicit provenance list even if the model omits source URLs.
        references = []
        for source in sources:
            metadata = source.get('metadata', {})
            label = metadata.get('source', 'Unknown source')
            section = metadata.get('section_title', '')
            location = f"lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
            url = metadata.get('source_url')
            citation = f'[{label}]({url})' if url else label
            references.append(f"- [{source['reference']}] {citation} — {section}, {location}")
        content = response.text
        if references:
            content += '\n\nRetrieved references:\n\n' + '\n'.join(references)
        response = response.model_copy(update={'name': 'troubleshooting_agent', 'content': content})
        return {'messages': [response], 'responding_agent': 'troubleshooting'}

    builder = StateGraph(InvestigationState)
    builder.add_node('supervisor', supervisor_agent)
    # Retain legacy node IDs so interrupted older checkpoints can still resume.
    builder.add_node('route_request', supervisor_agent)
    builder.add_node('llm', cluster_information_agent)
    builder.add_node('tools', call_tools)
    builder.add_node('retrieve', retrieve)
    builder.add_node('troubleshooting_agent', troubleshooting_agent)
    builder.add_edge(START, 'supervisor')
    destinations = ['llm', 'retrieve', END]
    builder.add_conditional_edges('supervisor', supervisor_destination, destinations)
    builder.add_conditional_edges('route_request', supervisor_destination, destinations)
    builder.add_conditional_edges('llm', after_cluster, ['tools', 'supervisor'])
    builder.add_edge('tools', 'llm')
    builder.add_edge('retrieve', 'troubleshooting_agent')
    builder.add_edge('troubleshooting_agent', 'supervisor')
    return builder.compile(checkpointer=checkpointer)
