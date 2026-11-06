# Kube Autopilot

FastAPI chat, LangGraph orchestration, and PostgreSQL conversation history.
Kubernetes tools are discovered from the independent [MCP repository](https://github.com/kube-pilot-ai/MCP)
over Streamable HTTP. This application does not import Kubernetes tools or RAG
code from another checkout.

## Local setup (PowerShell)

```powershell
cd C:\Users\kusum\Documents\Final_Project\kube-autopilot
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Fresh clone only: Copy-Item .env.example .env, then replace the example values.
docker compose up -d --wait
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Start MCP separately before this application.
If port 8000 is unavailable, add `--port 8001` instead.
Your local `.env` has already been migrated and remains Git-ignored. The app
needs only OpenAI, PostgreSQL, and MCP connection settings; it has no kubeconfig.
URL-encode special password characters in DATABASE_URL.

## Conversations

Use **New chat** and **Saved chats** to create and reopen conversations.
PostgreSQL stores UI messages and LangGraph checkpoints. Follow-up questions
in a conversation retain context. Edit SYSTEM_MESSAGE in app.py for instructions.
API: GET /conversations, POST /conversations, GET /conversations/{id}, POST /chat
with message and conversation_id, and POST /conversations/{id}/resume.
The local UI is for one user; API access exposes that user's saved conversations.

The optional CLI still runs with `.\.venv\Scripts\python.exe main.py` and uses
the SystemMessage in main.py. CLI runs do not persist conversation history.

## Database ownership

This repository owns only PostgreSQL, published on 127.0.0.1:5433.
The ignored compose.override.yaml on this machine reuses the pre-migration
Docker volume, preserving existing chat history. Fresh clones use a new named
volume automatically. Existing volume names retain their old prefix; that
does not mean the old repository is needed. Do not delete these volumes.
`docker compose stop` preserves the database. For fresh-clone managed volumes,
`docker compose down -v` deletes data.

## Other projects

- [MCP](https://github.com/kube-pilot-ai/MCP): independent Kubernetes tool server.
- [RAG](https://github.com/kube-pilot-ai/RAG): documents and Qdrant setup. Ingestion is not implemented yet.


## RAG retrieval

Start the independent RAG API from the RAG checkout before submitting questions:

```powershell
cd C:\Users\kusum\Documents\Final_Project\RAG
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8003
```

The chat .env uses RAG_SERVER_URL=http://127.0.0.1:8003 and RAG_AUTH_TOKEN,
which must match the RAG service token. Values have been configured locally.
Optional RAG_TOP_K (default 4) and RAG_MIN_SCORE (default 0.3) control retrieval.
Similarity scores are ranking signals, not calibrated confidence probabilities.

Graph flow: START -> Supervisor Agent -> selected specialist -> Supervisor Agent.
The cluster specialist can call MCP tools; troubleshooting delegation goes through
RAG retrieval before the troubleshooting specialist. The supervisor reviews the
worker result, requests a focused follow-up if necessary, or finishes.
For troubleshooting, the retrieval query includes the latest question, a short
previous question for follow-ups, the cluster agent's evidence handoff, and
bounded fresh tool observations. Explicit ImagePullBackOff or
OOMKilled terms activate matching symptom metadata filters. Multiple detected
symptoms use an OR filter. Qdrant applies the similarity threshold and top-k
limit. The RAG service enforces a context token budget and returns no vectors.

The Troubleshooting Agent receives the retrieved text, scores, and source metadata as
untrusted reference excerpts. Answers should cite relevant [R1], [R2], etc.
and include source locations. Retrieved guidance does not prove a condition in
the live cluster; MCP tools still provide cluster evidence. Each retrieval
replaces graph rag_matches; excerpts are not appended as duplicate chat messages.
If no chunks match, the model is told no documentation matched. If RAG is
unavailable, the request fails visibly rather than pretending to have retrieved
references. Existing conversations and MCP tools remain supported.

The app and RAG projects communicate over HTTP, never cross-repository imports.
This connection also applies to the command-line main.py flow. No UI changes
are needed; citations are included in the answer text and selected chunk
sections/scores are printed in the chat server terminal.


## Two specialist agents

`agents.py` defines independent role prompts and a validated request-routing
schema. Both agents use gpt-5.6-luna with separate instructions and invocations:

1. **Cluster Information Agent** selects read-only MCP tools, gathers current
   facts, and answers simple listings/status questions. For an investigation,
   it supplies an evidence handoff instead of composing a remediation plan.
2. **Troubleshooting Agent** receives that evidence and RAG source excerpts,
   distinguishes confirmed observations from suspected causes, and recommends
   prioritized fixes, manual commands/manifest examples, verification, and
   rollback considerations. It has no tools bound and cannot apply changes.

The Supervisor Agent classifies and delegates the request with a structured output. Lists,
status, and factual resource inspection use the information route. Failures,
root-cause questions, fixes, and remediation checks use the troubleshooting
route. Follow-up context is included in routing. Hypothetical questions can
reach the troubleshooter without cluster reads when the user requests that.

The cluster agent can perform up to six tool rounds per request. The
40-step graph limit bounds the supervisor-led run. Names of agents, tool calls,
and the handoff are printed to the server terminal. The final agent response
is returned through the existing chat API; PostgreSQL retains messages and
additional agent state. Existing completed conversations can continue normally.
The existing llm/tools/retrieve node identifiers are retained for checkpoints.

Example information request: `List pods in namespace rag-demo`.
Example troubleshooting request: `Investigate memory-demo in namespace rag-demo
and recommend remediation using the troubleshooting guides. Do not change the cluster.`

Restart the chat app to load this workflow. MCP and RAG remain independent
services; no new Kubernetes write operations or document changes were added.
No re-ingestion is needed solely for this agent implementation. Re-ingest any
previously edited troubleshooting guides if that has not already been done.


## Supervisor Agent

The application now has three agent roles in agents.py: Supervisor Agent,
Cluster Information Agent, and Troubleshooting Agent. The supervisor has its
own prompt and structured decision (intent, next_agent, task, reason). It does
not merely classify the first message: it reviews each completed worker response
and chooses the next specialist or finish. Specialist results, rather than an
extra rewritten supervisor answer, are returned to the user.

- Information request: supervisor -> cluster specialist/MCP -> supervisor -> finish.
- Live investigation: supervisor -> cluster specialist/MCP -> supervisor -> RAG ->
  troubleshooting specialist -> supervisor -> finish.
- Hypothetical/document-only request: supervisor may go directly to RAG and the
  troubleshooting specialist, respecting requests not to query the cluster.

A supervisor assignment is supplied to each delegated specialist. Coordination
state (next agent, task, reason, delegation count, request ID) is checkpointed
alongside the conversation. It resets for each new user request. Up to four
specialist delegations and six total cluster tool rounds are allowed per request,
with an overall limit of 40 graph steps. The final allowed investigation handoff
goes to troubleshooting so the user receives recommendations rather than an
endless collection loop. Legacy graph node names are retained for resuming older
checkpoints.

The supervisor has no cluster modification tools. The troubleshooting specialist
still recommends fixes only. No new external service, dependencies, database
migration, or document re-ingestion is required. Restart the chat app to load
this orchestration. The server terminal shows supervisor choices and reasons.
