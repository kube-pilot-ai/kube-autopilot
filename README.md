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
