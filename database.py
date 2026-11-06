from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def open_database(url):
    pool = ConnectionPool(
        conninfo=url, min_size=1, max_size=8, timeout=10,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    )
    try:
        pool.wait(timeout=15)
        with pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id UUID NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'error')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""CREATE INDEX IF NOT EXISTS chat_messages_conversation_idx
                            ON chat_messages(conversation_id, id)""")
        return pool
    except Exception:
        pool.close()
        raise


def save_message(conn, conversation_id, role, content):
    with conn.transaction():
        conn.execute(
            "INSERT INTO chat_messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )
        conn.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (conversation_id,))
