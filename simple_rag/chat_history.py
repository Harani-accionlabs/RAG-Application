import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict

DB_PATH = Path(tempfile.gettempdir()) / "marginalia_chat_history.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def create_conversation(conversation_id: str, session_id: str, label: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, session_id, label) VALUES (?, ?, ?)",
            (conversation_id, session_id, label),
        )
        conn.commit()
    finally:
        conn.close()


def list_conversations(session_id: str) -> List[Dict[str, str]]:
    """Returns this session's conversations, most recent first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, label, created_at FROM conversations WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [{"id": cid, "label": label, "created_at": created_at} for cid, label, created_at in rows]
    finally:
        conn.close()


def save_message(conversation_id: str, role: str, content: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def load_messages(conversation_id: str) -> List[Dict[str, str]]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]
    finally:
        conn.close()


def clear_conversation(conversation_id: str) -> None:
    """Clears messages within one conversation, keeping the conversation itself
    (so it stays listed, just empty) — used by the 'Clear Chat' button."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()