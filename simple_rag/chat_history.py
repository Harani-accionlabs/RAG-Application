import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict

DB_PATH = Path(tempfile.gettempdir()) / "marginalia_chat_history.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def save_message(session_id: str, role: str, content: str) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def load_messages(session_id: str) -> List[Dict[str, str]]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]
    finally:
        conn.close()


def clear_session(session_id: str) -> None:
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()