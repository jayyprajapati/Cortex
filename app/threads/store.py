"""SQLite-backed persistence for conversational threads.

One file at $CORTEX_DATA_DIR/threads.db (defaults to ./data/threads.db).
Schema bootstrapped on first connection. Writes serialized through a module-level
lock since stdlib sqlite3 is not safe for concurrent writes from multiple threads.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
_DATA_DIR = os.getenv("CORTEX_DATA_DIR") or _DEFAULT_DATA_DIR
_DB_PATH = os.path.join(_DATA_DIR, "threads.db")

_write_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    doc_ids_json TEXT NOT NULL DEFAULT '[]',
    title TEXT,
    summary TEXT,
    summary_up_to_message_idx INTEGER NOT NULL DEFAULT 0,
    clarification_pending INTEGER NOT NULL DEFAULT 0,
    clarification_context TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_threads_user
    ON threads(user_id, app_name, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT,
    grounded INTEGER,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread
    ON messages(thread_id, id);
"""

# Migrations applied once against existing databases that pre-date new columns.
_MIGRATIONS = [
    "ALTER TABLE threads ADD COLUMN clarification_pending INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE threads ADD COLUMN clarification_context TEXT",
]


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        os.makedirs(_DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(_SCHEMA)
        # Apply additive migrations idempotently (ignore "duplicate column" errors).
        for stmt in _MIGRATIONS:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore.
        _conn = conn
        logger.info("threads sqlite ready at %s", _DB_PATH)
        return _conn


def _now() -> int:
    return int(time.time())


def _row_to_thread(row: sqlite3.Row) -> Dict[str, Any]:
    keys = row.keys()
    return {
        "id": row["id"],
        "app_name": row["app_name"],
        "user_id": row["user_id"],
        "doc_ids": json.loads(row["doc_ids_json"] or "[]"),
        "title": row["title"],
        "summary": row["summary"],
        "summary_up_to_message_idx": row["summary_up_to_message_idx"],
        "clarification_pending": bool(row["clarification_pending"]) if "clarification_pending" in keys else False,
        "clarification_context": row["clarification_context"] if "clarification_context" in keys else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_message(row: sqlite3.Row) -> Dict[str, Any]:
    citations = None
    if row["citations_json"]:
        try:
            citations = json.loads(row["citations_json"])
        except json.JSONDecodeError:
            citations = None
    grounded: Optional[bool] = None
    if row["grounded"] is not None:
        grounded = bool(row["grounded"])
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "role": row["role"],
        "content": row["content"],
        "citations": citations,
        "grounded": grounded,
        "created_at": row["created_at"],
    }


def create_thread(
    app_name: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> str:
    thread_id = uuid.uuid4().hex
    now = _now()
    with _write_lock:
        _get_conn().execute(
            "INSERT INTO threads (id, app_name, user_id, doc_ids_json, title, "
            "summary_up_to_message_idx, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (
                thread_id,
                app_name,
                user_id,
                json.dumps(list(doc_ids or [])),
                title,
                now,
                now,
            ),
        )
    return thread_id


def get_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    row = _get_conn().execute(
        "SELECT * FROM threads WHERE id = ?", (thread_id,)
    ).fetchone()
    return _row_to_thread(row) if row else None


def list_threads(
    user_id: str,
    app_name: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    rows = _get_conn().execute(
        "SELECT * FROM threads WHERE user_id = ? AND app_name = ? "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, app_name, limit),
    ).fetchall()
    threads = [_row_to_thread(r) for r in rows]
    # Attach message_count cheaply
    if threads:
        ids = [t["id"] for t in threads]
        placeholders = ",".join(["?"] * len(ids))
        counts = _get_conn().execute(
            f"SELECT thread_id, COUNT(*) as c FROM messages "
            f"WHERE thread_id IN ({placeholders}) GROUP BY thread_id",
            ids,
        ).fetchall()
        count_map = {r["thread_id"]: r["c"] for r in counts}
        for t in threads:
            t["message_count"] = count_map.get(t["id"], 0)
    return threads


def get_thread_with_messages(thread_id: str) -> Optional[Dict[str, Any]]:
    thread = get_thread(thread_id)
    if not thread:
        return None
    rows = _get_conn().execute(
        "SELECT * FROM messages WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    thread["messages"] = [_row_to_message(r) for r in rows]
    return thread


def append_message(
    thread_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    grounded: Optional[bool] = None,
) -> int:
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role!r}")
    now = _now()
    with _write_lock:
        conn = _get_conn()
        cursor = conn.execute(
            "INSERT INTO messages (thread_id, role, content, citations_json, grounded, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                role,
                content,
                json.dumps(citations) if citations is not None else None,
                None if grounded is None else (1 if grounded else 0),
                now,
            ),
        )
        message_id = cursor.lastrowid
        conn.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, thread_id),
        )
    return message_id


def get_recent_messages(thread_id: str, n: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent n messages in chronological order (oldest first)."""
    rows = _get_conn().execute(
        "SELECT * FROM messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
        (thread_id, n),
    ).fetchall()
    return [_row_to_message(r) for r in reversed(rows)]


def count_messages(thread_id: str) -> int:
    row = _get_conn().execute(
        "SELECT COUNT(*) AS c FROM messages WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    return int(row["c"]) if row else 0


def update_summary(thread_id: str, summary: str, up_to_message_idx: int) -> None:
    now = _now()
    with _write_lock:
        _get_conn().execute(
            "UPDATE threads SET summary = ?, summary_up_to_message_idx = ?, updated_at = ? "
            "WHERE id = ?",
            (summary, up_to_message_idx, now, thread_id),
        )


def update_title(thread_id: str, title: str) -> None:
    now = _now()
    with _write_lock:
        _get_conn().execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, thread_id),
        )


# Columns that update_thread_meta is allowed to touch (allowlist guards against SQL injection).
_UPDATABLE_META_COLUMNS = frozenset({
    "clarification_pending",
    "clarification_context",
    "title",
    "summary",
    "summary_up_to_message_idx",
})


def update_thread_meta(thread_id: str, meta: Dict[str, Any]) -> None:
    """Update arbitrary metadata fields on a thread record.

    Only keys present in ``_UPDATABLE_META_COLUMNS`` are written; unknown keys
    are silently ignored so callers don't need to filter themselves.
    """
    allowed = {k: v for k, v in meta.items() if k in _UPDATABLE_META_COLUMNS}
    if not allowed:
        return
    now = _now()
    set_clause = ", ".join(f"{col} = ?" for col in allowed)
    values = list(allowed.values()) + [now, thread_id]
    with _write_lock:
        _get_conn().execute(
            f"UPDATE threads SET {set_clause}, updated_at = ? WHERE id = ?",
            values,
        )


def delete_thread(thread_id: str) -> bool:
    with _write_lock:
        conn = _get_conn()
        cursor = conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        return cursor.rowcount > 0
