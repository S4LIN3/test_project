from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def insert_chat(self, role: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_history (role, message, created_at) VALUES (?, ?, ?)",
                (role, message, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_recent_chat(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, message, created_at FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"role": row[0], "message": row[1], "created_at": row[2]} for row in rows
        ]

    def insert_run(self, module_name: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analysis_runs (module_name, summary, created_at) VALUES (?, ?, ?)",
                (module_name, summary, datetime.utcnow().isoformat()),
            )
            conn.commit()
