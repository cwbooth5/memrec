# memory_stack/db.py
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


class SqliteDB:
    def __init__(self, path: str = "memory.sqlite"):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()

        # User profiles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            timezone TEXT,
            style TEXT,
            preferences_json TEXT
        )
        """)

        # Workspace profiles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS workspace_profiles (
            workspace_id TEXT PRIMARY KEY,
            primary_language TEXT,
            repo_root TEXT,
            tools_json TEXT
        )
        """)

        # Episodes (log)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            workspace_id TEXT,
            timestamp TEXT,
            title TEXT,
            summary TEXT,
            tags_json TEXT
        )
        """)

        # Conversation messages
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            ts TEXT
        )
        """)

        # Conversation summaries
        cur.execute("""
        CREATE TABLE IF NOT EXISTS convo_summaries (
            thread_id TEXT PRIMARY KEY,
            summary TEXT
        )
        """)

        # Thread titles (LLM-generated short labels)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
        """)

        self.conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: Iterable[Any] = ()):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()

    @staticmethod
    def dumps(obj: Any) -> str:
        return json.dumps(obj or {})

    @staticmethod
    def loads(s: str | None) -> Any:
        return json.loads(s) if s else {}
