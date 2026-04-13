from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .db import SqliteDB


@dataclass
class UserProfile:
    user_id: str
    name: Optional[str] = None
    timezone: Optional[str] = None
    style: Optional[str] = None
    preferences: dict = field(default_factory=dict)


@dataclass
class WorkspaceProfile:
    workspace_id: str
    primary_language: Optional[str] = None
    repo_root: Optional[str] = None
    tools: dict = field(default_factory=dict)


@dataclass
class Episode:
    id: str
    timestamp: datetime
    user_id: str
    workspace_id: Optional[str]
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)


class ProfileStore:
    def __init__(self, db: SqliteDB):
        self.db = db

    def get_user_profile(self, user_id: str) -> UserProfile:
        rows = self.db.query(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        if not rows:
            return UserProfile(user_id=user_id)

        row = rows[0]
        return UserProfile(
            user_id=row["user_id"],
            name=row["name"],
            timezone=row["timezone"],
            style=row["style"],
            preferences=self.db.loads(row["preferences_json"]),
        )

    def upsert_user_profile(self, profile: UserProfile):
        self.db.execute(
            """
            INSERT INTO user_profiles (user_id, name, timezone, style, preferences_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name=excluded.name,
                timezone=excluded.timezone,
                style=excluded.style,
                preferences_json=excluded.preferences_json
            """,
            (
                profile.user_id,
                profile.name,
                profile.timezone,
                profile.style,
                self.db.dumps(profile.preferences),
            ),
        )

    def get_workspace_profile(self, workspace_id: str) -> WorkspaceProfile:
        rows = self.db.query(
            "SELECT * FROM workspace_profiles WHERE workspace_id = ?",
            (workspace_id,),
        )
        if not rows:
            return WorkspaceProfile(workspace_id=workspace_id)

        row = rows[0]
        return WorkspaceProfile(
            workspace_id=row["workspace_id"],
            primary_language=row["primary_language"],
            repo_root=row["repo_root"],
            tools=self.db.loads(row["tools_json"]),
        )

    def upsert_workspace_profile(self, profile: WorkspaceProfile):
        self.db.execute(
            """
            INSERT INTO workspace_profiles (workspace_id, primary_language, repo_root, tools_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                primary_language=excluded.primary_language,
                repo_root=excluded.repo_root,
                tools_json=excluded.tools_json
            """,
            (
                profile.workspace_id,
                profile.primary_language,
                profile.repo_root,
                self.db.dumps(profile.tools),
            ),
        )

# episodes

class EpisodicLog:
    def __init__(self, db: SqliteDB):
        self.db = db

    def append(self, ep: Episode):
        self.db.execute(
            """
            INSERT OR REPLACE INTO episodes
            (id, user_id, workspace_id, timestamp, title, summary, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep.id,
                ep.user_id,
                ep.workspace_id,
                ep.timestamp.isoformat(),
                ep.title,
                ep.summary,
                self.db.dumps(ep.tags),
            ),
        )

    def list_recent(
        self, user_id: str, workspace_id: Optional[str], limit: int = 20
    ) -> List[Episode]:
        if workspace_id:
            rows = self.db.query(
                """
                SELECT * FROM episodes
                WHERE user_id = ? AND workspace_id = ?
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
                """,
                (user_id, workspace_id, limit),
            )
        else:
            rows = self.db.query(
                """
                SELECT * FROM episodes
                WHERE user_id = ?
                ORDER BY datetime(timestamp) DESC
                LIMIT ?
                """,
                (user_id, limit),
            )

        episodes: List[Episode] = []
        for r in rows:
            episodes.append(
                Episode(
                    id=r["id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    user_id=r["user_id"],
                    workspace_id=r["workspace_id"],
                    title=r["title"],
                    summary=r["summary"],
                    tags=self.db.loads(r["tags_json"]),
                )
            )
        return episodes

# convos

class ConversationStore:
    def __init__(self, db: SqliteDB):
        self.db = db

    def add_turn(
        self,
        thread_id: str,
        user_msg: str,
        assistant_msg: Optional[str] = None,
        ts: Optional[datetime] = None,
    ):
        ts_str = (ts or datetime.utcnow()).isoformat()
        self.db.execute(
            "INSERT INTO messages (thread_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (thread_id, "user", user_msg, ts_str),
        )
        if assistant_msg is not None:
            self.db.execute(
                "INSERT INTO messages (thread_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (thread_id, "assistant", assistant_msg, ts_str),
            )

    def get_recent(self, thread_id: str, n_turns: int = 8) -> list[dict]:
        # each "turn" is user + assistant; we approximate by last 2*n rows
        rows = self.db.query(
            """
            SELECT role, content, ts
            FROM messages
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (thread_id, n_turns * 2),
        )
        # return in chronological order
        rows = list(rows)[::-1]
        return [
            {"role": r["role"], "content": r["content"], "ts": r["ts"]}
            for r in rows
        ]

    def get_summary(self, thread_id: str) -> str:
        rows = self.db.query(
            "SELECT summary FROM convo_summaries WHERE thread_id = ?",
            (thread_id,),
        )
        return rows[0]["summary"] if rows else ""

    def set_summary(self, thread_id: str, summary: str):
        self.db.execute(
            """
            INSERT INTO convo_summaries (thread_id, summary)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET summary=excluded.summary
            """,
            (thread_id, summary),
        )

    def list_threads(self, limit: int = 50) -> list[dict]:
        """Return all thread IDs sorted by most-recently-updated, with message counts."""
        rows = self.db.query(
            """
            SELECT thread_id, MAX(ts) AS last_ts, COUNT(*) AS msg_count
            FROM messages
            GROUP BY thread_id
            ORDER BY MAX(ts) DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {"thread_id": r["thread_id"], "last_ts": r["last_ts"], "msg_count": r["msg_count"]}
            for r in rows
        ]

    def get_all(self, thread_id: str) -> list[dict]:
        """Return every message in a thread in chronological order, including events."""
        import json as _json

        rows = self.db.query(
            "SELECT role, content, ts, events_json FROM messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        )
        result = []
        for r in rows:
            raw = r["events_json"] if r["events_json"] else "[]"
            try:
                events = _json.loads(raw)
            except Exception:
                events = []
            result.append(
                {"role": r["role"], "content": r["content"], "ts": r["ts"], "events": events}
            )
        return result

    def set_events(self, thread_id: str, events_json: str) -> None:
        """Attach serialised events to the most recent assistant message in a thread."""
        self.db.execute(
            """
            UPDATE messages SET events_json = ?
            WHERE id = (
                SELECT id FROM messages
                WHERE thread_id = ? AND role = 'assistant'
                ORDER BY id DESC LIMIT 1
            )
            """,
            (events_json, thread_id),
        )

    def search_messages(self, query: str, limit: int = 50) -> list[dict]:
        """Full-text LIKE search across all conversation messages.

        Returns rows sorted newest-first, each with thread_id, thread_title,
        role, content (full), and ts.  The caller is responsible for extracting
        a display excerpt.
        """
        rows = self.db.query(
            """
            SELECT m.thread_id, m.role, m.content, m.ts,
                   COALESCE(t.title, m.thread_id) AS thread_title
            FROM messages m
            LEFT JOIN thread_titles t ON m.thread_id = t.thread_id
            WHERE m.content LIKE ? AND m.role IN ('user', 'assistant')
            ORDER BY m.ts DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        )
        return [dict(r) for r in rows]

    def get_title(self, thread_id: str) -> Optional[str]:
        rows = self.db.query(
            "SELECT title FROM thread_titles WHERE thread_id = ?",
            (thread_id,),
        )
        return rows[0]["title"] if rows else None

    def set_title(self, thread_id: str, title: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
