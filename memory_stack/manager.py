from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .stores import (
    UserProfile,
    WorkspaceProfile,
    Episode,
    ProfileStore,
    EpisodicLog,
    ConversationStore,
)
from .vector_store import MemoryChunk, VectorStore


@dataclass
class MemoryContext:
    user_profile: UserProfile
    workspace_profile: Optional[WorkspaceProfile]
    convo_summary: str
    recent_turns: List[dict]
    retrieved_chunks: List[MemoryChunk]
    recent_episodes: List[Episode]


class MemoryManager:
    def __init__(
        self,
        profiles: ProfileStore,
        conv: ConversationStore,
        episodes: EpisodicLog,
        vectors: VectorStore,
    ):
        self.profiles = profiles
        self.conv = conv
        self.episodes = episodes
        self.vectors = vectors

    def read_context(
        self,
        user_id: str,
        workspace_id: Optional[str],
        thread_id: str,
        query: str,
    ) -> MemoryContext:
        user_profile = self.profiles.get_user_profile(user_id)
        workspace_profile = (
            self.profiles.get_workspace_profile(workspace_id)
            if workspace_id
            else None
        )

        convo_summary = self.conv.get_summary(thread_id)
        recent_turns = self.conv.get_recent(thread_id, n_turns=8)

        filters = {}
        if workspace_id:
            filters["workspace_id"] = workspace_id

        retrieved_chunks = self.vectors.search(query=query, k=5, filters=filters)

        recent_episodes = self.episodes.list_recent(user_id, workspace_id, limit=5)

        return MemoryContext(
            user_profile=user_profile,
            workspace_profile=workspace_profile,
            convo_summary=convo_summary,
            recent_turns=recent_turns,
            retrieved_chunks=retrieved_chunks,
            recent_episodes=recent_episodes,
        )

    def update_from_turn(
        self,
        user_id: str,
        workspace_id: Optional[str],
        thread_id: str,
        user_msg: str,
        assistant_msg: str,
    ):
        self.conv.add_turn(thread_id, user_msg, assistant_msg)
