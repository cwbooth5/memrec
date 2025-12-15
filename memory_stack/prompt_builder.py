from __future__ import annotations
from typing import List
from .manager import MemoryContext


def build_prompt(ctx: MemoryContext, user_message: str) -> str:
    # Profile info
    profile_lines = [
        f"User ID: {ctx.user_profile.user_id}",
    ]
    if ctx.user_profile.name:
        profile_lines.append(f"Name: {ctx.user_profile.name}")
    if ctx.user_profile.style:
        profile_lines.append(f"Preferred style: {ctx.user_profile.style}")
    if ctx.user_profile.preferences:
        profile_lines.append(f"Preferences: {ctx.user_profile.preferences}")

    profile_section = "\n".join(profile_lines)

    workspace_section = ""
    if ctx.workspace_profile:
        workspace_section = (
            "Workspace:\n"
            f"- ID: {ctx.workspace_profile.workspace_id}\n"
            f"- Root: {ctx.workspace_profile.repo_root}\n"
            f"- Primary language: {ctx.workspace_profile.primary_language}\n"
        )

    docs_section = ""
    if ctx.retrieved_chunks:
        lines: List[str] = ["Relevant design docs and notes:"]
        for ch in ctx.retrieved_chunks:
            src = ch.metadata.get("source", ch.metadata.get("doc_name", "unknown"))
            lines.append(f"[{src}] {ch.text}")
        docs_section = "\n".join(lines)

    episodes_section = ""
    if ctx.recent_episodes:
        lines = ["Recent project episodes:"]
        for ep in ctx.recent_episodes:
            lines.append(f"- {ep.timestamp.date()}: {ep.title} — {ep.summary}")
        episodes_section = "\n".join(lines)

    convo_section = "Conversation summary:\n"
    convo_section += (ctx.convo_summary or "(no summary yet)") + "\n\n"
    if ctx.recent_turns:
        convo_section += "Recent turns:\n"
        for m in ctx.recent_turns:
            convo_section += f"{m['role']}: {m['content']}\n"

    prompt = f"""You are a helpful coding/design assistant.

{profile_section}

{workspace_section}

{docs_section}

{episodes_section}

{convo_section}

User: {user_message}
Assistant:"""

    return prompt
