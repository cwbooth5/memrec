import os
from datetime import datetime

import dspy

from memory_stack.db import SqliteDB
from memory_stack.stores import ProfileStore, EpisodicLog, ConversationStore, Episode
from memory_stack.vector_store import VectorStore, MemoryChunk
from memory_stack.manager import MemoryManager
from memory_stack.prompt_builder import build_prompt


def setup_lm():
    """
    Configure DSPy with your LLM.

    Example: OpenAI-compatible local server at http://localhost:1234/v1
    running 'gpt-oss-20b' or whatever model name you've set up.
    """
    lm = dspy.LM(
        "openai/mistralai/ministral-3-3b",
        api_base="http://localhost:1234/v1",
        api_key=os.getenv("OPENAI_API_KEY", "local"),
    )
    dspy.configure(lm=lm)
    return lm


def main():
    lm = setup_lm()

    db = SqliteDB("memory.sqlite")
    profiles = ProfileStore(db)
    episodes = EpisodicLog(db)
    conv = ConversationStore(db)
    vectors = VectorStore(persist_dir="chroma_db")

    memory = MemoryManager(
        profiles=profiles,
        conv=conv,
        episodes=episodes,
        vectors=vectors,
    )

    user_id = "bill"
    workspace_id = "snapsafe-ios"
    thread_id = "thread-1"

    prof = profiles.get_user_profile(user_id)
    prof.name = "Bill"
    prof.style = "detailed"
    prof.preferences["code_style"] = "show runnable examples"
    profiles.upsert_user_profile(prof)

    wprof = profiles.get_workspace_profile(workspace_id)
    wprof.primary_language = "Swift"
    wprof.repo_root = "/Users/bill/src/SnapSafe"
    profiles.upsert_workspace_profile(wprof)

    vectors.add([
        MemoryChunk(
            id="design:snap_safe:1",
            text=(
                "SnapSafe is a secure camera/photo vault app. "
                "All photos must be stored encrypted on device-only storage. "
                "We never upload images to any cloud service. "
                "We use the Secure Enclave for key protection and biometric unlock."
            ),
            metadata={"workspace_id": workspace_id, "source": "SnapSafe design doc"},
        )
    ])

    # Seed one episode
    episodes.append(
        Episode(
            id="ep-1",
            timestamp=datetime.utcnow(),
            user_id=user_id,
            workspace_id=workspace_id,
            title="Decided on local-only encrypted storage",
            summary="We decided that SnapSafe will never sync photos to cloud; "
                    "encryption keys are protected in Secure Enclave with biometrics.",
            tags=["security", "storage"],
        )
    )

    # Simulate a user query
    user_message = " describe our most recent episode"
    user_message = "describe which memories you have and which types they are"

    # Read memory context
    ctx = memory.read_context(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
        query=user_message,
    )

    # Build prompt
    prompt = build_prompt(ctx, user_message)

    # Call the LM via DSPy
    completions = dspy.settings.lm(prompt, temperature=0.3)
    assistant_reply = completions[0]

    print("=== PROMPT ===")
    print(prompt)
    print("\n=== ASSISTANT REPLY ===")
    print(assistant_reply)

    # Update memory from this turn
    memory.update_from_turn(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
        user_msg=user_message,
        assistant_msg=assistant_reply,
    )


if __name__ == "__main__":
    main()
