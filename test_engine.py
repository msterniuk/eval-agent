"""
test_engine.py — Test the deployed Agent Engine directly.
Tools (including email) run INSIDE Agent Engine with the correct service account.

Usage:
    python test_engine.py
"""
import uuid
import vertexai
from vertexai import agent_engines
from google.adk.sessions import VertexAiSessionService
import config

vertexai.init(project=config.PROJECT_ID, location=config.REGION)


def query_engine(remote_app, message: str, user_id: str, session_id: str):
    """Send one message to the deployed Agent Engine and print streamed response."""
    print(f"\nYou: {message}")
    print("Agent: ", end="", flush=True)

    for event in remote_app.stream_query(
        user_id=user_id,
        session_id=session_id,
        message=message,
    ):
        # Agent Engine returns events as dicts
        if isinstance(event, dict):
            parts = (event.get("content") or {}).get("parts", [])
            for part in parts:
                text = part.get("text", "") if isinstance(part, dict) else ""
                if text:
                    print(text, end="", flush=True)
        elif hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text, end="", flush=True)

    print()  # newline after response


def main():
    resource = config.AGENT_ENGINE_RESOURCE
    if not resource:
        print("ERROR: AGENT_ENGINE_RESOURCE not set in .env")
        return

    print("=" * 60)
    print(f"  Agent Engine Test")
    print(f"  Resource: {resource}")
    print("=" * 60)

    import asyncio
    session_svc = VertexAiSessionService(project=config.PROJECT_ID, location=config.REGION)
    uid = f"user_{uuid.uuid4().hex[:8]}"
    session = asyncio.run(session_svc.create_session(app_name=resource, user_id=uid))
    sid = session.id
    print(f"  Session : {sid}")
    print(f"  User    : {uid}")

    remote_app = agent_engines.get(resource)

    query_engine(remote_app, "Hi", uid, sid)

    print("\nType 'exit' to quit.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if msg.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not msg:
            continue
        query_engine(remote_app, msg, uid, sid)


if __name__ == "__main__":
    main()
