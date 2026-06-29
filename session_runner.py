# session_runner.py — Test the agent locally or with Agent Engine

#filtering out some annoying warnings, comment this out for testing

import warnings
import logging

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)



import asyncio
import uuid
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService, InMemorySessionService
from google.genai.types import Content, Part
import config
from agent import root_agent


async def create_runner(
    use_agent_engine: bool = False,
    agent_engine_resource: str | None = None,
) -> tuple[Runner, str, str]:
    """
    Initialize a runner for the golden stays analysis agent.

    Args:
        use_agent_engine: If True, use VertexAiSessionService (requires deployed agent).
                         If False, use InMemorySessionService (local testing only).
        agent_engine_resource: Full Agent Engine resource name (required if use_agent_engine=True).

    Returns:
        (runner, user_id, session_id)
    """
    if use_agent_engine:
        resource = agent_engine_resource or config.AGENT_ENGINE_RESOURCE
        if not resource:
            raise ValueError(
                "AGENT_ENGINE_RESOURCE is not set. Run deploy.py first, "
                "then add the resource name to your .env file."
            )
        session_service = VertexAiSessionService(
            project=config.PROJECT_ID,
            location=config.REGION,
        )
        app_name = resource  # Agent Engine requires the full resource name as app_name
    else:
        # For local testing, use in-memory sessions
        session_service = InMemorySessionService()
        app_name = "goldenStaysAgent"

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )

    return runner, user_id, session.id


async def chat(runner: Runner, user_id: str, session_id: str, message: str) -> str:
    """Send one turn to the agent and return the final text response."""
    content = Content(role="user", parts=[Part(text=message)])
    response_parts: list[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_parts.append(part.text)

    return "".join(response_parts) or "[No response]"


async def main():
    # Always run locally with InMemorySessionService for local testing
    print("Initializing golden stays agent for local testing...")
    runner, user_id, session_id = await create_runner(use_agent_engine=False)
    print(f"✓ Agent ready [local]. (user_id={user_id}, session_id={session_id})")
    print("\nExample queries:")
    print("  - 'Find all the stays associated with a given Booking Conference Number.'")
    print("  - 'Give me 5 examples of hotels that have X IATA Number and use the GBP currency.'")
    print("  - 'Find the channel code that generated the highest average total room revenue per guest per night.'")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue
        response = await chat(runner, user_id, session_id, user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())