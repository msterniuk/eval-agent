# deploy.py — Deploy agent to Vertex AI Agent Engine (Google ADK)
import vertexai
from vertexai.agent_engines import AdkApp
from google.adk.sessions import VertexAiSessionService
import config
from agent import root_agent

# Compute SA of ca-sbox-es-aiml-demo-444 — has roles/composer.user on Composer
_SERVICE_ACCOUNT = "740246139241-compute@developer.gserviceaccount.com"


def deploy() -> str:
    """
    Package and deploy the Golden Stays agent to Vertex AI Agent Engine.
    Prints the resource name on success — add it to AGENT_ENGINE_RESOURCE in .env.
    """
    print("=" * 70)
    print("  Deploying Golden Stays agent to allow for basic testing  ")
    print("=" * 70)
    print(f"  Deploy Project : {config.PROJECT_ID}")
    print(f"  Region         : {config.REGION}")
    print(f"  Staging Bucket : {config.STAGING_BUCKET}")
    print(f"  BQ Project     : {config.BQ_PROJECT_ID}")
    print(f"  BQ Dataset     : {config.BQ_DATASET}")
    print(f"  Model          : {config.GEMINI_MODEL}")
    print(f"  Service Acct   : {_SERVICE_ACCOUNT}")
    print("=" * 70)

    client = vertexai.Client(project=config.PROJECT_ID, location=config.REGION)

    app = AdkApp(
        agent=root_agent,
        enable_tracing=False,
        session_service_builder=lambda p=config.PROJECT_ID, r=config.REGION: VertexAiSessionService(
            project=p, location=r
        ),
    )

    remote_app = client.agent_engines.create(
        agent=app,
        config={
            "display_name": "Golden_Stays_Agent_V0.0",
            "description": "Analyzes hotel stay data",
            "requirements": [
                "google-cloud-aiplatform[adk,reasoningengine]==1.146.0",
                "google-cloud-bigquery>=3.25.0",
                "python-dotenv>=1.0.0",
                "requests>=2.28.0",
            ],
            "extra_packages": ["tools", "config.py"],
            "staging_bucket": config.STAGING_BUCKET,
            "service_account": _SERVICE_ACCOUNT,
        },
    )

    resource_name = remote_app.api_resource.name
    print("\n  Deployment successful!")
    print(f"\nResource name:\n  {resource_name}")
    print("\n  Next step — add this to your .env file:")
    print(f"  AGENT_ENGINE_RESOURCE={resource_name}")
    print("\nThen run: python session_runner.py")

    return resource_name


if __name__ == "__main__":
    deploy()
