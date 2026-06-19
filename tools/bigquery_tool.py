# tools/bigquery_tool.py
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig

try:
    from .. import config
except ImportError:
    import config


def get_bigquery_toolset() -> BigQueryToolset:
    """
    Returns a BigQueryToolset for the ADK agent.
    compute_project_id ensures queries are billed/executed against the correct
    BQ project both locally and when deployed to Vertex AI Agent Engine.
    _use_invocation_cache is patched onto the instance because Agent Engine's
    runtime checks for this attribute before loading tools; if it is missing
    the entire toolset is silently dropped and the LLM hallucinates answers.
    """
    toolset = BigQueryToolset(
        bigquery_tool_config=BigQueryToolConfig(
            compute_project_id=config.BQ_PROJECT_ID,
        ),
    )
    # Patch missing attribute expected by Agent Engine runtime
    if not hasattr(toolset, "_use_invocation_cache"):
        toolset._use_invocation_cache = False
    return toolset