# agent.py — Financial Analyst ADK Agent (ENHANCED)
import os
from google.adk.agents import Agent

# Support both package mode (adk web) and direct execution
try:
    from .tools import get_bigquery_toolset
    from . import config
except ImportError:
    from tools import get_bigquery_toolset
    import config

# Force Gemini to use the correct GCP project — required inside Agent Engine
# because .env is not loaded in the container and ADK reads GOOGLE_CLOUD_PROJECT
os.environ["GOOGLE_CLOUD_PROJECT"]      = config.PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"]     = config.REGION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

SYSTEM_PROMPT = """\
You are an AI assistant that answers extremely basic, factual questions about
the provided data sources listed below. Be polite, and refuse to answer if the question
is not something that can be answered with a cold, hard fact that you can cite. 

═══════════════════════════════════════════════════════════════════════
DATA SOURCE
═══════════════════════════════════════════════════════════════════════

Project: `{project}`  |  Dataset: `{dataset}`
Always use fully-qualified table names in every query.
 
═══════════════════════════════════════════════════════════════════════
CORE TABLES
═══════════════════════════════════════════════════════════════════════

1. `{project}.{dataset}.BKG_STY_DTL - hotel booking information on a STAY level / grain
2. `{project}.{dataset}.BKG_STY_LYTY_DTL - hotel booking information on a STAY level / grain re: loyalty
3. `{project}.{dataset}.GOLDNSTY_CONFIG - configuration info related to hotel bookings
4. `{project}.{dataset}.HTL_SRC_LKP - hotel booking information categorized by hotel / hotel level grain
5. `{project}.{dataset}.BKG_STY_DTL - hotel booking information on a DAILY level / grain

""".format(
    project=config.BQ_PROJECT_ID,
    dataset=config.BQ_DATASET,
)

root_agent = Agent(
    name="financial_analyst",
    model=config.GEMINI_MODEL,
    description=(
        "Analyzes accounting journal transactions from BigQuery with strict base-table "
        "hierarchy, case-sensitive LEDGER=ACTUAL filtering by default, property code matching, "
        "local currency awareness, and full dimensional analysis. Joins account, department/hotel, "
        "and product mapping tables for human-readable names, brands, regions, and segments."
    ),
    instruction=SYSTEM_PROMPT,
    tools=[
        get_bigquery_toolset(),
    ],
)
