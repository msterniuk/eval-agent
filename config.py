# config.py — Centralized configuration loaded from environment
import os
from dotenv import load_dotenv

load_dotenv()

# GCP — Project where the agent is DEPLOYED
PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "ca-sbox-es-aiml-demo-444")
REGION: str = os.environ.get("GCP_REGION", "us-central1")
STAGING_BUCKET: str = os.environ.get("STAGING_BUCKET", "gs://ca-sbox-es-aiml-demo-444")

# BigQuery — Project/dataset where the DATA lives (can be a different project)
BQ_PROJECT_ID: str = os.environ.get("BQ_PROJECT_ID", "ca-app-shared-prd-444")
BQ_DATASET: str = os.environ.get("BQ_DATASET", "GOLDNSTY")

# Agent Engine — populated after deploy.py runs
AGENT_ENGINE_RESOURCE: str = os.environ.get("AGENT_ENGINE_RESOURCE", "")

# Model
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")