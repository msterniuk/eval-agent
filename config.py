# config.py — Centralized configuration loaded from environment
import os
from dotenv import load_dotenv

load_dotenv()

# GCP — Project where the agent is DEPLOYED
PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "some_dataset_anon")
REGION: str = os.environ.get("GCP_REGION", "some_region")
STAGING_BUCKET: str = os.environ.get("STAGING_BUCKET", "some_dataset_anon")

# BigQuery — Project/dataset where the DATA lives (can be a different project)
BQ_PROJECT_ID: str = os.environ.get("BQ_PROJECT_ID", "some_dataset_anon")
BQ_DATASET: str = os.environ.get("BQ_DATASET", "some_dataset_anon")

# Agent Engine — populated after deploy.py runs
AGENT_ENGINE_RESOURCE: str = os.environ.get("AGENT_ENGINE_RESOURCE", "")

# Model
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
