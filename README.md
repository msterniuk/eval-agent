# Financial Analyst Agent

A production-ready BigQuery agent built with Google's Agent Development Kit (ADK) that analyzes accounting journal transactions and headers.

## Overview

This agent provides AI-powered financial analysis of your journal data stored in BigQuery:
- **PS_JRNL_HEADER** — Journal entry headers with metadata (dates, batch IDs, status)
- **PS_JRNL_LN** — Journal line items with account details, amounts, and descriptions

The agent uses natural language to understand your questions and generates optimized SQL queries to analyze the data.

## Prerequisites

- **Python 3.11+**
- **Google Cloud SDK** installed and configured (`gcloud` CLI)
- **GCP Project**: `ca-sbox-es-science-444` with BigQuery access
- **Service Account** or Application Default Credentials (ADC)

### GCP Permissions Required

Your service account needs these IAM roles:
- `roles/bigquery.dataViewer` — Read access to BigQuery datasets
- `roles/bigquery.jobUser` — Run BigQuery queries
- `roles/storage.objectAdmin` — Write to the staging bucket (for deployment)
- `roles/aiplatform.user` — Deploy and run agents on Agent Engine

### Setup ADC (Application Default Credentials)

```bash
gcloud auth application-default login
```

This allows local scripts to authenticate to GCP without explicit credentials.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values (or leave defaults as-is)
cat .env
```

Default values are pre-filled for the `ca-sbox-es-science-444` project:
- `GCP_PROJECT_ID=ca-sbox-es-science-444`
- `BQ_DATASET=USL_POC`
- `STAGING_BUCKET=gs://ca-sbox-es-science-444`
- `GCP_REGION=us-central1`

### 3. Test Locally

Run the agent in local memory mode (no Agent Engine deployment required):

```bash
python session_runner.py
```

You'll see an interactive prompt:

```
Initializing financial_analyst agent for local testing...
✓ Agent ready.

Example queries:
  - 'Show me journal entries from the last 30 days'
  - 'What accounts have the highest transaction volumes?'
  - 'Summarize debit vs credit by account'

Type 'exit' to quit.

You: Show me the top 10 accounts by transaction count
Agent: [generates SQL and returns results]
```

### Sample Queries

Try these example questions:

```
- "Show me all journal entries for the last 30 days"
- "What are the top accounts by debit amount?"
- "List all transactions in batch ABC123"
- "Find entries with amounts greater than $100,000"
- "Summarize transactions by cost center"
- "Compare total debits vs credits by account"
```

## Architecture

### Local Testing
```
session_runner.py (InMemorySessionService)
         ↓
    agent.py (Root Agent)
         ↓
   BigQueryToolset
         ↓
    BigQuery API
```

### Cloud Deployment
```
Agent Engine (VertexAiSessionService)
         ↓
    agent.py (Root Agent)
         ↓
   BigQueryToolset
         ↓
    BigQuery API
```

## Deployment to Agent Engine

### Step 1: Deploy the Agent

```bash
python deploy.py
```

This packages your agent and deploys it to Vertex AI Agent Engine. On success, you'll see:

```
✅ Deployment successful!

Resource name:
  projects/ca-sbox-es-science-444/locations/us-central1/reasoningEngines/12345...

📝 Next step — add this to your .env file:
  AGENT_ENGINE_RESOURCE=projects/ca-sbox-es-science-444/locations/us-central1/reasoningEngines/12345...
```

### Step 2: Update .env

Copy the resource name and add it to `.env`:

```bash
AGENT_ENGINE_RESOURCE=projects/ca-sbox-es-science-444/locations/us-central1/reasoningEngines/12345...
```

### Step 3: Test Against Agent Engine

```bash
python session_runner.py
```

The script will detect `AGENT_ENGINE_RESOURCE` and use `VertexAiSessionService` automatically.

## File Structure

```
financial_analyst/
├── agent.py                 # Main ADK agent definition
├── config.py               # Configuration (GCP, BigQuery, models)
├── session_runner.py       # Interactive testing + Agent Engine runner
├── deploy.py              # Deployment script to Agent Engine
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .adk/
│   └── settings.json     # ADK Web UI configuration (local development)
├── tools/
│   ├── __init__.py
│   └── bigquery_tool.py  # BigQuery toolset wrapper
└── README.md             # This file
```

## Configuration

All configuration is managed via environment variables in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GCP_PROJECT_ID` | `ca-sbox-es-science-444` | GCP project ID |
| `GCP_REGION` | `us-central1` | Region for Agent Engine |
| `STAGING_BUCKET` | `gs://ca-sbox-es-science-444` | GCS bucket for deployment |
| `BQ_DATASET` | `USL_POC` | BigQuery dataset name |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM model to use |
| `AGENT_ENGINE_RESOURCE` | (empty) | Set after deployment |

## Development

### Local ADK Web UI (Advanced)

For a graphical interface during local development:

```bash
# Install MCP server for BigQuery
pip install mcp-server-bigquery uvx

# Start ADK Web UI
adk web
```

This opens a web interface on `http://localhost:8000` with visual query building.

**Note**: The MCP server configuration is in `.adk/settings.json`.

### Updating the Agent

1. Modify `agent.py` (system prompt, tools, etc.)
2. Test locally: `python session_runner.py`
3. Redeploy: `python deploy.py`

### Custom BigQuery Queries

The agent automatically generates SQL based on your questions. To debug or manually test:

```python
from google.cloud import bigquery

client = bigquery.Client(project="ca-sbox-es-science-444")
query = """
SELECT * FROM USL_POC.PS_JRNL_HEADER
LIMIT 10
"""
results = client.query(query).result()
for row in results:
    print(row)
```

## Troubleshooting

### Issue: "AGENT_ENGINE_RESOURCE is not set"

**Solution**: You haven't deployed yet or haven't updated `.env` after deployment. Run:
```bash
python deploy.py
```

Then copy the resource name to `.env`.

### Issue: "Authentication failed" or "Permission denied"

**Solution**: Ensure ADC is set up:
```bash
gcloud auth application-default login
```

And verify IAM roles are assigned to your service account.

### Issue: BigQuery queries timeout or fail

**Solution**: 
- Check that `BQ_DATASET=USL_POC` is set in `.env`
- Verify the tables `PS_JRNL_HEADER` and `PS_JRNL_LN` exist
- Check dataset permissions

### Issue: "Module not found" errors

**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## Security Notes

1. **API Keys & Credentials**:
   - Never commit `.env` to version control
   - Use Google Cloud Secret Manager for production credentials
   - Service accounts should have minimal necessary permissions

2. **BigQuery Access**:
   - Agent can only read data (SELECT queries only)
   - No INSERT, UPDATE, DELETE, or DROP permissions

3. **Agent Engine**:
   - Deployments are private to your GCP project
   - Sessions are authenticated via Vertex AI

## Next Steps

- **Customize the system prompt** in `agent.py` for domain-specific analysis
- **Add more tables** by extending the `bigquery_tool.py` configuration
- **Monitor deployments** via Cloud Logging and Vertex AI monitoring
- **Integrate with applications** using the Agent Engine REST API

## Support

For issues or questions:
1. Check the [ADK documentation](https://github.com/google/adk-python)
2. Review [Vertex AI Agent Engine docs](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)
3. Verify BigQuery dataset and table names

## License

This project uses Google Cloud APIs. See [Google Cloud Terms of Service](https://cloud.google.com/terms).
