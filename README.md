# Evaluation Agent Documentation

## Overview

### Purpose

The Evaluation Agent is a framework designed to evaluate deployed AI agents using a structured dataset of prompts and expected outputs.

The system:

1. Loads a dataset from `dataset.jsonl`
2. Sends prompts to a target agent
3. Evaluates responses using deterministic rules and LLM-assisted scoring
4. Generates CSV reports
5. Writes evaluation results into BigQuery
6. Produces detailed debugging output for analysis

The framework is agent-agnostic and can evaluate any deployed Vertex AI Agent Engine Natural Language -> SQL agent by changing the configured `AGENT_ENGINE_RESOURCE`. It would also be reasonably straightforward to expand this to other types of agents by modifying the scoring logic and retaining the overall architecture. 

***

# Quick Start

## Prerequisites

### Environment Variables

```env
GCP_PROJECT_ID=
GCP_REGION=
STAGING_BUCKET=

# BigQuery — project/dataset where the DATA lives
BQ_PROJECT_ID=
BQ_DATASET=

##Agent Engine — Fill in after running deploy.py
##AGENT_ENGINE_RESOURCE=


# Model Configuration
GEMINI_MODEL=
GOOGLE_GENAI_USE_VERTEXAI=

#Warnings are ignored during demo, comment out for testing
PYTHONWARNINGS="ignore"
```

***

### Run Evaluation

```bash
python eval.py
```

High-level execution:

```text
eval.py
    ↓
run_evaluation()
    ↓
create_runner()
    ↓
safe_chat()
    ↓
chat()
    ↓
Agent under testing
    ↓
is_correct()
    ↓
export_results_to_csv()
    ↓
export_results_to_bigquery()
```

***

# Architecture

## High-Level Architecture

```text
dataset.jsonl
        ↓

run_evaluation()
        ↓

create_runner()
        ↓

safe_chat()
        ↓
      
chat()
        ↓

Target Agent
(local or deployed)
        ↓

Response
        ↓

is_correct()
        ↓

Deterministic Evaluation
        ↓              ↘
Pass                 LLM Judge
                        ↓
                 Optional Override
                        ↓

Question Results
        ↓

CSV Export
        ↓

BigQuery Export
```

***

# Core Components

***

## run\_evaluation()

```python
def run_evaluation(
    dataset,
    num_trials=1,
    use_llm_judge=False,
    export_results=True,
    print_all_debug=True
)
```

### Purpose

Main evaluation loop.

Responsible for:

* Creating sessions
* Sending prompts
* Evaluating correctness
* Aggregating trial results
* Exporting results

### Inputs

| Parameter         | Description                            |
| ----------------- | -------------------------------------- |
| dataset           | Dataset loaded from JSONL              |
| num\_trials       | Number of attempts per prompt          |
| use\_llm\_judge   | Enable Gemini LLM as a Judge evaluation|
| export\_results   | Enable CSV/BigQuery export             |
| print\_all\_debug | Show all debug output vs failures only |

### Outputs

```python
results,
llm_runs,
llm_passes
```

***

## create\_runner()

```python
async def create_runner(
    use_agent_engine=False,
    agent_engine_resource=None
)
```

### Purpose

Creates a session-backed runner for either:

```text
Local Agent
```

or

```text
Vertex Agent Engine
```

### Inputs

| Parameter               | Description                     |
| ----------------------- | ------------------------------- |
| use\_agent\_engine      | Enable deployed agent           |
| agent\_engine\_resource | Full Agent Engine resource name |

### Returns

```python
runner,
user_id,
session_id
```

***

## safe\_chat()

```python
async def safe_chat(
    runner,
    user_id,
    session_id,
    prompt
)
```

### Purpose

Wrapper around chat execution.

Provides:

* Timeout protection
* Error handling
* Standardized response format

### Returns

```python
response_string
```

or

```python
TIMEOUT
```

or

```python
ERROR ...
```

***

## is\_correct()

```python
def is_correct(
    response,
    expected
)
```

### Purpose

Determines whether a response is correct.

### Evaluation Paths

```text
NUMERIC_SCALAR
STRING_OR_CODE
NAN_UNANSWERABLE
JSON_LIST
```

### Returns

```python
(
    correct,
    extracted,
    failure_category
)
```

***

### extracted

Value extracted from the response.

Examples:

```text
75.34
Bruce
557343
```

***

### failure\_category

Examples:

```text
VALUE_MISMATCH
NO_EXTRACTION
FAILURE_RESPONSE
HALLUCINATED_ANSWER
UNANSWERABLE
JSON_LIST_MATCH
JSON_LIST_MISMATCH
```

***

## classify\_expected()

```python
def classify_expected(expected)
```

### Purpose

Routes questions into evaluation strategies.

### Possible Types

```text
NUMERIC_SCALAR
STRING_OR_CODE
JSON_LIST
NAN_UNANSWERABLE
NULL_VALUE
```

***

## llm\_judge()

### Purpose

Gemini-based evaluation fallback.

Used to augment deterministic evaluation, especially when regex cannot extract clear answer.

### Evaluation Modes

```text
STANDARD
UNANSWERABLE
JSON_LIST
```

### Returns

```python
{
    "score": 1,
    "reasoning": "..."
}
```

***

# Result Structure

## Question-Level Results

Each question generates:

```python
{
    "run_id": ...,
    "question_id": ...,

    "prompt": ...,
    "expected": ...,

    "correct_count": ...,
    "accuracy": ...,

    "expected_type": ...,

    "failure_categories": ...,

    "llm_score": ...,
    "llm_reasoning": ...,
    "llm_override": ...,

    "question_start": ...,
    "question_end": ...,
    "duration_seconds": ...
}
```

***

## Trial-Level Results

Stored under:

```python
result["trials"]
```

Example:

```python
{
    "run_id": ...,
    "question_id": ...,
    "trial_id": ...,

    "correct": ...,

    "extracted": ...,
    "failure_category": ...,

    "response": ...,

    "refused": ...,

    "llm_score": ...,
    "llm_reasoning": ...,
    "llm_override": ...
}
```

***

# Debugging

## print\_debug\_details()

```python
def print_debug_details(
    results,
    print_all_debug=True
)
```

### Purpose

Primary debugging utility.

Prints:

* Prompt
* Expected value
* Expected type
* Trial outcomes
* Extracted values
* Failure categories
* Full responses
* LLM judge results
* Override information

### Modes

```python
print_all_debug=True
```

Print all questions.

```python
print_all_debug=False
```

Print failures only.

***

# BigQuery

## Question Table

```text
evaluation_question_level
```

Stores:

```text
One row per question
```

Primary key concept:

```text
run_id
question_id
```

***

## Trial Table

```text
evaluation_trial_level
```

Stores:

```text
One row per trial
```

Primary key concept:

```text
run_id
question_id
trial_id
```

***

# Evaluation Metadata

## run\_id

Generated:

```python
str(uuid4())
```

Purpose:

```text
Groups all questions from a single evaluation run.
```

***

## question\_id

Identifies a question within a run.

***

## trial\_id

Identifies a trial within a question.

***

# Typical Workflow

```text
1. Upload Dataset
        ↓
2. Run Evaluation
        ↓
3. Agent Generates Responses
        ↓
4. Responses Evaluated
        ↓
5. CSV Generated
        ↓
6. BigQuery Updated
        ↓
7. Analyze Results
```

***

### Multi-Agent Evaluation

This evaluation framework is largely agent agnostic: simply alter the environment variables and all changes should populate downstream into agent.py, system prompt, config.py, etc. The key change is the AGENT_ENGINE_RESOURCE, but also ensure variables such as your BQ project or staging bucket have not changed in the .env file. 

### Note on Agent 'Types' 
One important thing to note is that this evaluation process is custom fit for Natural Language -> SQL agents. The overall architecture and skeleton can support any Vertex AI agent, but for best results you may wish to modify key evaluation functions such as is_correct() and llm_judge() to accomodate different agent types. 
