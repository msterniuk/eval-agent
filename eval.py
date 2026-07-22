import json
import asyncio
import math
import re
import os
import pandas as pd
import csv
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import config
from session_runner import create_runner, chat
from google.cloud import bigquery
import vertexai 
from vertexai.generative_models import GenerativeModel

N_TRIALS = 1
USE_LLM_JUDGE = True
EXPORT_RESULTS = True
PRINT_ALL_DEBUG = True
USE_AGENT_ENGINE = True

#import and set up llm as a judge model 
vertexai.init(
    project = config.PROJECT_ID, 
    location = config.REGION
)

judge_model = GenerativeModel("gemini-2.5-flash")


#helps normalize text to improve regex detection
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

def classify_expected(expected):

    if expected is None:
        return "NULL_VALUE"
    
    if isinstance(expected, str):
        if expected.strip().lower() == "nan":
            return "NAN_UNANSWERABLE"

    if isinstance(expected, float) and math.isnan(expected):
        return "NAN_UNANSWERABLE"

    if isinstance(expected, (int, float)):
        return "NUMERIC_SCALAR"

    if isinstance(expected, str):

        # JSON object
        try:
            parsed = json.loads(expected)

            if isinstance(parsed, dict):
                return "JSON_OBJECT"

            if isinstance(parsed, list):
                return "JSON_LIST"

        except:
            pass

        return "STRING_OR_CODE"

    return "STRING_OR_CODE"

UNANSWERABLE_PATTERNS = [
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient information",
    "insufficient data",
    "not enough information",
    "not enough data",
    "unavailable in data sources",
    "missing required column",
    "missing required table",
    "column does not exist",
    "table does not exist",
    "requires clarification",
    "needs clarification",
    "ambiguous request",
    "cannot fulfill this request",
    "unable to fulfill this request",
    "please clarify",
    "could you please specify",
    "need more information",
    "i need to know",
    "which metric",
    "what metric",
    "do not contain information",
    "does not contain information"
]

FAILURE_PATTERNS = [
    "sql error",
    "query failed",
    "execution failed",
    "unable to execute",
    "failed to execute",
    "column not found",
    "column does not exist",
    "table not found",
    "table does not exist",
    "invalid identifier",
    "invalid column",
    "timeout",
    "timed out",
]

#should eventually become a config level var ( save in env or surface through config.py)
question_level_grain_source_table = (
        "ca-sbox-es-aiml-demo-444."
        "demo."
        "evaluation_question_level"
)

trial_level_grain_source_table = (
    "ca-sbox-es-aiml-demo-444."
    "demo."
    "evaluation_trial_level"
)

def is_valid_unanswerable_response(response: str) -> bool:
    response = response.lower()

    if any(
        pattern in response
        for pattern in UNANSWERABLE_PATTERNS
    ):
        return True

    clarification_phrases = [
        "please clarify",
        "could you please specify",
        "which metric",
        "what metric",
        "which type",
        "need more information",
        "need additional information",
        "i need to know",
        "i need more information",
        "cannot fulfill this request",
        "unable to fulfill this request",
        "do not contain information",
        "does not contain information",
        "do not contain data",
        "does not contain data",
        "not available in the available tables",
        "not available in the provided schema",
    ]

    return any(
        phrase in response
        for phrase in clarification_phrases
    )


def contains_failure_message(response: str) -> bool:
    response = response.lower()

    return any(
        pattern in response
        for pattern in FAILURE_PATTERNS
    )

def clean_response_for_extraction(response: str):

    response = re.sub( #removes the prompt version so the regex doesn't get confused
        r"PROMPT_VERSION:.*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"\[\d+\]",
        "",
        response
    )

    response = re.sub( #removes the dataset name
        r"prd-\d+",
        "",
        response,
        flags=re.IGNORECASE
    )

    return response

def isolate_results_section(response):

    match = re.search(
        r"Results(.*?)(Explanation|Follow-Up|$)",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1)

    return response

def extract_answer(response: str) -> str:

    response = clean_response_for_extraction(response)
    response = isolate_results_section(response)

    patterns = [

    # JSON output
    r'"\w+"\s*:\s*([\d,\.]+)',

    # f0_
    r"associated\s+f0_\s+is\s+([\d,\.]+)",

    # named metric
    r"associated\s+[A-Za-z0-9_]+\s+is\s+([\d,\.]+)",

    # answer is
    r"answer\s+is\s+([\d,\.]+)",

    # key = value
    r"[A-Za-z0-9_]+\s*=\s*([\d,\.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    return None


def is_correct(response: str, expected) -> tuple[bool, str, str]:
    rubric = load_rubric()
    # convert JSON string to dict
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            pass

    # unwrap dictionary expected values
    if isinstance(expected, dict):
        if len(expected) == 1:
            expected = next(iter(expected.values()))

    print(f"Normalized expected: {expected}")
    
    expected_type = classify_expected(expected)
    response_norm = normalize(response)
    expected_norm = normalize(str(expected))

    #JSON List Explicit Handling
    if expected_type == "JSON_LIST":

        if not USE_LLM_JUDGE:
            return False, None, "JSON_LIST_REQUIRES_LLM"

        llm_result = llm_judge(
            prompt=response,
            expected=expected,
            trials=[{"response": response}],
            rubric=rubric,
            evaluation_mode="JSON_LIST"
        )

        if str(llm_result["score"]).strip() == "1":
            return True, None, "JSON_LIST_MATCH"

        return False, None, "JSON_LIST_MISMATCH"

    # --------------------------------------------------
    # Handle unanswerable / NAN questions
    # --------------------------------------------------
    if expected_type == "NAN_UNANSWERABLE":

        refused = is_valid_unanswerable_response(response)

        if refused:
            return True, None, "UNANSWERABLE"
            
        # ambiguous cases go to LLM
        if USE_LLM_JUDGE:

            llm_result = llm_judge(
                "Determine whether the agent correctly refused to answer.",
                "UNANSWERABLE",
                [{"response": response}],
                rubric,
                evaluation_mode="UNANSWERABLE"
            )

            print(
                f"NAN DEBUG | refused={refused} | "
                f"llm_score={llm_result.get('score')}"
            )

            if llm_result.get("score") == 1:
                return True, None, "LLM_UNANSWERABLE_PASS"

        hallucinated = extract_answer(response)

        return (
            False,
            hallucinated,
            "HALLUCINATED_ANSWER"
        )


    # --------------------------------------------------
    # Handle SQL / tool failures
    # --------------------------------------------------
    if expected_type != "NAN_UNANSWERABLE":

        if contains_failure_message(response):
            return False, None, "FAILURE_RESPONSE"


    # Step 1: containment
    if expected_norm in response_norm:
        return True, str(expected), None

    # Step 2: extraction fallback
    extracted = extract_answer(response)

    if extracted is None:
        return False, None, "NO_EXTRACTION"

    extracted_norm = normalize(str(extracted))

    if extracted_norm == expected_norm:
        return True, extracted, None

    return False, extracted, "VALUE_MISMATCH"

def is_refusal(response):
    patterns = [
        r"\bi'm sorry\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bunfortunately\b",
        r"\bnot able to\b",
        r"\bno data\b",
        r"\binsufficient\b"
    ]

    response_lower = response.lower()

    return any(re.search(p, response_lower) for p in patterns)

def load_rubric(path="rubric.txt"):
    with open(path, "r") as f:
        return f.read()


def load_dataset(path="dataset.jsonl"):
    dataset = []
    with open(path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def convert_excel_to_jsonl(input_path="input.xlsx", output_path="dataset.jsonl"):
    df = pd.read_excel(input_path, engine="openpyxl")
    df = df.astype(str)  # ensure consistency

    with open(output_path, "w") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    print(f"✅ Converted {input_path} → {output_path}")


def export_results_to_csv(results, output_file="evaluation_results.csv"):
    """
    Export evaluation results to CSV.

    Args:
        results: list[dict]
        output_file: str
    """
    if not results:
        print("No results to export.")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results exported to {Path(output_file).resolve()}")


#this exports "results", aka the question level responses. trial level has a seperate export func
def export_question_level_to_bigquery(results):

    client = bigquery.Client()

    rows = []

    for r in results:

        rows.append({

            "run_id": r["run_id"],
            "question_id": r["question_id"],

            "prompt": r["prompt"],
            "expected": str(r["expected"]),
            "expected_type": r["expected_type"],

            "correct_count": r["correct_count"],
            "accuracy": r["accuracy"],

            "refusal_count": r["refusal_count"],

            "failure_categories": r["failure_categories"],

            "llm_score": r["llm_score"],
            "llm_reasoning": r["llm_reasoning"],
            "llm_override": r["llm_override"],

            "question_start": r["question_start"],
            "question_end": r["question_end"],
            "duration_seconds": r["duration_seconds"],

            "created_at": r["question_end"]
        })

    errors = client.insert_rows_json(
        question_level_grain_source_table,
        rows
    )

    if errors:
        print(errors)
    else:
        print(
            f"Successfully inserted {len(rows)} question rows"
        )



#this exports the trial level results
def export_trial_level_to_bigquery(results):

    client = bigquery.Client()

    rows = []

    for question in results:

        for trial in question["trials"]:

            rows.append({

                "run_id": trial["run_id"],
                "question_id": trial["question_id"],
                "trial_id": trial["trial_id"],

                "correct": trial["correct"],

                "extracted": trial["extracted"],
                "failure_category": trial["failure_category"],

                "response": trial["response"],

                "refused": trial["refused"],

                "llm_score": trial["llm_score"],
                "llm_reasoning": trial["llm_reasoning"],
                "llm_override": trial["llm_override"],

                # reuse question timestamp for partitioning
                "created_at": question["question_end"]
            })

    errors = client.insert_rows_json(
        trial_level_grain_source_table,
        rows
    )

    if errors:
        print("Trial table errors:")
        print(errors)

    else:
        print(
            f"Successfully inserted {len(rows)} trial rows"
        )


#just a wrapper that calls utilities to export results, responses into the two associated tables
def export_results_to_bigquery(results):

    export_question_level_to_bigquery(results)

    export_trial_level_to_bigquery(results)


def ensure_dataset_exists():
    if not os.path.exists("dataset.jsonl"):
        print("dataset.jsonl not found — generating from Excel...")
        
        if not os.path.exists("input.xlsx"):
            raise FileNotFoundError("❌ input.xlsx not found in working directory.")

        convert_excel_to_jsonl("input.xlsx", "dataset.jsonl")
    else:
        print("✅ dataset.jsonl already exists — skipping conversion")


async def safe_chat(runner, user_id, session_id, prompt):
    try:
        return await asyncio.wait_for(
            chat(runner, user_id, session_id, prompt),
            timeout=90
        )
    except asyncio.TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)}"
    
def parse_llm_judge_response(raw: str) -> dict:
    raw = raw.strip()

    # Remove markdown fences
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"\s*```\s*$",
        "",
        raw
    )

    # Try direct JSON parse
    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object in surrounding text
    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "score": None,
        "reasoning": f"Failed to parse LLM judge response: {raw[:1000]}"
    }

def llm_judge(prompt, expected, trials, rubric, evaluation_mode = "NORMAL"):

    trial_text = ""

    for i, t in enumerate(trials, start=1):
        trial_text += (
            f"\nTrial {i}\n"
            f"Response: {t['response']}\n"
        )

    if evaluation_mode == "UNANSWERABLE":

        judge_prompt = f"""
        The expected answer for this evaluation item is NAN_UNANSWERABLE.

        NAN_UNANSWERABLE means the agent should NOT provide a factual answer.

        The agent should instead do one of the following:

        - Explain that data is unavailable
        - Explain that required columns/tables are unavailable
        - Explain that the request cannot be answered from available data
        - Request clarification because the question is ambiguous
        - Explain that the request requires external information

        The agent does NOT need to literally say "nan".

        Agent response:

        {trial_text}

        Return ONLY valid JSON:

        {{
            "score": 0,
            "reasoning": ""
        }}

        score = 1 if the agent appropriately declined, refused, explained limitations, or requested clarification.

        score = 0 if the agent attempted to provide a factual answer.
        """
        
    elif evaluation_mode == "JSON_LIST":

        judge_prompt = f"""
        The expected answer is a JSON list.

        QUESTION:
        {prompt}

        EXPECTED JSON:
        {expected}

        AGENT RESPONSE:
        {trial_text}

        Determine whether the agent response contains the same information,
        even if:

        - formatting differs
        - field ordering differs
        - column names differ slightly
        - the response is prose rather than JSON

        Return ONLY valid JSON:

        {{
            "score": 0,
            "reasoning": ""
        }}

        score = 1 if the information content matches.
        score = 0 otherwise.
        """
    else:

        judge_prompt = f"""
        {rubric}

        QUESTION:
        {prompt}

        EXPECTED ANSWER:
        {expected}

        AGENT RESPONSES:
        {trial_text}

        Return ONLY valid JSON.

        Do NOT return markdown.
        Do NOT return code fences.
        Do NOT return explanatory text.

        Required format:

        {{
            "score": 0,
            "reasoning": ""
        }}

        score = 1 if the answer is semantically correct.
        score = 0 otherwise.
        """
        judge_prompt = f"""
        {rubric}

        QUESTION:
        {prompt}

        EXPECTED ANSWER:
        {expected}

        AGENT RESPONSES:
        {trial_text}

        Return ONLY valid JSON.

        Do NOT return markdown.
        Do NOT return code fences.
        Do NOT return explanatory text.

        Required format:

        {{
            "score": 0,
            "reasoning": ""
        }}

        score = 1 if the answer is semantically correct.
        score = 0 otherwise.
        """

    try:
        response = judge_model.generate_content(judge_prompt)
    except Exception as e:
        return {
            "score": None,
            "reasoning": f"Gemini call failed: {e}"
        }

    try:
        parsed = parse_llm_judge_response(response.text)

        print("\n--- RAW LLM RESPONSE ---")
        print(response.text)

        print("\n--- PARSED LLM RESPONSE ---")
        print(parsed)

        return parsed

    except Exception as e:
        return {
            "score": None,
            "reasoning": f"Failed while processing judge result with error {e}"
        }


#main evaluation loop 
def run_evaluation(
        dataset,
        num_trials = 1, 
        use_llm_judge = False, 
        export_results = True,
        print_all_debug = True
        ):
    results = []
    rubric = load_rubric()
    run_id = str(uuid4())

    for i, case in enumerate(dataset, start=1):
        prompt = case["prompt"]
        expected = case["expected_value"]
        question_start = datetime.now()

        trial_results = []
        responses = []

        for t in range(num_trials):
            runner, user_id, session_id = asyncio.run(
                create_runner(use_agent_engine=USE_AGENT_ENGINE,
                              agent_engine_resource=config.AGENT_ENGINE_RESOURCE)
            )

            response = asyncio.run(
                safe_chat(runner, user_id, session_id, prompt)
            )
            
            print_debug_details(results, print_all_debug)
            
            if response in ["TIMEOUT"] or response.startswith("ERROR"):
                correct = False
                result_detail = None
                extracted = None
                refused = False

            else: 
                failure_category = None
                correct, result_detail, failure_category = is_correct(response, expected) 
                refused = is_refusal(response)

            status = "PASS" if correct else "FAIL"
            print(f"Q{i} Trial {t + 1}/{num_trials}: {status}")


            llm_score = None
            llm_reasoning = None
            llm_override = None

            if not correct:    
                print("\n--- FAILURE DEBUG ---")
                print(f"Q{i} Trial {t+1}")
                print(f"Expected : {expected}")
                print(f"Extracted: {result_detail}")
                print(f"Response : {response[:1000]}")
                print("---------------------\n")

                llm_result = llm_judge(
                prompt=prompt,
                expected=expected,
                trials=[{"response": response}],
                rubric=rubric
                )

                llm_score = llm_result["score"]
                llm_reasoning = llm_result["reasoning"]
                failure_category = result_detail 

                print("\n--- TRIAL LLM RESULT ---")
                print(llm_result)
                print("SCORE TYPE:", type(llm_result["score"]))
                print("-------------------------")


                if str(llm_result["score"]).strip() == "1":
                    print("LLM OVERRIDE: PASS")
                    correct = True
                    llm_override = True
                
            trial_results.append(correct)
            responses.append({
                "run_id": run_id,
                "question_id" : i,
                "trial_id" : t + 1, 
                "response": response,
                "extracted": result_detail,
                "failure_category": failure_category,
                "correct": correct, 
                "refused": refused, 
                "llm_score" : llm_score, 
                "llm_reasoning" : llm_reasoning,
                "llm_override" : llm_override
            })

        correct_count = sum(trial_results)
        failure_categories = [
            r["failure_category"]
              for r in responses
              if r["failure_category"] is not None
        ]
        refusal_count = sum(t["refused"] for t in responses)
        accuracy = correct_count / N_TRIALS
        expected_type = classify_expected(expected)

        question_llm_score = None

        for r in responses:
            if r["llm_score"] is not None:
                question_llm_score = r["llm_score"]
                break
        
        question_llm_reasoning = None

        for r in responses:
            if r["llm_reasoning"] is not None:
                question_llm_reasoning = r["llm_reasoning"]
                break

        question_llm_override = False

        for r in responses:
            if r["llm_override"]:
                question_llm_override = True
                break
        
        question_end = datetime.now() 
        duration_in_seconds = (
        question_end - question_start
        ).total_seconds()
        
        results.append({
            "run_id": run_id,
            "question_id": i,
            "prompt": prompt,
            "expected": expected,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "trials": responses,
            "expected_type" : expected_type, 
            "refusal_count": refusal_count,
            "failure_categories" : failure_categories, 
            "llm_score": question_llm_score,
            "llm_reasoning": question_llm_reasoning,
            "llm_override" : llm_override,
            "question_start": question_start.isoformat(),
            "question_end": question_end.isoformat(),
            "duration_seconds": duration_in_seconds,
        })

        llm_runs = sum(
            1 for r in results
            if r["llm_score"] is not None
        )

        llm_passes = sum(
            1 for r in results
            if r["llm_score"] == 1
        )


        print(f"Q{i}: {correct_count}/{N_TRIALS} correct ({accuracy:.2f})")

    return results, llm_runs, llm_passes



def print_debug_details(results, print_all_debug = True):
    print("\n--- DEBUG BREAKDOWN ---")

    for r in results:

        if (not print_all_debug and r["correct_count"] == N_TRIALS):
            continue

        print(f"\n--- Q{r['question_id']} SCORED ({r['correct_count']}/{N_TRIALS}) ---")
        print(f"Current Prompt: {r['prompt']}")
        print(f"Expected Response: {r['expected']}")
        print(f"Expected Type: {r['expected_type']}")
        print(f"Failure Categories: {r["failure_categories"]}")

        for i, trial in enumerate(r["trials"], start=1):
            status = "✅" if trial["correct"] else "❌"
            print(f"LLM Override: {trial['llm_override']}")
            print(f"\n--- Trial {i}: {status} ---")
            print("--- Extracted Value Used For Scoring: ---\n", trial["extracted"])
            print(
                "--- Failure Category: ---\n",
                trial["failure_category"]
            )
            print("--- The Full Response: ---\n", trial["response"])

            if trial["llm_score"] is not None:
                print("--- Trial LLM Judge ---")
                print("Score:", trial["llm_score"])
                print("Reasoning:", trial["llm_reasoning"])


#runs the evaluation + guards against import runs
if __name__ == "__main__":

    ensure_dataset_exists()
    dataset = load_dataset("dataset.jsonl")
    results, llm_runs, llm_passes = run_evaluation(
        dataset,
        num_trials=N_TRIALS,
        use_llm_judge=USE_LLM_JUDGE,
        export_results=EXPORT_RESULTS,
        print_all_debug=PRINT_ALL_DEBUG,
    )

    total_correct = sum(r["correct_count"] for r in results)    
    total_refusals = sum(r["refusal_count"] for r in results)

    
    total_questions = len(results)
    total_trials = total_questions * N_TRIALS

    refusal_rate = total_refusals / total_trials
    avg_accuracy = total_correct / total_trials

    print("\n--- Configuration ---")
    print(f"LLM judge enabled: {USE_LLM_JUDGE}")
    print(f"Number of trials: {N_TRIALS}")

    print("\n--- Summary ---")
    print(f"{total_correct} / {total_trials} correct (across all trials)")
    print(f"Average accuracy per question: {avg_accuracy:.2f}")
    print(f"Total refusals: {total_refusals}")
    print(f"Refusal rate: {refusal_rate:.2f}")

    print(f"Total LLM Runs: {llm_runs}")
    print(f"Total LLM Passes: {llm_passes}")

    #toggle whether to save results to csv or not 
    if EXPORT_RESULTS:
        export_results_to_csv(results)
        export_results_to_bigquery(results)




