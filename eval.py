import json
import asyncio
import re
import os
import pandas as pd
import csv
from pathlib import Path
import config
from session_runner import create_runner, chat
import vertexai 
from vertexai.generative_models import GenerativeModel

N_TRIALS = 1
USE_LLM_JUDGE = True
EXPORT_RESULTS = True

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

#tries common patterns and compares expected result against them
import re

def extract_answer(response: str) -> str:

    patterns = [
        # "The associated f0_ is 17577."
        r"associated\s+f0_\s+is\s+([\d,\.]+)",

        # "total_revenue = 12345"
        r"[A-Za-z0-9_]+\s*=\s*([\d,\.]+)",

        # "The answer is 17577"
        r"answer\s+is\s+([\d,\.]+)",

        # "is 17577"
        r"\bis\s+([\d,\.]+)",

        # JSON-style output
        r'"\w+"\s*:\s*([\d,\.]+)',

        # standalone large number
        r"\b(\d+(?:\.\d+)?)\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    return None


def is_correct(response: str, expected) -> tuple[bool, str]:
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

    response_norm = normalize(response)
    expected_norm = normalize(str(expected))

    # Step 1: containment
    if expected_norm in response_norm:
        return True, str(expected)

    # Step 2: extraction fallback
    extracted = extract_answer(response)

    if extracted is None:
        return False, None

    extracted_norm = normalize(str(extracted))

    return extracted_norm == expected_norm, extracted

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


import json

def llm_judge(prompt, expected, trials, rubric):

    trial_text = ""

    for i, t in enumerate(trials, start=1):
        trial_text += (
            f"\nTrial {i}\n"
            f"Response: {t['response']}\n"
        )

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

    response = judge_model.generate_content(judge_prompt)

    try:
        response = judge_model.generate_content(judge_prompt)
    except Exception as e:
        return {
            "score": None,
            "reasoning": f"Gemini call failed: {e}"
        }

    try:
        return json.loads(response.text)
    except Exception:
        return {
            "score": None,
            "reasoning": response.text
        }


#main evaluation loop 
def run_evaluation(dataset):
    results = []
    rubric = load_rubric()

    for i, case in enumerate(dataset, start=1):
        prompt = case["prompt"]
        expected = case["expected_value"]

        trial_results = []
        responses = []

        for t in range(N_TRIALS):
            runner, user_id, session_id = asyncio.run(
                create_runner(use_agent_engine=False)
            )

            response = asyncio.run(
                safe_chat(runner, user_id, session_id, prompt)
            )
            
            print_debug_details(results)
            
            if response in ["TIMEOUT"] or response.startswith("ERROR"):
                correct = False
                extracted = None
                refused = False

            else: 
                correct, extracted = is_correct(response, expected) 
                refused = is_refusal(response)

            status = "PASS" if correct else "FAIL"
            print(f"Q{i} Trial {t + 1}/{N_TRIALS}: {status}")

            if not correct:    
                print("\n--- FAILURE DEBUG ---")
                print(f"Q{i} Trial {t+1}")
                print(f"Expected : {expected}")
                print(f"Extracted: {extracted}")
                print(f"Response : {response[:500]}")
                print("---------------------\n")

            trial_results.append(correct)
            responses.append({
                "response": response,
                "extracted": extracted,
                "correct": correct, 
                "refused": refused
            })

        correct_count = sum(trial_results)
        refusal_count = sum(t["refused"] for t in responses)
        accuracy = correct_count / N_TRIALS

        
        llm_result = None

        if USE_LLM_JUDGE and correct_count < N_TRIALS:
            llm_result = llm_judge(prompt, expected, responses, rubric)

        
        results.append({
            "id": i,
            "prompt": prompt,
            "expected": expected,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "trials": responses,
            "refusal_count": refusal_count,
            "llm_score": llm_result["score"] if llm_result else None,
            "llm_reasoning": llm_result["reasoning"] if llm_result else None
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



def print_debug_details(results):
    print("\n--- DEBUG BREAKDOWN ---")

    for r in results:
        if r["accuracy"] < 1:
            print(f"\n--- Q{r['id']} FAILED ({r['correct_count']}/{N_TRIALS}) ---")
            print(f"Prompt: {r['prompt']}")
            print(f"Expected: {r['expected']}")

            for i, trial in enumerate(r["trials"], start=1):
                status = "✅" if trial["correct"] else "❌"
                print(f"\n--- Trial {i}: {status} ---")
                print("--- What Was Extracted: ---\n", trial["extracted"])
                print("--- The Full Response: ---\n")
                print(trial["response"])

            
            if r["llm_score"] is not None:
                    print("\n--- LLM JUDGE ---")
                    print("Score:", r["llm_score"])
                    print("Reasoning:", r["llm_reasoning"])

                    deterministic_pass = r["correct_count"] == N_TRIALS
                    if deterministic_pass != bool(r["llm_score"]):
                        print("⚠️ DISAGREEMENT BETWEEN RULES AND LLM")





#runs the evaluation + guards against import runs
if __name__ == "__main__":

    ensure_dataset_exists()
    dataset = load_dataset("dataset.jsonl")
    results, llm_runs, llm_passes = run_evaluation(dataset)

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




