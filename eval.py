import json
import asyncio
import re
import os
import pandas as pd
from session_runner import create_runner, chat

N_TRIALS = 3

#helps normalize text to improve regex detection
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

#tries common patterns and compares expected result against them
def extract_answer(response: str) -> str:
    # Basic name extraction (adjust later if needed)
    patterns = [
        r"is ([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+)",
        r"([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+) is",
        r":[ ]*([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+)"
    ]

    for p in patterns:
        match = re.search(p, response)
        if match:
            return match.group(1)

    return response.strip()


def is_correct(response: str, expected: str) -> tuple[bool, str]:
    response_norm = normalize(response)
    expected_norm = normalize(expected)

    # Step 1: containment (primary check)
    if expected_norm in response_norm:
        return True, expected

    # Step 2: regex extraction fallback
    extracted = extract_answer(response)
    extracted_norm = normalize(extracted)

    return extracted_norm == expected_norm, extracted

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
    df = pd.read_excel(input_path)
    df = df.astype(str)  # ensure consistency

    with open(output_path, "w") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    print(f"✅ Converted {input_path} → {output_path}")


def llm_judge(prompt, expected, trials, rubric):
    trial_text = ""

    for i, t in enumerate(trials, 1):
        trial_text += f"\nTrial {i}:\n{t['response']}\n"

    judge_prompt = f"""
    {rubric}

    --- QUESTION ---
    {prompt}

    --- EXPECTED ANSWER ---
    {expected}

    --- AGENT RESPONSES ---
    {trial_text}

    Evaluate whether the agent is correct overall.

    Return JSON:
    {{"score": 0 or 1, "reasoning": "..."}}
    """

    # TODO: replace with actual LLM call
    return {
        "score": None,
        "reasoning": "LLM not implemented yet"
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
                chat(runner, user_id, session_id, prompt)
            )

            correct, extracted = is_correct(response, expected)

            trial_results.append(correct)
            responses.append({
                "response": response,
                "extracted": extracted,
                "correct": correct
            })

        correct_count = sum(trial_results)
        accuracy = correct_count / N_TRIALS

        
        llm_result = None

        if correct_count < N_TRIALS:
            #need to actually connect llm, probably gemini-1.5-flash
            llm_result = llm_judge(prompt, expected, responses, rubric)


        
        results.append({
            "id": i,
            "prompt": prompt,
            "expected": expected,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "trials": responses,
            "llm_score": llm_result["score"] if llm_result else None,
            "llm_reasoning": llm_result["reasoning"] if llm_result else None
        })


        print(f"Q{i}: {correct_count}/{N_TRIALS} correct ({accuracy:.2f})")

    return results



def print_failure_details(result):
    print("\n--- FAILURE BREAKDOWN ---")

    for r in results:
        if r["accuracy"] < 1:
            print(f"\nQ{r['id']} FAILED ({r['correct_count']}/{N_TRIALS})")
            print(f"Prompt: {r['prompt']}")
            print(f"Expected: {r['expected']}")

            for i, trial in enumerate(r["trials"], start=1):
                status = "✅" if trial["correct"] else "❌"
                print(f"\nTrial {i}: {status}")
                print("Extracted:", trial["extracted"])
                print("Response:")
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
    dataset = load_dataset("dataset.jsonl")
    results = run_evaluation(dataset)

    total_correct = sum(r["correct_count"] for r in results)    
    total = len(results)
    
    
    total_questions = len(results)
    total_trials = total_questions * N_TRIALS

    total_correct = sum(r["correct_count"] for r in results)

    print("\n--- Summary ---")
    print(f"{total_correct} / {total_trials} correct (across all trials)")
