import json
import asyncio
import re
import os
import pandas as pd
from session_runner import create_runner, chat

N_TRIALS = 1

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
    df = pd.read_excel(input_path)
    df = df.astype(str)  # ensure consistency

    with open(output_path, "w") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")

    print(f"✅ Converted {input_path} → {output_path}")



def ensure_dataset_exists():
    if not os.path.exists("dataset.jsonl"):
        print("dataset.jsonl not found — generating from Excel...")
        
        if not os.path.exists("input.xlsx"):
            raise FileNotFoundError("❌ input.xlsx not found in working directory.")

        convert_excel_to_jsonl("input.xlsx", "dataset.jsonl")
    else:
        print("✅ dataset.jsonl already exists — skipping conversion")



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
        print(f"currently running eval on Q{i}")
        prompt = case["prompt"]
        expected = case["expected_value"]

        trial_results = []
        responses = []

        for t in range(N_TRIALS):
            print("am creating running rn")
            runner, user_id, session_id = asyncio.run(
                create_runner(use_agent_engine=False)
            )

            print("runner has been creating, chat is next")
            response = asyncio.run(
                chat(runner, user_id, session_id, "What is 2+2?")
            )
            
            print("chat has been created and has responded")
            correct, extracted = is_correct(response, expected) 
            refused = is_refusal(response)


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
            "refusal_count": refusal_count,
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

    ensure_dataset_exists()
    dataset = load_dataset("dataset.jsonl")
    print("loaded dataset, about to run eval")
    results = run_evaluation(dataset)
    print("eval has been run, now aggregating results")

    total_correct = sum(r["correct_count"] for r in results)    
    total_refusals = sum(r["refusal_count"] for r in results)

    
    total_questions = len(results)
    total_trials = total_questions * N_TRIALS

    refusal_rate = total_refusals / total_trials
    avg_accuracy = total_correct / total_trials

    print("\n--- Summary ---")
    print(f"{total_correct} / {total_trials} correct (across all trials)")
    print(f"Average accuracy per question: {avg_accuracy:.2f}")
    print(f"Total refusals: {total_refusals}")
    print(f"Refusal rate: {refusal_rate:.2f}")



