import json
import asyncio
import re
from session_runner import create_runner, chat


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


def load_dataset(path="dataset.jsonl"):
    dataset = []
    with open(path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


#main evaluation loop 
def run_evaluation(dataset):
    results = []

    for i, case in enumerate(dataset, start=1):
        prompt = case["prompt"]
        expected = case["expected_value"]

        # Create fresh session per question
        runner, user_id, session_id = asyncio.run(
            create_runner(use_agent_engine=False)
        )

        # Call agent
        response = asyncio.run(
            chat(runner, user_id, session_id, prompt)
        )

        # Evaluate
        correct, extracted = is_correct(response, expected)

        results.append({
            "id": i,
            "prompt": prompt,
            "raw_response": response,
            "extracted": extracted,
            "expected": expected,
            "correct": correct
        })

        status = "✅" if correct else "❌"
        print(f"{status} Q{i}: {prompt}")

    return results


#runs the evaluation + guards against import runs
if __name__ == "__main__":
    dataset = load_dataset("dataset.jsonl")
    results = run_evaluation(dataset)

    correct_count = sum(r["correct"] for r in results)
    total = len(results)

    print("\n--- Summary ---")
    print(f"{correct_count} / {total} correct")