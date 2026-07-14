from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from eval import (
    ensure_dataset_exists,
    load_dataset,
    run_evaluation,
    export_results_to_csv,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    # overwrite existing input.xlsx
    with open("input.xlsx", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # force dataset regeneration from the newly uploaded file
    dataset_path = Path("dataset.jsonl")

    if dataset_path.exists():
        dataset_path.unlink()

    ensure_dataset_exists()

    dataset = load_dataset("dataset.jsonl")

    return {
        "status": "success",
        "filename": file.filename,
    }


@app.post("/run-eval")
async def run_eval(
    num_trials: int = Form(...),
    use_llm_judge: bool = Form(False),
    export_results: bool = Form(True),
):

    try:
        dataset = load_dataset("dataset.jsonl")

        results, llm_runs, llm_passes = run_evaluation(
            dataset=dataset,
            num_trials=num_trials,
            use_llm_judge=use_llm_judge,
            export_results=export_results,
        )

        # mimic current __main__ behavior
        if export_results:
            export_results_to_csv(results)

        total_correct = sum(
            r["correct_count"]
            for r in results
        )

        total_refusals = sum(
            r["refusal_count"]
            for r in results
        )

        total_questions = len(results)
        total_trials = total_questions * num_trials

        avg_accuracy = (
            total_correct / total_trials
            if total_trials > 0
            else 0
        )

        return {
            "status": "success",
            "total_questions": total_questions,
            "total_trials": total_trials,
            "total_correct": total_correct,
            "total_refusals": total_refusals,
            "avg_accuracy": avg_accuracy,
            "llm_runs": llm_runs,
            "llm_passes": llm_passes,
            "export_results": export_results,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/download-results")
def download_results():

    csv_path = Path("evaluation_results.csv")

    if not csv_path.exists():
        return {
            "status": "error",
            "message": "evaluation_results.csv not found"
        }

    return FileResponse(
        path=str(csv_path),
        filename="evaluation_results.csv",
        media_type="text/csv",
    )