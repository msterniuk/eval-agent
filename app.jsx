import { useState } from "react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);

  const [uploadStatus, setUploadStatus] = useState("");

  const [isUploaded, setIsUploaded] = useState(false);

  const [isRunningEval, setIsRunningEval] = useState(false);

  const [evalStatus, setEvalStatus] = useState("");

  const [evalComplete, setEvalComplete] = useState(false);

  const [results, setResults] = useState(null);

  const handleUpload = async () => {

    if (!selectedFile) {
        return;
    }

    try {

        setUploadStatus("Uploading dataset...");

        const formData = new FormData();

        formData.append(
            "file",
            selectedFile
        );

        const response = await fetch(
            "http://localhost:8000/upload",
            {
                method: "POST",
                body: formData,
            }
        );

        const data = await response.json();

        if (data.status === "success") {

            setIsUploaded(true);

            setUploadStatus(
                `Upload successful: ${data.filename}`
            );

        } else {

            setUploadStatus(
                `Upload failed: ${data.message}`
            );

        }

    } catch (err) {

        setUploadStatus(
            `Upload failed: ${err.message}`
        );

    }
    };

  const handleRunEval = async () => {

    try {

        setIsRunningEval(true);

        setEvalStatus(
            "Evaluation is running, please wait..."
        );

        const formData = new FormData();

        formData.append(
            "num_trials",
            1
        );

        formData.append(
            "use_llm_judge",
            true
        );

        formData.append(
            "export_results",
            true
        );

        const response = await fetch(
            "http://localhost:8000/run-eval",
            {
                method: "POST",
                body: formData,
            }
        );

        const data = await response.json();

        setResults(data);

        setEvalComplete(true);

        setEvalStatus(
            "Evaluation complete ✅"
        );

    } catch (err) {

        setEvalStatus(
            `Evaluation failed: ${err.message}`
        );

    } finally {

        setIsRunningEval(false);

    }
    };

  const handleDownload = () => {
    window.open(
        "http://localhost:8000/download-results",
        "_blank"
    );
    };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Evaluation Agent</h1>

      <hr />

      <h3>Dataset</h3>

      <input
        type="file"
        accept=".xlsx,.jsonl"
        onChange={(e) => {
          setSelectedFile(e.target.files[0]);
        }}
      />

      <br />
      <br />

      <button
        onClick={handleUpload}
        disabled={!selectedFile || isUploaded}
      >
        Upload Dataset
      </button>

      <br />
      <br />

      <button
        onClick={handleRunEval}
        disabled={!isUploaded || isRunningEval}
      >
        Run Evaluation
      </button>

      <br />
      <br />

      <button
        onClick={handleDownload}
        disabled={!evalComplete}
      >
        Download Results
      </button>

      <hr />

      <p>
        Selected File:{" "}
        {selectedFile ? selectedFile.name : "None"}
      </p>

      <p>
        Upload Status:{" "}
        {isUploaded ? "Uploaded ✅" : "Not Uploaded"}
      </p>
      <p>
        {uploadStatus}
      </p>

      <p>{evalStatus}</p>

      {results && (
        <div
          style={{
            border: "1px solid #ccc",
            padding: "1rem",
            marginTop: "1rem",
            maxWidth: "400px",
          }}
        >
          <h3>Evaluation Summary</h3>

          <p>
            Accuracy:{" "}
            {(results.avg_accuracy * 100).toFixed(1)}%
          </p>

          <p>
            Correct:{" "}
            {results.total_correct}
          </p>

          <p>
            Questions:{" "}
            {results.total_questions}
          </p>

          <p>
            LLM Runs:{" "}
            {results.llm_runs}
          </p>

          <p>
            LLM Passes:{" "}
            {results.llm_passes}
          </p>
        </div>
      )}
    </div>
  );
}
