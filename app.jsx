import { useState } from "react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);

  const [isUploaded, setIsUploaded] = useState(false);

  const [isRunningEval, setIsRunningEval] = useState(false);

  const [evalComplete, setEvalComplete] = useState(false);

  const [results, setResults] = useState(null);

  const handleUpload = () => {
    setIsUploaded(true);
  };

  const handleRunEval = () => {
    setIsRunningEval(true);

    setTimeout(() => {
      setIsRunningEval(false);
      setEvalComplete(true);

      // placeholder data for commit 1
      setResults({
        avg_accuracy: 0.42,
        total_correct: 21,
        total_questions: 49,
        llm_runs: 10,
        llm_passes: 3,
      });
    }, 3000);
  };

  const handleDownload = () => {
    alert("Download placeholder");
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

      {isRunningEval && (
        <p>
          Evaluation is running, please wait...
        </p>
      )}

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
