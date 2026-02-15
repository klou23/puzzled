import { useState } from "react";
import CameraCapture from "../components/CameraCapture";

export default function CameraPage() {
  const [stepId, setStepId] = useState(1);
  const [feedback, setFeedback] = useState("");
  const [lastResult, setLastResult] = useState<boolean | null>(null);

  function handleResult(data: any) {
    setFeedback(data.feedback);
    setLastResult(data.match);
    if (data.match) setStepId((prev) => prev + 1);
  }

  return (
    <div style={{ maxWidth: 420, margin: "24px auto", padding: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <h2 style={{ margin: 0 }}>Step {stepId}</h2>
        {lastResult !== null && (
          <div
            style={{
              padding: "8px 16px",
              borderRadius: 8,
              backgroundColor: lastResult ? "#28a745" : "#dc3545",
              color: "white",
              fontWeight: "bold",
              fontSize: 18,
            }}
          >
            {lastResult ? "TRUE" : "FALSE"}
          </div>
        )}
      </div>

      <CameraCapture stepId={stepId} onResult={handleResult} />

      {feedback && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            borderRadius: 8,
            backgroundColor: "#f3f3f3",
          }}
        >
          {feedback}
        </div>
      )}
    </div>
  );
}
