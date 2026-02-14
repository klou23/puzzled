import { useState } from "react";
import CameraCapture from "../components/CameraCapture";

export default function CameraPage() {
  const [stepId, setStepId] = useState(1);
  const [feedback, setFeedback] = useState("");

  function handleResult(data: any) {
    setFeedback(data.feedback);
    if (data.match) setStepId((prev) => prev + 1);
  }

  return (
    <div style={{ maxWidth: 420, margin: "24px auto", padding: 16 }}>
      <h2>Test Live Camera</h2>

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
