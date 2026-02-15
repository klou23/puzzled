import { useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";

type Props = {
  stepId: number;
  onResult: (data: any) => void;
};

const API_URL = import.meta.env.VITE_API_URL;
const RUNPOD_URL = import.meta.env.VITE_RUNPOD_URL;
const RUNPOD_API_KEY = import.meta.env.VITE_RUNPOD_API_KEY;
const USE_RUNPOD = import.meta.env.VITE_USE_RUNPOD === "true";

export default function CameraCapture({ stepId, onResult }: Props) {
  const webcamRef = useRef<Webcam>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // iPhone/Safari friendly constraints
  const videoConstraints = useMemo(
    () => ({
      facingMode: { ideal: "environment" }, // back camera
      width: { ideal: 1280 },
      height: { ideal: 720 },
    }),
    []
  );

  const captureAndUpload = async () => {
    setError("");
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setError("Could not capture image. Make sure the camera is running.");
      return;
    }

    setLoading(true);

    try {
      let data;

      if (USE_RUNPOD && RUNPOD_URL) {
        // RunPod serverless format - send base64 directly
        const base64Data = imageSrc.split(",")[1]; // Remove "data:image/jpeg;base64," prefix

        const response = await fetch(RUNPOD_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${RUNPOD_API_KEY}`,
          },
          body: JSON.stringify({
            input: {
              image_base64: base64Data,
              step_id: stepId,
            },
          }),
        });

        if (!response.ok) {
          throw new Error(`RunPod API error: ${response.status}`);
        }

        const result = await response.json();
        // RunPod wraps response in "output" field
        data = result.output || result;
      } else {
        // Standard REST API format
        const res = await fetch(imageSrc);
        const blob = await res.blob();
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

        const form = new FormData();
        form.append("image", file);
        form.append("stepId", String(stepId));

        const response = await fetch(`${API_URL}/verify/verify-step`, {
          method: "POST",
          body: form,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        data = await response.json();
      }

      onResult(data);

      if ("speechSynthesis" in window && data.feedback) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(data.feedback));
      }
    } catch (e) {
      setError("Upload failed. Check your API URL / HTTPS / CORS and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      {!cameraOn ? (
        <button
          onClick={() => setCameraOn(true)}
          style={{
            marginTop: 12,
            padding: "12px 16px",
            borderRadius: 8,
            border: "none",
            backgroundColor: "#007bff",
            color: "white",
            fontSize: 16,
            width: "100%",
          }}
        >
          Take Picture
        </button>
      ) : (
        <>
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            playsInline
            style={{ width: "100%", borderRadius: 12 }}
          />

          <button
            onClick={captureAndUpload}
            disabled={loading}
            style={{
              marginTop: 12,
              padding: "12px 16px",
              borderRadius: 8,
              border: "none",
              backgroundColor: loading ? "#6c757d" : "#007bff",
              color: "white",
              fontSize: 16,
              width: "100%",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            Capture & Verify
          </button>

          <button
            onClick={() => {
              setCameraOn(false);
              setError("");
            }}
            style={{
              marginTop: 10,
              padding: "10px 16px",
              borderRadius: 8,
              border: "1px solid #ddd",
              backgroundColor: "black",
              fontSize: 14,
              width: "100%",
            }}
          >
            Cancel
          </button>
        </>
      )}

      {loading && <p>Checking step...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
