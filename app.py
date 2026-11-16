
from __future__ import annotations

import pandas as pd
import gradio as gr

from avdd import config, fusion, model
from avdd.pipeline import analyze_video, environment_report


def _header() -> str:
    env = environment_report()
    bundle = model.load_bundle()
    if bundle:
        m = bundle["metrics"]
        model_line = (
            f"**Trained classifier loaded** — Random Forest from "
            f"`{bundle.get('saved_to')}` (trained {bundle.get('trained_at')}, "
            f"{m['n_samples']} videos, held-out accuracy {m['accuracy']:.3f}, "
            f"F1 {m['f1']:.3f})."
        )
    else:
        model_line = (
            "**Feature analysis mode** — no trained model found. "
            "Train one with `python scripts/train_model.py` to enable predictions."
        )
    return (
        "# Audio-Visual Deepfake Detection (prototype)\n\n"
        f"{model_line}\n\n"
        f"Face backend: `{env['face_backend']}` · OpenCV {env['opencv']} · "
        f"Librosa {env['librosa']} · scikit-learn {env['scikit_learn']} · "
        f"FFmpeg: `{env['ffmpeg']}`"
    )


def _summary_md(result: dict) -> str:
    if result.get("status") != "ok":
        return f"### ❌ Could not analyse this file\n\n**{result.get('message')}**"

    lines = []
    pred = result.get("prediction")
    if pred:
        emoji = "🚩" if pred["label"] == "fake" else "✅"
        lines.append(
            f"## {emoji} Prediction: **{pred['label'].upper()}** "
            f"(P(fake) = {pred['probability_fake']:.2f})"
        )
    else:
        lines.append("## 🔬 Feature analysis")
        lines.append("*No trained model is loaded, so the measurements below are shown "
                     "without a classification.*")

    lines.append(result["explanation"])
    return "\n".join(lines)


def _features_df(result: dict) -> pd.DataFrame:
    names = result.get("feature_names") or []
    vec = result.get("feature_vector") or []
    rows = []
    for n, v in zip(names, vec):
        rows.append({
            "group": fusion.GROUPS.get(n, "?"),
            "feature": n,
            "value": None if v is None or v != v else round(float(v), 5),
        })
    return pd.DataFrame(rows)


def run(video_path, backend: str, progress=gr.Progress()):
    if not video_path:
        return "### Upload a video first.", [], pd.DataFrame(), {}

    def tick(msg):
        try:
            progress(0.0, desc=msg)
        except Exception:
            pass

    result = analyze_video(video_path, backend=backend, progress=tick)
    gallery = [p for p in (result.get("plots") or {}).values()]
    details = {
        "mode": result.get("mode_label"),
        "video": result.get("video"),
        "audio_summary": result.get("audio", {}).get("summary"),
        "visual_summary": {
            "backend": result.get("visual", {}).get("backend"),
            "frames_analysed": result.get("visual", {}).get("n_frames_analysed"),
            "faces_detected": result.get("visual", {}).get("n_frames_detected"),
            "detection_rate": result.get("visual", {}).get("face_detect_rate"),
        },
        "sync": {k: result.get("sync", {}).get(k) for k in
                 ("status", "best_lag_ms", "peak_correlation",
                  "zero_lag_correlation", "verdict", "reason")},
        "prediction": (result.get("prediction") or {}).get("label"),
        "warnings": result.get("warnings"),
        "result_json": result.get("result_json"),
    }
    return _summary_md(result), gallery, _features_df(result), details


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Audio-Visual Deepfake Detection") as demo:
        gr.Markdown(_header())
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Upload a video (.mp4 / .avi / .mov / .mkv)",
                                 sources=["upload"])
                backend = gr.Radio(
                    ["auto", "mediapipe", "opencv_haar"],
                    value="auto",
                    label="Face backend (auto falls back to Haar if MediaPipe fails)",
                )
                btn = gr.Button("Analyse video", variant="primary")
                gr.Markdown(
                    "**Tips**\n"
                    "- A short clip (5–20 s) with a visible, front-facing face "
                    "analyses fastest.\n"
                    "- Only the first 60 s are processed.\n"
                    "- Videos without audio or without a face are handled and "
                    "explained, they do not crash."
                )
            with gr.Column(scale=2):
                summary = gr.Markdown("Upload a video and press **Analyse video**.")
                gallery = gr.Gallery(label="Extracted features and signals",
                                     columns=2, height="auto")
                with gr.Accordion("Feature vector (every number fed to the classifier)",
                                  open=False):
                    table = gr.Dataframe(headers=["group", "feature", "value"],
                                         interactive=False, wrap=True)
                with gr.Accordion("Run details (JSON)", open=False):
                    details = gr.JSON()

        btn.click(run, inputs=[video, backend],
                  outputs=[summary, gallery, table, details])
    return demo


if __name__ == "__main__":
    import os

    # GRADIO_SHARE=1 gives a public link (needed inside Colab).
    build_ui().launch(share=os.getenv("GRADIO_SHARE", "0") == "1")
