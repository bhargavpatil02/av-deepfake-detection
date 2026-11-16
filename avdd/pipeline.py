"""
End-to-end pipeline.

    Input video
      -> Module 1  validate + extract audio + sample frames
      -> Module 2  audio features        (Librosa)
      -> Module 3  visual/lip features   (MediaPipe or Haar)
      -> Module 4  audio-visual sync     (energy vs mouth signal correlation)
      -> Module 5  fuse + classify       (Random Forest, only if trained)
      -> results dict + PNG plots + result.json

Every stage is wrapped so a failure in one module degrades the report instead of
crashing the run. If no trained model is available the run reports the measured
features without a real/fake decision.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

import numpy as np

from . import config, fusion, model, sync, viz
from .audio_feats import extract_audio_features
from .io_video import (
    VideoError,
    extract_audio,
    iter_sampled_frames,
    sampling_summary,
    validate_video,
    find_ffmpeg,
)
from .visual_feats import extract_visual_features




def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if not isinstance(v, (np.ndarray,))}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _mouth_series(visual: dict):
    """Pick the best available per-frame mouth signal for Module 4."""
    records = [r for r in (visual.get("records") or []) if r.get("detected")]
    if not records:
        return None, None, None
    for key, label in (("mouth_open_norm", "mouth opening / face height"),
                       ("mar", "mouth aspect ratio"),
                       ("motion", "mouth pixel motion")):
        vals = [r.get(key) for r in records]
        if sum(v is not None for v in vals) >= 5:
            times = np.array([r["t"] for r in records], dtype=float)
            series = np.array([np.nan if v is None else v for v in vals], dtype=float)
            return times, series, label
    return None, None, None


def _fmt(value, spec: str, missing: str = "not measured") -> str:
    """Format a possibly-absent feature without crashing or inventing a value."""
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def _explanation(result: dict) -> str:
    """Plain-language summary shown in the UI and saved to the report."""
    video = result["video"]
    audio = result["audio"]
    visual = result["visual"]
    lines: list[str] = []

    lines.append("### What was analysed")
    lines.append(
        f"- File: **{video['file']}**, {video['resolution']}, "
        f"{video['duration_s']:.1f}s, {video['fps']:.1f} fps."
    )
    lines.append(
        f"- Frames sampled for visual analysis: **{video['analysed_frames']}** "
        f"(~{video['analysed_fps']:.1f} fps, stride {video['frame_stride']})."
    )

    lines.append("\n### Audio (Module 2)")
    if audio["ok"]:
        s = audio["summary"]
        lines.append(
            f"- {s['n_features']} features from {s['n_frames']} frames: MFCC, "
            f"zero-crossing rate, spectral centroid, roll-off, RMS energy, "
            f"bandwidth, flatness.\n"
            f"- Mean RMS energy {s['mean_rms']}, mean spectral centroid "
            f"{s['mean_spectral_centroid_hz']} Hz."
        )
    else:
        lines.append(f"- **Audio not analysed:** {audio.get('error')}")

    lines.append("\n### Visual / lips (Module 3)")
    if visual["ok"]:
        backend = visual.get("backend")
        lines.append(
            f"- Backend **{backend}**; a face was found in "
            f"{visual['n_frames_detected']}/{visual['n_frames_analysed']} analysed "
            f"frames ({visual['face_detect_rate'] * 100:.0f}%)."
        )
        if backend == "mediapipe":
            f = visual["features"]
            lines.append(
                f"- Mean mouth aspect ratio {_fmt(f.get('mean_mar'), '.3f')}, mean "
                f"normalised mouth opening {_fmt(f.get('mean_mouth_open_norm'), '.4f')}, "
                f"mean |delta MAR| {_fmt(f.get('mean_abs_dmar'), '.4f')}, mouth pixel "
                f"motion {_fmt(f.get('mean_mouth_pixel_motion'), '.2f')}/s."
            )
        else:
            lines.append(
                "- Landmark features are unavailable on this backend, so mouth "
                "opening/MAR are reported as *not measured* rather than guessed."
            )
    else:
        lines.append(f"- **Visual analysis failed:** {visual.get('error')}")

    lines.append("\n### Audio-visual synchronisation (Module 4)")
    if result["sync"]["status"] == "computed":
        lines.append(f"- {result['sync']['interpretation']}")
    else:
        lines.append(f"- Not computed: {result['sync'].get('reason')}")

    lines.append("\n### Decision (Module 5)")
    if result["prediction"]:
        p = result["prediction"]
        lines.append(
            f"- **Prediction: {p['label'].upper()}** "
            f"(P(fake) = {p['probability_fake']:.2f}, confidence {p['confidence']:.2f})."
        )
        m = p["model_metrics"]
        lines.append(
            f"- Model: Random Forest trained on {m['n_samples']} videos; held-out "
            f"test accuracy {m['accuracy']:.3f}, F1 {m['f1']:.3f} "
            f"(test n={m['n_test']})."
        )
        for note in p["notes"]:
            lines.append(f"- ⚠️ {note}")
    else:
        lines.append(
            "- **Feature analysis only** - no trained model bundle found, so no "
            "real/fake label is produced. Run `python scripts/train_model.py` on "
            "labelled videos to enable predictions."
        )

    if result.get("warnings"):
        lines.append("\n### Warnings")
        for w in result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)


def analyze_video(
    video_path: str | Path,
    model_path: str | Path | None = None,
    backend: str = "auto",
    out_dir: str | Path | None = None,
    save_plots: bool = True,
    progress=None,
) -> dict:
    """
    Run the full analysis on one video and return a result dictionary.

    Never raises for bad input: returns {"status": "error", "code": ..., "message": ...}.
    """
    def tick(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    result: dict = {
        "status": "error",
        "mode": "demo",
        "warnings": [],
        "prediction": None,
        "plots": {},
    }
    try:
        # ---------------------------------------------------------- Module 1
        tick("Validating video...")
        meta = validate_video(video_path)
        meta = sampling_summary(meta)
        workdir = Path(out_dir) if out_dir else config.OUTPUT_DIR / Path(meta.path).stem
        workdir.mkdir(parents=True, exist_ok=True)

        result["video"] = meta.to_dict()
        result["warnings"].extend(meta.warnings)

        tick("Extracting audio track...")
        wav_path = workdir / "audio.wav"
        audio_ext = extract_audio(meta.path, wav_path)
        if not audio_ext["ok"]:
            result["warnings"].append(audio_ext["message"])
        elif audio_ext["source"] != "system":
            result["warnings"].append(
                f"FFmpeg provided by {audio_ext['source']} (no system FFmpeg found)."
            )

        # ---------------------------------------------------------- Module 2
        audio: dict = {"ok": False, "error": audio_ext["message"], "features": {},
                       "names": [], "summary": {}}
        if audio_ext["ok"]:
            tick("Extracting audio features (Librosa)...")
            audio = extract_audio_features(wav_path)
            if not audio["ok"]:
                result["warnings"].append(f"Audio features failed: {audio.get('error')}")
        result["audio"] = {
            "ok": audio.get("ok", False),
            "error": None if audio.get("ok") else audio.get("error"),
            "summary": audio.get("summary", {}),
            "features": audio.get("features", {}),
            "n_features": len(audio.get("names", [])),
        }

        # ---------------------------------------------------------- Module 3
        tick("Detecting faces and lip landmarks...")
        try:
            frames = iter_sampled_frames(meta, resize_width=config.VISUAL_WORK_WIDTH)
            visual = extract_visual_features(frames, backend=backend)
        except VideoError as exc:
            visual = {"ok": False, "error": exc.message, "records": [],
                      "features": {}, "notes": [], "backend": "unavailable",
                      "n_frames_analysed": 0, "n_frames_detected": 0,
                      "face_detect_rate": 0.0}
        result["visual"] = {
            "ok": visual.get("ok", False),
            "backend": visual.get("backend"),
            "error": visual.get("error"),
            "notes": visual.get("notes", []),
            "features": visual.get("features", {}),
            "n_frames_analysed": visual.get("n_frames_analysed", 0),
            "n_frames_detected": visual.get("n_frames_detected", 0),
            "face_detect_rate": visual.get("face_detect_rate", 0.0),
        }
        if visual.get("notes"):
            result["warnings"].extend(visual["notes"])

        # ---------------------------------------------------------- Module 4
        tick("Correlating audio and mouth motion...")
        if audio.get("ok") and visual.get("ok"):
            vt, vs, label = _mouth_series(visual)
            if vt is None:
                sync_res = sync.unavailable("no usable mouth signal available.")
            else:
                sync_res = sync.compute_av_sync(
                    audio["times"], audio["rms"], vt, vs, label
                )
        else:
            sync_res = sync.unavailable(
                "requires both an audio track and detected faces."
            )
        result["sync"] = sync_res

        # ---------------------------------------------------------- Module 5
        tick("Fusing features...")
        fused = fusion.fuse(audio.get("features"), visual.get("features"))
        result["fusion"] = fused["counts"] | {
            "missing_blocks": fused["missing_blocks"],
            "n_missing": len(fused["missing"]),
        }

        bundle = model.load_bundle(model_path)
        if bundle is not None:
            tick("Running trained classifier...")
            try:
                pred = model.predict(bundle, fused["vector"])
                pred["model_path"] = bundle.get("saved_to")
                result["prediction"] = pred
                result["mode"] = "classifier"
            except Exception as exc:
                result["warnings"].append(f"Classifier could not run: {exc}")
        result["model_available"] = bundle is not None
        result["mode_label"] = (
            "Trained classifier" if result["mode"] == "classifier"
            else "Feature analysis (no trained model)"
        )

        # ---------------------------------------------------------- plots
        if save_plots:
            tick("Generating plots...")
            plots = {}
            if audio.get("ok"):
                p = viz.plot_audio(str(wav_path), audio, workdir / "audio_overview.png")
                if p:
                    plots["audio_overview"] = p
                p = viz.plot_spectrogram(str(wav_path), workdir / "audio_spectrogram.png")
                if p:
                    plots["audio_spectrogram"] = p
            p = viz.plot_mouth(visual, workdir / "visual_mouth_motion.png")
            if p:
                plots["visual_mouth_motion"] = p
            p = viz.plot_sync(sync_res, workdir / "sync_correlation.png")
            if p:
                plots["sync_correlation"] = p
            p = viz.plot_feature_groups(fused["counts"]["audio_features"],
                                        fused["counts"]["visual_features"],
                                        workdir / "feature_vector.png")
            if p:
                plots["feature_vector"] = p
            result["plots"] = plots

        result["status"] = "ok"
        result["output_dir"] = str(workdir)
        result["feature_names"] = fused["names"]
        result["feature_vector"] = fused["vector"].tolist()
        result["analyzed_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        result["explanation"] = _explanation(result)

        # -------------------------------------------------- persist artifacts
        json_path = workdir / "result.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(result), fh, indent=2)
        result["result_json"] = str(json_path)
        try:
            (workdir / "report.md").write_text(result["explanation"], encoding="utf-8")
        except Exception:
            pass
        if wav_path.exists():
            with open(workdir / ".keep", "w") as fh:
                fh.write("")
        return result

    except VideoError as exc:
        result.update(status="error", code=exc.code, message=exc.message)
        result["explanation"] = f"### Cannot analyse this file\n\n**{exc.message}**"
        return result
    except Exception as exc:  # unexpected bug: report, do not crash the UI
        result.update(status="error", code="internal_error",
                      message=f"{type(exc).__name__}: {exc}")
        result["explanation"] = (
            "### Analysis failed\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Check that FFmpeg is available and the video is not corrupted."
        )
        return result


def environment_report() -> dict:
    """Small helper used by the UI and the notebook to show the setup."""
    import cv2
    import librosa
    import sklearn

    backend = "unavailable"
    try:
        import mediapipe as mp  # noqa: F401

        backend = f"mediapipe {mp.__version__}"
    except Exception:
        backend = "mediapipe not installed (Haar fallback)"

    ffmpeg, ffmpeg_src = find_ffmpeg()
    return {
        "python": __import__("sys").version.split()[0],
        "opencv": cv2.__version__,
        "librosa": librosa.__version__,
        "scikit_learn": sklearn.__version__,
        "face_backend": backend,
        "ffmpeg": ffmpeg_src,
        "model_found": config.DEFAULT_MODEL_PATH.exists(),
    }
