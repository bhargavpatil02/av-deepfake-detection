"""
Command-line demo: analyse one video end to end and print the report.

Usage
-----
    python scripts/run_demo.py samples/sample_face_audio.mp4
    python scripts/run_demo.py my_video.mp4 --backend opencv_haar
    python scripts/run_demo.py my_video.mp4 --model models/av_deepfake_rf.joblib
    python scripts/run_demo.py my_video.mp4 --no-plots

Plots and result.json are written to outputs/<video_name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avdd import config  # noqa: E402
from avdd.pipeline import analyze_video, environment_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse a video for AV deepfake cues.")
    ap.add_argument("video", help="path to the video file")
    ap.add_argument("--model", default=None,
                    help="model bundle (default: models/av_deepfake_rf.joblib if present)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "mediapipe", "opencv_haar"])
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of a report")
    args = ap.parse_args()

    if args.json:
        import json

        result = analyze_video(args.video, model_path=args.model, backend=args.backend,
                               save_plots=not args.no_plots)
        safe = {k: v for k, v in result.items()
                if k not in ("plots", "explanation", "feature_vector", "feature_names")}
        print(json.dumps(safe, indent=2, default=str))
        return 0 if result.get("status") == "ok" else 1

    env = environment_report()
    print("=" * 72)
    print(" Audio-Visual Deepfake Detection - prototype")
    print("=" * 72)
    print(f" OpenCV {env['opencv']} | Librosa {env['librosa']} | "
          f"scikit-learn {env['scikit_learn']}")
    print(f" Face backend: {env['face_backend']} | FFmpeg: {env['ffmpeg']}")
    print(f" Trained model: {'found' if env['model_found'] else 'none (demo mode)'}")
    print(f" Analysing: {args.video}")
    print("-" * 72)

    def progress(msg: str) -> None:
        print(f"  ... {msg}")

    result = analyze_video(args.video, model_path=args.model, backend=args.backend,
                           save_plots=not args.no_plots, progress=progress)

    print("-" * 72)
    if result.get("status") != "ok":
        print(f" FAILED [{result.get('code')}]: {result.get('message')}")
        return 1

    print(result["explanation"].replace("###", "\n##").replace("**", ""))
    print("-" * 72)
    print(f" Outputs: {result.get('output_dir')}")
    for name, path in (result.get("plots") or {}).items():
        print(f"   {name:22s} {path}")
    print(f" Feature vector: {len(result['feature_names'])} dimensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
