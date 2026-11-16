"""
Train the baseline classifier from labelled videos.

Usage
-----
    python scripts/train_model.py                     # data/real + data/fake
    python scripts/train_model.py --data path/to/data
    python scripts/train_model.py --csv reports/features.csv --reuse-csv

It writes:
    reports/features.csv        one row per video (all fused features)
    reports/metrics.json        held-out metrics + feature importances
    models/av_deepfake_rf.joblib  the bundle used by the UI / pipeline

If fewer than 10 usable videos are found, the script stops and tells you to keep
collecting data - it will not fabricate a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avdd import config  # noqa: E402
from avdd.dataset import build_csv, list_videos  # noqa: E402
from avdd.model import ModelError, train_from_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Random Forest baseline.")
    ap.add_argument("--data", default=str(config.DATA_DIR), help="folder with real/ and fake/")
    ap.add_argument("--csv", default=str(config.REPORTS_DIR / "features.csv"))
    ap.add_argument("--out", default=str(config.DEFAULT_MODEL_PATH))
    ap.add_argument("--backend", default="auto", choices=["auto", "mediapipe", "opencv_haar"])
    ap.add_argument("--reuse-csv", action="store_true",
                    help="skip feature extraction and train on an existing CSV")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N videos")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not args.reuse_csv:
        videos = list_videos(args.data)
        if not videos:
            print(f"No videos found under {args.data}/real and {args.data}/fake.")
            print("Add labelled clips, or run the demo mode:")
            print("   python scripts/run_demo.py samples/sample_face_audio.mp4")
            return 2
        print(f"Extracting features from {len(videos)} videos "
              f"(backend={args.backend}) ...\n")
        stats = build_csv(args.data, csv_path, backend=args.backend, limit=args.limit)
        if stats["rows_written"] < 10:
            print(f"\nOnly {stats['rows_written']} usable rows. Collect more videos "
                  "before training; nothing was saved.")
            return 2

    try:
        bundle = train_from_csv(csv_path, args.out)
    except ModelError as exc:
        print(f"Training aborted: {exc}")
        return 2

    m = bundle["metrics"]
    print("\n=== Held-out test results ===")
    print(f"  videos used      : {m['n_samples']} (real={m['n_real']}, fake={m['n_fake']})")
    print(f"  train / test     : {m['n_train']} / {m['n_test']}")
    print(f"  accuracy         : {m['accuracy']:.4f}")
    print(f"  precision (fake) : {m['precision']:.4f}")
    print(f"  recall    (fake) : {m['recall']:.4f}")
    print(f"  F1        (fake) : {m['f1']:.4f}")
    if "roc_auc" in m:
        print(f"  ROC-AUC          : {m['roc_auc']:.4f}")
    if "cv_accuracy_mean" in m:
        print(f"  {m['cv_folds']}-fold CV (train split): "
              f"{m['cv_accuracy_mean']:.4f} +/- {m['cv_accuracy_std']:.4f}")
    tn, fp = m["confusion_matrix"][0]
    fn, tp = m["confusion_matrix"][1]
    print("\n  Confusion matrix (rows = truth, cols = prediction)")
    print("                 pred real   pred fake")
    print(f"    true real    {tn:9d}   {fp:9d}")
    print(f"    true fake    {fn:9d}   {tp:9d}")
    print("\n  Top features:")
    for f in bundle["top_features"][:8]:
        print(f"    {f['feature']:32s} {f['importance']:.4f}")

    out = Path(args.out)
    metrics_path = config.REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(
        {"metrics": m, "top_features": bundle["top_features"],
         "trained_at": bundle["trained_at"], "dataset_csv": str(csv_path)},
        indent=2), encoding="utf-8")
    print(f"\nModel saved to : {out}")
    print(f"Metrics saved  : {metrics_path}")
    print("\nReminder: these numbers describe THIS dataset and split only. Do not")
    print("present them as general deepfake-detection performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
