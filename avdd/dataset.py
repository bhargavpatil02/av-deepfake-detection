"""
Dataset helpers: turn a folder of labelled videos into a feature CSV.

Expected layout (folder names are the labels):

    data/
      real/  real_001.mp4  real_002.mp4 ...
      fake/  fake_001.mp4  fake_002.mp4 ...

Frames from the same video stay together as one row, and the train/test split in
`model.train_from_csv` is done on rows (videos), never on frames, so there is no
leakage between a training video and a test video.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from . import config, fusion
from .audio_feats import extract_audio_features
from .io_video import (
    VideoError,
    extract_audio,
    iter_sampled_frames,
    sampling_summary,
    validate_video,
)
from .visual_feats import extract_visual_features

LABELS = {"real": "real", "genuine": "real", "fake": "fake", "manipulated": "fake"}


def list_videos(data_dir: str | Path) -> list[tuple[str, Path]]:
    data_dir = Path(data_dir)
    items: list[tuple[str, Path]] = []
    for folder, label in LABELS.items():
        d = data_dir / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                items.append((label, p))
    return items


def extract_row(video_path: str | Path, label: str, backend: str = "auto") -> dict:
    """
    Extract one fused feature row for a video.

    Returns {"label", "source_file", <feature columns...>, "__error__"?}.
    Missing features are NaN, never invented values.
    """
    row: dict = {"label": label, "source_file": Path(video_path).name}
    tmpdir = Path(tempfile.mkdtemp(prefix="avdd_row_"))
    try:
        meta = sampling_summary(validate_video(video_path))
        wav = tmpdir / "a.wav"
        audio_ext = extract_audio(meta.path, wav)
        audio = extract_audio_features(wav) if audio_ext["ok"] else {"ok": False}

        frames = iter_sampled_frames(meta, resize_width=config.VISUAL_WORK_WIDTH)
        visual = extract_visual_features(frames, backend=backend)

        fused = fusion.fuse(audio.get("features") if audio.get("ok") else None,
                            visual.get("features") if visual.get("ok") else None)
        for name, value in zip(fused["names"], fused["vector"]):
            row[name] = float(value) if np.isfinite(value) else np.nan
        row["audio_ok"] = bool(audio.get("ok"))
        row["visual_ok"] = bool(visual.get("ok"))
        row["visual_backend"] = visual.get("backend")
        return row
    except VideoError as exc:
        row["__error__"] = f"{exc.code}: {exc.message}"
        return row
    except Exception as exc:
        row["__error__"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        for f in tmpdir.glob("*"):
            f.unlink(missing_ok=True)
        tmpdir.rmdir()


def build_csv(
    data_dir: str | Path,
    out_csv: str | Path,
    backend: str = "auto",
    limit: int | None = None,
    progress=print,
) -> dict:
    """Extract features for every labelled video and write a CSV."""
    items = list_videos(data_dir)
    if limit:
        items = items[:limit]
    if not items:
        raise FileNotFoundError(
            f"No labelled videos found. Create {data_dir}/real/ and {data_dir}/fake/ "
            "and put .mp4 files inside."
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    audio_ok = visual_ok = 0

    for i, (label, path) in enumerate(items, 1):
        progress(f"[{i}/{len(items)}] {label:4s} {path.name}")
        row = extract_row(path, label, backend=backend)
        if "__error__" in row:
            failed.append((path.name, row["__error__"]))
            progress(f"        skipped: {row['__error__']}")
            continue
        audio_ok += int(row.get("audio_ok", False))
        visual_ok += int(row.get("visual_ok", False))
        rows.append(row)

    if not rows:
        raise RuntimeError("Every video failed; nothing to write.")

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "videos_found": len(items),
        "rows_written": len(rows),
        "failed": failed,
        "audio_ok": audio_ok,
        "visual_ok": visual_ok,
        "csv": str(out_csv),
    }
    progress(
        f"\nWrote {len(rows)} rows to {out_csv}\n"
        f"  audio usable: {audio_ok}/{len(rows)} | visual usable: {visual_ok}/{len(rows)}"
    )
    if failed:
        progress(f"  {len(failed)} video(s) failed and were skipped.")
    return stats
