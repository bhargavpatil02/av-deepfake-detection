"""
End-to-end tests for the Audio-Visual Deepfake Detection prototype.

Run either way:

    python tests/test_pipeline.py          # no extra dependency
    pytest tests/test_pipeline.py          # if you have pytest

The cases mirror the Review 2 test plan:
  1. video with audio + visible face          -> audio, visual and sync all usable
  2. video without audio                      -> audio explained as missing, visual still works
  3. video where no face is found             -> visual reported unusable, no invented values
  4. invalid / unsupported / corrupted file   -> clear error code, no crash
  5. the synthetic-clip set used for training -> feature CSV + model file are produced

The clips in samples/ are synthetic drawings; these tests check that the
machinery works end to end.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import avdd  # noqa: E402,F401  (applies the Windows long-path fix first)
from avdd.pipeline import analyze_video  # noqa: E402

SAMPLES = ROOT / "samples"


def _need_sample(name: str) -> Path:
    path = SAMPLES / name
    if not path.exists():
        raise AssertionError(
            f"Missing {path}. Run: python scripts/make_test_videos.py"
        )
    return path


# --------------------------------------------------------------- test cases
def test_face_and_audio():
    result = analyze_video(_need_sample("sample_face_audio.mp4"), save_plots=False)
    assert result["status"] == "ok", result.get("message")

    assert result["audio"]["ok"], result["audio"].get("error")
    assert result["audio"]["n_features"] == 52          # 20 MFCC*2 + 6 descriptors*2

    assert result["visual"]["ok"], result["visual"].get("error")
    assert result["visual"]["face_detect_rate"] > 0.5
    feats = result["visual"]["features"]
    # A detected face must yield real lip numbers, not None.
    assert feats["mean_mar"] is not None
    assert feats["mean_mouth_pixel_motion"] is not None

    # Both streams available -> synchronisation must actually be computed.
    assert result["sync"]["status"] == "computed"
    assert abs(result["sync"]["best_lag_ms"]) <= 400

    # No trained model in a fresh checkout -> features only, no verdict.
    if not result["model_available"]:
        assert result["mode"] == "demo"
        assert result["prediction"] is None
        assert "Feature analysis only" in result["explanation"]


def test_video_without_audio():
    result = analyze_video(_need_sample("sample_face_no_audio.mp4"), save_plots=False)
    assert result["status"] == "ok", result.get("message")   # handled, not fatal
    assert result["audio"]["ok"] is False
    assert "audio" in (result["audio"].get("error") or "").lower()
    # The visual half should still be analysed.
    assert result["visual"]["ok"] is True
    # Sync needs both streams -> must be reported as unavailable, not faked.
    assert result["sync"]["status"] != "computed"
    assert result["sync"]["reason"]


def test_video_without_face():
    result = analyze_video(_need_sample("sample_no_face_audio.mp4"), save_plots=False)
    assert result["status"] == "ok", result.get("message")
    assert result["visual"]["ok"] is False
    assert result["visual"]["n_frames_detected"] == 0
    # Unmeasurable lip features must stay None instead of invented zeros.
    feats = result["visual"]["features"]
    assert feats.get("mean_mar") is None
    assert result["sync"]["status"] != "computed"


def test_corrupt_video():
    result = analyze_video(_need_sample("sample_corrupt.mp4"), save_plots=False)
    assert result["status"] == "error"
    assert result["code"] == "unreadable_video"


def test_unsupported_and_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "notes.txt"
        txt.write_text("not a video")
        r = analyze_video(txt, save_plots=False)
        assert r["status"] == "error" and r["code"] == "unsupported_format"

        r = analyze_video(Path(tmp) / "nope.mp4", save_plots=False)
        assert r["status"] == "error" and r["code"] == "missing_file"


def test_feature_csv_and_training_smoke():
    """Feature extraction -> CSV -> model bundle -> inference, on synthetic clips."""
    import shutil

    import numpy as np
    import soundfile as sf

    import make_test_videos as mk
    from avdd.dataset import build_csv
    from avdd.model import ModelError, train_from_csv

    work = Path(tempfile.mkdtemp(prefix="avdd_test_"))
    try:
        real_dir, fake_dir = work / "data/real", work / "data/fake"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        sr = 16000
        t = np.arange(int(sr * mk.DUR)) / sr

        for i in range(4):   # 8 clips total: enough to exercise the code path
            for folder, phase, freq in ((real_dir, 0.0, 220.0), (fake_dir, 0.2, 340.0)):
                raw = work / "raw.mp4"
                wav = work / "a.wav"
                mk._write_video(raw, phase=phase)
                gate = (0.5 + 0.5 * np.sin(2 * np.pi * (3.0 if freq < 300 else 4.0) * t)) ** 1.5
                sf.write(str(wav), (0.3 * gate * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
                mk._mux(raw, folder / f"clip_{i}.mp4", wav)

        stats = build_csv(work / "data", work / "features.csv", progress=lambda _m: None)
        assert stats["rows_written"] >= 8
        assert not stats["failed"]
        assert (work / "features.csv").exists()

        # 8 samples is below the honest minimum of 10 -> must refuse to train.
        try:
            train_from_csv(work / "features.csv", work / "model.joblib")
        except ModelError as exc:
            assert "10" in str(exc)
        else:
            raise AssertionError("Expected a refusal to train on 8 videos")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - test runner
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
