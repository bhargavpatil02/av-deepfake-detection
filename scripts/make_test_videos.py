"""
Generate the synthetic test videos used by tests/test_pipeline.py.

    python scripts/make_test_videos.py

Creates (in samples/):
  sample_face_audio.mp4          animated cartoon face + 3 Hz pulsed tone
  sample_face_audio_desynced.mp4 same face, audio shifted 300 ms (sync test)
  sample_face_no_audio.mp4       same face, silent
  sample_no_face_audio.mp4       FFmpeg test pattern + tone (no face at all)
  sample_corrupt.mp4             random bytes with a .mp4 extension

Why synthetic? No external dataset is downloaded. These clips exercise the
error paths and the audio/lip pipeline. They are NOT deepfakes: use them to show
that features are extracted, never to claim detection accuracy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avdd.io_video import find_ffmpeg  # noqa: E402

SAMPLES = ROOT / "samples"
W, H, FPS, DUR = 640, 480, 25, 6


def _face_frame(t: float, phase: float = 0.0, rng=None) -> np.ndarray:
    """Draw a simple frontal face; the mouth opens at 3 Hz."""
    frame = np.full((H, W, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (W, H), (70, 60, 55), -1)

    bob = int(3 * np.sin(2 * np.pi * 0.7 * t))
    cx, cy = W // 2, H // 2 - 10 + bob
    skin = (168, 196, 226)

    # head + ears + neck
    cv2.ellipse(frame, (cx, cy), (120, 155), 0, 0, 360, skin, -1)
    cv2.ellipse(frame, (cx - 120, cy + 10), (18, 36), 0, 0, 360, skin, -1)
    cv2.ellipse(frame, (cx + 120, cy + 10), (18, 36), 0, 0, 360, skin, -1)
    cv2.rectangle(frame, (cx - 34, cy + 150), (cx + 34, H), (150, 178, 208), -1)

    # eyebrows, eyes
    for sx in (-45, 45):
        cv2.ellipse(frame, (cx + sx, cy - 40), (26, 12), 0, 0, 360, (55, 55, 60), -1)
        cv2.ellipse(frame, (cx + sx, cy - 8), (22, 14), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(frame, (cx + sx, cy - 8), 8, (60, 45, 35), -1)
        cv2.circle(frame, (cx + sx, cy - 8), 3, (10, 10, 10), -1)
    # nose
    cv2.ellipse(frame, (cx, cy + 45), (12, 22), 0, 0, 360, (140, 172, 205), -1)

    # mouth: height pulsing at 3 Hz (this is the lip signal Module 3 measures)
    open_amt = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * (t - phase))
    mouth_h = int(6 + 30 * open_amt)
    cv2.ellipse(frame, (cx, cy + 105), (52, mouth_h), 0, 0, 360, (30, 30, 70), -1)
    cv2.ellipse(frame, (cx, cy + 105 - mouth_h // 3), (44, max(3, mouth_h // 3)),
                0, 0, 360, (60, 70, 120), -1)
    # teeth hint
    cv2.rectangle(frame, (cx - 40, cy + 105 - mouth_h),
                  (cx + 40, cy + 105 - mouth_h + 5), (235, 235, 235), -1)

    if rng is not None:
        noise = rng.normal(0, 4, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    cv2.putText(frame, "SYNTHETIC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return frame


def _write_video(path: Path, phase: float = 0.0) -> None:
    rng = np.random.default_rng(0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * DUR):
        out.write(_face_frame(i / FPS, phase=phase, rng=rng))
    out.release()


def _pulse_wav(path: Path) -> None:
    """220 Hz tone gated at 3 Hz - a crude stand-in for speech energy."""
    import soundfile as sf

    sr = 16000
    t = np.arange(int(sr * DUR)) / sr
    gate = (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) ** 1.5
    tone = 0.35 * gate * np.sin(2 * np.pi * 220 * t)
    sf.write(str(path), tone.astype(np.float32), sr)


def _mux(video_in: Path, video_out: Path, audio: Path | None) -> bool:
    ffmpeg, _ = find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(video_in)]
    if audio is not None:
        cmd += ["-i", str(audio), "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", str(video_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("   ffmpeg error:", proc.stderr.strip()[-300:])
        return False
    return True


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ffmpeg, src = find_ffmpeg()
    if ffmpeg is None:
        print("FFmpeg not available; install imageio-ffmpeg first.")
        return 1
    print(f"Using FFmpeg from: {src}")

    tone = SAMPLES / "_tone.wav"
    _pulse_wav(tone)

    print("1/5 face + synchronised audio ...")
    tmp = SAMPLES / "_face_sync_raw.mp4"
    _write_video(tmp)
    ok = _mux(tmp, SAMPLES / "sample_face_audio.mp4", tone)

    print("2/5 face + audio delayed 300 ms (desync demo) ...")
    slow = SAMPLES / "_tone_shifted.wav"
    import soundfile as sf

    y, sr = sf.read(str(tone))
    shift = int(0.3 * sr)
    sf.write(str(slow), np.concatenate([np.zeros(shift), y])[: len(y)].astype(np.float32), sr)
    tmp2 = SAMPLES / "_face_desync_raw.mp4"
    _write_video(tmp2, phase=0.0)
    ok = _mux(tmp2, SAMPLES / "sample_face_audio_desynced.mp4", slow) and ok

    print("3/5 face without audio ...")
    _write_video(SAMPLES / "sample_face_no_audio.mp4")

    print("4/5 test pattern (no face) + audio ...")
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={DUR}",
           "-i", str(tone), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-loglevel", "error",
           str(SAMPLES / "sample_no_face_audio.mp4")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("   ffmpeg error:", proc.stderr.strip()[-300:])

    print("5/5 corrupt file ...")
    rng = np.random.default_rng(1)
    (SAMPLES / "sample_corrupt.mp4").write_bytes(rng.integers(0, 255, 20000, dtype=np.uint8).tobytes())

    for f in SAMPLES.glob("_*"):
        f.unlink(missing_ok=True)

    print("\nCreated:")
    for f in sorted(SAMPLES.glob("sample_*.mp4")):
        print(f"  {f.name:38s} {f.stat().st_size / 1024:8.1f} KB")
    print("\nNote: these are synthetic clips, not deepfakes. Use them to test the")
    print("pipeline and error handling, not to measure detection accuracy.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
