"""
Module 1 - Video input and preprocessing.

Responsibilities
----------------
* Validate an uploaded file (extension, size, decodability).
* Probe basic metadata with OpenCV (fps, frame count, resolution).
* Extract the audio track to a mono 16 kHz WAV with FFmpeg.
  A system FFmpeg is used when available, otherwise the binary bundled
  with the `imageio-ffmpeg` pip package, so no manual install is needed.
* Stream sampled frames with a single decode pass (memory friendly).

Every failure mode raises `VideoError`, which carries a short machine
readable `code` so the pipeline/UI can explain what went wrong.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import config


class VideoError(Exception):
    """Raised for any user-facing input problem (bad file, no audio, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    analysed_fps: float = 0.0
    stride: int = 1
    analysed_frames: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": Path(self.path).name,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "resolution": f"{self.width}x{self.height}",
            "duration_s": round(self.duration_s, 3),
            "analysed_fps": round(self.analysed_fps, 3),
            "frame_stride": self.stride,
            "analysed_frames": self.analysed_frames,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------- ffmpeg
def find_ffmpeg() -> tuple[str | None, str]:
    """Return (path_to_ffmpeg, source_label)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe, "system"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"
    except Exception:  # package missing or no binary for this platform
        return None, "missing"


# ------------------------------------------------------------------ validation
def validate_video(path: str | Path) -> VideoMeta:
    """Check the file is a supported, decodable video and return its metadata."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise VideoError("missing_file", f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise VideoError(
            "unsupported_format",
            f"Unsupported format '{ext or '(none)'}'. Supported: {allowed}",
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise VideoError("empty_file", "The file is empty (0 bytes).")
    if size_mb > config.MAX_FILE_SIZE_MB:
        raise VideoError(
            "file_too_large",
            f"File is {size_mb:.1f} MB; limit is {config.MAX_FILE_SIZE_MB} MB.",
        )

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(
                "unreadable_video",
                "OpenCV could not open the file. It may be corrupted or use an "
                "unsupported codec.",
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(
                "unreadable_video",
                "The video opened but no frame could be decoded.",
            )
    finally:
        cap.release()

    warnings: list[str] = []
    if fps <= 0 or not np.isfinite(fps):
        fps = 25.0
        warnings.append("Container reports no FPS; assuming 25 fps.")
    if frame_count <= 0:
        warnings.append("Container reports no frame count; duration is approximate.")

    duration = frame_count / fps if frame_count > 0 else 0.0
    if duration <= 0:
        raise VideoError("empty_video", "The video contains no frames.")
    if duration > config.MAX_ANALYSIS_DURATION_S:
        warnings.append(
            f"Video is {duration:.1f}s; only the first "
            f"{config.MAX_ANALYSIS_DURATION_S:.0f}s are analysed."
        )

    return VideoMeta(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width or frame.shape[1],
        height=height or frame.shape[0],
        duration_s=duration,
        warnings=warnings,
    )


# --------------------------------------------------------------- frame sampling
def plan_sampling(meta: VideoMeta) -> tuple[int, float]:
    """Decide the frame stride so we get ~VISUAL_FPS_TARGET with a hard cap."""
    stride = max(1, int(round(meta.fps / config.VISUAL_FPS_TARGET)))
    usable = min(meta.duration_s, config.MAX_ANALYSIS_DURATION_S)
    n_planned = max(1, int(usable * meta.fps / stride))
    while n_planned > config.VISUAL_MAX_FRAMES:  # keep runtime bounded
        stride += 1
        n_planned = max(1, int(usable * meta.fps / stride))
    return stride, meta.fps / stride


def iter_sampled_frames(
    meta: VideoMeta, resize_width: int | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame) for every stride-th frame, one decode pass."""
    stride, _ = plan_sampling(meta)
    cap = cv2.VideoCapture(meta.path)
    idx = 0
    emitted = 0
    try:
        while emitted < config.VISUAL_MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / meta.fps
            if t > config.MAX_ANALYSIS_DURATION_S:
                break
            if idx % stride == 0:
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                yield t, frame
                emitted += 1
            idx += 1
    finally:
        cap.release()

    if emitted == 0:
        raise VideoError("unreadable_video", "No frames could be decoded.")


def sampling_summary(meta: VideoMeta) -> VideoMeta:
    """Fill in the sampling fields on the metadata object."""
    stride, analysed_fps = plan_sampling(meta)
    usable = min(meta.duration_s, config.MAX_ANALYSIS_DURATION_S)
    meta.stride = stride
    meta.analysed_fps = analysed_fps
    meta.analysed_frames = int(usable * meta.fps / stride)
    return meta


# ---------------------------------------------------------------------- audio
def extract_audio(video_path: str | Path, out_wav: str | Path) -> dict:
    """
    Extract the audio track to mono 16 kHz WAV.

    Returns {"ok": bool, "path": str|None, "source": str, "message": str}.
    Never raises: a video without audio is a normal, handled case.
    """
    video_path = Path(video_path)
    out_wav = Path(out_wav)
    ffmpeg, source = find_ffmpeg()
    if ffmpeg is None:
        return {
            "ok": False,
            "path": None,
            "source": source,
            "message": "FFmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.",
        }

    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-i", str(video_path),
        "-vn",                              # drop video
        "-ac", "1",                         # mono
        "-ar", str(config.AUDIO_SR),        # 16 kHz
        "-f", "wav",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0 or not out_wav.exists():
        if "does not contain any stream" in stderr or "Output file is empty" in stderr:
            return {"ok": False, "path": None, "source": source,
                    "message": "The video has no audio track."}
        tail = stderr.splitlines()[-3:] if stderr else ["unknown error"]
        return {"ok": False, "path": None, "source": source,
                "message": "Audio extraction failed: " + " | ".join(tail)}

    # Confirm the WAV really holds samples (some silent tracks decode to ~0 length).
    try:
        with wave.open(str(out_wav), "rb") as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
        duration = n_frames / float(sr) if sr else 0.0
    except Exception as exc:
        return {"ok": False, "path": None, "source": source,
                "message": f"Extracted WAV could not be read: {exc}"}

    if duration < 0.05:
        return {"ok": False, "path": None, "source": source,
                "message": "The audio track is empty or shorter than 50 ms."}

    return {"ok": True, "path": str(out_wav), "source": source,
            "message": f"Audio extracted ({duration:.2f}s @ {config.AUDIO_SR} Hz)."}
