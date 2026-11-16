"""
Module 2 - Audio feature extraction with Librosa.

Frame-level descriptors are computed for the whole clip, then aggregated to a
fixed-length vector (mean + standard deviation per descriptor) so that any clip
length yields the same number of numbers for the classifier.

We also keep the frame-level RMS envelope: it is the speech-energy signal used
by Module 4 (audio-visual synchronisation) and drawn in the UI plots.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from . import config

AUDIO_FEATURE_NAMES: list[str] = []  # filled by build_feature_names()


def build_feature_names() -> list[str]:
    """Stable, ordered names for the aggregated audio vector."""
    names = [f"mfcc{i}_{stat}" for i in range(config.AUDIO_N_MFCC) for stat in ("mean", "std")]
    for base in ("zcr", "spec_centroid", "spec_rolloff", "rms", "spec_bw", "spec_flatness"):
        names += [f"{base}_mean", f"{base}_std"]
    return names


def _safe(m: np.ndarray, name: str, out: dict) -> None:
    """Aggregate one frame-level matrix into mean/std, guarding NaN/inf."""
    arr = np.asarray(m, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out[f"{name}_mean"] = float(arr.mean())
    if arr.ndim == 1:
        out[f"{name}_std"] = float(arr.std())
    else:
        # 2-D (e.g. MFCC coeffs x frames): aggregate over time only.
        out[f"{name}_mean"] = float(arr.mean(axis=1).mean())


def extract_audio_features(wav_path: str | Path) -> dict:
    """
    Extract audio features from a mono WAV file.

    Returns {"ok": bool, ...}. On success it contains:
      features : dict[str, float]      aggregated fixed-length descriptors
      names    : list[str]             aligned feature order
      times    : np.ndarray            frame times (seconds)
      rms      : np.ndarray            frame-level RMS envelope
      summary  : dict                  human-readable numbers for the UI
      error    : str                   only when ok is False
    """
    wav_path = Path(wav_path)
    result: dict = {"ok": False, "features": {}, "names": build_feature_names()}
    try:
        y, sr = librosa.load(str(wav_path), sr=config.AUDIO_SR, mono=True)
    except Exception as exc:
        result["error"] = f"Could not decode audio: {exc}"
        return result

    if y.size == 0 or not np.any(np.isfinite(y)):
        result["error"] = "Decoded audio contains no samples."
        return result
    duration = len(y) / float(sr)
    if duration < 0.10:
        result["error"] = f"Audio too short for analysis ({duration:.2f}s)."
        return result
    if float(np.max(np.abs(y))) < 1e-5:
        result["error"] = "Audio track is silent (all samples ~0)."
        return result

    kwargs = dict(n_fft=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP)
    feats: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.AUDIO_N_MFCC, **kwargs)
    for i in range(config.AUDIO_N_MFCC):
        feats[f"mfcc{i}_mean"] = float(np.nan_to_num(mfcc[i].mean()))
        feats[f"mfcc{i}_std"] = float(np.nan_to_num(mfcc[i].std()))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=config.AUDIO_HOP, frame_length=config.AUDIO_N_FFT)
    _safe(zcr[0], "zcr", feats)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, **kwargs)[0]
    _safe(centroid, "spec_centroid", feats)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, **kwargs)[0]
    _safe(rolloff, "spec_rolloff", feats)

    rms = librosa.feature.rms(y=y, frame_length=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP)[0]
    _safe(rms, "rms", feats)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, **kwargs)[0]
    _safe(bandwidth, "spec_bw", feats)

    flatness = librosa.feature.spectral_flatness(y=y, **kwargs)[0]
    _safe(flatness, "spec_flatness", feats)

    times = librosa.frames_to_time(np.arange(rms.shape[0]), sr=sr, hop_length=config.AUDIO_HOP)

    # Keep only the declared names, in order, so the vector is reproducible.
    names = build_feature_names()
    missing = [n for n in names if n not in feats]
    if missing:
        result["error"] = f"Internal feature mismatch, missing: {missing[:5]}"
        return result
    vector = np.array([feats[n] for n in names], dtype=np.float64)

    result.update(
        ok=True,
        features={n: float(feats[n]) for n in names},
        names=names,
        vector=vector,
        times=times,
        rms=np.nan_to_num(rms.astype(np.float64)),
        duration=duration,
        sr=sr,
        summary={
            "duration_s": round(duration, 2),
            "sample_rate_hz": sr,
            "n_frames": int(rms.shape[0]),
            "mean_rms": round(float(feats["rms_mean"]), 5),
            "mean_zcr": round(float(feats["zcr_mean"]), 4),
            "mean_spectral_centroid_hz": round(float(feats["spec_centroid_mean"]), 1),
            "mean_spectral_rolloff_hz": round(float(feats["spec_rolloff_mean"]), 1),
            "mean_spectral_flatness": round(float(feats["spec_flatness_mean"]), 5),
            "n_features": len(names),
        },
    )
    return result
