"""
Module 4 - Audio-visual synchronisation analysis.

Idea
----
Speech energy rises and falls as the mouth opens and closes. If the audio and
the mouth were produced by different sources (e.g. a dubbed/re-generated
audio track), the two signals tend to drift apart, so the correlation between
the audio energy envelope and the mouth-opening signal weakens or the best
alignment lag moves away from 0.

We therefore compute:
  * the Pearson correlation at zero lag,
  * the normalised cross-correlation over +/-SYNC_MAX_LAG_S and its peak lag.

The analysis runs at the frame sampling rate of the prototype (~12 fps, so a
lag resolution of about 80 ms) and reports one global lag for the clip.
"""

from __future__ import annotations

import numpy as np

from . import config


def _resample(values: np.ndarray, times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(times)
    values, times = values[finite], times[finite]
    order = np.argsort(times)
    values, times = values[order], times[order]
    return np.interp(grid, times, values)


def _z(x: np.ndarray) -> np.ndarray | None:
    s = float(np.std(x))
    if not np.isfinite(s) or s < 1e-9:
        return None
    return (x - float(np.mean(x))) / s


def unavailable(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "best_lag_ms": None,
        "peak_correlation": None,
        "zero_lag_correlation": None,
        "interpretation": (
            "Synchronisation analysis unavailable - " + reason
        ),
    }


def compute_av_sync(
    audio_times: np.ndarray,
    audio_env: np.ndarray,
    vis_times: np.ndarray,
    vis_series: np.ndarray,
    signal_label: str,
) -> dict:
    """Correlate the audio energy envelope with a mouth-motion signal."""
    audio_times = np.asarray(audio_times, dtype=float)
    audio_env = np.asarray(audio_env, dtype=float)
    vis_times = np.asarray(vis_times, dtype=float)
    vis_series = np.asarray(vis_series, dtype=float)

    ok = np.isfinite(audio_env) & np.isfinite(audio_times)
    okv = np.isfinite(vis_series) & np.isfinite(vis_times)
    if ok.sum() < 10:
        return unavailable("too little usable audio (need at least ~0.3 s).")
    if okv.sum() < 5:
        return unavailable("too few frames with usable mouth measurements.")
    audio_times, audio_env = audio_times[ok], audio_env[ok]
    vis_times, vis_series = vis_times[okv], vis_series[okv]

    t0 = max(float(audio_times.min()), float(vis_times.min()))
    t1 = min(float(audio_times.max()), float(vis_times.max()))
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 - t0) < config.SYNC_MIN_OVERLAP_S:
        return unavailable("audio and video overlap for less than "
                           f"{config.SYNC_MIN_OVERLAP_S:.1f}s.")

    grid = np.arange(t0, t1, 1.0 / config.SYNC_FPS)
    if grid.size < config.SYNC_MIN_SAMPLES:
        return unavailable("overlap too short after resampling.")

    a = _resample(audio_env, audio_times, grid)
    v = _resample(vis_series, vis_times, grid)
    az, vz = _z(a), _z(v)
    if az is None or vz is None:
        return unavailable("one of the signals is constant, correlation undefined.")

    full = np.correlate(vz, az, mode="full") / len(grid)
    lags_s = np.arange(-len(az) + 1, len(az)) / config.SYNC_FPS
    window = np.abs(lags_s) <= config.SYNC_MAX_LAG_S
    if not np.any(window):
        return unavailable("lag window smaller than one sample.")
    lags_w, corr_w = lags_s[window], full[window]
    peak_i = int(np.argmax(corr_w))
    best_lag_ms = float(lags_w[peak_i] * 1000.0)
    peak_corr = float(np.clip(corr_w[peak_i], -1.0, 1.0))
    zero_i = int(np.argmin(np.abs(lags_w)))
    zero_corr = float(np.clip(corr_w[zero_i], -1.0, 1.0))

    if abs(best_lag_ms) <= 80 and peak_corr >= 0.35:
        verdict = "aligned: best lag is within ~1 frame and correlation is moderate/strong."
    elif peak_corr >= 0.30 and abs(best_lag_ms) > 80:
        verdict = (f"shifted: best correlation at {best_lag_ms:+.0f} ms - "
                   "mouth motion trails/leads speech by more than one frame.")
    elif peak_corr < 0.15:
        verdict = "weak: mouth motion hardly tracks speech energy."
    else:
        verdict = "inconclusive: correlation is weak-to-moderate; inspect the plots."

    return {
        "status": "computed",
        "signal": signal_label,
        "n_samples": int(grid.size),
        "overlap_s": round(float(t1 - t0), 2),
        "best_lag_ms": round(best_lag_ms, 1),
        "peak_correlation": round(peak_corr, 3),
        "zero_lag_correlation": round(zero_corr, 3),
        "verdict": verdict,
        "interpretation": (
            f"Audio energy vs {signal_label}: zero-lag r={zero_corr:.3f}, "
            f"peak r={peak_corr:.3f} at {best_lag_ms:+.0f} ms. {verdict}"
        ),
        # data kept for plotting
        "grid": grid,
        "audio_norm": az,
        "mouth_norm": vz,
        "lags_ms": lags_w * 1000.0,
        "lags_corr": corr_w,
    }
