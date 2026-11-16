"""
Matplotlib visualisations for the UI and the report.

All figures are written to PNG files (Agg backend, so this works headless in
Colab and on a server) and returned as paths; the Gradio UI displays them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_audio(wav_path: str, audio: dict, out_png: Path) -> str | None:
    """Waveform + RMS envelope + spectral centroid over time."""
    try:
        import librosa
        import librosa.display

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 5.6), sharex=True)

        t = np.arange(len(y)) / sr
        axes[0].plot(t, y, linewidth=0.4, color="#2563eb")
        axes[0].set_ylabel("amplitude")
        axes[0].set_title("Audio waveform (16 kHz mono)")

        hits = np.where(audio["rms"] > 0.5 * audio["rms"].max())[0]
        axes[1].plot(audio["times"], audio["rms"], color="#16a34a", linewidth=0.9)
        axes[1].set_ylabel("RMS energy")
        axes[1].set_title(f"RMS energy envelope  (speech-active frames: {len(hits)})")

        import librosa.feature as lf

        from . import config

        kwargs = dict(n_fft=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP)
        cent = lf.spectral_centroid(y=y, sr=sr, **kwargs)[0]
        ct = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=config.AUDIO_HOP)
        ax = axes[2]
        ax.plot(ct, cent, color="#9333ea", linewidth=0.8)
        ax.set_ylabel("Hz")
        ax.set_xlabel("time (s)")
        ax.set_title("Spectral centroid")
        fig.suptitle("Module 2 - audio features", fontsize=10, y=1.0)
        return _save(fig, out_png)
    except Exception:
        return None


def plot_spectrogram(wav_path: str, out_png: Path) -> str | None:
    """Mel spectrogram + MFCC heatmap."""
    try:
        import librosa
        import librosa.display

        from . import config

        y, sr = librosa.load(wav_path, sr=config.AUDIO_SR, mono=True)
        kwargs = dict(n_fft=config.AUDIO_N_FFT, hop_length=config.AUDIO_HOP)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, **kwargs)
        S_db = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.AUDIO_N_MFCC, **kwargs)

        fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=config.AUDIO_HOP,
                                       x_axis="time", ax=axes[0], cmap="magma")
        axes[0].set_title("Mel spectrogram (dB)")
        fig.colorbar(img, ax=axes[0], format="%+2.0f dB")

        img2 = librosa.display.specshow(mfcc, sr=sr, hop_length=config.AUDIO_HOP,
                                        x_axis="time", ax=axes[1], cmap="coolwarm")
        axes[1].set_title("MFCC (20 coefficients)")
        fig.colorbar(img2, ax=axes[1])
        return _save(fig, out_png)
    except Exception:
        return None


def plot_mouth(visual: dict, out_png: Path) -> str | None:
    """Per-frame mouth-opening / motion signals with detection overlay."""
    try:
        records = visual.get("records") or []
        if not records:
            return None
        times = np.array([r["t"] for r in records], dtype=float)
        detected = np.array([bool(r["detected"]) for r in records])

        def series(key):
            vals = np.array([np.nan if r.get(key) is None else r[key] for r in records],
                            dtype=float)
            return vals

        fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
        ax = axes[0]
        opened = series("mouth_open_norm")
        mar = series("mar")
        plotted = False
        if np.isfinite(opened).any():
            ax.plot(times, opened, "o-", ms=2.5, lw=0.9, label="mouth opening / face height")
            plotted = True
        if np.isfinite(mar).any():
            ax.plot(times, mar, "s-", ms=2, lw=0.8, color="#f59e0b", label="mouth aspect ratio")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No landmark backend - mouth opening not measurable\n"
                              "(Haar backend: face box only)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
        else:
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylabel("normalised")
        ax.set_title(f"Module 3 - lip signal (backend: {visual.get('backend')}, "
                     f"faces in {int(detected.sum())}/{len(detected)} frames)")

        ax2 = axes[1]
        motion = series("motion")
        if np.isfinite(motion).any():
            ax2.plot(times, motion, color="#dc2626", lw=1.0, label="mouth pixel motion / s")
            ax2.legend(loc="upper right", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "No mouth motion measured", transform=ax2.transAxes,
                     ha="center", va="center")
        # shade frames without a detected face so failures are visible, not hidden
        for t, det in zip(times, detected):
            if not det:
                for a in axes:
                    a.axvspan(t - 0.02, t + 0.02, color="#e5e7eb", alpha=0.5, lw=0)
        ax2.set_xlabel("time (s)  |  grey = no face detected")
        ax2.set_title("Frame-to-frame mouth-region motion")
        return _save(fig, out_png)
    except Exception:
        return None


def plot_sync(sync: dict, out_png: Path) -> str | None:
    """Overlay the two signals and draw the cross-correlation curve."""
    if sync.get("status") != "computed":
        return None
    try:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=False)
        grid = sync["grid"]
        axes[0].plot(grid, sync["audio_norm"], lw=1.0, label="audio energy (z)")
        axes[0].plot(grid, sync["mouth_norm"], lw=1.0, label="mouth signal (z)")
        axes[0].set_title(f"Module 4 - signals over the overlap ({sync['signal']})")
        axes[0].set_xlabel("time (s)")
        axes[0].legend(fontsize=7)

        axes[1].plot(sync["lags_ms"], sync["lags_corr"], color="#0f766e", lw=1.2)
        axes[1].axvline(0, color="grey", ls="--", lw=0.8, label="zero lag")
        axes[1].axvline(sync["best_lag_ms"], color="#dc2626", ls="-", lw=0.9,
                        label=f"best {sync['best_lag_ms']:+.0f} ms")
        axes[1].set_xlabel("lag (ms)")
        axes[1].set_ylabel("correlation")
        axes[1].set_title(f"Cross-correlation (peak r={sync['peak_correlation']:.2f})")
        axes[1].legend(fontsize=7)
        return _save(fig, out_png)
    except Exception:
        return None


def plot_feature_groups(audio_names: int, visual_names: int, out_png: Path) -> str | None:
    """Simple bar chart of how many features each modality contributes."""
    try:
        fig, ax = plt.subplots(figsize=(4.4, 2.6))
        ax.bar(["audio", "visual", "fused"], [audio_names, visual_names,
                                              audio_names + visual_names],
               color=["#2563eb", "#f59e0b", "#334155"])
        ax.set_ylabel("number of features")
        ax.set_title("Fused feature vector")
        return _save(fig, out_png)
    except Exception:
        return None
