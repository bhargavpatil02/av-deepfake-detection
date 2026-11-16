"""
Generate the Review 2 diagrams (PNG, no external tools needed).

    python scripts/make_diagrams.py

Writes:
    reports/architecture.png   system architecture (the six modules and the data flow)
    reports/flowchart.png      per-video decision flow, including every error path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

C_AUDIO = "#dbeafe"
C_VISUAL = "#fef3c7"
C_CORE = "#e0e7ff"
C_UI = "#dcfce7"
C_ERR = "#fee2e2"
C_SYNC = "#fae8ff"
C_EDGE = "#334155"


def box(ax, x, y, w, h, text, color="#ffffff", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015",
        linewidth=1.1, edgecolor=C_EDGE, facecolor=color, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35)


def diamond(ax, cx, cy, w, h, text, color="#fffbeb", fs=8.5):
    ax.add_patch(plt.Polygon(
        [[cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2], [cx - w / 2, cy]],
        closed=True, facecolor=color, edgecolor=C_EDGE, linewidth=1.1, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3,
            linespacing=1.25)


def arrow(ax, p0, p1, text="", fs=7.5, rad=0.0, tx=0.0, ty=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
        color=C_EDGE, linestyle=ls, zorder=1, connectionstyle=f"arc3,rad={rad}",
    ))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + tx, (p0[1] + p1[1]) / 2 + ty, text,
                ha="center", va="center", fontsize=fs, color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95),
                zorder=4)


# --------------------------------------------------------------- architecture
def architecture() -> Path:
    fig, ax = plt.subplots(figsize=(13.6, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Audio-Visual Deepfake Detection — System Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.918,
            "Input video → preprocessing → audio features + lip features → "
            "audio-visual sync → fusion → classifier → report",
            ha="center", va="center", fontsize=9, color="#475569")

    box(ax, 0.012, 0.46, 0.105, 0.17, "Input video\n\n.mp4 · .avi · .mov\n.mkv · .webm",
        color="#f1f5f9", fs=8.5, bold=True)

    box(ax, 0.155, 0.445, 0.155, 0.25,
        "Module 1\nVideo preprocessing\n\n"
        "• format / size validation\n• FFmpeg audio extraction\n"
        "• frame sampling (~12 fps)", color=C_CORE, fs=8.2)

    box(ax, 0.355, 0.655, 0.20, 0.20,
        "Module 2\nAudio features (Librosa)\n\n"
        "MFCC (20) · zero-crossing rate\nspectral centroid / roll-off\n"
        "RMS energy · bandwidth · flatness", color=C_AUDIO, fs=8.2)
    box(ax, 0.585, 0.678, 0.125, 0.155, "52 audio\nfeatures\n(mean + std)",
        color=C_AUDIO, fs=8.5, bold=True)

    box(ax, 0.355, 0.195, 0.20, 0.215,
        "Module 3\nVisual / lip features\n\n"
        "face detection\n(MediaPipe → Haar)\n"
        "mouth opening · MAR\nlip-corner + pixel motion", color=C_VISUAL, fs=8.2)
    box(ax, 0.585, 0.225, 0.125, 0.155, "16 visual\nfeatures\n(None if\nunmeasured)",
        color=C_VISUAL, fs=8.2, bold=True)

    box(ax, 0.745, 0.655, 0.115, 0.155,
        "Module 4\nAV sync\n\nenergy vs mouth\ncross-correlation",
        color=C_SYNC, fs=8.0)
    box(ax, 0.885, 0.668, 0.10, 0.13, "sync\nscore\n+ lag", color=C_SYNC,
        fs=8.2, bold=True)

    box(ax, 0.745, 0.415, 0.24, 0.10,
        "real / fake + P(fake)\n+ held-out metrics\n(only with a trained bundle)",
        color=C_CORE, fs=7.6)
    box(ax, 0.745, 0.205, 0.24, 0.185,
        "Module 5\nFusion + classification\n\n"
        "68-dim fused vector (NaN if missing)\n"
        "median imputation → Random Forest", color=C_CORE, fs=8.2)

    box(ax, 0.20, 0.045, 0.36, 0.085,
        "Module 6 — Gradio UI + report\n"
        "feature table · plots · explanation · result.json",
        color=C_UI, fs=8.4, bold=True)

    arrow(ax, (0.117, 0.545), (0.155, 0.545))
    arrow(ax, (0.310, 0.625), (0.355, 0.735), "audio track", ty=0.045, tx=-0.012)
    arrow(ax, (0.310, 0.500), (0.355, 0.320), "sampled frames", ty=-0.045, tx=0.010)
    arrow(ax, (0.555, 0.755), (0.585, 0.755))
    arrow(ax, (0.555, 0.300), (0.585, 0.300))
    arrow(ax, (0.710, 0.755), (0.745, 0.740))
    arrow(ax, (0.860, 0.740), (0.885, 0.740))
    arrow(ax, (0.710, 0.700), (0.745, 0.420), "audio + visual\nfeature vector",
          rad=0.18, tx=0.045, ty=0.005)
    arrow(ax, (0.710, 0.300), (0.745, 0.345))
    arrow(ax, (0.935, 0.668), (0.935, 0.515), "soft cue", tx=0.048)
    arrow(ax, (0.745, 0.250), (0.560, 0.120), rad=0.10)

    ax.text(0.5, 0.010,
        "Module 5 produces a verdict only when a trained model bundle exists; "
        "otherwise the run reports the measured features only.",
        ha="center", va="bottom", fontsize=8, color="#475569")

    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return OUT / "architecture.png"


# ------------------------------------------------------------------ flowchart
def flowchart() -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 12.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Detection pipeline — per-video flowchart",
                 fontsize=14, fontweight="bold", y=0.995)

    cx = 0.40

    # 1. start
    box(ax, cx - 0.11, 0.950, 0.22, 0.038, "Upload / select video",
        color="#f1f5f9", bold=True, fs=9)
    arrow(ax, (cx, 0.950), (cx, 0.928))

    # 2. format check
    diamond(ax, cx, 0.900, 0.30, 0.056, "Supported format,\nsize, decodable?")
    arrow(ax, (cx + 0.15, 0.900), (0.740, 0.900), "no")
    box(ax, 0.740, 0.868, 0.22, 0.058, "Return clear\nerror code", color=C_ERR, fs=8)
    arrow(ax, (cx, 0.872), (cx, 0.865), "yes")

    # 3. extract
    box(ax, cx - 0.15, 0.815, 0.30, 0.050,
        "Extract audio track (FFmpeg)\n+ sample frames (~12 fps)", color=C_CORE, fs=8.2)
    arrow(ax, (cx, 0.815), (cx, 0.773))

    # 4. audio present?
    diamond(ax, cx, 0.745, 0.30, 0.056, "Audio track\npresent?")
    arrow(ax, (cx + 0.15, 0.745), (0.740, 0.745), "no")
    box(ax, 0.740, 0.713, 0.22, 0.058,
        "Skip Modules 2 & 4\nnote it in the report", color=C_ERR, fs=7.8)
    arrow(ax, (cx, 0.717), (cx, 0.705), "yes")

    # 5. audio features
    box(ax, cx - 0.15, 0.655, 0.30, 0.050,
        "Module 2 — Librosa features\n(MFCC, ZCR, centroid, RMS, …)", color=C_AUDIO, fs=8.2)
    arrow(ax, (cx, 0.655), (cx, 0.620))

    # 6. face found?
    diamond(ax, cx, 0.590, 0.30, 0.060, "Face detected in\n≥ 5 frames?")
    arrow(ax, (cx + 0.15, 0.590), (0.740, 0.590), "no")
    box(ax, 0.740, 0.556, 0.22, 0.068,
        "Visual features left\nas None (never\ninvented)", color=C_ERR, fs=7.6)
    arrow(ax, (cx, 0.560), (cx, 0.550), "yes")

    # 7. lip features
    box(ax, cx - 0.15, 0.500, 0.30, 0.050,
        "Module 3 — lip landmarks\n(MAR, mouth opening, motion)", color=C_VISUAL, fs=8.2)
    arrow(ax, (cx, 0.500), (cx, 0.465))

    # 8. sync
    box(ax, cx - 0.16, 0.415, 0.32, 0.050,
        "Module 4 — audio vs mouth\ncross-correlation (lag + r)", color=C_SYNC, fs=8.2)
    arrow(ax, (cx, 0.415), (cx, 0.380))

    # 9. fusion
    box(ax, cx - 0.16, 0.330, 0.32, 0.050,
        "Module 5a — fuse into a 68-dim vector\n(missing values stay NaN)", color=C_CORE, fs=8.2)
    arrow(ax, (cx, 0.330), (cx, 0.293))

    # 10. model available?
    diamond(ax, cx, 0.265, 0.32, 0.056, "Trained model\nbundle available?")
    arrow(ax, (cx + 0.16, 0.265), (0.760, 0.188), "no", rad=0.12, tx=0.02)
    box(ax, 0.760, 0.160, 0.21, 0.056,
        "FEATURE ANALYSIS\nno prediction", color="#fffbeb", fs=7.6)
    arrow(ax, (cx, 0.237), (cx, 0.215), "yes")

    # 11. classifier
    box(ax, cx - 0.18, 0.160, 0.36, 0.055,
        "Module 5b — Random Forest\n→ real / fake + P(fake) + metrics", color=C_CORE, fs=8.2)

    # 12. report
    box(ax, 0.17, 0.045, 0.46, 0.075,
        "Module 6 — report + plots + result.json\n"
        "features · sync info · verdict (if any) · plots",
        color=C_UI, fs=7.8, bold=True)
    arrow(ax, (cx, 0.160), (cx, 0.120))
    arrow(ax, (0.865, 0.160), (0.63, 0.090), rad=0.12)

    fig.savefig(OUT / "flowchart.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return OUT / "flowchart.png"


if __name__ == "__main__":
    a = architecture()
    f = flowchart()
    print("wrote", a)
    print("wrote", f)
