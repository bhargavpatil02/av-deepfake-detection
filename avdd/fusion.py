"""
Module 5a - Feature fusion.

Audio descriptors and visual descriptors are concatenated into one fixed-length
vector. Missing values are represented as NaN (never as invented numbers); the
classifier bundle stores median values from its training set and uses them as
imputation, and any prediction made with imputed blocks is flagged in the UI.
"""

from __future__ import annotations

import numpy as np

from .audio_feats import AUDIO_FEATURE_NAMES, build_feature_names
from .visual_feats import VISUAL_FEATURE_NAMES

AUDIO_NAMES = build_feature_names()
VISUAL_NAMES = VISUAL_FEATURE_NAMES
FUSED_NAMES = AUDIO_NAMES + VISUAL_NAMES

GROUPS = {
    **{n: "audio" for n in AUDIO_NAMES},
    **{n: "visual" for n in VISUAL_NAMES},
}


def fuse(audio_features: dict | None, visual_features: dict | None) -> dict:
    """
    Concatenate the two feature dicts into one ordered vector.

    Returns:
      names        : ordered feature names (always FUSED_NAMES)
      vector       : np.ndarray float64, NaN where a feature is missing
      missing      : list of feature names that are NaN
      missing_blocks : e.g. ["visual"] if the whole visual block is absent
      counts       : how many features came from each modality
    """
    audio_features = audio_features or {}
    visual_features = visual_features or {}

    vector = np.full(len(FUSED_NAMES), np.nan, dtype=np.float64)
    for i, name in enumerate(FUSED_NAMES):
        src = audio_features if name in AUDIO_NAMES else visual_features
        val = src.get(name)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            vector[i] = fval

    missing = [n for n, v in zip(FUSED_NAMES, vector) if not np.isfinite(v)]
    missing_blocks = []
    if audio_features and all(not np.isfinite(vector[AUDIO_NAMES.index(n)]) for n in AUDIO_NAMES):
        missing_blocks.append("audio")
    if not audio_features:
        missing_blocks.append("audio")
    if not visual_features or all(
        not np.isfinite(vector[FUSED_NAMES.index(n)]) for n in VISUAL_NAMES
    ):
        missing_blocks.append("visual")

    counts = {
        "audio_features": int(sum(np.isfinite(vector[i]) for i, n in enumerate(FUSED_NAMES)
                                  if n in AUDIO_NAMES)),
        "visual_features": int(sum(np.isfinite(vector[i]) for i, n in enumerate(FUSED_NAMES)
                                   if n in VISUAL_NAMES)),
        "total": len(FUSED_NAMES),
    }
    return {
        "names": FUSED_NAMES,
        "vector": vector,
        "missing": missing,
        "missing_blocks": missing_blocks,
        "counts": counts,
    }


def as_dataframe(fused: dict, audio_summary: dict | None = None) -> "object":
    """Return a tidy pandas frame (group, feature, value) for the UI table."""
    import pandas as pd

    rows = []
    for name, value in zip(fused["names"], fused["vector"]):
        rows.append({
            "group": GROUPS[name],
            "feature": name,
            "value": None if not np.isfinite(value) else round(float(value), 5),
        })
    return pd.DataFrame(rows)
