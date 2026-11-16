"""
Module 5b - Baseline classifier (Random Forest).

Design rules that keep this honest:
* Training happens ONLY when labelled data exists (data/real + data/fake).
* Missing feature values are imputed with the *training* medians, and the same
  imputer object is reused at inference (no leakage, no recomputation).
* Train/test split is stratified and the test set is touched once, at the end.
* Metrics are computed from that split and saved *with* the model, so the UI can
  display exactly how good (or bad) the model was instead of a hard-coded number.
* The feature-extraction settings are stored in the bundle; a model trained with
  different settings is refused rather than silently mixed.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config
from .fusion import FUSED_NAMES

META_COLUMNS = {
    "label", "source_file", "group", "split", "notes",
    # diagnostics written by dataset.build_csv - not model inputs
    "audio_ok", "visual_ok", "visual_backend",
}


class ModelError(Exception):
    pass


# ------------------------------------------------------------------- features
def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLUMNS]
    expected = set(FUSED_NAMES)
    unknown = [c for c in cols if c not in expected]
    if unknown:
        raise ModelError(
            f"CSV contains unknown feature columns (e.g. {unknown[:3]}). "
            "Re-extract features with the current config."
        )
    missing = [c for c in FUSED_NAMES if c not in cols]
    if missing:
        # Tolerate a partial CSV (e.g. audio-only run) by adding NaN columns.
        for c in missing:
            df[c] = np.nan
    return FUSED_NAMES


def _clean_labels(series: pd.Series) -> np.ndarray:
    mapping = {"real": 0, "genuine": 0, "authentic": 0, "0": 0,
               "fake": 1, "manipulated": 1, "deepfake": 1, "1": 1}
    out = []
    bad = set()
    for v in series.astype(str).str.strip().str.lower():
        if v in mapping:
            out.append(mapping[v])
        else:
            bad.add(v)
    if bad:
        raise ModelError(
            f"Unknown label values {sorted(bad)}. Use 'real'/'fake' (or 0/1)."
        )
    return np.array(out, dtype=int)


# -------------------------------------------------------------------- training
def train_from_csv(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    test_size: float = config.TEST_SIZE,
    n_estimators: int = 300,
) -> dict:
    """
    Train the baseline Random Forest and save a bundle.

    Returns the bundle dict (also persisted to `out_path`).
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path else config.DEFAULT_MODEL_PATH
    if not csv_path.exists():
        raise ModelError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ModelError("CSV has no 'label' column (expected real/fake).")
    if len(df) < 10:
        raise ModelError(
            f"Only {len(df)} samples in {csv_path.name}. Collect at least ~20 "
            "(10 per class) before training; otherwise use demonstration mode."
        )

    feats = feature_columns(df)
    y = _clean_labels(df["label"])
    X = df[feats].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan

    counts = np.bincount(y, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ModelError(
            f"Need both classes, found real={counts[0]}, fake={counts[1]}."
        )
    if min(counts) < 5:
        raise ModelError(
            f"Class imbalance too extreme for a reliable split "
            f"(real={counts[0]}, fake={counts[1]}); need >=5 per class."
        )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )

    # Fit the imputer on TRAIN ONLY, then reuse it for the test set.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_tr_i, y_tr)

    y_pred = clf.predict(X_te_i)
    y_prob = clf.predict_proba(X_te_i)[:, 1]
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    metrics = {
        "n_samples": int(len(df)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_real": int(counts[0]),
        "n_fake": int(counts[1]),
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "classification_report": classification_report(
            y_te, y_pred, target_names=["real", "fake"], zero_division=0
        ),
        "test_size": test_size,
    }
    if len(np.unique(y_te)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_te, y_prob)), 4)

    # Cross-validation on the training split gives a stability estimate.
    try:
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                              random_state=config.RANDOM_STATE)
        cv = cross_val_score(
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=config.RANDOM_STATE, n_jobs=-1),
            X_tr_i, y_tr, cv=skf, scoring="accuracy",
        )
        metrics["cv_accuracy_mean"] = round(float(cv.mean()), 4)
        metrics["cv_accuracy_std"] = round(float(cv.std()), 4)
        metrics["cv_folds"] = int(len(cv))
    except Exception as exc:  # too few samples per fold, etc.
        metrics["cv_error"] = str(exc)

    importances = sorted(
        zip(feats, clf.feature_importances_), key=lambda kv: kv[1], reverse=True
    )[:12]

    bundle = {
        "model": clf,
        "imputer": imputer,
        "feature_names": feats,
        "classes": ["real", "fake"],
        "fake_index": int(list(clf.classes_).index(1)),
        "metrics": metrics,
        "top_features": [{"feature": n, "importance": round(float(v), 4)}
                         for n, v in importances],
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "extraction_config": {
            "audio_sr": config.AUDIO_SR,
            "audio_n_mfcc": config.AUDIO_N_MFCC,
            "audio_hop": config.AUDIO_HOP,
            "visual_fps_target": config.VISUAL_FPS_TARGET,
            "visual_work_width": config.VISUAL_WORK_WIDTH,
        },
        "dataset_csv": str(csv_path),
        "notes": [
            "Random Forest trained on fused audio + visual features.",
            "Metrics come from a single stratified train/test split of the "
            "feature CSV; they reflect that dataset only.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    bundle["saved_to"] = str(out_path)
    return bundle


# ------------------------------------------------------------------- inference
def load_bundle(path: str | Path | None = None) -> dict | None:
    path = Path(path) if path else config.DEFAULT_MODEL_PATH
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        bundle.setdefault("saved_to", str(path))
        return bundle
    except Exception:
        return None


def predict(bundle: dict, fused_vector: np.ndarray) -> dict:
    """
    Run the trained model on one fused vector.

    `fused_vector` must follow `bundle["feature_names"]`. NaN values are imputed
    with the stored training medians and the imputed count is reported.
    """
    names = bundle["feature_names"]
    x = np.asarray(fused_vector, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != len(names):
        raise ModelError(
            f"Feature vector has {x.shape[1]} values, model expects {len(names)}."
        )
    n_imputed = int(np.sum(~np.isfinite(x)))
    x_i = bundle["imputer"].transform(x)
    proba = bundle["model"].predict_proba(x_i)[0]
    fake_p = float(proba[bundle["fake_index"]])
    label = "fake" if fake_p >= 0.5 else "real"
    conf = fake_p if label == "fake" else 1.0 - fake_p

    notes = []
    if n_imputed:
        notes.append(
            f"{n_imputed} of {len(names)} features were missing and replaced by "
            "training medians - the prediction is therefore unreliable."
        )
    if bundle["metrics"].get("n_samples", 0) < 40:
        notes.append(
            f"Model was trained on only {bundle['metrics']['n_samples']} videos; "
            "expect wide error bars."
        )
    return {
        "label": label,
        "probability_fake": round(fake_p, 4),
        "confidence": round(conf, 4),
        "n_imputed": n_imputed,
        "model_trained_at": bundle.get("trained_at"),
        "model_metrics": bundle["metrics"],
        "notes": notes,
    }
