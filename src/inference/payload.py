"""Shared model-payload loading and ensembled prediction.

Used by both the Kaggle submission generator (`scripts/generate_submission.py`) and the
label-free production batch scorer (`scripts/pipeline/run_production_inference.py`) --
previously `run_production_inference.py` reached directly into `generate_submission.py`
for these two functions (a script importing internals from another script); this module
is the single, shared home for them instead.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.ops.validate_model_payload import validate_payload


def load_validated_payload(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model payload not found: {model_path}")

    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{model_path} is a bare {type(payload).__name__}, not the Phase-2 payload dict "
            "({'models', 'feature_cols', 'threshold', ...}) scripts/ops/validate_model_payload.py "
            "expects. This is the pre-Phase-2 model format -- re-run the matching "
            "scripts/pipeline/train_*.py to produce a valid payload before generating a submission. "
            "Refusing to proceed."
        )

    problems = validate_payload(payload)
    if problems:
        raise ValueError(f"Invalid model payload at {model_path}: {problems}")
    return payload


def predict_proba_ensemble(payload: dict, features: pd.DataFrame) -> np.ndarray:
    """Average predict_proba across CV folds (mean-of-folds ensembling)."""
    feature_matrix = features[payload["feature_cols"]]
    fold_preds = [
        np.mean([model.predict_proba(feature_matrix)[:, 1] for model in fold_models], axis=0)
        for fold_models in payload["models"]
    ]
    return np.mean(fold_preds, axis=0)
