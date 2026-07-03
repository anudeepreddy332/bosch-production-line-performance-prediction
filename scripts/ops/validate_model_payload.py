"""Validate the model artifact "payload" contract introduced in Phase 2
(feature/model-contract-and-persistence).

Checks that a model artifact dict has the shape BoschPredictor / TwoStagePredictor
already expect (``{"models": [[...]], "feature_cols": [...], "threshold": ...}``)
plus the richer provenance fields this phase adds (model_name, oof_mcc,
fold_metrics, created_at_utc, training_rows, data_fingerprint).

Runs two checks, neither of which requires the full data pipeline or any
unlabeled test data:
1. A self-test that drives the REAL train_lightgbm_oof + build_model_payload
   code on a tiny synthetic labeled dataset, then round-trips the resulting
   payload through joblib.dump -> joblib.load, the same write/read pair used
   by train_*.py and BoschPredictor.load/TwoStagePredictor.load. (Earlier in
   this phase, BoschPredictor/TwoStagePredictor used plain pickle.load, which
   this same self-test caught failing on any payload containing a fitted
   LightGBM model -- joblib.dump's container format for objects holding numpy
   arrays is not plain-pickle-readable. That is why both loaders were changed
   to joblib.load; see src/inference/predictor.py and two_stage_predictor.py.)
2. A best-effort, non-fatal check of whatever is currently in models/*.pkl.

KNOWN GAP (documented here, not fixed in this phase): this script does NOT
instantiate BoschPredictor.load() end-to-end. BoschPredictor requires a fitted
FeaturePipeline backed by data/features/selected_features_top150.txt,
selected_categorical_top100.txt, and train_selected.parquet, none of which are
present in this repo snapshot -- and the active feature builders
(build_dataset_baseline.py / build_dataset_g.py / build_dataset_h.py) produce a
different, smaller feature set than FeaturePipeline emits. Reconciling that
mismatch is a separate, later effort (see CLAUDE.md "Two distinct inference code
paths"). This script validates the part of the contract that IS achievable
today: payload structure and joblib.dump<->joblib.load loadability.

dataset_h additionally needs data/features/dataset_h_lookup.json (a train-derived
lookup artifact, gitignored/regenerable via scripts/build_dataset_h.py -- see
src/features/dataset_h_pipeline.py) to run inference on unlabeled rows. That file
is not part of the models/dataset_h_model.pkl payload itself, so this script also
cross-checks it against the payload's data_fingerprint when checking
dataset_h_model.pkl, and fails with a precise, actionable error if the lookup is
missing or was fit from a different training run.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.dataset_h_pipeline import load_dataset_h_lookup, validate_dataset_h_lookup_compatibility
from src.training.modeling import build_model_payload, train_lightgbm_oof

ROOT = Path(__file__).resolve().parents[1]
DATASET_H_LOOKUP_PATH = ROOT / "data" / "features" / "dataset_h_lookup.json"

REQUIRED_KEYS = {
    "models",
    "feature_cols",
    "threshold",
    "model_name",
    "oof_mcc",
    "fold_metrics",
    "created_at_utc",
    "training_rows",
    "data_fingerprint",
}


def validate_payload(payload: object) -> list[str]:
    """Return a list of problems with `payload`; an empty list means valid."""
    if not isinstance(payload, dict):
        return [f"payload is not a dict (got {type(payload).__name__})"]

    problems: list[str] = []

    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        problems.append(f"missing required keys: {sorted(missing)}")

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        problems.append("'models' must be a non-empty list of per-fold model lists")
    else:
        for i, fold in enumerate(models):
            if not isinstance(fold, list) or not fold:
                problems.append(f"models[{i}] must be a non-empty list of fitted models")
                continue
            for j, model in enumerate(fold):
                if not hasattr(model, "predict_proba"):
                    problems.append(f"models[{i}][{j}] has no predict_proba method")

    feature_cols = payload.get("feature_cols")
    if (
        not isinstance(feature_cols, list)
        or not feature_cols
        or not all(isinstance(c, str) for c in feature_cols)
    ):
        problems.append("'feature_cols' must be a non-empty list of strings")
    elif len(set(feature_cols)) != len(feature_cols):
        problems.append("'feature_cols' contains duplicates")

    threshold = payload.get("threshold")
    if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
        problems.append(f"'threshold' must be a float in [0, 1], got {threshold!r}")

    oof_mcc = payload.get("oof_mcc")
    if not isinstance(oof_mcc, (int, float)) or not (-1.0 <= float(oof_mcc) <= 1.0):
        problems.append(f"'oof_mcc' must be a float in [-1, 1], got {oof_mcc!r}")

    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        problems.append("'model_name' must be a non-empty string")

    fold_metrics = payload.get("fold_metrics")
    if not isinstance(fold_metrics, list) or not fold_metrics:
        problems.append("'fold_metrics' must be a non-empty list")
    else:
        for i, fm in enumerate(fold_metrics):
            if not isinstance(fm, dict) or not {"fold", "rows", "best_threshold", "mcc"} <= fm.keys():
                problems.append(f"fold_metrics[{i}] missing expected keys (fold/rows/best_threshold/mcc)")
        if isinstance(models, list) and models and len(models) != len(fold_metrics):
            problems.append(
                f"len(models)={len(models)} does not match len(fold_metrics)={len(fold_metrics)}"
            )

    training_rows = payload.get("training_rows")
    if not isinstance(training_rows, int) or isinstance(training_rows, bool) or training_rows <= 0:
        problems.append(f"'training_rows' must be a positive int, got {training_rows!r}")

    created_at_utc = payload.get("created_at_utc")
    if not isinstance(created_at_utc, str):
        problems.append("'created_at_utc' must be an ISO timestamp string")
    else:
        try:
            datetime.fromisoformat(created_at_utc)
        except ValueError:
            problems.append(f"'created_at_utc' does not parse as ISO datetime: {created_at_utc!r}")

    return problems


def _build_synthetic_dataset(rows: int = 300, n_chunks: int = 30, seed: int = 0) -> pd.DataFrame:
    """Tiny, fully labeled, in-memory dataset -- NOT real Bosch data, NOT
    unlabeled test data. Only used to exercise train_lightgbm_oof structurally."""
    rng = np.random.default_rng(seed)
    rows_per_chunk = rows // n_chunks
    chunk_id = np.repeat(np.arange(n_chunks, dtype=np.int32), rows_per_chunk)
    return pd.DataFrame(
        {
            "Id": np.arange(1, rows_per_chunk * n_chunks + 1, dtype=np.int64),
            "Response": (rng.random(rows_per_chunk * n_chunks) < 0.05).astype(np.int8),
            "chunk_id": chunk_id,
            "feat_a": rng.normal(size=rows_per_chunk * n_chunks).astype(np.float32),
            "feat_b": rng.normal(size=rows_per_chunk * n_chunks).astype(np.float32),
        }
    )


def _run_self_test(tmp_dir: Path) -> tuple[dict, list[str]]:
    df = _build_synthetic_dataset()
    result, fold_models = train_lightgbm_oof(
        df=df,
        feature_cols=["feat_a", "feat_b"],
        model_name="smoke_test",
        output_oof_path=tmp_dir / "oof.parquet",
        output_importance_path=tmp_dir / "importance.csv",
    )
    payload = build_model_payload(result, fold_models)
    problems = validate_payload(payload)

    dump_path = tmp_dir / "payload.pkl"
    joblib.dump(payload, dump_path)
    reloaded = joblib.load(dump_path)
    problems += [f"round-trip: {p}" for p in validate_payload(reloaded)]
    if len(reloaded.get("models", [])) != len(payload["models"]):
        problems.append("round-trip: models length changed after joblib.dump -> joblib.load")

    return payload, problems


def main() -> int:
    print(__doc__)
    overall_ok = True

    print("=== Self-test: tiny synthetic dataset through train_lightgbm_oof + build_model_payload ===")
    with tempfile.TemporaryDirectory() as tmp:
        payload, problems = _run_self_test(Path(tmp))
    if problems:
        overall_ok = False
        print(f"FAIL ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
    else:
        print(
            f"PASS: payload structurally valid and joblib.dump<->joblib.load round-trip-safe "
            f"(model_name={payload['model_name']!r}, folds={len(payload['models'])}, "
            f"training_rows={payload['training_rows']}, oof_mcc={payload['oof_mcc']:.4f})"
        )

    print("\n=== Best-effort check of committed models/*.pkl (if present) ===")
    models_dir = ROOT / "models"
    pkl_files = sorted(models_dir.glob("*.pkl")) if models_dir.exists() else []
    if not pkl_files:
        print("No models/*.pkl found on disk -- nothing to check.")
    for pkl_path in pkl_files:
        try:
            obj = joblib.load(pkl_path)
        except Exception as exc:  # noqa: BLE001 - best-effort inspection, report and continue
            print(f"  {pkl_path.name}: COULD NOT LOAD ({exc})")
            continue
        if not isinstance(obj, dict):
            print(
                f"  {pkl_path.name}: bare {type(obj).__name__} on disk, NOT a payload dict. "
                f"This is the pre-Phase-2 format; it will only become a payload dict after "
                f"the next training run with the current train_*.py code."
            )
            continue
        file_problems = validate_payload(obj)
        if file_problems:
            overall_ok = False
            print(f"  {pkl_path.name}: INVALID payload -- {file_problems}")
        else:
            print(f"  {pkl_path.name}: valid payload (model_name={obj.get('model_name')!r})")

        if obj.get("model_name") == "dataset_h":
            print(f"    checking dataset_h lookup dependency: {DATASET_H_LOOKUP_PATH}")
            try:
                lookup = load_dataset_h_lookup(DATASET_H_LOOKUP_PATH)
            except (FileNotFoundError, ValueError) as exc:
                overall_ok = False
                print(f"    FAIL: {exc}")
            else:
                lookup_problems = validate_dataset_h_lookup_compatibility(obj, lookup)
                if lookup_problems:
                    overall_ok = False
                    print(f"    FAIL: {lookup_problems}")
                else:
                    print(
                        f"    PASS: lookup present and data_fingerprint matches "
                        f"({lookup['data_fingerprint']}) -- dataset_h inference is runnable."
                    )

    print(
        "\nNOTE: BoschPredictor.load() is intentionally NOT exercised end-to-end here -- it "
        "needs a fitted FeaturePipeline (data/features/selected_features_top150.txt, "
        "selected_categorical_top100.txt, train_selected.parquet) that is not present in this "
        "repo, and the active feature builders emit a different, smaller feature set than "
        "FeaturePipeline does. See this script's module docstring and CLAUDE.md for details."
    )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
