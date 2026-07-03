"""E1 experiment: train LightGBM on dataset_e1 (dataset_h + per-station sensor means).

Identical hyperparameters and CV config as train_dataset_h.py — one variable changed:
the feature set. See docs/research/decisions.md DR-004.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.dataset_h_pipeline import DATASET_H_FEATURE_COLS
from src.logger import setup_logger
from src.training.cv import verify_persisted_fold_assignment
from src.training.modeling import build_model_payload, train_lightgbm_oof
from src.training.summary import update_training_summary

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"

_SENSOR_FEATURE_PREFIXES = ("sensor_mean_", "sensor_nonull_count", "sensor_std")


def _e1_feature_cols(df: pd.DataFrame) -> list[str]:
    sensor_cols = [
        c for c in df.columns
        if c.startswith("sensor_mean_") or c in {"sensor_nonull_count", "sensor_std"}
    ]
    return DATASET_H_FEATURE_COLS + sorted(sensor_cols)


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_e1.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Missing dataset_e1.parquet. Run scripts/build_dataset_e1.py first."
        )

    df = pd.read_parquet(dataset_path)
    verify_persisted_fold_assignment(df)

    feature_cols = _e1_feature_cols(df)
    logger.info(
        "E1 feature set: %d total (%d from dataset_h + %d sensor)",
        len(feature_cols),
        len(DATASET_H_FEATURE_COLS),
        len(feature_cols) - len(DATASET_H_FEATURE_COLS),
    )

    result, fold_models = train_lightgbm_oof(
        df=df,
        feature_cols=feature_cols,
        model_name="dataset_e1",
        output_oof_path=FEATURES_DIR / "oof_predictions_dataset_e1.parquet",
        output_importance_path=OUTPUTS_DIR / "feature_importance_dataset_e1.csv",
    )

    payload = build_model_payload(result, fold_models)
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "dataset_e1_model.pkl"
    joblib.dump(payload, model_path)
    logger.info("Saved model payload to %s", model_path)

    update_training_summary(SUMMARY_PATH, "dataset_e1", result)

    logger.info(
        "E1 training complete. OOF MCC=%.5f threshold=%.2f",
        result["oof_mcc"],
        result["best_threshold"],
    )
    logger.info("Fold MCCs: %s", [f"{m['mcc']:.5f}" for m in result["fold_metrics"]])


if __name__ == "__main__":
    main()
