from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib

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

# Canonical column list lives in src/features/dataset_h_pipeline.py so the test-side
# feature builder (scripts/build_test_dataset_h.py) can never drift from what this
# script trains the model on.
FEATURE_COLS = DATASET_H_FEATURE_COLS


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_h.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Missing dataset_h.parquet. Run scripts/build_dataset_h.py first.")

    df = pd.read_parquet(dataset_path)
    verify_persisted_fold_assignment(df)

    result, fold_models = train_lightgbm_oof(
        df=df,
        feature_cols=FEATURE_COLS,
        model_name="dataset_h",
        output_oof_path=FEATURES_DIR / "oof_predictions_dataset_h.parquet",
        output_importance_path=OUTPUTS_DIR / "feature_importance_dataset_h.csv",
    )
    payload = build_model_payload(result, fold_models)
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "dataset_h_model.pkl"
    joblib.dump(payload, model_path)

    logger.info(f"Saved model payload ({len(fold_models)} fold models) to {model_path}")

    update_training_summary(SUMMARY_PATH, "dataset_h", result)
    logger.info("Dataset H training complete.")


if __name__ == "__main__":
    main()
