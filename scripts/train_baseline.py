from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib

from src.logger import setup_logger
from src.training.modeling import train_lightgbm_oof
from src.training.summary import update_training_summary

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"

FEATURE_COLS = [
    "start_time",
    "duration",
    "feature_mean",
    "records_last_1hr",
    "records_last_24hr",
    "density_ratio",
    "chunk_id",
    "chunk_size",
]


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_baseline.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Missing dataset_baseline.parquet. Run scripts/build_dataset_baseline.py first.")

    df = pd.read_parquet(dataset_path)

    result, model = train_lightgbm_oof(
        df=df,
        feature_cols=FEATURE_COLS,
        model_name="baseline",
        output_oof_path=FEATURES_DIR / "oof_predictions_baseline.parquet",
        output_importance_path=OUTPUTS_DIR / "feature_importance_baseline.csv",
    )
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "baseline_model.pkl"
    joblib.dump(model, model_path)

    logger.info(f"Saved model to {model_path}")

    update_training_summary(SUMMARY_PATH, "baseline", result)
    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
