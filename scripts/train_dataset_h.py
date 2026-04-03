from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    "transition_fail_rate_mean",
    "transition_fail_rate_max",
    "transition_fail_rate_std",
    "station_risk_mean",
    "path_count",
    "pair_cooccur_mean",
    "pair_cooccur_max",
    "pair_cooccur_std",
]


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_h.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Missing dataset_h.parquet. Run scripts/build_dataset_h.py first.")

    df = pd.read_parquet(dataset_path)

    result = train_lightgbm_oof(
        df=df,
        feature_cols=FEATURE_COLS,
        model_name="dataset_h",
        output_oof_path=FEATURES_DIR / "oof_predictions_dataset_h.parquet",
        output_importance_path=OUTPUTS_DIR / "feature_importance_dataset_h.csv",
    )

    update_training_summary(SUMMARY_PATH, "dataset_h", result)
    logger.info("Dataset H training complete.")


if __name__ == "__main__":
    main()
