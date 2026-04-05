from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.logger import setup_logger
from src.training.modeling import train_lightgbm_oof
from src.training.summary import read_training_summary, update_training_summary

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"


META_FEATURES = [
    "baseline_pred",
    "dataset_g_pred",
    "dataset_h_pred",
    "mean_prediction",
    "std_prediction",
    "max_prediction",
    "agreement_count",
]


def _get_threshold(summary: dict, model_key: str, default: float = 0.5) -> float:
    return float(summary.get("models", {}).get(model_key, {}).get("best_threshold", default))


def main() -> None:
    baseline_oof_path = FEATURES_DIR / "oof_predictions_baseline.parquet"
    g_oof_path = FEATURES_DIR / "oof_predictions_dataset_g.parquet"
    h_oof_path = FEATURES_DIR / "oof_predictions_dataset_h.parquet"
    baseline_dataset_path = FEATURES_DIR / "dataset_baseline.parquet"

    required = [baseline_oof_path, g_oof_path, h_oof_path, baseline_dataset_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for meta model: {missing}")

    base = pd.read_parquet(baseline_oof_path)[["Id", "Response", "oof_pred"]].rename(
        columns={"oof_pred": "baseline_pred"}
    )
    g = pd.read_parquet(g_oof_path)[["Id", "oof_pred"]].rename(columns={"oof_pred": "dataset_g_pred"})
    h = pd.read_parquet(h_oof_path)[["Id", "oof_pred"]].rename(columns={"oof_pred": "dataset_h_pred"})
    chunk = pd.read_parquet(baseline_dataset_path)[["Id", "chunk_id", "chunk_size"]]

    df = base.merge(g, on="Id", how="inner", validate="one_to_one")
    df = df.merge(h, on="Id", how="inner", validate="one_to_one")
    df = df.merge(chunk, on="Id", how="inner", validate="one_to_one")

    pred_mat = df[["baseline_pred", "dataset_g_pred", "dataset_h_pred"]].to_numpy(dtype=np.float32)
    df["mean_prediction"] = pred_mat.mean(axis=1).astype(np.float32)
    df["std_prediction"] = pred_mat.std(axis=1).astype(np.float32)
    df["max_prediction"] = pred_mat.max(axis=1).astype(np.float32)

    summary = read_training_summary(SUMMARY_PATH)
    thr_base = _get_threshold(summary, "baseline")
    thr_g = _get_threshold(summary, "dataset_g")
    thr_h = _get_threshold(summary, "dataset_h")

    df["agreement_count"] = (
        (df["baseline_pred"] >= thr_base).astype(np.int8)
        + (df["dataset_g_pred"] >= thr_g).astype(np.int8)
        + (df["dataset_h_pred"] >= thr_h).astype(np.int8)
    ).astype(np.int8)

    meta_dataset = df[["Id", "Response", "chunk_id", "chunk_size", *META_FEATURES]].copy()
    meta_dataset_path = FEATURES_DIR / "meta_dataset.parquet"
    meta_dataset.to_parquet(meta_dataset_path, index=False)

    result, model = train_lightgbm_oof(
        df=meta_dataset,
        feature_cols=META_FEATURES,
        model_name="meta_model",
        output_oof_path=FEATURES_DIR / "oof_predictions_final.parquet",
        output_importance_path=OUTPUTS_DIR / "feature_importance_meta_model.csv",
    )
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "meta_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved meta model to {model_path}")

    result["meta_dataset_path"] = str(meta_dataset_path)
    result["base_thresholds"] = {
        "baseline": thr_base,
        "dataset_g": thr_g,
        "dataset_h": thr_h,
    }

    update_training_summary(SUMMARY_PATH, "meta_model", result)
    logger.info("Meta model training complete.")


if __name__ == "__main__":
    main()
