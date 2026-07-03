"""E1a/E1b/E1c decomposition: attribute the source of E1's +0.009 OOF MCC gain.

Three arms, identical CV/hyperparams, differ only in the sensor feature representation:
  E1a: global dispersion only  — sensor_std + sensor_nonull_count (18 features total)
  E1b: presence only           — 50 binary station presence flags (66 features total)
  E1c: value only              — 50 station means, NaN filled by per-station median (66 features)

Comparators: dataset_h (0.1534), full E1 (0.1627).
Pre-registered attribution rules: docs/research/decisions.md DR-005 §3 / DR-006.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

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
MODEL_DIR = ROOT / "models"

_E1_BASELINE = 0.15337
_E1_CEILING = 0.16270

_H_FOLD_MCCS = [0.13813, 0.13536, 0.18114, 0.15277, 0.18498]
_E1_FOLD_MCCS = [0.15592, 0.14006, 0.18937, 0.15144, 0.18688]


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _station_mean_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("sensor_mean_"))


def _build_e1a_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Global dispersion only: sensor_std + sensor_nonull_count."""
    cols = DATASET_H_FEATURE_COLS + ["sensor_std", "sensor_nonull_count"]
    return cols, df[cols + ["Id", "Response", "cv_fold"]].copy()


def _build_e1b_features(df: pd.DataFrame, station_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Presence only: 50 binary station presence flags, no values."""
    presence_names = [f"sensor_present_{c.removeprefix('sensor_mean_')}" for c in station_cols]
    out = df[["Id", "Response", "cv_fold"] + DATASET_H_FEATURE_COLS].copy()
    for pname, mcol in zip(presence_names, station_cols):
        out[pname] = df[mcol].notna().astype(np.uint8)
    cols = DATASET_H_FEATURE_COLS + presence_names
    return cols, out


def _build_e1c_features(df: pd.DataFrame, station_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Value only: station means with NaN filled by per-station global median.

    Global (not fold-wise) median is correct here: it is a non-target statistic of the
    raw feature distribution, so including validation rows in the median does not leak
    target information. Equivalent to global mean-centering of raw features.
    """
    out = df[["Id", "Response", "cv_fold"] + DATASET_H_FEATURE_COLS].copy()
    for col in station_cols:
        median_val = df[col].median()  # NaN-skipping, globally
        filled = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        out[col] = filled.astype(np.float32)
    cols = DATASET_H_FEATURE_COLS + station_cols
    return cols, out


def _run_arm(
    arm_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    logger.info(
        "=== ARM %s: %d features, %d rows ===", arm_name, len(feature_cols), len(df)
    )
    mem_before = _memory_gb()
    t0 = time.perf_counter()

    result, fold_models = train_lightgbm_oof(
        df=df,
        feature_cols=feature_cols,
        model_name=arm_name,
        output_oof_path=FEATURES_DIR / f"oof_predictions_{arm_name}.parquet",
        output_importance_path=OUTPUTS_DIR / f"feature_importance_{arm_name}.csv",
    )

    elapsed = time.perf_counter() - t0
    mem_peak = _memory_gb()

    payload = build_model_payload(result, fold_models)
    MODEL_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(payload, MODEL_DIR / f"{arm_name}_model.pkl")

    update_training_summary(SUMMARY_PATH, arm_name, result)

    fold_mccs = [m["mcc"] for m in result["fold_metrics"]]
    arm_result = {
        "arm": arm_name,
        "features": len(feature_cols),
        "oof_mcc": result["oof_mcc"],
        "fold_mccs": fold_mccs,
        "elapsed_s": elapsed,
        "mem_peak_gb": mem_peak,
        "mem_delta_gb": mem_peak - mem_before,
    }
    return arm_result


def _report(results: list[dict]) -> None:
    h_mccs = _H_FOLD_MCCS
    e1_mccs = _E1_FOLD_MCCS

    header = f"{'Arm':<12} {'Feats':>5} {'OOF MCC':>9} {'Δ vs h':>8} {'Δ vs E1':>8} {'Folds↑':>7} {'Runtime':>8} {'MemΔ':>6}"
    logger.info("\n%s", header)
    logger.info("%s", "-" * len(header))

    for r in results:
        delta_h = r["oof_mcc"] - _E1_BASELINE
        delta_e1 = r["oof_mcc"] - _E1_CEILING
        folds_up = sum(1 for a, b in zip(r["fold_mccs"], h_mccs) if a > b)
        logger.info(
            "%-12s %5d  %9.5f  %+8.5f  %+8.5f  %5d/5  %6.0fs  %+5.2fGB",
            r["arm"], r["features"], r["oof_mcc"],
            delta_h, delta_e1,
            folds_up, r["elapsed_s"], r["mem_delta_gb"],
        )

    logger.info("\nFold-by-fold detail:")
    fold_header = f"{'Fold':<5} {'dataset_h':>10} {'full E1':>10}" + "".join(
        f" {r['arm']:>12}" for r in results
    )
    logger.info(fold_header)
    for i, (h, e1) in enumerate(zip(h_mccs, e1_mccs)):
        row = f"  {i:<3} {h:>10.5f} {e1:>10.5f}"
        for r in results:
            row += f" {r['fold_mccs'][i]:>12.5f}"
        logger.info(row)


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_e1.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Missing dataset_e1.parquet. Run scripts/build_dataset_e1.py first."
        )

    logger.info("Loading dataset_e1 for decomposition arms")
    df = pd.read_parquet(dataset_path)
    verify_persisted_fold_assignment(df)

    station_cols = _station_mean_cols(df)
    logger.info("Station columns: %d", len(station_cols))

    logger.info("Building E1a feature matrix (global dispersion only)")
    e1a_cols, df_e1a = _build_e1a_features(df)

    logger.info("Building E1b feature matrix (presence only)")
    e1b_cols, df_e1b = _build_e1b_features(df, station_cols)

    logger.info("Building E1c feature matrix (value only, median-filled)")
    e1c_cols, df_e1c = _build_e1c_features(df, station_cols)

    # Quick sanity: E1c should have no NaN in sensor columns
    e1c_sensor_nans = df_e1c[station_cols].isna().sum().sum()
    assert e1c_sensor_nans == 0, f"E1c has {e1c_sensor_nans} NaNs in station cols after imputation"
    logger.info("E1c imputation sanity check passed (0 NaNs in station cols)")

    results = []
    for arm_name, arm_df, arm_cols in [
        ("dataset_e1a", df_e1a, e1a_cols),
        ("dataset_e1b", df_e1b, e1b_cols),
        ("dataset_e1c", df_e1c, e1c_cols),
    ]:
        result = _run_arm(arm_name, arm_df, arm_cols)
        results.append(result)

    _report(results)
    logger.info(
        "\nComparators: dataset_h=%.5f, full_E1=%.5f",
        _E1_BASELINE, _E1_CEILING,
    )


if __name__ == "__main__":
    main()
