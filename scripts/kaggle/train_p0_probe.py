"""P0 (KDR-007): train the raw-numeric probe cells (regression anchor, Cell B, Cell C).

Reuses `src.kaggle.wide_modeling.train_wide_lgbm_oof` (frozen CV/OOF/threshold/determinism
contract, KDR-007 SS5a) and `src.training.modeling.build_model_payload` unchanged for the
on-disk model artifact.

Modes:
  --mode regression            LEGACY_LGB_PARAMS on K5-A's exact 51-column feature stack,
                                against dataset_h_dup_train.parquet (no raw build required).
                                Must reproduce K5-A's honest OOF MCC (0.32506) -- validates
                                wide_modeling.py is a faithful superset of train_lightgbm_oof
                                before any Cell B/C result is trusted.
  --mode cell_b                968 raw numeric + 6 aggregates only (974 features),
                                LEGACY_LGB_PARAMS. Standalone raw ceiling -- informational,
                                not decision-gating (KDR-007 SS3).
  --mode cell_c_default         Cell B + K5-A's 51 columns (1025 features), LEGACY_LGB_PARAMS.
  --mode cell_c_high_capacity  Same feature set as cell_c_default, HIGH_CAPACITY_LGB_PARAMS
                                (n_estimators=2500, lr=0.02) -- capacity-unbound guard against
                                a false-negative decision (KDR-007 SS5).

Requires (cell_b / cell_c_*): scripts/kaggle/build_dataset_p0_raw.py has already been run.
Requires (regression): scripts/kaggle/build_duplicate_dataset.py has already been run (K5).

Outputs (gitignored): outputs/kaggle/models/p0_{mode}_model.pkl

Reproduce:
  PYTHONPATH=. python scripts/kaggle/train_p0_probe.py --mode regression
  PYTHONPATH=. python scripts/kaggle/train_p0_probe.py --mode cell_b
  PYTHONPATH=. python scripts/kaggle/train_p0_probe.py --mode cell_c_default
  PYTHONPATH=. python scripts/kaggle/train_p0_probe.py --mode cell_c_high_capacity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

from src.features.dataset_h_pipeline import DATASET_H_FEATURE_COLS
from src.kaggle.duplicate_features import DUPLICATE_FEATURE_COLS
from src.kaggle.magic_features import POSITION_ONLY_MAGIC_COLS
from src.kaggle.wide_modeling import (
    HIGH_CAPACITY_EARLY_STOPPING_ROUNDS,
    HIGH_CAPACITY_LGB_PARAMS,
    K5A_HONEST_OOF_MCC,
    LEGACY_EARLY_STOPPING_ROUNDS,
    train_wide_lgbm_oof,
)
from src.logger import setup_logger
from src.training.modeling import build_model_payload

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs" / "kaggle"
MODEL_DIR = OUTPUTS_DIR / "models"

TRAIN_NUMERIC_RAW = PROCESSED_DIR / "train_numeric.parquet"
DUP_TRAIN = FEATURES_DIR / "dataset_h_dup_train.parquet"
P0_RAW_TRAIN = FEATURES_DIR / "dataset_p0_raw_train.parquet"

_NON_FEATURE_COLS = {"Id", "Response"}

# Exact K5-A feature stack (51 cols) -- the regression anchor and Cell C's "existing" half.
K5A_FEATURE_COLS = [*DATASET_H_FEATURE_COLS, *POSITION_ONLY_MAGIC_COLS, *DUPLICATE_FEATURE_COLS]
AGGREGATE_COLS = ["p0_nnz", "p0_row_min", "p0_row_max", "p0_row_mean", "p0_row_std", "p0_row_sum"]

REGRESSION_TOLERANCE = 1e-4

# KDR-007 SS3 decision bands (delta over K5-A honest OOF 0.32506).
H_RAW_DOMINANT_DELTA = 0.030
H_RAW_MODEST_DELTA = 0.010


def _raw_numeric_feature_cols() -> list[str]:
    names = pq.ParquetFile(TRAIN_NUMERIC_RAW).schema_arrow.names
    return [n for n in names if n not in _NON_FEATURE_COLS]


def _run(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    lgb_params: dict[str, object] | None,
    early_stopping_rounds: int,
) -> dict[str, object]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset missing {len(missing)} required columns for {model_name}: {missing[:10]}")

    result, fold_models = train_wide_lgbm_oof(
        df=df,
        feature_cols=feature_cols,
        model_name=model_name,
        output_oof_path=OUTPUTS_DIR / f"oof_predictions_{model_name}.parquet",
        output_importance_path=OUTPUTS_DIR / f"feature_importance_{model_name}.csv",
        lgb_params=lgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )
    payload = build_model_payload(result, fold_models)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}_model.pkl"
    joblib.dump(payload, model_path)

    print(f"model={model_name}")
    print(f"feature_count={len(feature_cols)}")
    print(f"model_path={model_path}")
    print(f"data_fingerprint={payload['data_fingerprint']}")
    print(f"threshold={payload['threshold']}")
    print(f"oof_mcc={payload['oof_mcc']:.5f}")
    print(f"capacity_bound={result['capacity_bound']}")
    print(f"fold_best_iterations={result['fold_best_iterations']} / n_estimators={result['n_estimators']}")

    return {"result": result, "payload": payload}


def run_regression() -> None:
    if not DUP_TRAIN.exists():
        raise FileNotFoundError(f"Missing {DUP_TRAIN}. Run scripts/kaggle/build_duplicate_dataset.py first (K5).")
    df = pd.read_parquet(DUP_TRAIN)
    out = _run(
        df=df,
        feature_cols=K5A_FEATURE_COLS,
        model_name="p0_regression_check",
        lgb_params=None,
        early_stopping_rounds=LEGACY_EARLY_STOPPING_ROUNDS,
    )
    oof_mcc = out["payload"]["oof_mcc"]
    delta = oof_mcc - K5A_HONEST_OOF_MCC
    passed = abs(delta) <= REGRESSION_TOLERANCE
    print(f"K5A_reference_oof_mcc={K5A_HONEST_OOF_MCC:.5f}")
    print(f"delta={delta:+.5f} tolerance={REGRESSION_TOLERANCE}")
    print("REGRESSION " + ("PASS" if passed else "FAIL"))
    if not passed:
        raise RuntimeError(
            f"wide_modeling.train_wide_lgbm_oof with LEGACY_LGB_PARAMS on K5-A's exact 51-column "
            f"feature stack produced oof_mcc={oof_mcc:.5f}, expected {K5A_HONEST_OOF_MCC:.5f} "
            f"(delta={delta:+.5f}, tolerance={REGRESSION_TOLERANCE}). wide_modeling.py is not a "
            f"faithful superset of train_lightgbm_oof -- do not trust Cell B/C results until fixed."
        )


def run_cell_b() -> None:
    if not P0_RAW_TRAIN.exists():
        raise FileNotFoundError(f"Missing {P0_RAW_TRAIN}. Run scripts/kaggle/build_dataset_p0_raw.py first.")
    df = pd.read_parquet(P0_RAW_TRAIN)
    raw_cols = _raw_numeric_feature_cols()
    feature_cols = [*raw_cols, *AGGREGATE_COLS]
    _run(
        df=df,
        feature_cols=feature_cols,
        model_name="p0_cell_b",
        lgb_params=None,
        early_stopping_rounds=LEGACY_EARLY_STOPPING_ROUNDS,
    )


def run_cell_c(high_capacity: bool) -> None:
    if not P0_RAW_TRAIN.exists():
        raise FileNotFoundError(f"Missing {P0_RAW_TRAIN}. Run scripts/kaggle/build_dataset_p0_raw.py first.")
    df = pd.read_parquet(P0_RAW_TRAIN)
    raw_cols = _raw_numeric_feature_cols()
    feature_cols = [*raw_cols, *AGGREGATE_COLS, *K5A_FEATURE_COLS]
    model_name = "p0_cell_c_high_capacity" if high_capacity else "p0_cell_c_default"
    out = _run(
        df=df,
        feature_cols=feature_cols,
        model_name=model_name,
        lgb_params=HIGH_CAPACITY_LGB_PARAMS if high_capacity else None,
        early_stopping_rounds=HIGH_CAPACITY_EARLY_STOPPING_ROUNDS if high_capacity else LEGACY_EARLY_STOPPING_ROUNDS,
    )
    delta = out["payload"]["oof_mcc"] - K5A_HONEST_OOF_MCC
    print(f"delta_vs_K5A={delta:+.5f} (K5A_honest_oof_mcc={K5A_HONEST_OOF_MCC:.5f})")
    if delta >= H_RAW_DOMINANT_DELTA:
        print(f"H_raw_dominant band (delta >= +{H_RAW_DOMINANT_DELTA:.3f})")
    elif delta >= H_RAW_MODEST_DELTA:
        print(f"H_raw_modest band (+{H_RAW_MODEST_DELTA:.3f} <= delta < +{H_RAW_DOMINANT_DELTA:.3f})")
    else:
        print(f"H_raw_not_dominant band (delta < +{H_RAW_MODEST_DELTA:.3f}) -- check capacity_bound before treating as falsifying")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P0 raw-numeric probe cell (KDR-007).")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["regression", "cell_b", "cell_c_default", "cell_c_high_capacity"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "regression":
        run_regression()
    elif args.mode == "cell_b":
        run_cell_b()
    elif args.mode == "cell_c_default":
        run_cell_c(high_capacity=False)
    elif args.mode == "cell_c_high_capacity":
        run_cell_c(high_capacity=True)


if __name__ == "__main__":
    main()
