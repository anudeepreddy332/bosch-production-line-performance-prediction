"""P1 (KDR-008): train the station-temporal experimental matrix (R0-R6).

Reuses `src.kaggle.wide_modeling.train_wide_lgbm_oof` and
`src.training.modeling.build_model_payload` unchanged -- no trainer-contract amendment
(KDR-008 SS5). `wide_modeling.py` is not modified anywhere in P1.

Modes:
  --mode regression   R0. LEGACY_LGB_PARAMS on K5-A's exact 51-column feature stack, against
                       dataset_h_dup_train.parquet. Must reproduce K5-A's honest OOF MCC
                       (0.32506) before any P1 result is trusted.
  --mode p1_d          R1. Base 1025 (P0 Cell C) + Family D 57 = 1082 features,
                       HIGH_CAPACITY_LGB_PARAMS.
  --mode p1_ds          R2. R1 + Family S/L 108 = 1190 features, HIGH_CAPACITY_LGB_PARAMS.
  --mode tune           R3-R6. Generic tuning run over a chosen feature set (p1_d or p1_ds)
                       with CLI-supplied num_leaves/min_child_samples/n_estimators/
                       learning_rate -- the exact grid is pre-registered in KDR-008 SS4, but
                       which of {p1_d, p1_ds} is "best" (and therefore which feature set R3-R6
                       tune against) and whether R4/R6 run at all are decisions made only
                       after R1/R2/R3 results are known, per KDR-008 SS4's conditional-skip
                       rule -- not hardcoded here.

Requires: scripts/kaggle/build_dataset_p0_raw.py and
          scripts/kaggle/build_dataset_p1_station_temporal.py have already been run.
Requires (regression): scripts/kaggle/build_duplicate_dataset.py has already been run (K5).

Outputs (gitignored): outputs/kaggle/models/p1_{mode}_model.pkl

Reproduce (R0-R2; R3-R6 are examples -- see KDR-008 SS4 for the full grid and the R4
conditional-skip rule):
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode regression
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode p1_d
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode p1_ds
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode tune --feature-set p1_ds --num-leaves 127
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode tune --feature-set p1_ds --num-leaves 255
  PYTHONPATH=. python scripts/kaggle/train_p1.py --mode tune --feature-set p1_ds --num-leaves 127 --min-child-samples 20
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
P1_DT_TRAIN = FEATURES_DIR / "dataset_p1_dt_train.parquet"

_NON_FEATURE_COLS = {"Id", "Response"}

# Exact K5-A feature stack (51 cols) -- the regression anchor and the base stack's "existing" half.
K5A_FEATURE_COLS = [*DATASET_H_FEATURE_COLS, *POSITION_ONLY_MAGIC_COLS, *DUPLICATE_FEATURE_COLS]
AGGREGATE_COLS = ["p0_nnz", "p0_row_min", "p0_row_max", "p0_row_mean", "p0_row_std", "p0_row_sum"]

REGRESSION_TOLERANCE = 1e-4

# KDR-008 SS2 decision bands (delta over P0 Cell C high-capacity honest OOF).
P0_CELL_C_HC_OOF_MCC = 0.37892
H_STRONG_DELTA = 0.020
H_MODERATE_DELTA = 0.007


def _raw_numeric_feature_cols() -> list[str]:
    names = pq.ParquetFile(TRAIN_NUMERIC_RAW).schema_arrow.names
    return [n for n in names if n not in _NON_FEATURE_COLS]


def _base_feature_cols(raw_cols: list[str]) -> list[str]:
    return [*raw_cols, *AGGREGATE_COLS, *K5A_FEATURE_COLS]


def _load_train_frame() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if not P0_RAW_TRAIN.exists():
        raise FileNotFoundError(f"Missing {P0_RAW_TRAIN}. Run scripts/kaggle/build_dataset_p0_raw.py first.")
    if not P1_DT_TRAIN.exists():
        raise FileNotFoundError(
            f"Missing {P1_DT_TRAIN}. Run scripts/kaggle/build_dataset_p1_station_temporal.py first."
        )

    raw_cols = _raw_numeric_feature_cols()
    base_cols = ["Id", "Response", "cv_fold", *_base_feature_cols(raw_cols)]
    dupes = {c for c in base_cols if base_cols.count(c) > 1}
    if dupes:
        raise RuntimeError(f"duplicate column names in base_cols: {sorted(dupes)}")

    base = pd.read_parquet(P0_RAW_TRAIN, columns=base_cols)
    dt = pd.read_parquet(P1_DT_TRAIN)
    d_cols = [c for c in dt.columns if c.startswith("d_")]
    sl_cols = [c for c in dt.columns if c.startswith("s_") or c.startswith("l_")]

    df = base.merge(dt, on="Id", how="left", validate="one_to_one")
    assert len(df) == len(base), "row count changed merging Family D/S/L onto base"
    assert df[d_cols[0]].notna().sum() > 0, "Family D merge produced an entirely-NaN column -- Id mismatch"

    return df, raw_cols, d_cols, sl_cols


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


def _report_delta(oof_mcc: float) -> None:
    delta = oof_mcc - P0_CELL_C_HC_OOF_MCC
    print(f"delta_vs_P0_Cell_C_HC={delta:+.5f} (P0_cell_c_hc_oof_mcc={P0_CELL_C_HC_OOF_MCC:.5f})")
    if delta >= H_STRONG_DELTA:
        print(f"H_station_temporal_strong band (delta >= +{H_STRONG_DELTA:.3f})")
    elif delta >= H_MODERATE_DELTA:
        print(f"H_station_temporal_moderate band (+{H_MODERATE_DELTA:.3f} <= delta < +{H_STRONG_DELTA:.3f})")
    else:
        print(f"H_station_temporal_weak band (delta < +{H_MODERATE_DELTA:.3f})")


def run_regression() -> None:
    if not DUP_TRAIN.exists():
        raise FileNotFoundError(f"Missing {DUP_TRAIN}. Run scripts/kaggle/build_duplicate_dataset.py first (K5).")
    df = pd.read_parquet(DUP_TRAIN)
    out = _run(
        df=df,
        feature_cols=K5A_FEATURE_COLS,
        model_name="p1_regression_check",
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
            f"(delta={delta:+.5f}, tolerance={REGRESSION_TOLERANCE}). Do not trust P1 results until fixed."
        )


def run_p1_d() -> None:
    df, raw_cols, d_cols, _sl_cols = _load_train_frame()
    feature_cols = [*_base_feature_cols(raw_cols), *d_cols]
    out = _run(
        df=df,
        feature_cols=feature_cols,
        model_name="p1_d",
        lgb_params=HIGH_CAPACITY_LGB_PARAMS,
        early_stopping_rounds=HIGH_CAPACITY_EARLY_STOPPING_ROUNDS,
    )
    _report_delta(out["payload"]["oof_mcc"])


def run_p1_ds() -> None:
    df, raw_cols, d_cols, sl_cols = _load_train_frame()
    feature_cols = [*_base_feature_cols(raw_cols), *d_cols, *sl_cols]
    out = _run(
        df=df,
        feature_cols=feature_cols,
        model_name="p1_ds",
        lgb_params=HIGH_CAPACITY_LGB_PARAMS,
        early_stopping_rounds=HIGH_CAPACITY_EARLY_STOPPING_ROUNDS,
    )
    _report_delta(out["payload"]["oof_mcc"])


def run_tune(
    feature_set: str,
    num_leaves: int,
    min_child_samples: int,
    n_estimators: int,
    learning_rate: float,
    early_stopping_rounds: int,
    run_name: str | None,
) -> None:
    df, raw_cols, d_cols, sl_cols = _load_train_frame()
    if feature_set == "p1_d":
        feature_cols = [*_base_feature_cols(raw_cols), *d_cols]
    elif feature_set == "p1_ds":
        feature_cols = [*_base_feature_cols(raw_cols), *d_cols, *sl_cols]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    tune_params = {
        **HIGH_CAPACITY_LGB_PARAMS,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    model_name = run_name or f"p1_tune_{feature_set}_leaves{num_leaves}_mcs{min_child_samples}_est{n_estimators}"
    out = _run(
        df=df,
        feature_cols=feature_cols,
        model_name=model_name,
        lgb_params=tune_params,
        early_stopping_rounds=early_stopping_rounds,
    )
    _report_delta(out["payload"]["oof_mcc"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P1 station-temporal run (KDR-008).")
    parser.add_argument("--mode", required=True, choices=["regression", "p1_d", "p1_ds", "tune"])
    parser.add_argument("--feature-set", choices=["p1_d", "p1_ds"], help="Required for --mode tune")
    parser.add_argument("--num-leaves", type=int, help="Required for --mode tune")
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--early-stopping-rounds", type=int, default=HIGH_CAPACITY_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "regression":
        run_regression()
    elif args.mode == "p1_d":
        run_p1_d()
    elif args.mode == "p1_ds":
        run_p1_ds()
    elif args.mode == "tune":
        if args.feature_set is None or args.num_leaves is None:
            raise SystemExit("--mode tune requires --feature-set and --num-leaves")
        run_tune(
            feature_set=args.feature_set,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            early_stopping_rounds=args.early_stopping_rounds,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
