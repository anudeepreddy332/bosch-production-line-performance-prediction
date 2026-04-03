from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.training.cv import ChunkCVConfig, assign_fold_ids, make_chunk_aware_splits

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"


BASELINE_COLUMNS = [
    "start_time",
    "duration",
    "feature_mean",
    "records_last_1hr",
    "records_last_24hr",
    "density_ratio",
    "chunk_id",
    "chunk_size",
]


def _rolling_rate_from_train_to_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    global_mean: float,
    window: int = 10_000,
) -> pd.Series:
    tr = train_df[["start_time", "Response"]].copy()
    tr["start_time"] = pd.to_numeric(tr["start_time"], errors="coerce").fillna(-1e9)
    tr = tr.sort_values("start_time", kind="mergesort").reset_index(drop=True)

    tr["rolling_fail_rate"] = (
        tr["Response"].rolling(window=window, min_periods=1).mean().shift(1).fillna(global_mean).astype(np.float32)
    )

    tr_ref = tr[["start_time", "rolling_fail_rate"]].sort_values("start_time", kind="mergesort")

    valid_tmp = valid_df[["start_time"]].copy().reset_index()
    valid_tmp["start_time"] = pd.to_numeric(valid_tmp["start_time"], errors="coerce").fillna(-1e9)
    valid_tmp = valid_tmp.sort_values("start_time", kind="mergesort")

    merged = pd.merge_asof(valid_tmp, tr_ref, on="start_time", direction="backward")
    out = merged.set_index("index")["rolling_fail_rate"].reindex(valid_df.index)
    return out.fillna(global_mean).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dataset G with OOF-safe target features.")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    baseline_path = FEATURES_DIR / "dataset_baseline.parquet"
    path_meta_path = FEATURES_DIR / "path_metadata.parquet"

    if not baseline_path.exists() or not path_meta_path.exists():
        raise FileNotFoundError("Missing baseline artifacts. Run scripts/build_dataset_baseline.py first.")

    baseline_df = pd.read_parquet(baseline_path)
    meta_df = pd.read_parquet(path_meta_path)

    df = baseline_df.merge(
        meta_df[["Id", "path_signature", "path_count"]],
        on="Id",
        how="inner",
        validate="one_to_one",
    )

    n = len(df)
    global_mean = float(df["Response"].mean())

    chunk_failure_rate = np.full(n, global_mean, dtype=np.float32)
    signature_failure_rate = np.full(n, global_mean, dtype=np.float32)
    path_failure_rate = np.full(n, global_mean, dtype=np.float32)
    rolling_fail_rate = np.full(n, global_mean, dtype=np.float32)

    cv_cfg = ChunkCVConfig(n_splits=args.n_splits, random_state=42, shuffle=True)
    splits = make_chunk_aware_splits(df, target_col="Response", group_col="chunk_id", config=cv_cfg)
    fold_ids = assign_fold_ids(n_rows=n, splits=splits)

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        tr = df.iloc[train_idx]
        va = df.iloc[valid_idx]

        chunk_rate_map = tr.groupby("chunk_id")["Response"].mean()
        sig_rate_map = tr.groupby("path_signature")["Response"].mean()

        chunk_failure_rate[valid_idx] = va["chunk_id"].map(chunk_rate_map).fillna(global_mean).to_numpy(dtype=np.float32)
        signature_failure_rate[valid_idx] = (
            va["path_signature"].map(sig_rate_map).fillna(global_mean).to_numpy(dtype=np.float32)
        )
        path_failure_rate[valid_idx] = signature_failure_rate[valid_idx]

        fold_roll = _rolling_rate_from_train_to_valid(
            train_df=tr,
            valid_df=va,
            global_mean=global_mean,
            window=10_000,
        )
        rolling_fail_rate[valid_idx] = fold_roll.to_numpy(dtype=np.float32)

        logger.info(
            "Dataset G fold=%d train=%d valid=%d", fold_idx, len(train_idx), len(valid_idx)
        )

    out = df[["Id", "Response", *BASELINE_COLUMNS]].copy()
    out["chunk_failure_rate"] = chunk_failure_rate
    out["rolling_fail_rate_w10000"] = rolling_fail_rate
    out["signature_failure_rate"] = signature_failure_rate
    out["path_failure_rate"] = path_failure_rate
    out["duration_x_path_failure_rate"] = (
        out["duration"].to_numpy(dtype=np.float32) * out["path_failure_rate"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    out["feature_mean_x_duration"] = (
        out["feature_mean"].to_numpy(dtype=np.float32) * out["duration"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    out["cv_fold"] = fold_ids.astype(np.int16)

    output_path = FEATURES_DIR / "dataset_g.parquet"
    out.to_parquet(output_path, index=False)

    logger.info("Saved Dataset G: %s rows=%d", output_path, len(out))


if __name__ == "__main__":
    main()
