"""P0 (KDR-007): build the raw-numeric probe dataset.

Reads Production's raw `{train,test}_numeric.parquet` **read-only** (never written to),
enforces float32 end-to-end, computes 6 NaN-aware row-level aggregates
(`p0_nnz`, `p0_row_min`, `p0_row_max`, `p0_row_mean`, `p0_row_std`, `p0_row_sum`) over the
968 raw numeric feature columns, then additively merges onto the existing
`dataset_h_dup_{train,test}.parquet` (Id, Response, chunk_id, `DATASET_H_FEATURE_COLS`,
`POSITION_ONLY_MAGIC_COLS`, `DUPLICATE_FEATURE_COLS`, `DUPLICATE_LABEL_COLS` -- label cols
are carried through for provenance but excluded from Cell B/C training, KDR-007 SS4).
`cv_fold` (train-only) is additionally merged in from `dataset_h.parquet` to enable the
persisted-fold verification guard in `src.kaggle.wide_modeling`.

No feature engineering beyond the 6 global aggregates happens here -- station/date
engineering and feature selection are explicitly out of scope for P0 (KDR-007 SS4).

Outputs (gitignored):
  data/features/dataset_p0_raw_train.parquet
  data/features/dataset_p0_raw_test.parquet

Reproduce (slow step: two ~4.6 GB float32 raw-numeric reads, sequential to bound peak RAM):
  PYTHONPATH=. python scripts/kaggle/build_dataset_p0_raw.py
"""
from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features.dataset_h_pipeline import DATASET_H_FEATURE_COLS
from src.kaggle.duplicate_features import DUPLICATE_FEATURE_COLS, DUPLICATE_LABEL_COLS
from src.kaggle.magic_features import POSITION_ONLY_MAGIC_COLS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

TRAIN_NUMERIC_RAW = PROCESSED_DIR / "train_numeric.parquet"
TEST_NUMERIC_RAW = PROCESSED_DIR / "test_numeric.parquet"
TRAIN_BASE_IN = FEATURES_DIR / "dataset_h_dup_train.parquet"
TEST_BASE_IN = FEATURES_DIR / "dataset_h_dup_test.parquet"
DATASET_H_IN = FEATURES_DIR / "dataset_h.parquet"  # source of train-only cv_fold
TRAIN_OUT = FEATURES_DIR / "dataset_p0_raw_train.parquet"
TEST_OUT = FEATURES_DIR / "dataset_p0_raw_test.parquet"

NON_FEATURE_COLS = {"Id", "Response"}

AGGREGATE_COLS = ["p0_nnz", "p0_row_min", "p0_row_max", "p0_row_mean", "p0_row_std", "p0_row_sum"]
_AGGREGATE_FLOAT_COLS = [c for c in AGGREGATE_COLS if c != "p0_nnz"]

BASE_KEEP_COLS = [
    *DATASET_H_FEATURE_COLS,
    *POSITION_ONLY_MAGIC_COLS,
    *DUPLICATE_FEATURE_COLS,
    *DUPLICATE_LABEL_COLS,
]


def _raw_numeric_feature_cols(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return [n for n in names if n not in NON_FEATURE_COLS]


def _assert_float32(df: pd.DataFrame, cols: list[str], label: str) -> None:
    bad = [c for c in cols if df[c].dtype != np.float32]
    if bad:
        raise RuntimeError(f"{label}: expected float32 for {len(bad)} columns, found other dtypes: {bad[:10]}")


def _load_raw_numeric(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["Id", *feature_cols])
    if df["Id"].dtype != np.int64:
        df["Id"] = df["Id"].astype(np.int64)
    float_cols = [c for c in feature_cols if df[c].dtype != np.float32]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    _assert_float32(df, feature_cols, f"raw numeric read ({path.name})")
    return df


def _compute_aggregates(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    values = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    mask = ~np.isnan(values)
    nnz = mask.sum(axis=1).astype(np.int16)
    has_any = nnz > 0

    # nanmin/nanmax/nanmean/nanstd raise "All-NaN slice"/"Mean of empty slice" RuntimeWarnings
    # (via the warnings module, not floating-point errstate) for all-NaN rows even though the
    # `has_any` mask already routes those rows to an explicit NaN below -- suppress the noise.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_min = np.where(has_any, np.nanmin(values, axis=1), np.nan).astype(np.float32)
        row_max = np.where(has_any, np.nanmax(values, axis=1), np.nan).astype(np.float32)
        row_mean = np.where(has_any, np.nanmean(values, axis=1), np.nan).astype(np.float32)
        row_std = np.where(has_any, np.nanstd(values, axis=1), np.nan).astype(np.float32)
        row_sum = np.where(has_any, np.nansum(values, axis=1), np.nan).astype(np.float32)

    agg = pd.DataFrame(
        {
            "Id": df["Id"].to_numpy(dtype=np.int64),
            "p0_nnz": nnz,
            "p0_row_min": row_min,
            "p0_row_max": row_max,
            "p0_row_mean": row_mean,
            "p0_row_std": row_std,
            "p0_row_sum": row_sum,
        }
    )
    _assert_float32(agg, _AGGREGATE_FLOAT_COLS, "aggregates")
    return agg


def _build_side(
    raw_path: Path,
    base_df: pd.DataFrame,
    raw_feature_cols: list[str],
    keep_cols: list[str],
    out_path: Path,
    side: str,
) -> None:
    dupes = {c for c in keep_cols if keep_cols.count(c) > 1}
    if dupes:
        raise RuntimeError(f"[{side}] duplicate column names in keep_cols: {sorted(dupes)}")

    print(f"[{side}] reading raw numeric matrix (read-only, float32-enforced) ...")
    raw = _load_raw_numeric(raw_path, raw_feature_cols)
    print(f"[{side}] computing 6 global aggregates ...")
    agg = _compute_aggregates(raw, raw_feature_cols)

    raw_full = raw.merge(agg, on="Id", how="left", validate="one_to_one")
    assert len(raw_full) == len(raw), f"[{side}] row count changed during aggregate merge"
    del raw, agg
    gc.collect()

    out = base_df[["Id", *keep_cols]].merge(raw_full, on="Id", how="left", validate="one_to_one")
    assert len(out) == len(base_df), f"[{side}] row count changed during raw merge onto base"
    del raw_full
    gc.collect()

    _assert_float32(out, [*raw_feature_cols, *_AGGREGATE_FLOAT_COLS], f"final {side} output")
    # NaN in a single raw sensor column is expected Bosch sparsity (a part may skip that
    # station), not a merge failure -- row-count asserts above are what guard the merge itself.
    first_col_nan = int(out[raw_feature_cols[0]].isna().sum()) if raw_feature_cols else 0
    print(f"[{side}] rows={len(out)} cols={len(out.columns)} (NaN rows in raw col {raw_feature_cols[0]!r}: {first_col_nan} -- expected sparsity, not a merge signal)")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{side}] wrote {out_path}")
    del out
    gc.collect()


def main() -> None:
    for p in (TRAIN_NUMERIC_RAW, TEST_NUMERIC_RAW, TRAIN_BASE_IN, TEST_BASE_IN, DATASET_H_IN):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}.")

    train_raw_cols = _raw_numeric_feature_cols(TRAIN_NUMERIC_RAW)
    test_raw_cols = _raw_numeric_feature_cols(TEST_NUMERIC_RAW)
    if train_raw_cols != test_raw_cols:
        raise RuntimeError("train_numeric.parquet and test_numeric.parquet raw feature columns differ in name/order")
    print(f"raw numeric feature columns: {len(train_raw_cols)}")

    train_base = pd.read_parquet(TRAIN_BASE_IN)
    test_base = pd.read_parquet(TEST_BASE_IN)
    print(f"train_base rows={len(train_base)} cols={len(train_base.columns)}")
    print(f"test_base rows={len(test_base)} cols={len(test_base.columns)}")

    cv_fold = pd.read_parquet(DATASET_H_IN, columns=["Id", "cv_fold"])
    train_base = train_base.merge(cv_fold, on="Id", how="left", validate="one_to_one")
    assert train_base["cv_fold"].isna().sum() == 0, "cv_fold merge left NaN rows -- Id mismatch with dataset_h.parquet"
    train_base["cv_fold"] = train_base["cv_fold"].astype(np.int16)
    print(f"merged cv_fold onto train_base (train-only, from {DATASET_H_IN.name})")

    # chunk_id is already part of DATASET_H_FEATURE_COLS (inside BASE_KEEP_COLS) -- do not add it again.
    train_keep_cols = ["Response", "cv_fold", *BASE_KEEP_COLS]
    test_keep_cols = [*BASE_KEEP_COLS]

    _build_side(TRAIN_NUMERIC_RAW, train_base, train_raw_cols, train_keep_cols, TRAIN_OUT, side="train")
    del train_base
    gc.collect()

    _build_side(TEST_NUMERIC_RAW, test_base, test_raw_cols, test_keep_cols, TEST_OUT, side="test")

    print(f"raw_numeric_cols={len(train_raw_cols)} aggregate_cols={len(AGGREGATE_COLS)} base_keep_cols={len(BASE_KEEP_COLS)}")


if __name__ == "__main__":
    main()
