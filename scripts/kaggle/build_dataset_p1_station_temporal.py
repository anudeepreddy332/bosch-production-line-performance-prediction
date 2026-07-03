"""P1 (KDR-008): build the station-temporal feature side table.

Family D (57 cols): per-station date offsets (`d_st__{L}_{S}`, 52), weekly position at
start/end (`d_week_pos_start`/`d_week_pos_end`, mod 1680), station count (`d_nstations`),
and inter-station transit deltas (`d_transit_mean`/`d_transit_max`). Computed from a single
full read of `{train,test}_date.parquet`, reusing `start_time`/`duration` from the existing
`dataset_p0_raw_{side}.parquet` (never recomputed). No `d_end_time` column is materialized --
it would be a perfect linear combination of `start_time` + `duration`, already in the base
stack (KDR-008 SS3).

Family S/L (108 cols): per-station and per-line row-wise mean/std over the raw numeric
matrix (`s_mean__{L}_{S}`, `s_std__{L}_{S}`, `l_mean__{L}`, `l_std__{L}`), computed from
`dataset_p0_raw_{side}.parquet` (no re-read of `train/test_numeric.parquet`). Station/line
group counts are derived mechanically from the schema (asserted: 52 date stations, 50
numeric stations, 4 lines) -- never hardcoded. `L3_S32`'s single numeric column has its std
left as NaN by design (undefined spread over one sample), not computed via `nanstd`.

Family D and Family S/L are two strictly sequential passes within each side -- the date
DataFrame is fully released before the raw-numeric-derived DataFrame is read, following the
proven P0/`build_dataset_baseline.py` single-full-read pattern (no column-batching: batching
by column risks splitting a single station's date columns, e.g. L1_S25's 333, across batch
boundaries and silently corrupting the per-station reduction).

Outputs (gitignored):
  data/features/dataset_p1_dt_train.parquet
  data/features/dataset_p1_dt_test.parquet

Reproduce (two sequential full-matrix passes per side; ~9.5GB peak for the Family D pass,
~5-6GB peak for the Family S/L pass, KDR-008 SS5):
  PYTHONPATH=. python scripts/kaggle/build_dataset_p1_station_temporal.py
"""
from __future__ import annotations

import gc
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

TRAIN_DATE_RAW = PROCESSED_DIR / "train_date.parquet"
TEST_DATE_RAW = PROCESSED_DIR / "test_date.parquet"
TRAIN_NUMERIC_RAW = PROCESSED_DIR / "train_numeric.parquet"
TEST_NUMERIC_RAW = PROCESSED_DIR / "test_numeric.parquet"
TRAIN_P0_RAW_IN = FEATURES_DIR / "dataset_p0_raw_train.parquet"
TEST_P0_RAW_IN = FEATURES_DIR / "dataset_p0_raw_test.parquet"
TRAIN_OUT = FEATURES_DIR / "dataset_p1_dt_train.parquet"
TEST_OUT = FEATURES_DIR / "dataset_p1_dt_test.parquet"

WEEK_PERIOD = 1680  # 1 week in the raw 6-minute date unit (src/features/pipeline.py precedent)

DATE_COL_PATTERN = re.compile(r"^(L\d+)_(S\d+)_D\d+$")
NUMERIC_COL_PATTERN = re.compile(r"^(L\d+)_(S\d+)_F\d+$")

EXPECTED_DATE_STATIONS = 52
EXPECTED_NUMERIC_STATIONS = 50
EXPECTED_LINES = 4
EXPECTED_FAMILY_D_COLS = 57
EXPECTED_FAMILY_SL_COLS = 108

NON_FEATURE_COLS = {"Id", "Response"}


def _raw_numeric_feature_cols(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return [n for n in names if n not in NON_FEATURE_COLS]


def _station_groups(columns: list[str], pattern: re.Pattern) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        m = pattern.match(col)
        if not m:
            raise RuntimeError(f"Column {col!r} does not match expected station pattern {pattern.pattern!r}")
        key = f"{m.group(1)}_{m.group(2)}"
        groups.setdefault(key, []).append(col)
    return dict(sorted(groups.items()))


def _line_groups(columns: list[str], pattern: re.Pattern) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        m = pattern.match(col)
        if not m:
            raise RuntimeError(f"Column {col!r} does not match expected line pattern {pattern.pattern!r}")
        key = m.group(1)
        groups.setdefault(key, []).append(col)
    return dict(sorted(groups.items()))


def _assert_float32(df: pd.DataFrame, cols: list[str], label: str) -> None:
    bad = [c for c in cols if df[c].dtype != np.float32]
    if bad:
        raise RuntimeError(f"{label}: expected float32 for {len(bad)} columns, found other dtypes: {bad[:10]}")


def _build_family_d(date_path: Path, start_duration: pd.DataFrame, side: str) -> pd.DataFrame:
    print(f"[{side}] Family D: reading full date matrix (single full read, P0 precedent) ...")
    date_df = pd.read_parquet(date_path)
    if date_df["Id"].dtype != np.int64:
        date_df["Id"] = date_df["Id"].astype(np.int64)
    date_cols = [c for c in date_df.columns if c != "Id"]
    float_cols = [c for c in date_cols if date_df[c].dtype != np.float32]
    if float_cols:
        date_df[float_cols] = date_df[float_cols].astype(np.float32)
    _assert_float32(date_df, date_cols, f"[{side}] date matrix read")

    station_groups = _station_groups(date_cols, DATE_COL_PATTERN)
    if len(station_groups) != EXPECTED_DATE_STATIONS:
        raise RuntimeError(
            f"[{side}] expected {EXPECTED_DATE_STATIONS} date stations, found {len(station_groups)}"
        )
    station_keys = sorted(station_groups.keys())

    ids = date_df["Id"].to_numpy(dtype=np.int64, copy=True)

    merged = date_df[["Id"]].merge(start_duration, on="Id", how="left", validate="one_to_one")
    assert len(merged) == len(date_df), f"[{side}] row count changed merging start_time/duration"
    n_nan_start = int(merged["start_time"].isna().sum())
    print(
        f"[{side}] start_time NaN rows after merge: {n_nan_start} -- genuine Bosch sparsity (rows whose "
        f"full date-matrix row is all-NaN, i.e. start_time's own row-min over all date cols is NaN), not "
        f"a merge failure; the row-count assert above is what guards the merge itself. Offsets/week-position "
        f"for these rows are correctly NaN by propagation (no valid time anchor)."
    )
    start_time = merged["start_time"].to_numpy(dtype=np.float32, copy=False)
    duration = merged["duration"].to_numpy(dtype=np.float32, copy=False)
    del merged

    offsets = np.empty((len(date_df), len(station_keys)), dtype=np.float32)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, key in enumerate(station_keys):
            values = date_df[station_groups[key]].to_numpy(dtype=np.float32, copy=False)
            has_any = (~np.isnan(values)).any(axis=1)
            station_min = np.where(has_any, np.nanmin(values, axis=1), np.nan).astype(np.float32)
            offsets[:, i] = station_min - start_time

    del date_df
    gc.collect()

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sorted_offsets = np.sort(offsets, axis=1)  # NaNs sort to the end (ascending)
        diffs = np.diff(sorted_offsets, axis=1)  # NaN wherever either neighbor is NaN
        has_two = (~np.isnan(diffs)).any(axis=1)
        transit_mean = np.where(has_two, np.nanmean(diffs, axis=1), np.nan).astype(np.float32)
        transit_max = np.where(has_two, np.nanmax(diffs, axis=1), np.nan).astype(np.float32)

    d_nstations = (~np.isnan(offsets)).sum(axis=1).astype(np.int16)
    week_pos_start = np.mod(start_time, WEEK_PERIOD).astype(np.float32)
    week_pos_end = np.mod(start_time + duration, WEEK_PERIOD).astype(np.float32)

    out = pd.DataFrame({"Id": ids})
    for i, key in enumerate(station_keys):
        out[f"d_st__{key}"] = offsets[:, i]
    out["d_week_pos_start"] = week_pos_start
    out["d_week_pos_end"] = week_pos_end
    out["d_nstations"] = d_nstations
    out["d_transit_mean"] = transit_mean
    out["d_transit_max"] = transit_max

    d_float_cols = [c for c in out.columns if c not in ("Id", "d_nstations")]
    _assert_float32(out, d_float_cols, f"[{side}] Family D output")
    n_feature_cols = len(out.columns) - 1
    if n_feature_cols != EXPECTED_FAMILY_D_COLS:
        raise RuntimeError(f"[{side}] Family D expected {EXPECTED_FAMILY_D_COLS} feature cols, got {n_feature_cols}")
    print(f"[{side}] Family D: {n_feature_cols} features, rows={len(out)}")
    return out


def _build_family_sl(p0_raw_path: Path, raw_cols: list[str], side: str) -> pd.DataFrame:
    print(f"[{side}] Family S/L: reading raw numeric cols from {p0_raw_path.name} (reuse of P0 artifact) ...")
    df = pd.read_parquet(p0_raw_path, columns=["Id", *raw_cols])
    if df["Id"].dtype != np.int64:
        df["Id"] = df["Id"].astype(np.int64)
    float_cols = [c for c in raw_cols if df[c].dtype != np.float32]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    _assert_float32(df, raw_cols, f"[{side}] Family S/L raw read")

    station_groups = _station_groups(raw_cols, NUMERIC_COL_PATTERN)
    line_groups = _line_groups(raw_cols, NUMERIC_COL_PATTERN)
    if len(station_groups) != EXPECTED_NUMERIC_STATIONS:
        raise RuntimeError(
            f"[{side}] expected {EXPECTED_NUMERIC_STATIONS} numeric stations, found {len(station_groups)}"
        )
    if len(line_groups) != EXPECTED_LINES:
        raise RuntimeError(f"[{side}] expected {EXPECTED_LINES} lines, found {len(line_groups)}")

    out = pd.DataFrame({"Id": df["Id"].to_numpy(dtype=np.int64)})
    n_rows = len(out)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key in sorted(station_groups):
            cols = station_groups[key]
            values = df[cols].to_numpy(dtype=np.float32, copy=False)
            if values.shape[1] == 1:
                # KDR-008 SS3: a single-column station has an undefined spread. Left as NaN by
                # design rather than via nanstd (which would silently return 0.0 for ddof=0 on
                # a single sample) -- L3_S32 is the only such station in this schema.
                out[f"s_mean__{key}"] = values[:, 0]
                out[f"s_std__{key}"] = np.full(n_rows, np.nan, dtype=np.float32)
            else:
                has_any = (~np.isnan(values)).any(axis=1)
                out[f"s_mean__{key}"] = np.where(has_any, np.nanmean(values, axis=1), np.nan).astype(np.float32)
                out[f"s_std__{key}"] = np.where(has_any, np.nanstd(values, axis=1), np.nan).astype(np.float32)
        for key in sorted(line_groups):
            values = df[line_groups[key]].to_numpy(dtype=np.float32, copy=False)
            has_any = (~np.isnan(values)).any(axis=1)
            out[f"l_mean__{key}"] = np.where(has_any, np.nanmean(values, axis=1), np.nan).astype(np.float32)
            out[f"l_std__{key}"] = np.where(has_any, np.nanstd(values, axis=1), np.nan).astype(np.float32)

    del df
    gc.collect()

    sl_float_cols = [c for c in out.columns if c != "Id"]
    _assert_float32(out, sl_float_cols, f"[{side}] Family S/L output")
    n_feature_cols = len(out.columns) - 1
    if n_feature_cols != EXPECTED_FAMILY_SL_COLS:
        raise RuntimeError(f"[{side}] Family S/L expected {EXPECTED_FAMILY_SL_COLS} feature cols, got {n_feature_cols}")
    print(f"[{side}] Family S/L: {n_feature_cols} features, rows={len(out)}")
    return out


def _build_side(
    date_path: Path,
    p0_raw_path: Path,
    raw_cols: list[str],
    out_path: Path,
    side: str,
) -> None:
    start_duration = pd.read_parquet(p0_raw_path, columns=["Id", "start_time", "duration"])
    if start_duration["Id"].dtype != np.int64:
        start_duration["Id"] = start_duration["Id"].astype(np.int64)

    family_d = _build_family_d(date_path, start_duration, side=side)
    del start_duration
    gc.collect()

    family_sl = _build_family_sl(p0_raw_path, raw_cols, side=side)

    out = family_d.merge(family_sl, on="Id", how="left", validate="one_to_one")
    assert len(out) == len(family_d) == len(family_sl), f"[{side}] row count mismatch merging Family D + Family S/L"
    del family_d, family_sl
    gc.collect()

    expected_cols = 1 + EXPECTED_FAMILY_D_COLS + EXPECTED_FAMILY_SL_COLS
    if len(out.columns) != expected_cols:
        raise RuntimeError(f"[{side}] expected {expected_cols} output columns, got {len(out.columns)}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{side}] wrote {out_path} rows={len(out)} cols={len(out.columns)}")
    del out
    gc.collect()


def main() -> None:
    for p in (TRAIN_DATE_RAW, TEST_DATE_RAW, TRAIN_NUMERIC_RAW, TEST_NUMERIC_RAW, TRAIN_P0_RAW_IN, TEST_P0_RAW_IN):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}.")

    train_date_cols = [n for n in pq.ParquetFile(TRAIN_DATE_RAW).schema_arrow.names if n != "Id"]
    test_date_cols = [n for n in pq.ParquetFile(TEST_DATE_RAW).schema_arrow.names if n != "Id"]
    if train_date_cols != test_date_cols:
        raise RuntimeError("train_date.parquet and test_date.parquet date columns differ in name/order")

    train_raw_cols = _raw_numeric_feature_cols(TRAIN_NUMERIC_RAW)
    test_raw_cols = _raw_numeric_feature_cols(TEST_NUMERIC_RAW)
    if train_raw_cols != test_raw_cols:
        raise RuntimeError("train_numeric.parquet and test_numeric.parquet raw feature columns differ in name/order")

    print(f"date columns: {len(train_date_cols)} | raw numeric feature columns: {len(train_raw_cols)}")

    _build_side(TRAIN_DATE_RAW, TRAIN_P0_RAW_IN, train_raw_cols, TRAIN_OUT, side="train")
    _build_side(TEST_DATE_RAW, TEST_P0_RAW_IN, test_raw_cols, TEST_OUT, side="test")

    print(f"family_d_cols={EXPECTED_FAMILY_D_COLS} family_sl_cols={EXPECTED_FAMILY_SL_COLS}")


if __name__ == "__main__":
    main()
