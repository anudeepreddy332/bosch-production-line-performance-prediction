from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CORE_FEATURE_COLUMNS = [
    "start_time",
    "duration",
    "feature_mean",
    "records_last_1hr",
    "records_last_24hr",
    "density_ratio",
    "chunk_id",
    "chunk_size",
]


@dataclass(frozen=True)
class CorePipelineConfig:
    chunk_size_rows: int = 10_000


def _fill_start_time(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64, copy=True)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.arange(len(arr), dtype=np.float64)

    min_value = float(arr[finite_mask].min())
    arr[~finite_mask] = min_value - 1.0
    return arr


def _rolling_count_by_time(start_time: np.ndarray, window_size: float) -> np.ndarray:
    n = len(start_time)
    sort_idx = np.argsort(start_time, kind="mergesort")
    sorted_time = start_time[sort_idx]

    counts_sorted = np.zeros(n, dtype=np.int32)
    left = 0
    for right in range(n):
        current = sorted_time[right]
        while left <= right and (current - sorted_time[left]) > window_size:
            left += 1
        counts_sorted[right] = right - left + 1

    counts = np.zeros(n, dtype=np.int32)
    counts[sort_idx] = counts_sorted
    return counts


def _build_chunk_columns(start_time: np.ndarray, chunk_size_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size_rows <= 0:
        raise ValueError("chunk_size_rows must be a positive integer.")

    n = len(start_time)
    sort_idx = np.argsort(start_time, kind="mergesort")

    chunk_id_sorted = (np.arange(n, dtype=np.int64) // int(chunk_size_rows)).astype(np.int32)
    chunk_id = np.zeros(n, dtype=np.int32)
    chunk_id[sort_idx] = chunk_id_sorted

    chunk_sizes = pd.Series(chunk_id).value_counts(sort=False).sort_index()
    chunk_size = pd.Series(chunk_id).map(chunk_sizes).astype(np.int32).to_numpy()
    return chunk_id, chunk_size


def build_core_features(df: pd.DataFrame, config: CorePipelineConfig | None = None) -> pd.DataFrame:
    """
    Build the lean core feature block used across all downstream datasets.

    Required input columns: Id, start_time, duration, feature_mean
    Optional input column: Response
    """
    cfg = config or CorePipelineConfig()
    required = {"Id", "start_time", "duration", "feature_mean"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for core features: {missing}")

    out = pd.DataFrame({"Id": df["Id"].astype(np.int64)})
    if "Response" in df.columns:
        out["Response"] = df["Response"].fillna(0).astype(np.int8)

    out["start_time"] = pd.to_numeric(df["start_time"], errors="coerce").astype(np.float32)
    out["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).astype(np.float32)
    out["feature_mean"] = pd.to_numeric(df["feature_mean"], errors="coerce").fillna(0.0).astype(np.float32)

    safe_start = _fill_start_time(out["start_time"].to_numpy(dtype=np.float64, copy=False))
    out["records_last_1hr"] = _rolling_count_by_time(safe_start, window_size=1.0).astype(np.int32)
    out["records_last_24hr"] = _rolling_count_by_time(safe_start, window_size=24.0).astype(np.int32)
    out["density_ratio"] = (
        out["records_last_1hr"].to_numpy(dtype=np.float32)
        / np.maximum(out["records_last_24hr"].to_numpy(dtype=np.float32), 1.0)
    ).astype(np.float32)

    chunk_id, chunk_size = _build_chunk_columns(safe_start, chunk_size_rows=cfg.chunk_size_rows)
    out["chunk_id"] = chunk_id
    out["chunk_size"] = chunk_size

    ordered_cols = ["Id"]
    if "Response" in out.columns:
        ordered_cols.append("Response")
    ordered_cols.extend(CORE_FEATURE_COLUMNS)

    return out[ordered_cols]
