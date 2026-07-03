"""E1 experiment: build dataset_e1 by adding per-station sensor means to dataset_h.

Adds 52 features to dataset_h's 16:
- sensor_mean_L{l}_S{s} (50 features): mean of non-null numeric readings at each station.
  NaN for unvisited stations — LightGBM handles NaN natively, so missingness structure
  (which stations were visited) is encoded for free without any explicit encoding.
- sensor_nonull_count: total non-null sensor readings per part (measurement breadth).
- sensor_std: std across all non-null sensor readings (distributional spread).

See docs/research/decisions.md DR-004 for justification of these choices.
"""
from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from src.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

_NUMERIC_COL_PAT = re.compile(r"^(L\d+_S\d+)_F\d+$")


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _station_groups(sensor_cols: list[str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for col in sensor_cols:
        m = _NUMERIC_COL_PAT.match(col)
        if m:
            key = m.group(1)
            groups.setdefault(key, []).append(col)
    return sorted(groups.items())


def _build_sensor_features(numeric_path: Path, batch_size: int) -> pd.DataFrame:
    pf = pq.ParquetFile(numeric_path)
    all_cols = pf.schema.names
    sensor_cols = [c for c in all_cols if c not in {"Id", "Response"}]
    station_groups = _station_groups(sensor_cols)
    station_keys = [k for k, _ in station_groups]

    logger.info(
        "Sensor columns=%d station groups=%d", len(sensor_cols), len(station_groups)
    )

    result_chunks: list[pd.DataFrame] = []
    total_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size), start=1):
        chunk = batch.to_pandas()
        ids = pd.to_numeric(chunk["Id"], errors="coerce").astype(np.int64)
        sensor_vals = chunk[sensor_cols].astype(np.float32)

        sensor_nonull_count = sensor_vals.notna().sum(axis=1).astype(np.int16)
        sensor_std = sensor_vals.std(axis=1, skipna=True).astype(np.float32)

        station_means: dict[str, np.ndarray] = {}
        for key, cols in station_groups:
            block = sensor_vals[cols]
            station_means[f"sensor_mean_{key}"] = (
                block.mean(axis=1, skipna=True).astype(np.float32).to_numpy()
            )

        out = pd.DataFrame({"Id": ids})
        out["sensor_nonull_count"] = sensor_nonull_count.to_numpy()
        out["sensor_std"] = sensor_std.to_numpy()
        for key in station_keys:
            out[f"sensor_mean_{key}"] = station_means[f"sensor_mean_{key}"]

        result_chunks.append(out)
        total_rows += len(out)

        if batch_idx == 1 or batch_idx % 5 == 0:
            logger.info(
                "Sensor batch=%d rows=%d mem=%.2fGB", batch_idx, total_rows, _memory_gb()
            )

        del chunk, sensor_vals, station_means, out
        gc.collect()

    logger.info("Sensor feature pass complete: total_rows=%d", total_rows)
    return pd.concat(result_chunks, ignore_index=True)


def main() -> None:
    numeric_path = PROCESSED_DIR / "train_numeric.parquet"
    dataset_h_path = FEATURES_DIR / "dataset_h.parquet"

    for p in [numeric_path, dataset_h_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run prepare_data.py and build_dataset_h.py first."
            )

    logger.info("Building sensor features from %s", numeric_path)
    sensor_df = _build_sensor_features(numeric_path, batch_size=20_000)

    logger.info("Loading dataset_h from %s", dataset_h_path)
    dataset_h = pd.read_parquet(dataset_h_path)

    logger.info("Merging sensor features with dataset_h (rows=%d)", len(dataset_h))
    merged = dataset_h.merge(sensor_df, on="Id", how="inner", validate="one_to_one")

    if len(merged) != len(dataset_h):
        raise RuntimeError(
            f"Row count mismatch after merge: dataset_h={len(dataset_h)} merged={len(merged)}"
        )

    output_path = FEATURES_DIR / "dataset_e1.parquet"
    merged.to_parquet(output_path, index=False)

    logger.info(
        "Saved dataset_e1: %s rows=%d cols=%d", output_path, len(merged), len(merged.columns)
    )
    sensor_feat_cols = [c for c in merged.columns if c.startswith("sensor_")]
    logger.info("Sensor feature columns added: %d", len(sensor_feat_cols))


if __name__ == "__main__":
    main()
