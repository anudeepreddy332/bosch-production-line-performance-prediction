from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from src.features.core_pipeline import CorePipelineConfig, build_core_features
from src.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

STATION_COL_PATTERN = re.compile(r"^L\d+_S\d+_D\d+$")


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _append_parquet(writer: pq.ParquetWriter | None, frame: pd.DataFrame, output_path: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy", use_dictionary=True)

    if table.schema != writer.schema:
        table = table.cast(writer.schema, safe=False)

    writer.write_table(table)
    return writer


def _station_groups(date_columns: list[str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for col in date_columns:
        if not STATION_COL_PATTERN.match(col):
            continue
        line, station, _ = col.split("_")
        key = f"{line}_{station}"
        groups.setdefault(key, []).append(col)

    return sorted(groups.items(), key=lambda x: x[0])


def _mask_to_signature(mask: np.ndarray, station_keys: np.ndarray) -> str:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return "__none__"
    return "|".join(station_keys[idx].tolist())


def _build_numeric_core(numeric_path: Path, temp_output: Path, batch_size: int) -> None:
    pf = pq.ParquetFile(numeric_path)
    cols = pf.schema.names
    sensor_cols = [c for c in cols if c not in {"Id", "Response"}]

    writer: pq.ParquetWriter | None = None
    rows = 0

    for idx, batch in enumerate(pf.iter_batches(batch_size=batch_size), start=1):
        chunk = batch.to_pandas()
        out = pd.DataFrame(
            {
                "Id": pd.to_numeric(chunk["Id"], errors="coerce").astype(np.int64),
                "Response": pd.to_numeric(chunk["Response"], errors="coerce").fillna(0).astype(np.int8),
                "feature_mean": chunk[sensor_cols].mean(axis=1, skipna=True).astype(np.float32),
            }
        )

        writer = _append_parquet(writer, out, temp_output)
        rows += len(out)

        if idx == 1 or idx % 10 == 0:
            logger.info("Numeric pass chunk=%d rows=%d mem=%.2fGB", idx, rows, _memory_gb())

        del chunk, out
        gc.collect()

    if writer is None:
        raise RuntimeError("No numeric batches were processed.")
    writer.close()


def _build_date_core(date_path: Path, temp_output: Path, batch_size: int) -> None:
    pf = pq.ParquetFile(date_path)
    cols = pf.schema.names
    date_cols = [c for c in cols if c != "Id"]
    station_groups = _station_groups(date_cols)
    station_keys = np.array([k for k, _ in station_groups], dtype=object)

    writer: pq.ParquetWriter | None = None
    rows = 0

    for idx, batch in enumerate(pf.iter_batches(batch_size=batch_size), start=1):
        chunk = batch.to_pandas()

        start_time = chunk[date_cols].min(axis=1, skipna=True).astype(np.float32)
        end_time = chunk[date_cols].max(axis=1, skipna=True).astype(np.float32)
        duration = (end_time - start_time).fillna(0.0).astype(np.float32)

        if station_groups:
            station_presence = np.column_stack(
                [chunk[group_cols].notna().any(axis=1).to_numpy(dtype=bool) for _, group_cols in station_groups]
            )
            path_signature = [
                _mask_to_signature(mask=row_mask, station_keys=station_keys) for row_mask in station_presence
            ]
            station_count = station_presence.sum(axis=1).astype(np.int16)
        else:
            station_presence = None
            path_signature = ["__none__"] * len(chunk)
            station_count = np.zeros(len(chunk), dtype=np.int16)

        out = pd.DataFrame(
            {
                "Id": pd.to_numeric(chunk["Id"], errors="coerce").astype(np.int64),
                "start_time": start_time,
                "duration": duration,
                "path_signature": path_signature,
                "station_count": station_count,
            }
        )

        writer = _append_parquet(writer, out, temp_output)
        rows += len(out)

        if idx == 1 or idx % 10 == 0:
            logger.info("Date pass chunk=%d rows=%d mem=%.2fGB", idx, rows, _memory_gb())

        del chunk, out, station_presence, path_signature
        gc.collect()

    if writer is None:
        raise RuntimeError("No date batches were processed.")
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline dataset with lean core features.")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--chunk-size-rows", type=int, default=10_000)
    args = parser.parse_args()

    numeric_path = PROCESSED_DIR / "train_numeric.parquet"
    date_path = PROCESSED_DIR / "train_date.parquet"

    if not numeric_path.exists() or not date_path.exists():
        raise FileNotFoundError(
            "Missing required processed files. Run scripts/prepare_data.py first to create train_numeric.parquet and train_date.parquet"
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_numeric = FEATURES_DIR / "_tmp_numeric_core.parquet"
    tmp_date = FEATURES_DIR / "_tmp_date_core.parquet"

    logger.info("Building numeric core features from %s", numeric_path)
    _build_numeric_core(numeric_path, tmp_numeric, batch_size=args.batch_size)

    logger.info("Building date core + path metadata from %s", date_path)
    _build_date_core(date_path, tmp_date, batch_size=args.batch_size)

    logger.info("Merging temp datasets")
    numeric_df = pd.read_parquet(tmp_numeric)
    date_df = pd.read_parquet(tmp_date)

    merged = numeric_df.merge(date_df, on="Id", how="inner", validate="one_to_one")
    merged = merged.sort_values("Id", kind="mergesort").reset_index(drop=True)

    path_counts = merged["path_signature"].value_counts(dropna=False)
    merged["path_count"] = merged["path_signature"].map(path_counts).astype(np.int32)

    core = build_core_features(
        merged[["Id", "Response", "start_time", "duration", "feature_mean"]],
        config=CorePipelineConfig(chunk_size_rows=args.chunk_size_rows),
    )

    baseline_output = FEATURES_DIR / "dataset_baseline.parquet"
    core.to_parquet(baseline_output, index=False)

    meta_output = FEATURES_DIR / "path_metadata.parquet"
    metadata = merged[["Id", "path_signature", "station_count", "path_count"]].merge(
        core[["Id", "chunk_id", "chunk_size"]], on="Id", how="inner", validate="one_to_one"
    )
    metadata.to_parquet(meta_output, index=False)

    for temp_path in [tmp_numeric, tmp_date]:
        if temp_path.exists():
            temp_path.unlink()

    logger.info("Saved baseline dataset: %s rows=%d", baseline_output, len(core))
    logger.info("Saved path metadata: %s", meta_output)


if __name__ == "__main__":
    main()
