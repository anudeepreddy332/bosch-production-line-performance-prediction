from __future__ import annotations

import argparse
import gc
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from src.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_ZIP_PATH = Path.home() / "Downloads" / "bosch-production-line-performance.zip"


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def safe_unzip(zip_path: Path, destination: Path, overwrite: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)
    dest_root = destination.resolve()

    logger.info("Unzipping dataset: %s -> %s", zip_path, destination)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        logger.info("ZIP entries to extract: %d", len(members))

        for idx, member in enumerate(members, start=1):
            target_path = (destination / member.filename).resolve()
            if not str(target_path).startswith(str(dest_root)):
                raise ValueError(f"Unsafe ZIP entry blocked: {member.filename}")

            if target_path.exists() and not overwrite:
                logger.info("[%d/%d] Exists, skipping: %s", idx, len(members), target_path.name)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("[%d/%d] Extracted: %s", idx, len(members), target_path.name)


def iter_csv_files(raw_dir: Path) -> list[Path]:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {raw_dir}")
    return csv_files


def optimize_chunk_dtypes(chunk: pd.DataFrame) -> pd.DataFrame:
    for col in chunk.columns:
        if col == "Response":
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype("int8")
            continue

        if col == "Id":
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("int64")
            continue

        if pd.api.types.is_float_dtype(chunk[col]):
            chunk[col] = chunk[col].astype("float32")
        elif pd.api.types.is_integer_dtype(chunk[col]):
            chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

    return chunk


def convert_csv_to_parquet_incremental(
    csv_path: Path,
    parquet_path: Path,
    chunksize: int,
    overwrite: bool = False,
    log_every: int = 10,
) -> None:
    if parquet_path.exists() and not overwrite:
        logger.info("Parquet exists, skipping: %s", parquet_path)
        return

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting CSV -> Parquet (chunked): %s", csv_path)
    start_time = time.time()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    chunk_count = 0

    try:
        for chunk_count, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunksize, low_memory=False), start=1
        ):
            chunk = optimize_chunk_dtypes(chunk)
            total_rows += len(chunk)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            else:
                if table.schema != writer.schema:
                    table = table.cast(writer.schema, safe=False)

            writer.write_table(table)

            if chunk_count == 1 or (chunk_count % log_every) == 0:
                logger.info(
                    "%s chunks=%d rows=%d mem=%.2fGB",
                    csv_path.name,
                    chunk_count,
                    total_rows,
                    _memory_gb(),
                )

            del chunk, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start_time
    size_mb = parquet_path.stat().st_size / (1024 ** 2)
    logger.info(
        "Completed %s -> %s | rows=%d chunks=%d size=%.1fMB elapsed=%.1fs",
        csv_path.name,
        parquet_path.name,
        total_rows,
        chunk_count,
        size_mb,
        elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely unzip Bosch data and convert CSV files to chunked Parquet.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Path to Bosch ZIP archive.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV rows processed per chunk.")
    parser.add_argument("--skip-unzip", action="store_true", help="Skip unzip step and only run CSV->Parquet conversion.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted/parquet files.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_unzip:
        safe_unzip(args.zip_path, RAW_DIR, overwrite=args.overwrite)

    csv_files = iter_csv_files(RAW_DIR)
    logger.info("CSV files discovered: %d", len(csv_files))

    for csv_path in csv_files:
        parquet_name = f"{csv_path.stem}.parquet"
        parquet_path = PROCESSED_DIR / parquet_name
        convert_csv_to_parquet_incremental(
            csv_path=csv_path,
            parquet_path=parquet_path,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
        )

    logger.info("Data preparation complete. Processed Parquet directory: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
