from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
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
PROVENANCE_PATH = PROCESSED_DIR / "PROVENANCE.json"


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


def _categorical_target_schema(csv_path: Path) -> pa.Schema:
    """Fixed per-column Arrow schema for a Bosch *_categorical.csv file.

    Bosch categorical columns are extremely sparse (many are 100% null in any
    given 50k-row chunk) and hold alphanumeric level codes (e.g. "T1") in
    other chunks. Without a fixed target schema, pandas/pyarrow infer an
    all-null chunk's column as float/null type, and the writer then locks
    that in from chunk 1 -- a later chunk with real string values fails to
    cast into that narrower type. Declaring every non-Id column as pa.string()
    up front makes every chunk's schema identical regardless of nullness.
    """
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    fields = [
        pa.field("Id", pa.int64()) if col == "Id" else pa.field(col, pa.string())
        for col in header_cols
    ]
    return pa.schema(fields)


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
    sample_rows: int | None = None,
) -> tuple[int, str]:
    """Convert a CSV file to chunked Parquet.

    Returns ``(rows, action)`` where ``action`` is ``"generated"`` if this call
    actually (re)wrote the parquet file from the CSV this run, or
    ``"skipped_existing"`` if an existing parquet file was left as-is (in which
    case ``rows`` is the row count read from that existing file's metadata —
    NOT evidence that the existing file matches the current ``sample_rows``
    setting; the caller must not assume a skipped file is full data).

    If ``sample_rows`` is None (the default), the FULL CSV is converted —
    there is no implicit row cap. Passing ``sample_rows`` explicitly caps
    the output to the first N rows (reading only as many chunks as needed),
    which is the only way a sampled/truncated parquet can be produced.
    """
    if parquet_path.exists() and not overwrite:
        existing_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        logger.warning(
            "Parquet exists, skipping (kept as-is): %s | existing_rows=%d. "
            "This file may be a STALE or PARTIAL artifact from a previous run "
            "(e.g. a dev sample). Pass --overwrite to regenerate it from the "
            "current CSV with the current --sample-rows setting.",
            parquet_path,
            existing_rows,
        )
        return existing_rows, "skipped_existing"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Converting CSV -> Parquet (chunked): %s%s",
        csv_path,
        f" | sample_rows={sample_rows} (EXPLICIT SAMPLE, not full data)" if sample_rows is not None else " | full data, no row cap",
    )
    start_time = time.time()

    is_categorical = "categorical" in csv_path.stem.lower()
    target_schema = _categorical_target_schema(csv_path) if is_categorical else None
    read_kwargs = {"dtype": str} if is_categorical else {}

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    chunk_count = 0

    try:
        for chunk_count, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, **read_kwargs), start=1
        ):
            if sample_rows is not None and total_rows >= sample_rows:
                break

            if sample_rows is not None and total_rows + len(chunk) > sample_rows:
                chunk = chunk.iloc[: sample_rows - total_rows]

            chunk = optimize_chunk_dtypes(chunk)
            total_rows += len(chunk)

            table = pa.Table.from_pandas(chunk, schema=target_schema, preserve_index=False)
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

            if sample_rows is not None and total_rows >= sample_rows:
                break
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
    logger.info("Final row count for %s: %d", parquet_path.name, total_rows)
    return total_rows, "generated"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _load_previous_sample_evidence(prev_path: Path) -> dict[str, int]:
    """Read a previous PROVENANCE.json (if any) and return ``{filename: rows}``
    for files that previous evidence indicates were a SAMPLE (not full data).

    Only the "this was a sample" direction is ever trusted across runs. A
    previous claim of full data is deliberately never carried forward as
    verified — that asymmetry is the whole point: under-claiming completeness
    (calling something a sample when unsure) is safe, but over-claiming it
    (calling a never-reverified file "full data") is exactly the false-evidence
    bug this function exists to prevent.
    """
    if not prev_path.exists():
        return {}
    try:
        prev = json.loads(prev_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    sample_rows_by_file: dict[str, int] = {}

    prev_files = prev.get("files")
    if isinstance(prev_files, dict):
        for name, info in prev_files.items():
            if isinstance(info, dict) and info.get("full_data_status") in ("generated_sample", "sampled"):
                rows = info.get("rows")
                if isinstance(rows, int):
                    sample_rows_by_file[name] = rows
        return sample_rows_by_file

    # Old (pre-fix) schema: a single global is_full_data + flat row_counts.
    # Only honor this if it explicitly claims sample data; an old-schema
    # is_full_data=true is exactly the unverified claim this fix exists to
    # stop trusting, so it is intentionally ignored here.
    if prev.get("is_full_data") is False:
        row_counts = prev.get("row_counts")
        if isinstance(row_counts, dict):
            for name, rows in row_counts.items():
                if isinstance(rows, int):
                    sample_rows_by_file[name] = rows
    return sample_rows_by_file


def _overall_status(file_statuses: list[str]) -> str:
    """Roll per-file full_data_status values up into one overall status."""
    full_like = {"generated_full"}
    sample_like = {"generated_sample", "sampled"}
    has_full = any(s in full_like for s in file_statuses)
    has_sample = any(s in sample_like for s in file_statuses)
    has_unverified = any(s == "unverified" for s in file_statuses)

    if has_full and has_sample:
        return "mixed"
    if has_unverified:
        return "unknown"
    if has_full:
        return "full_data"
    if has_sample:
        return "sample"
    return "unknown"


def write_provenance(
    file_info: dict[str, dict],
    requested_sample_rows: int | None,
    sample_tag: str | None,
) -> None:
    """Write a small JSON manifest documenting how data/processed/*.parquet
    was produced, so downstream readers never have to guess whether the
    committed parquet files are full data or a sample.

    ``file_info`` maps parquet filename -> {"rows": int, "action": "generated"
    | "skipped_existing"}, as produced by convert_csv_to_parquet_incremental
    for THIS run. Critically: a file's full_data_status is only ever
    "generated_full" when it was actually (re)converted from its CSV during
    this exact run with no --sample-rows cap. Omitting --sample-rows has NO
    effect on files that were skipped because a parquet already existed —
    those are classified independently of the current CLI flags.
    """
    prev_sample_evidence = _load_previous_sample_evidence(PROVENANCE_PATH)

    files: dict[str, dict] = {}
    for name, info in file_info.items():
        rows = info["rows"]
        action = info["action"]
        if action == "generated":
            status = "generated_full" if requested_sample_rows is None else "generated_sample"
        else:
            prev_rows = prev_sample_evidence.get(name)
            status = "sampled" if prev_rows is not None and prev_rows == rows else "unverified"
        files[name] = {"rows": rows, "action": action, "full_data_status": status}

    overall_status = _overall_status([f["full_data_status"] for f in files.values()])
    is_full_data = {"full_data": True, "sample": False}.get(overall_status)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "requested_sample_rows": requested_sample_rows,
        "sample_tag": sample_tag,
        "status": overall_status,
        "is_full_data": is_full_data,
        "row_counts": {name: f["rows"] for name, f in files.items()},
        "files": files,
        "note": (
            "Per-file full_data_status: 'generated_full' = this file was freshly "
            "converted from its source CSV during THIS run with no --sample-rows "
            "cap (directly observed). 'generated_sample' = freshly converted THIS "
            "run with an explicit --sample-rows cap (directly observed). 'sampled' "
            "= this file already existed and was SKIPPED (not regenerated) this "
            "run, but a previous PROVENANCE.json recorded the exact same row count "
            "for this filename as a sample, carried forward as a conservative, "
            "safe-direction claim. 'unverified' = this file already existed and "
            "was skipped, and there is no trustworthy evidence it is a sample; "
            "this is also used whenever a previous record claimed FULL data for a "
            "skipped file, because a claim of full data is never itself treated as "
            "proof -- that asymmetry is intentional: omitting --sample-rows must "
            "never imply full data for a file that was not actually regenerated "
            "this run. Overall 'status' is 'full_data' only if every file is "
            "generated_full; 'sample' only if every file is generated_sample or "
            "sampled; 'mixed' if some files are confirmed full and others "
            "confirmed sample; 'unknown' if any file is unverified. 'is_full_data' "
            "is a derived convenience boolean (true/false only when status is "
            "unambiguous; null when status is 'mixed' or 'unknown', since no "
            "boolean can honestly represent those cases)."
        ),
    }
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROVENANCE_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote provenance manifest: %s (status=%s)", PROVENANCE_PATH, overall_status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely unzip Bosch data and convert CSV files to chunked Parquet.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Path to Bosch ZIP archive.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV rows processed per chunk.")
    parser.add_argument("--skip-unzip", action="store_true", help="Skip unzip step and only run CSV->Parquet conversion.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted/parquet files.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=(
            "If set, cap EACH processed parquet file to the first N rows "
            "(deterministic). Default is None, meaning FULL data with NO "
            "truncation. This is the only way a sampled/dev dataset is produced."
        ),
    )
    parser.add_argument(
        "--sample-tag",
        type=str,
        default=None,
        help="Free-text label describing sample intent (e.g. 'dev'). Recorded in PROVENANCE.json only; has no effect on processing.",
    )
    args = parser.parse_args()

    if args.sample_rows is None:
        logger.info("No --sample-rows given: processing FULL data (no row cap).")
    else:
        logger.warning(
            "--sample-rows=%d set: each processed parquet will be capped to %d rows "
            "(sample_tag=%s). This is an EXPLICIT sample, not full data.",
            args.sample_rows,
            args.sample_rows,
            args.sample_tag,
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_unzip:
        safe_unzip(args.zip_path, RAW_DIR, overwrite=args.overwrite)

    csv_files = iter_csv_files(RAW_DIR)
    logger.info("CSV files discovered: %d", len(csv_files))

    file_info: dict[str, dict] = {}
    for csv_path in csv_files:
        parquet_name = f"{csv_path.stem}.parquet"
        parquet_path = PROCESSED_DIR / parquet_name
        rows, action = convert_csv_to_parquet_incremental(
            csv_path=csv_path,
            parquet_path=parquet_path,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
            sample_rows=args.sample_rows,
        )
        file_info[parquet_name] = {"rows": rows, "action": action}

    write_provenance(file_info=file_info, requested_sample_rows=args.sample_rows, sample_tag=args.sample_tag)

    logger.info("Data preparation complete. Processed Parquet directory: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
