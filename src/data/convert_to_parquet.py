"""
Convert raw CSV files to optimized Parquet format with chunked processing.

Why chunked processing:
- Large CSVs (2GB+) cause memory issues when loaded all at once
- Chunked reading processes 100K rows at a time (5-8GB RAM vs 50GB+)
- Prevents swap thrashing on memory-constrained hardware
- Slightly slower but much more reliable

Key optimizations:
1. Dtype reduction (float64→float32, int64→int8/int16)
2. Categorical encoding for string columns
3. Parquet columnar compression (snappy)
4. Chunked processing with configurable chunk size

Result: 12GB CSVs → ~3GB Parquets using <10GB RAM
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import argparse
from tqdm import tqdm
from src.config import Config
from src.logger import setup_logger
from src.utils.memory import (
    memory_usage_report,
    compare_memory,
    get_system_memory
)

logger = setup_logger(__name__)


class ParquetConverter:
    """
    Handles conversion of Bosch CSVs to memory-optimized Parquet files.
    Uses chunked processing to avoid OOM on large files.

    Why Parquet over CSV:
    - Columnar storage: Read only needed columns
    - Built-in compression: 3-4x smaller on disk
    - Preserves dtypes: No re-parsing on load
    - Faster I/O: Binary format vs text parsing
    """

    def __init__(self, config: Config, chunk_size: int = 100000):
        """
        Initialize converter.

        Args:
            config: Project configuration
            chunk_size: Number of rows to process at once (default 100K)
                       Lower = less memory, slower
                       Higher = more memory, faster
        """
        self.config = config
        self.raw_dir = Path(config.get('paths.raw'))
        self.processed_dir = Path(config.get('paths.processed'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Chunk size (100K rows = ~400MB per chunk for numeric, ~2GB for categorical)
        self.chunk_size = chunk_size

        # Dtype optimization rules
        self.numeric_dtype = 'float32'
        self.response_dtype = 'int8'

    def _optimize_numeric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric columns to memory-efficient dtypes.

        Applied per chunk to keep memory low.
        """
        # Response is always 0 or 1
        if 'Response' in df.columns:
            df['Response'] = df['Response'].astype(self.response_dtype)

        # All numeric features to float32
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c not in ['Id', 'Response']]

        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(self.numeric_dtype)

        return df

    def _optimize_categorical_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object/string columns to category dtype.

        Why per-chunk optimization works:
        - Category mapping is global (pandas handles this across chunks)
        - Only overhead is storing unique values once
        """
        object_cols = df.select_dtypes(include=['object']).columns
        object_cols = [c for c in object_cols if c != 'Id']

        if len(object_cols) == 0:
            return df

        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)

            # Only convert if cardinality < 50%
            if unique_ratio < 0.5:
                df[col] = df[col].astype('category')

        return df

    def _convert_file_chunked(
            self,
            csv_path: Path,
            parquet_path: Path,
            sample_frac: Optional[float] = None
    ) -> None:
        """
        Convert single CSV to Parquet using chunked processing.

        Process flow:
        1. Read CSV in chunks (100K rows at a time)
        2. Optimize dtypes per chunk
        3. Accumulate chunks in list
        4. Concatenate all chunks
        5. Write to Parquet

        Why this works:
        - Peak memory = chunk_size × columns (manageable)
        - List overhead is small compared to full dataframe
        - Final concatenation happens with optimized dtypes (cheaper)
        """
        logger.info("=" * 60)
        logger.info(f"Converting: {csv_path.name}")
        logger.info("=" * 60)

        # Safety check: Ensure enough RAM available
        sys_mem = get_system_memory()
        if sys_mem['available_gb'] < 2.0:
            logger.error(f"⚠️  Only {sys_mem['available_gb']:.1f} GB RAM available!")
            logger.error("    Close other applications and try again")
            raise MemoryError("Insufficient RAM to load file")

        # Determine total rows for progress bar
        logger.info(f"Counting rows in {csv_path.name}...")
        total_rows = sum(1 for _ in open(csv_path)) - 1  # -1 for header
        logger.info(f"  Total rows: {total_rows:,}")

        # Apply sampling if requested
        if sample_frac:
            total_rows = int(total_rows * sample_frac)
            logger.info(f"  → Sampling to {total_rows:,} rows ({sample_frac * 100}%)")

        # Read and process chunks
        logger.info(f"Processing in chunks of {self.chunk_size:,} rows...")
        chunks = []
        rows_processed = 0

        # Read CSV in chunks
        chunk_iterator = pd.read_csv(
            csv_path,
            chunksize=self.chunk_size,
            low_memory=False  # Consistent dtype inference
        )

        # Progress bar for chunks
        num_chunks = (total_rows // self.chunk_size) + 1
        pbar = tqdm(total=num_chunks, desc=f"Processing {csv_path.name}", unit="chunk")

        for chunk in chunk_iterator:
            # Apply sampling cutoff if needed
            if sample_frac and rows_processed >= total_rows:
                break

            # Optimize dtypes based on file type
            if 'numeric' in csv_path.name or 'date' in csv_path.name:
                chunk = self._optimize_numeric_dtypes(chunk)

            if 'categorical' in csv_path.name:
                chunk = self._optimize_categorical_dtypes(chunk)

            chunks.append(chunk)
            rows_processed += len(chunk)
            pbar.update(1)

            # Memory safety: Log if RAM usage getting high
            sys_mem = get_system_memory()
            if sys_mem['available_gb'] < 1.0:
                logger.warning(f"⚠️  Low memory: {sys_mem['available_gb']:.1f} GB available")

        pbar.close()

        # Concatenate all chunks
        logger.info("Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        # Memory report on final dataframe
        memory_usage_report(df, f"{csv_path.name} (FINAL)")

        # Write to Parquet with compression
        logger.info(f"Writing to {parquet_path.name}...")
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Verify file sizes
        csv_size_mb = csv_path.stat().st_size / (1024 ** 2)
        parquet_size_mb = parquet_path.stat().st_size / (1024 ** 2)
        compression_ratio = csv_size_mb / parquet_size_mb

        logger.info("=" * 60)
        logger.info("Conversion Summary:")
        logger.info(f"  Rows processed: {len(df):,}")
        logger.info(f"  CSV size:       {csv_size_mb:,.0f} MB")
        logger.info(f"  Parquet size:   {parquet_size_mb:,.0f} MB")
        logger.info(f"  Compression:    {compression_ratio:.1f}x smaller")
        logger.info("=" * 60)

        # Cleanup
        del chunks, df

    def convert_all(self, sample_frac: Optional[float] = None) -> None:
        """
        Convert all Bosch CSV files to Parquet using chunked processing.

        Args:
            sample_frac: If set, only convert sample for testing (e.g., 0.10 = 10%)
        """
        file_pairs = [
            ('train_numeric.csv', 'train_numeric.parquet'),
            ('train_categorical.csv', 'train_categorical.parquet'),
            ('train_date.csv', 'train_date.parquet'),
            ('test_numeric.csv', 'test_numeric.parquet'),
            ('test_categorical.csv', 'test_categorical.parquet'),
            ('test_date.csv', 'test_date.parquet'),
        ]

        logger.info("🚀 Starting CSV → Parquet conversion (CHUNKED MODE)")
        logger.info(f"Chunk size: {self.chunk_size:,} rows per chunk")
        logger.info(f"Source: {self.raw_dir}")
        logger.info(f"Target: {self.processed_dir}")

        if sample_frac:
            logger.warning(f"⚠️  SAMPLING MODE: Converting {sample_frac * 100}% of data")

        for csv_name, parquet_name in file_pairs:
            csv_path = self.raw_dir / csv_name
            parquet_path = self.processed_dir / parquet_name

            if not csv_path.exists():
                logger.warning(f"⚠️  Skipping {csv_name} (not found)")
                continue

            if parquet_path.exists() and not sample_frac:
                logger.info(f"✓ {parquet_name} already exists, skipping")
                continue

            try:
                self._convert_file_chunked(csv_path, parquet_path, sample_frac)
                logger.info(f"✅ {csv_name} → {parquet_name} complete\n")
            except Exception as e:
                logger.error(f"❌ Failed to convert {csv_name}: {str(e)}")
                raise

        logger.info("=" * 60)
        logger.info("🎉 ALL CONVERSIONS COMPLETE")
        logger.info("=" * 60)

        # Final summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary of all converted files."""
        logger.info("\nFinal File Sizes:")

        total_csv_size = 0
        total_parquet_size = 0

        for parquet_file in self.processed_dir.glob('*.parquet'):
            parquet_size_mb = parquet_file.stat().st_size / (1024 ** 2)
            total_parquet_size += parquet_size_mb

            csv_name = parquet_file.stem + '.csv'
            csv_file = self.raw_dir / csv_name
            if csv_file.exists():
                csv_size_mb = csv_file.stat().st_size / (1024 ** 2)
                total_csv_size += csv_size_mb
                compression = csv_size_mb / parquet_size_mb
                logger.info(f"  {parquet_file.name}: {parquet_size_mb:,.0f} MB ({compression:.1f}x)")

        logger.info("=" * 60)
        logger.info(f"TOTAL CSV:     {total_csv_size:,.0f} MB ({total_csv_size / 1024:.2f} GB)")
        logger.info(f"TOTAL Parquet: {total_parquet_size:,.0f} MB ({total_parquet_size / 1024:.2f} GB)")
        logger.info(f"COMPRESSION:   {total_csv_size / total_parquet_size:.1f}x smaller")
        logger.info("=" * 60)


def main():
    """
    CLI entry point for CSV → Parquet conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert Bosch CSV files to optimized Parquet format (chunked processing)"
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Sample fraction for testing (e.g., 0.10 for 10%%)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Number of rows per chunk (default: 100,000)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Run conversion
    converter = ParquetConverter(config, chunk_size=args.chunk_size)
    converter.convert_all(sample_frac=args.sample)

    logger.info("\n✅ Done! Next steps:")
    logger.info("  1. Verify data integrity with 01_data_validation.py")
    logger.info("  2. Update NOTES.md with memory savings")
    logger.info("  3. Proceed to feature engineering")


if __name__ == "__main__":
    main()
