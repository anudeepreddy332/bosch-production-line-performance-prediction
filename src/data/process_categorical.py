"""
Process categorical data with proper global uniqueness check.

Problem from Day 3:
- We checked uniqueness PER CHUNK (100K rows)
- A column with 100 global unique values might have 60 unique in one chunk
- Result: Only 12% converted to category dtype

Solution:
- Two-pass approach:
  1. First pass: Scan all chunks, collect global unique counts
  2. Second pass: Convert appropriate columns based on global stats

Why this matters:
- Categorical features are CRITICAL (Kaggle winners used them)
- Proper category dtype: 10x memory savings
- Enables LightGBM categorical feature support
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Set
from tqdm import tqdm
from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


def process_categorical_two_pass(
        input_path: str = "data/raw/train_categorical.csv",
        output_path: str = "data/processed/train_categorical_fixed.parquet",
        chunksize: int = 100000,
        max_unique_ratio: float = 0.5
):
    """
    Process categorical data with two-pass global uniqueness check.

    Args:
        input_path: Raw CSV path
        output_path: Output parquet path
        chunksize: Rows per chunk
        max_unique_ratio: Convert to category if unique_values/total_rows < this
    """
    logger.info("=" * 60)
    logger.info("CATEGORICAL DATA PROCESSING (TWO-PASS)")
    logger.info("=" * 60)

    # Count total rows
    logger.info(f"\nCounting rows in {input_path}...")
    total_rows = sum(1 for _ in open(input_path)) - 1  # -1 for header
    logger.info(f"  Total rows: {total_rows:,}")

    # PASS 1: Collect global unique values
    logger.info("\n" + "=" * 60)
    logger.info("PASS 1: COLLECTING GLOBAL UNIQUE VALUES")
    logger.info("=" * 60)

    unique_values = {}  # {column: set of unique values}

    logger.info(f"Scanning {total_rows:,} rows in chunks of {chunksize:,}...")

    chunk_iter = pd.read_csv(input_path, chunksize=chunksize, dtype=str)
    for i, chunk in enumerate(tqdm(chunk_iter, desc="Pass 1")):
        # Skip Id column
        cat_cols = [c for c in chunk.columns if c != 'Id']

        for col in cat_cols:
            if col not in unique_values:
                unique_values[col] = set()

            # Add unique values from this chunk (drop nulls)
            unique_values[col].update(chunk[col].dropna().unique())

    logger.info(f"\n✅ Pass 1 complete!")

    # Determine which columns to convert
    logger.info("\nAnalyzing global uniqueness...")
    convert_to_category = []

    for col, uniques in unique_values.items():
        n_unique = len(uniques)
        unique_ratio = n_unique / total_rows

        if unique_ratio < max_unique_ratio:
            convert_to_category.append(col)

    logger.info(f"\nGlobal uniqueness analysis:")
    logger.info(f"  Total columns: {len(unique_values)}")
    logger.info(f"  Columns to convert to category: {len(convert_to_category)}")
    logger.info(f"  Columns to keep as object: {len(unique_values) - len(convert_to_category)}")
    logger.info(f"  Conversion rate: {len(convert_to_category) / len(unique_values) * 100:.1f}%")

    # Log examples
    logger.info("\nTop 10 lowest-cardinality columns (best for category):")
    sorted_cols = sorted(unique_values.items(), key=lambda x: len(x[1]))
    for col, uniques in sorted_cols[:10]:
        ratio = len(uniques) / total_rows
        logger.info(f"  {col}: {len(uniques)} unique ({ratio * 100:.3f}%)")

    logger.info("\nTop 10 highest-cardinality columns (keep as object):")
    for col, uniques in sorted_cols[-10:]:
        ratio = len(uniques) / total_rows
        logger.info(f"  {col}: {len(uniques)} unique ({ratio * 100:.3f}%)")

    # PASS 2: Read and convert
    logger.info("\n" + "=" * 60)
    logger.info("PASS 2: CONVERTING AND SAVING")
    logger.info("=" * 60)

    chunks_processed = []

    logger.info(f"Processing chunks with dtype optimization...")
    chunk_iter = pd.read_csv(input_path, chunksize=chunksize)

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Pass 2")):
        # Convert appropriate columns to category
        for col in convert_to_category:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype('category')

        chunks_processed.append(chunk)

    # Concatenate all chunks
    logger.info("\nConcatenating chunks...")
    df_final = pd.concat(chunks_processed, ignore_index=True)

    logger.info(f"  Final shape: {df_final.shape}")

    # Memory report
    logger.info("\nMemory usage:")
    memory_mb = df_final.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"  Total: {memory_mb:.2f} MB ({memory_mb / 1024:.2f} GB)")

    # Dtype breakdown
    logger.info("\nColumn dtypes:")
    dtype_counts = df_final.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")

    # Save
    logger.info(f"\nSaving to {output_path}...")
    df_final.to_parquet(output_path, compression='snappy', index=False)

    # Verify
    file_size_mb = Path(output_path).stat().st_size / 1024 ** 2
    compression_ratio = (memory_mb / file_size_mb) if file_size_mb > 0 else 0

    logger.info(f"✅ Saved successfully!")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}x")

    return df_final


if __name__ == "__main__":
    config = Config("config/config.yaml")

    # Process train categorical
    logger.info("Processing train_categorical.csv...")
    process_categorical_two_pass(
        input_path="data/raw/train_categorical.csv",
        output_path="data/processed/train_categorical_fixed.parquet"
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ CATEGORICAL PROCESSING COMPLETE")
    logger.info("=" * 60)
