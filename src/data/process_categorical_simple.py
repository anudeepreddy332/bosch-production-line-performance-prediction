"""
Simple single-pass categorical processing.

Why this works:
- No chunking = no concat issues
- Direct conversion to category dtype
- 14GB RAM is fine
"""

import pandas as pd
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


def process_categorical_simple(
        input_path: str = "data/raw/train_categorical.csv",
        output_path: str = "data/processed/train_categorical_fixed.parquet"
):
    """Single-pass categorical processing."""

    logger.info("=" * 60)
    logger.info("CATEGORICAL PROCESSING (SIMPLE APPROACH)")
    logger.info("=" * 60)

    # Load entire file
    logger.info(f"\nLoading {input_path}...")
    logger.info("(This takes ~60 seconds)")
    df = pd.read_csv(input_path)

    logger.info(f"  Shape: {df.shape}")
    memory_usage_report(df, "Original (object dtype)")

    # Convert all to category (except Id)
    logger.info("\nConverting to category dtype...")
    cat_cols = [c for c in df.columns if c != 'Id']

    for col in cat_cols:
        df[col] = df[col].astype('category')

    logger.info(f"✅ Converted {len(cat_cols)} columns to category")
    memory_usage_report(df, "After category conversion")

    # Save
    logger.info(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! File size: {file_size:.2f} MB")

    # Verify dtypes
    logger.info("\nVerifying dtypes...")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")

    logger.info("\n" + "=" * 60)
    logger.info("✅ COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    from pathlib import Path

    process_categorical_simple()
