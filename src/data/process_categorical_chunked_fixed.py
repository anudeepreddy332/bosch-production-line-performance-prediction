"""
Fixed chunked categorical processing.

Strategy:
1. Read in chunks (memory-safe)
2. Concat chunks (might have mixed dtypes)
3. Force convert ALL to category AFTER concat
4. Save parquet
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


def process_categorical_chunked_fixed(
        input_path: str = "data/raw/train_categorical.csv",
        output_path: str = "data/processed/train_categorical_fixed.parquet",
        chunksize: int = 100000
):
    """
    Chunked processing with post-concat category conversion.
    """
    logger.info("=" * 60)
    logger.info("CATEGORICAL PROCESSING (CHUNKED + POST-FIX)")
    logger.info("=" * 60)

    # 1. Read and concat chunks
    logger.info(f"\nReading {input_path} in chunks...")

    chunks = []
    chunk_iter = pd.read_csv(input_path, chunksize=chunksize)

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Loading chunks")):
        chunks.append(chunk)

    logger.info(f"✅ Loaded {len(chunks)} chunks")

    # 2. Concat
    logger.info("\nConcatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory

    logger.info(f"  Shape: {df.shape}")
    memory_usage_report(df, "After concat (before conversion)")

    # 3. FORCE convert ALL to category (this is the fix!)
    logger.info("\nConverting ALL columns to category...")
    cat_cols = [c for c in df.columns if c != 'Id']

    logger.info(f"Converting {len(cat_cols)} columns...")
    for i, col in enumerate(tqdm(cat_cols, desc="Converting")):
        # Convert any dtype to category
        df[col] = df[col].astype('category')

    logger.info(f"✅ Converted {len(cat_cols)} columns")
    memory_usage_report(df, "After category conversion")

    # 4. Verify dtypes
    logger.info("\nVerifying dtypes...")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")

    # Check for any non-category columns (except Id)
    non_cat = df.select_dtypes(exclude=['category', 'int64']).columns
    if len(non_cat) > 0:
        logger.warning(f"⚠️ Found {len(non_cat)} non-category columns:")
        for col in non_cat[:10]:
            logger.warning(f"    {col}: {df[col].dtype}")
    else:
        logger.info("✅ All columns are category dtype!")

    # 5. Save
    logger.info(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! File size: {file_size:.2f} MB")

    logger.info("\n" + "=" * 60)
    logger.info("✅ COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    process_categorical_chunked_fixed()
