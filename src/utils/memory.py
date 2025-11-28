"""
Memory profiling utilities for tracking DataFrame memory usage.

Why this exists:
- MacBook has 16GB RAM, dataset needs 60GB without optimization
- Need to measure impact of each optimization (dtype changes, parquet conversion)
- Provides evidence for technical decisions in case study

"Production ML requires resource awareness. This gives us
metrics to justify our optimization choices."
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import psutil
import os
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_system_memory() -> Dict[str, float]:
    """
    Get current system memory usage.

    Returns:
        Dict with total, available, used memory in GB, and usage percentage

    Why: Need to know if we're close to OOM (Out Of Memory) crashes
    """
    memory = psutil.virtual_memory()

    return {
        'total_gb': memory.total / (1024 ** 3),
        'available_gb': memory.available / (1024 ** 3),
        'used_gb': memory.used / (1024 ** 3),
        'percent_used': memory.percent
    }


def get_dataframe_memory(df: pd.DataFrame, deep: bool = True) -> Dict[str, float]:
    """
    Calculate detailed memory usage of a DataFrame.

    Args:
        df: DataFrame to analyze
        deep: If True, introspect object dtypes (slower but accurate for strings)

    Returns:
        Dict with total memory, per-column breakdown, and dtype summary

    Why deep=True:
    - object/string columns hide their true size without deep introspection
    - Worth the 2-3 second cost for accurate measurements

    "Without deep=True, pandas underestimates string column
    memory by 5-10x. We need real numbers to make optimization decisions."
    """
    memory_usage = df.memory_usage(deep=deep)
    total_mb = memory_usage.sum() / (1024 ** 2)

    # Per-column breakdown (top 10 memory consumers)
    column_memory = memory_usage.sort_values(ascending=False).head(10)
    column_memory_mb = {col: mem / (1024 ** 2) for col, mem in column_memory.items()}

    # Dtype summary (how much memory each dtype uses in total)
    dtype_groups = df.dtypes.value_counts()
    dtype_memory = {}
    for dtype in dtype_groups.index:
        cols_of_dtype = df.select_dtypes(include=[dtype]).columns
        dtype_memory[str(dtype)] = df[cols_of_dtype].memory_usage(deep=deep).sum() / (1024 ** 2)

    return {
        'total_mb': total_mb,
        'total_gb': total_mb / 1024,
        'column_memory_mb': column_memory_mb,
        'dtype_memory_mb': dtype_memory
    }


def compare_memory(
        original_df: pd.DataFrame,
        optimized_df: pd.DataFrame,
        label: str = "Optimization"
) -> None:
    """
    Compare memory usage before and after optimization.

    Args:
        original_df: DataFrame before optimization
        optimized_df: DataFrame after optimization
        label: Description of what optimization was applied

    Why: Provides clear metrics for case study and interview discussions

    """
    original_mem = get_dataframe_memory(original_df)
    optimized_mem = get_dataframe_memory(optimized_df)

    savings_mb = original_mem['total_mb'] - optimized_mem['total_mb']
    savings_pct = (savings_mb / original_mem['total_mb']) * 100

    logger.info("=" * 60)
    logger.info(f"{label} - Memory Comparison")
    logger.info("=" * 60)
    logger.info(f"BEFORE: {original_mem['total_mb']:.2f} MB ({original_mem['total_gb']:.2f} GB)")
    logger.info(f"AFTER:  {optimized_mem['total_mb']:.2f} MB ({optimized_mem['total_gb']:.2f} GB)")
    logger.info(f"SAVINGS: {savings_mb:.2f} MB ({savings_pct:.1f}% reduction)")
    logger.info("=" * 60)

    # Dtype breakdown
    logger.info("\nDtype Memory Usage (BEFORE):")
    for dtype, mem in original_mem['dtype_memory_mb'].items():
        logger.info(f"  {dtype}: {mem:.2f} MB")

    logger.info("\nDtype Memory Usage (AFTER):")
    for dtype, mem in optimized_mem['dtype_memory_mb'].items():
        logger.info(f"  {dtype}: {mem:.2f} MB")


def memory_usage_report(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive memory usage report for a DataFrame.

    Args:
        df: DataFrame to analyze
        name: Name to display in report

    Why: One function to get all memory insights - used throughout project
    """
    mem_info = get_dataframe_memory(df)
    sys_info = get_system_memory()

    logger.info("=" * 60)
    logger.info(f"Memory Report: {name}")
    logger.info("=" * 60)
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    logger.info(f"Memory: {mem_info['total_mb']:.2f} MB ({mem_info['total_gb']:.2f} GB)")
    logger.info(
        f"System: {sys_info['used_gb']:.1f} GB / {sys_info['total_gb']:.1f} GB ({sys_info['percent_used']:.1f}% used)")
    logger.info("=" * 60)

    logger.info("\nTop 10 Memory-Heavy Columns:")
    for col, mem_mb in mem_info['column_memory_mb'].items():
        pct = (mem_mb / mem_info['total_mb']) * 100
        logger.info(f"  {col}: {mem_mb:.2f} MB ({pct:.1f}%)")

    logger.info("\nMemory by Dtype:")
    for dtype, mem_mb in mem_info['dtype_memory_mb'].items():
        pct = (mem_mb / mem_info['total_mb']) * 100
        logger.info(f"  {dtype}: {mem_mb:.2f} MB ({pct:.1f}%)")


def estimate_full_dataset_memory(sample_df: pd.DataFrame, total_rows: int) -> float:
    """
    Estimate memory for full dataset based on sample.

    Args:
        sample_df: Sample DataFrame (e.g., first 10K rows)
        total_rows: Total number of rows in full dataset

    Returns:
        Estimated memory in GB for full dataset

    Why: Helps us decide if optimizations are sufficient before loading full data

    "Before spending 20 minutes loading 1.2M rows, we test on
    10K rows and extrapolate. If estimate shows 40GB needed and we have 16GB,
    we know we need more optimization first."
    """
    sample_rows = len(sample_df)
    sample_mem_gb = get_dataframe_memory(sample_df)['total_gb']

    estimated_full_gb = sample_mem_gb * (total_rows / sample_rows)

    logger.info(f"Sample: {sample_rows:,} rows = {sample_mem_gb:.2f} GB")
    logger.info(f"Estimated full dataset ({total_rows:,} rows): {estimated_full_gb:.2f} GB")

    sys_mem = get_system_memory()
    if estimated_full_gb > sys_mem['available_gb']:
        logger.warning(f"⚠️  Estimated {estimated_full_gb:.1f} GB exceeds available {sys_mem['available_gb']:.1f} GB!")
        logger.warning("    → Need more optimization or will crash")
    else:
        logger.info(f"✓ Fits in available memory ({sys_mem['available_gb']:.1f} GB)")

    return estimated_full_gb


if __name__ == "__main__":
    # Self-test: Create sample DataFrame and profile it
    logger.info("Memory utility self-test")

    test_df = pd.DataFrame({
        'int64_col': np.random.randint(0, 100, 10000),
        'float64_col': np.random.randn(10000),
        'object_col': ['string_' + str(i) for i in range(10000)]
    })

    memory_usage_report(test_df, "Test DataFrame")

    # Test optimization
    optimized_df = test_df.copy()
    optimized_df['int64_col'] = optimized_df['int64_col'].astype('int8')
    optimized_df['float64_col'] = optimized_df['float64_col'].astype('float32')
    optimized_df['object_col'] = optimized_df['object_col'].astype('category')

    compare_memory(test_df, optimized_df, "Dtype Optimization Test")
