"""
Merge leak features with existing feature set.

Current features: 250 (150 numeric + 100 categorical)
+ Leak features: 23
= Total: 273 features

Expected MCC: 0.35-0.40
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

# Columns that must never enter training — future information only available
# at label-creation time, not at inference time on unseen parts
FUTURE_LEAK_COLS = ['next_response_in_chunk']

def main():
    logger.info("=" * 60)
    logger.info("MERGE LEAK FEATURES WITH EXISTING FEATURES")
    logger.info("=" * 60)

    # 1. Load existing features (150 numeric + 100 categorical)
    logger.info("\n1. Loading existing features...")
    df_numeric = pd.read_parquet(project_root / "data/features/train_selected_top150.parquet")
    logger.info(f"  Numeric shape: {df_numeric.shape}")

    df_cat = pd.read_parquet(project_root / "data/features/train_categorical_top100.parquet")
    logger.info(f"  Categorical shape: {df_cat.shape}")

    # Merge numeric + categorical
    df_existing = df_numeric.merge(df_cat, on='Id', how='left')
    logger.info(f"  Combined Shape: {df_existing.shape}")
    memory_usage_report(df_existing, "Existing Features")

    # 2. Load leak features
    logger.info("\n2. Loading leak features...")
    df_leak = pd.read_parquet(project_root / "data/features/train_leak_features.parquet")
    logger.info(f"  Leak shape: {df_leak.shape}")
    memory_usage_report(df_leak, "Leak Features")

    # Drop future-leak columns before merge — these use information
    # that wouldn't be available when scoring unseen test parts
    cols_to_drop = [c for c in FUTURE_LEAK_COLS if c in df_leak.columns]
    if cols_to_drop:
        df_leak = df_leak.drop(columns=cols_to_drop)
        logger.info(f"  Dropped future-leak cols: {cols_to_drop}")
    logger.info(f"  Leak shape (clean): {df_leak.shape}")
    memory_usage_report(df_leak, "Leak Features")

    # 3. Merge
    logger.info("\n" + "=" * 60)
    logger.info("MERGING")
    logger.info("=" * 60)

    df_all = df_existing.merge(df_leak, on='Id', how='left')

    # Sanity checks
    assert df_all.shape[0] == df_existing.shape[0], "Row count changed after merge — check for duplicate Ids in leak file"
    n_features = df_all.shape[1] - 2  # exclude Id + Response

    logger.info(f"\n  Numeric features:     150")
    logger.info(f"  Categorical features: 100")
    logger.info(f"  Leak features:        {df_leak.shape[1] - 1}")
    logger.info(f"  Total features:       {n_features}")
    logger.info(f"  Total shape:          {df_all.shape}")
    memory_usage_report(df_all, "All Features")

    # Null check - flag any new nulls introduced by the merge
    leak_cols = [c for c in df_leak.columns if c!= 'Id']
    null_counts = df_all[leak_cols].isnull().sum()
    high_null = null_counts[null_counts > 0]
    if len(high_null) > 0:
        logger.warning(f"  ⚠️  Null values found in leak columns:")
        logger.warning(f"\n{high_null.to_string()}")
    else:
        logger.info(f"  ✅ No nulls in leak columns")


    # 4. Save
    output_path = project_root / "data/features/train_all_features_with_leaks.parquet"
    logger.info(f"\nSaving to {output_path}...")
    df_all.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! File size: {file_size:.2f} MB")

    # 5. Summary
    logger.info("\n" + "=" * 60)
    logger.info("READY FOR TRAINING")
    logger.info("=" * 60)
    logger.info(f"\n✅ Total features: {df_all.shape[1] - 2}")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"\n🎯 NEXT STEP: Run notebooks/08_retrain_with_leaks.py")


if __name__ == "__main__":
    main()
