"""
Script 21: Merge Phase 5 Feature Matrix

Adds 7 temporal features from script 20 to Phase 4 feature matrix.

Input:  train_phase4_features.parquet  (1,183,747 × 421)
        train_temporal_features.parquet (1,183,747 × 8)
Output: train_phase5_features.parquet  (1,183,747 × 428)
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("SCRIPT 21: MERGE PHASE 5 FEATURES")
    logger.info("=" * 60)

    # 1. Load Phase 4 base
    logger.info("\n1. Loading Phase 4 feature matrix...")
    df = pd.read_parquet(project_root / "data/features/train_phase4_features.parquet")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Failures: {df['Response'].sum():,}")

    # 2. Load temporal features
    logger.info("\n2. Loading temporal features...")
    temporal = pd.read_parquet(project_root / "data/features/train_temporal_features.parquet")
    logger.info(f"  Shape: {temporal.shape}")
    temporal_cols = [c for c in temporal.columns if c!= 'Id']
    logger.info(f"  New features: {temporal_cols}")

    # 3. Merge
    logger.info("\n3. Merging...")
    before_cols = df.shape[1]
    df = df.merge(temporal, on='Id', how='left')
    after_cols = df.shape[1]
    logger.info(f"  Columns: {before_cols} → {after_cols} (+{after_cols - before_cols})")
    logger.info(f"  Shape after merge: {df.shape}")

    # 4. Null check
    null_counts = df[temporal_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"  Nulls introduced by merge:")
        for col, n in null_counts[null_counts > 0].items():
            logger.warning(f"    {col}: {n:,}")
        # Fill with median
        for col in temporal_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        logger.info(f"  Nulls after fill: {df[temporal_cols].isnull().sum().sum()}")
    else:
        logger.info("  ✅ No nulls after merge")

    # 5. Sanity check
    assert df.shape[0] == 1_183_747, f"Row count mismatch: {df.shape[0]}"
    assert df['Response'].sum() == 6879, f"Failure count mismatch"
    logger.info("  ✅ Row count and failure count intact")

    # 6. Memory report
    memory_usage_report(df, "Phase 5 Feature Matrix")

    # 7. Save
    output_path = project_root / "data/features/train_phase5_features.parquet"
    logger.info(f"\n4. Saving → {output_path}")
    df.to_parquet(output_path, compression='snappy', index=False)
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"  ✅ Saved ({file_size:.2f} MB)")
    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"\n  Feature breakdown:")
    logger.info(f"    Phase 4 base features:    418")
    logger.info(f"    + Temporal (script 20):     7")
    logger.info(f"    = Phase 5 total:          425")
    logger.info(f"\n🎯 NEXT: python notebooks/22_retrain_phase5.py")

if __name__ == '__main__':
    main()
