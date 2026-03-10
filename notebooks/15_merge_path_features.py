"""
Merge path features (phase 3) into the main feature matrix.

Input:
    - data/features/train_all_features_with_leaks.parquet (273 features)
    - data/features/train_path_features.parquet (58 features)

Output:
    - data/features/train_phase3_features.parquet (273 + 58 = 331 features)
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
    logger.info("PHASE 3: MERGE PATH FEATURES")
    logger.info("=" * 60)

    # Load base feature matrix
    # This is the 273-column parquet from phase 2:
    # 150 numeric + 100 target-encoded categorical + chunk/leak features
    logger.info("\n1. Loading base feature matrix (Phase 2)...")
    df_base = pd.read_parquet(
        project_root / "data/features/train_all_features_with_leaks.parquet"
    )
    logger.info(f"   Shape: {df_base.shape}")

    # Load path features
    logger.info("\n2. Loading path features (Phase 3)...")
    df_path = pd.read_parquet(
        project_root / "data/features/train_path_features.parquet"
    )
    logger.info(f"   Shape: {df_path.shape}")
    logger.info(f"   Columns: {list(df_path.columns[:10])}...")

    # Merge on Id
    logger.info("\n3. Merging on Id...")
    df_merged = df_base.merge(df_path, on='Id', how='left')
    logger.info(f"   Merged shape: {df_merged.shape}")

    # sanity check: row count must not change after merge. If it does, there are duplicate Ids.
    assert len(df_merged) == len(df_base), \
        f"❌ Row count changed after merge: {len(df_base)} → {len(df_merged)}"
    logger.info("   ✅ Row count intact after merge")

    # Check for any nulls introduced by the merge in path cols
    path_cols = [c for c in df_path.columns if c!= 'Id']
    null_counts = df_merged[path_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        logger.warning(f"   ⚠️  Nulls in path columns after merge:")
        for col, cnt in cols_with_nulls.items():
            logger.warning(f"      {col}: {cnt:,} nulls")
    else:
        logger.info("   ✅ No nulls introduced by merge")

    # Feature breakdown
    logger.info("\n4. Feature breakdown...")
    all_cols = [c for c in df_merged.columns if c not in
                ['Id', 'Response', 'next_response_in_chunk']]

    visited_cols = [c for c in all_cols if c.startswith('visited_')]
    nstation_cols = [c for c in all_cols if c.startswith('n_stations_')]
    path_stat_cols = [c for c in all_cols if c.startswith('path_')]
    other_cols = [c for c in all_cols if c not in
                  visited_cols + nstation_cols + path_stat_cols]

    logger.info(f"   Phase 2 features (numeric/categorical/chunk): {len(other_cols)}")
    logger.info(f"   Station presence flags:                        {len(visited_cols)}")
    logger.info(f"   Line station counts:                           {len(nstation_cols)}")
    logger.info(f"   Path statistics:                               {len(path_stat_cols)}")
    logger.info(f"   ─────────────────────────────────────────────────")
    logger.info(f"   TOTAL features:                                {len(all_cols)}")

    memory_usage_report(df_merged, "Phase 3 Feature Matrix")

    # Save
    output_path = project_root / "data/features/train_phase3_features.parquet"
    df_merged.to_parquet(output_path, compression='snappy', index=False)
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"\n✅ Saved → {output_path} ({file_size:.2f} MB)")
    logger.info(f"\n🎯 NEXT: python notebooks/16_retrain_phase3.py")

if __name__ == '__main__':
    main()










