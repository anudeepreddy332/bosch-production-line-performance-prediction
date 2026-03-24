"""
Phase 4 Script 18: Merge all features for Phase 4

Sources:
    - train_phase3_features.parquet -> 332 features (current baseline)
    - train_date_features.parquet -> 79 features (real timestamps, script 17)
    - train_time_features.parquet -> 8 clean cols (record windows + id_mod + shift

Drops from time features:
    - rolling_mean_target_* -> globally leaky (labels used)
    - distance_to_next_* -> future leakage
    - distance_to_last_* -> globally leaky
    - day_of_week -> replaced by real timestamp version
    - hour_of_day -> replaced by real timestamp version

Output: data/features/train_phase4_features.parquet
Expected: ~ 360 features
"""
import gc
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

# Need 8 clean cols from train_time_features
CLEAN_TIME_COLS = [
    'Id',
    'records_last_25',
    'records_next_25',
    'records_last_240',
    'records_next_240',
    'records_last_1680',
    'records_next_1680',
    'id_mod_1680',
    'shift',
]

def main():
    logger.info("=" * 60)
    logger.info("PHASE 4 — SCRIPT 18: MERGE FEATURES")
    logger.info("=" * 60)

    # 1. Load phase 3 baseline
    logger.info("\n1. Loading train_phase3_features.parquet...")
    df = pd.read_parquet(project_root / "data/features/train_phase3_features.parquet")
    logger.info(f"   Shape: {df.shape}")
    memory_usage_report(df, "Phase 3 Features")

    # 2. Load date features
    logger.info("\n2. Loading train_date_features.parquet...")
    df_date = pd.read_parquet(project_root / "data/features/train_date_features.parquet")
    logger.info(f"   Shape: {df_date.shape}")

    # 3. Load 8 clean cols from time features
    logger.info("\n3. Loading clean cols from train_time_features.parquet...")
    df_time = pd.read_parquet(project_root / "data/features/train_time_features.parquet",
                              columns=CLEAN_TIME_COLS)
    logger.info(f"   Shape: {df_time.shape}")
    logger.info(f"   Cols: {[c for c in df_time.columns if c != 'Id']}")

    # 4. Merge date features
    logger.info("\n4. Merging date features...")
    before = df.shape[1]
    df = df.merge(df_date, on='Id', how='left')
    logger.info(f"   Cols added: {df.shape[1] - before}")
    logger.info(f"   Shape after date merge: {df.shape}")

    del df_date
    gc.collect()

    # 5. Merge clean time cols
    logger.info("\n5. Merging clean time cols...")
    before = df.shape[1]
    df = df.merge(df_time, on='Id', how='left')
    logger.info(f"   Cols added: {df.shape[1] - before}")
    logger.info(f"   Shape after time merge: {df.shape}")

    del df_time
    gc.collect()

    # 6. Sanity checks
    logger.info("\n6. Sanity checks...")

    # Duplicate columns
    dupes = [c for c in df.columns if df.columns.tolist().count(c) > 1]
    if dupes:
        logger.warning(f"   ⚠️  Duplicate columns found: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"   Deduplicated → {df.shape}")
    else:
        logger.info("   ✅ No duplicate columns")

    # Null check
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    logger.info(f"   Cols with nulls: {len(null_cols)}")
    if len(null_cols) > 0:
        logger.info(f"   Filling {len(null_cols)} cols with median...")
        for col in null_cols.index:
            if col == 'Id':
                continue
            df[col] = df[col].fillna(df[col].median())
        logger.info(f"   Nulls after fill: {df.isnull().sum().sum()}")

    # Row count must be unchanged
    assert df.shape[0] == 1183747, f"Row count changed! Got {df.shape[0]}"
    logger.info(f"   ✅ Row count intact: {df.shape[0]:,}")

    # 7. Summary
    feature_cols = [c for c in df.columns if c not in ['Id', 'Response']]
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 FEATURES SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Phase 3 baseline:     332 features")
    logger.info(f"  Date features added:   79 features")
    logger.info(f"  Clean time cols added:  8 features")
    logger.info(f"  Total features:       {len(feature_cols)}")
    logger.info(f"  Total cols (w/ Id+Response): {df.shape[1]}")

    memory_usage_report(df, "Phase 4 Features")

    # 8. Save
    output_path = project_root / "data/features/train_phase4_features.parquet"
    df.to_parquet(output_path, compression='snappy', index=False)
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"\n✅ Saved → {output_path} ({file_size:.2f} MB)")
    logger.info(f"\n🎯 NEXT: python notebooks/19_retrain_phase4.py")


if __name__ == '__main__':
    main()

