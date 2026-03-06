"""
Merge time features with all existing features.

Current: 274 features (150 num + 100 cat + 24 leak)
+ Time: ~22 features
= Total: ~296 features
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
    logger.info("MERGE TIME FEATURES - DAY 9")
    logger.info("=" * 60)

    # Load existing (with leaks)
    logger.info("\nLoading existing features (with leaks)...")
    df_existing = pd.read_parquet("data/features/train_all_features_with_leaks.parquet")
    logger.info(f"  Shape: {df_existing.shape}")

    # Load time features
    logger.info("\nLoading time features...")
    df_time = pd.read_parquet("data/features/train_time_features.parquet")
    logger.info(f"  Shape: {df_time.shape}")

    # Merge
    logger.info("\nMerging...")
    df_all = df_existing.merge(df_time, on='Id', how='left')

    # Drop duplicate Response if exists
    if 'Response_x' in df_all.columns:
        df_all = df_all.drop(columns=['Response_y']).rename(columns={'Response_x': 'Response'})

    logger.info(f"\nMerged shape: {df_all.shape}")
    logger.info(f"  Total features: {df_all.shape[1] - 2}")

    memory_usage_report(df_all, "All Features")

    # Save
    output_path = "data/features/train_all_features_complete.parquet"
    logger.info(f"\nSaving to {output_path}...")
    df_all.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! ({file_size:.2f} MB)")

    logger.info("\n" + "=" * 60)
    logger.info("READY FOR TRAINING")
    logger.info("=" * 60)
    logger.info(f"\n✅ Total features: {df_all.shape[1] - 2}")
    logger.info(f"\n🎯 NEXT: python notebooks/09_retrain_with_time.py")


if __name__ == "__main__":
    main()
