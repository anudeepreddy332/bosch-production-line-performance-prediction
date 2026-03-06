"""
Select top 100 target-encoded categorical features by correlation with Response.

Why 100:
- Balance between signal and memory
- Top features have strongest predictive power
- 100 * 9MB = ~900MB (safe for 16GB RAM)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("SELECT TOP 100 CATEGORICAL FEATURES")
    logger.info("=" * 60)

    # Load target-encoded categorical
    logger.info("\nLoading target-encoded categorical...")
    df_cat = pd.read_parquet("data/features/train_categorical_target_encoded.parquet")
    logger.info(f"  Shape: {df_cat.shape}")

    # Load Response
    logger.info("\nLoading Response...")
    df_numeric = pd.read_parquet("data/features/train_selected_top150.parquet")
    response = df_numeric[['Id', 'Response']]

    # Merge to get Response
    logger.info("\nMerging with Response...")
    df = df_cat.merge(response, on='Id', how='left')
    logger.info(f"  Merged shape: {df.shape}")

    # Calculate correlation for each categorical feature
    logger.info("\n" + "=" * 60)
    logger.info("CALCULATING CORRELATIONS")
    logger.info("=" * 60)

    cat_cols = [c for c in df.columns if c.endswith('_target_enc')]
    logger.info(f"\nCalculating correlations for {len(cat_cols)} features...")

    correlations = {}
    for col in tqdm(cat_cols, desc="Computing correlations"):
        # Calculate correlation with Response
        mask = ~df[col].isna() & ~df['Response'].isna()
        if mask.sum() > 0:
            corr = np.corrcoef(df[col][mask], df['Response'][mask])[0, 1]
            correlations[col] = abs(corr)  # Use absolute value
        else:
            correlations[col] = 0.0

    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\n✅ Calculated correlations for {len(sorted_features)} features")

    # Show top 20
    logger.info(f"\nTop 20 categorical features by correlation:")
    for i, (feat, corr) in enumerate(sorted_features[:20], 1):
        logger.info(f"  {i:2d}. {feat:50s} |corr|={corr:.4f}")

    # Show distribution
    corr_values = [c for _, c in sorted_features]
    logger.info(f"\nCorrelation distribution:")
    logger.info(f"  Top 10:  {np.mean(corr_values[:10]):.4f} ± {np.std(corr_values[:10]):.4f}")
    logger.info(f"  Top 50:  {np.mean(corr_values[:50]):.4f} ± {np.std(corr_values[:50]):.4f}")
    logger.info(f"  Top 100: {np.mean(corr_values[:100]):.4f} ± {np.std(corr_values[:100]):.4f}")
    logger.info(f"  All:     {np.mean(corr_values):.4f} ± {np.std(corr_values):.4f}")

    # Select top 100
    top_100_features = [f[0] for f in sorted_features[:100]]

    logger.info("\n" + "=" * 60)
    logger.info("SELECTING TOP 100 FEATURES")
    logger.info("=" * 60)

    df_selected = df_cat[['Id'] + top_100_features].copy()
    logger.info(f"\nSelected shape: {df_selected.shape}")
    logger.info(f"Memory usage: {df_selected.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Save
    output_path = "data/features/train_categorical_top100.parquet"
    logger.info(f"\nSaving to {output_path}...")
    df_selected.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! File size: {file_size:.2f} MB")

    # Save feature names
    feature_names_path = "data/features/selected_categorical_top100.txt"
    with open(feature_names_path, 'w') as f:
        for feat in top_100_features:
            f.write(f"{feat}\n")

    logger.info(f"✅ Feature names saved to: {feature_names_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ READY TO MERGE WITH NUMERIC FEATURES!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
