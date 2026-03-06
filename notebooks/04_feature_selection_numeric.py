"""
Select top K features by importance from baseline model.

Why:
- We have 465 features, but only ~150 contribute meaningfully
- Bottom 315 features add noise and slow training
- Dropping them often IMPROVES MCC (prevents overfitting)

Process:
1. Load baseline model (from Day 6)
2. Extract feature importances
3. Keep top 150 features
4. Save filtered dataset
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


def select_top_features(
        top_k: int = 150,
        model_path: str = "data/models/baseline_lgbm.pkl",
        data_path: str = "data/features/train_engineered.parquet",
        output_path: str = "data/features/train_selected_top150.parquet"
):
    """
    Select top K features by LightGBM importance.

    Args:
        top_k: Number of features to keep (default 150)
        model_path: Path to trained baseline model
        data_path: Path to engineered features
        output_path: Where to save selected features
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION: TOP K BY IMPORTANCE")
    logger.info("=" * 60)

    # 1. Load baseline model
    logger.info(f"\nLoading model from {model_path}...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']

    logger.info(f"  Model has {len(feature_names)} features")

    # 2. Get feature importances
    logger.info("\nExtracting feature importances...")
    importances = model.feature_importances_

    # Create importance dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Log stats
    logger.info(f"  Top feature: {feat_imp.iloc[0]['feature']} (importance={feat_imp.iloc[0]['importance']:.0f})")
    logger.info(f"  Bottom feature: {feat_imp.iloc[-1]['feature']} (importance={feat_imp.iloc[-1]['importance']:.0f})")
    logger.info(f"  Zero-importance features: {(feat_imp['importance'] == 0).sum()}")

    # 3. Select top K
    logger.info(f"\nSelecting top {top_k} features...")
    top_features = feat_imp.head(top_k)['feature'].tolist()

    # Calculate importance coverage
    total_importance = importances.sum()
    top_importance = top_features_importance = feat_imp.head(top_k)['importance'].sum()
    coverage = (top_importance / total_importance) * 100

    logger.info(f"  Top {top_k} features capture {coverage:.1f}% of total importance")

    # Log top 20
    logger.info(f"\nTop 20 selected features:")
    for i, row in feat_imp.head(20).iterrows():
        logger.info(f"  {i + 1}. {row['feature']:50s} {row['importance']:>8.0f}")

    # 4. Load data and filter
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"  Original shape: {df.shape}")

    # Keep Id, Response, and top K features
    cols_to_keep = ['Id', 'Response'] + top_features
    df_selected = df[cols_to_keep]

    logger.info(f"  Selected shape: {df_selected.shape}")
    logger.info(
        f"  Reduction: {df.shape[1]} → {df_selected.shape[1]} columns ({(1 - df_selected.shape[1] / df.shape[1]) * 100:.1f}% fewer)")

    # 5. Save
    logger.info(f"\nSaving to {output_path}...")
    df_selected.to_parquet(output_path, compression='snappy', index=False)

    # Save feature list
    feat_list_path = Path(output_path).parent / 'selected_features_top150.txt'
    with open(feat_list_path, 'w') as f:
        for feat in top_features:
            f.write(f"{feat}\n")

    logger.info(f"✅ Feature selection complete!")
    logger.info(f"  Selected features: {len(top_features)}")
    logger.info(f"  Importance coverage: {coverage:.1f}%")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  Feature list: {feat_list_path}")

    return df_selected, top_features


if __name__ == "__main__":
    select_top_features(top_k=150)
