"""
Feature selection utilities for Bosch production line data.

Removes features with:
1. High null percentage (>99% nulls = insufficient data)
2. Zero variance (all same value = no signal)
3. Duplicate columns (correlation = 1.0 = redundant)

Why feature selection matters:
- 4,268 features overwhelm models (curse of dimensionality)
- Most features are sparse (90%+ nulls) = noise, not signal
- Training time scales with feature count
- Interpretability improves with fewer features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from src.config import Config
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


class FeatureSelector:
    """
    Select informative features by removing noise and redundancy.

    Three-step filtering:
    1. Null-based: Drop columns with >threshold% nulls
    2. Variance-based: Drop columns with zero variance
    3. Correlation-based: Drop perfect duplicates (corr=1.0)
    """

    def __init__(self, config: Config):
        self.config = config
        self.processed_dir = Path(config.get('paths.processed'))
        self.features_dir = Path(config.get('paths.features'))
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Selection thresholds
        self.null_threshold = 0.99  # Drop if >99% nulls
        self.variance_threshold = 0.0  # Drop if all same value
        self.correlation_threshold = 0.999  # Drop if perfect duplicate

        # Track what gets removed (for documentation)
        self.removal_stats = {
            'high_null': [],
            'zero_variance': [],
            'duplicates': []
        }

    def remove_high_null_features(
            self,
            df: pd.DataFrame,
            threshold: float = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with null percentage above threshold.

        Why 99%: If only 1% of parts have a measurement, insufficient
        signal to learn patterns. Model will overfit to those rare cases.

        Args:
            df: Input DataFrame
            threshold: Null percentage threshold (0-1), defaults to self.null_threshold

        Returns:
            Filtered DataFrame, list of removed columns
        """
        if threshold is None:
            threshold = self.null_threshold

        logger.info(f"Removing features with >{threshold * 100}% nulls...")

        # Calculate null percentage per column
        null_pct = df.isnull().sum() / len(df)

        # Identify high-null columns (exclude Id, Response)
        protected_cols = ['Id', 'Response']
        high_null_cols = null_pct[
            (null_pct > threshold) & (~null_pct.index.isin(protected_cols))
            ].index.tolist()

        logger.info(f"  Found {len(high_null_cols)} high-null features")

        # Log a few examples with their null percentages
        if len(high_null_cols) > 0:
            examples = null_pct[high_null_cols].sort_values(ascending=False).head(5)
            logger.info("  Examples (highest null %):")
            for col, pct in examples.items():
                logger.info(f"    {col}: {pct * 100:.2f}% nulls")

        # Remove them
        df_filtered = df.drop(columns=high_null_cols)
        self.removal_stats['high_null'] = high_null_cols

        logger.info(f"  Remaining features: {df_filtered.shape[1] - len(protected_cols)}")

        return df_filtered, high_null_cols

    def remove_zero_variance_features(
            self,
            df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features where all non-null values are identical.

        Why: If all parts have same measurement, feature provides zero
        information for distinguishing failures from normal parts.

        Example: L0_S0_F10 = 5.0 for all 1.18M parts → useless
        """
        logger.info("Removing zero-variance features...")

        # Get numeric columns only (can't check variance on categoricals)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        protected_cols = ['Id', 'Response']
        numeric_cols = [c for c in numeric_cols if c not in protected_cols]

        zero_var_cols = []
        for col in numeric_cols:
            # Check variance on non-null values
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.var() == 0:
                zero_var_cols.append(col)

        logger.info(f"  Found {len(zero_var_cols)} zero-variance features")

        if len(zero_var_cols) > 0:
            logger.info("  Examples:")
            for col in zero_var_cols[:5]:
                unique_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'N/A'
                logger.info(f"    {col}: all values = {unique_val}")

        df_filtered = df.drop(columns=zero_var_cols)
        self.removal_stats['zero_variance'] = zero_var_cols

        logger.info(f"  Remaining features: {df_filtered.shape[1] - 2}")  # -2 for Id, Response

        return df_filtered, zero_var_cols

    def remove_duplicate_features(
            self,
            df: pd.DataFrame,
            sample_frac: float = 0.1
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove perfectly correlated features (correlation = 1.0).

        Why: If L0_S0_F10 and L0_S0_F12 always have same value,
        they measure the same thing → keep one, drop the other.

        Args:
            sample_frac: Use sample for correlation (full data is slow)
                        0.1 = 118K rows, sufficient for detecting duplicates
        """
        logger.info("Removing duplicate features...")

        # Only check numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        protected_cols = ['Id', 'Response']
        numeric_cols = [c for c in numeric_cols if c not in protected_cols]

        if len(numeric_cols) == 0:
            logger.info("  No numeric features to check")
            return df, []

        # Sample for speed (correlation stable with 10% sample)
        if sample_frac < 1.0:
            df_sample = df[numeric_cols].sample(frac=sample_frac, random_state=42)
            logger.info(f"  Using {len(df_sample):,} row sample for correlation")
        else:
            df_sample = df[numeric_cols]

        # Compute correlation matrix
        logger.info("  Computing correlation matrix...")
        corr_matrix = df_sample.corr().abs()

        # Find pairs with correlation > threshold
        # Upper triangle only (avoid double-counting)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns with correlation > 0.999 (essentially 1.0)
        duplicate_cols = [
            col for col in upper_tri.columns
            if any(upper_tri[col] > self.correlation_threshold)
        ]

        logger.info(f"  Found {len(duplicate_cols)} duplicate features")

        if len(duplicate_cols) > 0:
            # Log examples of what's being dropped
            logger.info("  Examples of duplicates:")
            for col in duplicate_cols[:5]:
                # Find what it's correlated with
                high_corr = upper_tri[col][upper_tri[col] > self.correlation_threshold]
                if len(high_corr) > 0:
                    partner = high_corr.idxmax()
                    corr_val = high_corr.max()
                    logger.info(f"    {col} ≈ {partner} (corr={corr_val:.4f})")

        df_filtered = df.drop(columns=duplicate_cols)
        self.removal_stats['duplicates'] = duplicate_cols

        logger.info(f"  Remaining features: {df_filtered.shape[1] - 2}")

        return df_filtered, duplicate_cols

    def select_features(
            self,
            include_categorical: bool = False,
            include_date: bool = True
    ) -> pd.DataFrame:
        """
        Run full feature selection pipeline on train data.

        Process:
        1. Load train_numeric.parquet (always included)
        2. Optionally load train_date.parquet (time features)
        3. Optionally load train_categorical.parquet (heavy, usually skip)
        4. Merge on Id
        5. Apply three filters (null, variance, duplicates)
        6. Save to data/features/train_selected.parquet

        Args:
            include_categorical: Include categorical features (14GB RAM)
            include_date: Include date features (5GB RAM)

        Returns:
            Filtered DataFrame
        """
        logger.info("=" * 60)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("=" * 60)

        # Load numeric (always included)
        logger.info("Loading train_numeric.parquet...")
        df_numeric = pd.read_parquet(self.processed_dir / 'train_numeric.parquet')
        logger.info(f"  Shape: {df_numeric.shape}")
        memory_usage_report(df_numeric, "train_numeric")

        df = df_numeric.copy()

        # Optionally add date features
        if include_date:
            logger.info("\nLoading train_date.parquet...")
            df_date = pd.read_parquet(self.processed_dir / 'train_date.parquet')
            logger.info(f"  Shape: {df_date.shape}")

            # Merge (drop Id from df_date to avoid duplicate)
            df = df.merge(df_date, on='Id', how='left')
            logger.info(f"  Merged shape: {df.shape}")

        # Optionally add categorical features (WARNING: 14GB RAM)
        if include_categorical:
            logger.info("\nLoading train_categorical.parquet...")
            df_cat = pd.read_parquet(self.processed_dir / 'train_categorical.parquet')
            logger.info(f"  Shape: {df_cat.shape}")

            df = df.merge(df_cat, on='Id', how='left')
            logger.info(f"  Merged shape: {df.shape}")

        # Initial state
        logger.info("\n" + "=" * 60)
        logger.info(f"BEFORE SELECTION: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        logger.info("=" * 60)

        # Filter 1: High nulls
        df, high_null = self.remove_high_null_features(df)

        # Filter 2: Zero variance
        df, zero_var = self.remove_zero_variance_features(df)

        # Filter 3: Duplicates
        df, duplicates = self.remove_duplicate_features(df)

        # Final state
        logger.info("\n" + "=" * 60)
        logger.info(f"AFTER SELECTION: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        logger.info("=" * 60)

        # Summary
        total_removed = len(high_null) + len(zero_var) + len(duplicates)
        logger.info(f"\nRemoved {total_removed} features:")
        logger.info(f"  High null (>99%):   {len(high_null)}")
        logger.info(f"  Zero variance:      {len(zero_var)}")
        logger.info(f"  Duplicates:         {len(duplicates)}")
        logger.info(f"  Remaining:          {df.shape[1] - 2}")  # -2 for Id, Response

        # Save filtered data
        output_path = self.features_dir / 'train_selected.parquet'
        logger.info(f"\nSaving to {output_path}...")
        df.to_parquet(output_path, compression='snappy', index=False)

        logger.info(f"✅ Feature selection complete")

        # Save removal stats for documentation
        self._save_removal_stats()

        return df

    def _save_removal_stats(self):
        """Save list of removed features to CSV for documentation."""
        stats_path = self.features_dir / 'removed_features.csv'

        records = []
        for reason, cols in self.removal_stats.items():
            for col in cols:
                records.append({'feature': col, 'removal_reason': reason})

        if len(records) > 0:
            pd.DataFrame(records).to_csv(stats_path, index=False)
            logger.info(f"  Removal stats saved to {stats_path}")


def main():
    """CLI entry point for feature selection."""
    parser = argparse.ArgumentParser(
        description="Select informative features from Bosch data"
    )
    parser.add_argument(
        '--include-categorical',
        action='store_true',
        help='Include categorical features (requires 14GB RAM)'
    )
    parser.add_argument(
        '--no-date',
        action='store_true',
        help='Exclude date features'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Run feature selection
    selector = FeatureSelector(config)
    df_selected = selector.select_features(
        include_categorical=args.include_categorical,
        include_date=not args.no_date
    )

    logger.info("\n✅ Done! Next steps:")
    logger.info("  1. Run EDA on selected features (02_eda.py)")
    logger.info("  2. Engineer leak + time features (Day 5)")
    logger.info("  3. Train baseline model (Day 6)")


if __name__ == "__main__":
    main()
