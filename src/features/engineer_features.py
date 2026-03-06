"""
Feature engineering for Bosch production line data.

Creates four categories of features:
1. Routing features: Which stations visited, coverage patterns
2. Time features: Duration, dwell time, processing speed
3. Aggregation features: Station-level statistics
4. Null pattern features: Sparsity as signal

Why feature engineering matters MORE than algorithms:
- Winning Kaggle solutions: 80% feature engineering, 20% model tuning
- Tree models (LightGBM) can't create interactions → we must
- Domain knowledge (ME thinking) translates to features

Key principle: NO TARGET LEAKAGE
- Never use Response to create features
- Time-based features use PAST data only (no future peeking)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict
import argparse
from tqdm import tqdm
from src.config import Config
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Create predictive features from raw sensor data.

    Philosophy:
    - Features should be interpretable (explainable to engineers)
    - Features should be stable (work on new data)
    - Features should be fast to compute (production-ready)
    """

    def __init__(self, config: Config):
        self.config = config
        self.processed_dir = Path(config.get('paths.processed'))
        self.features_dir = Path(config.get('paths.features'))

    def load_data(self) -> pd.DataFrame:
        """Load selected features from Day 4."""
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)

        data_path = self.features_dir / 'train_selected.parquet'
        logger.info(f"Loading from {data_path}...")
        df = pd.read_parquet(data_path)
        logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        return df

    def create_routing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on which stations parts visit.

        Insight from EDA:
        - 69.5% sparsity means routing is important
        - Station 32 (2% coverage, 4.5% failure) = routing signal
        - Different product types take different paths

        Features:
        - Total stations visited (overall coverage)
        - Stations per line (L0, L1, L2, L3 coverage separately)
        - Rare station visits (hit S32, S38, etc.)
        - Critical station visits (hit S29, S30, S33, etc.)
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREATING ROUTING FEATURES")
        logger.info("=" * 60)

        # Get all feature columns (exclude Id, Response)
        feature_cols = [c for c in df.columns if c not in ['Id', 'Response']]

        # Parse features into lines and stations
        features_by_line = {0: [], 1: [], 2: [], 3: []}
        stations_all = set()

        for feat in feature_cols:
            parts = feat.split('_')
            if len(parts) >= 3:
                line = int(parts[0][1:])
                station = int(parts[1][1:])
                features_by_line[line].append(feat)
                stations_all.add(station)

        logger.info(f"Found {len(stations_all)} unique stations")

        # Feature 1: Total stations visited
        logger.info("Creating: total_stations_visited...")
        df['total_stations_visited'] = 0
        for station in stations_all:
            station_cols = [c for c in feature_cols if f'_S{station}_' in c]
            if station_cols:
                df['total_stations_visited'] += df[station_cols].notnull().any(axis=1).astype(int)

        # Feature 2-5: Stations visited per line
        for line in [0, 1, 2, 3]:
            logger.info(f"Creating: L{line}_stations_visited...")
            line_cols = features_by_line[line]
            if line_cols:
                # Count unique stations on this line
                line_stations = set()
                for col in line_cols:
                    station = int(col.split('_')[1][1:])
                    line_stations.add(station)

                df[f'L{line}_stations_visited'] = 0
                for station in line_stations:
                    station_cols = [c for c in line_cols if f'_S{station}_' in c]
                    if station_cols:
                        df[f'L{line}_stations_visited'] += df[station_cols].notnull().any(axis=1).astype(int)

        # Feature 6: Coverage ratio (% of possible stations visited)
        logger.info("Creating: coverage_ratio...")
        df['coverage_ratio'] = df['total_stations_visited'] / len(stations_all)

        # Feature 7-10: Line-specific coverage ratios
        for line in [0, 1, 2, 3]:
            line_cols = features_by_line[line]
            if line_cols:
                line_stations = set(int(c.split('_')[1][1:]) for c in line_cols)
                df[f'L{line}_coverage_ratio'] = df[f'L{line}_stations_visited'] / len(line_stations)

        # Feature 11-14: Critical stations (from EDA: >50% coverage)
        critical_stations = [29, 30, 33, 34, 37]  # From EDA
        logger.info("Creating: critical_station_coverage...")
        df['critical_station_visits'] = 0
        for station in critical_stations:
            station_cols = [c for c in feature_cols if f'_S{station}_' in c]
            if station_cols:
                df['critical_station_visits'] += df[station_cols].notnull().any(axis=1).astype(int)
        df['critical_station_coverage'] = df['critical_station_visits'] / len(critical_stations)

        # Feature 15-18: Rare stations (from EDA: <10% coverage)
        rare_stations = [32, 38, 43, 44, 49, 50]  # From EDA
        logger.info("Creating: rare_station_visits...")
        df['rare_station_visits'] = 0
        df['visited_S32'] = 0  # Station 32 specifically (4.5% failure rate!)

        for station in rare_stations:
            station_cols = [c for c in feature_cols if f'_S{station}_' in c]
            if station_cols:
                visited = df[station_cols].notnull().any(axis=1).astype(int)
                df['rare_station_visits'] += visited
                if station == 32:
                    df['visited_S32'] = visited

        # Feature 19: Visited Station 24 (top predictive station from EDA)
        logger.info("Creating: visited_S24...")
        s24_cols = [c for c in feature_cols if '_S24_' in c]
        if s24_cols:
            df['visited_S24'] = df[s24_cols].notnull().any(axis=1).astype(int)

        logger.info(f"✅ Created {19} routing features")
        return df

    def create_null_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on sparsity patterns.
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREATING NULL PATTERN FEATURES")
        logger.info("=" * 60)

        # FIX: Only parse ORIGINAL features (not engineered ones)
        # Engineered features don't have L{line}_S{station} format
        feature_cols = [c for c in df.columns if c not in ['Id', 'Response']]

        # Filter to only L{line}_S{station}_F{feature} format
        original_features = []
        for col in feature_cols:
            parts = col.split('_')
            # Valid original features have at least 3 parts: L0_S0_F10
            if len(parts) >= 3 and parts[0].startswith('L') and parts[1].startswith('S'):
                original_features.append(col)

        # Parse by line
        features_by_line = {0: [], 1: [], 2: [], 3: []}
        for feat in original_features:  # ← Use original_features, not feature_cols
            parts = feat.split('_')
            if len(parts) >= 3:
                line = int(parts[0][1:])
                features_by_line[line].append(feat)

        # Feature 1-4: Sparsity ratio per line
        for line in [0, 1, 2, 3]:
            logger.info(f"Creating: L{line}_sparsity...")
            line_cols = features_by_line[line]
            if line_cols:
                df[f'L{line}_sparsity'] = df[line_cols].isnull().sum(axis=1) / len(line_cols)

        # Feature 5: Overall sparsity
        logger.info("Creating: overall_sparsity...")
        df['overall_sparsity'] = df[original_features].isnull().sum(axis=1) / len(original_features)

        # Feature 6: Non-null feature count
        logger.info("Creating: non_null_feature_count...")
        df['non_null_feature_count'] = df[original_features].notnull().sum(axis=1)

        logger.info(f"✅ Created {6} null pattern features")
        return df

    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregations of sensor values.

        Why:
        - Individual sensors are noisy
        - Aggregates (mean, std, min, max) capture overall state
        - Station-level aggregates show if station is "normal" or "anomalous"

        Focus on high-signal stations from EDA:
        - L1_S24 (9/20 top features)
        - L2_S26, L2_S27 (also in top 20)
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREATING AGGREGATION FEATURES")
        logger.info("=" * 60)

        feature_cols = [c for c in df.columns if c not in ['Id', 'Response']]

        # Overall aggregates (all numeric features)
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        logger.info("Creating: overall aggregates (mean, std, min, max)...")
        df['feature_mean'] = df[numeric_cols].mean(axis=1)
        df['feature_std'] = df[numeric_cols].std(axis=1)
        df['feature_min'] = df[numeric_cols].min(axis=1)
        df['feature_max'] = df[numeric_cols].max(axis=1)
        df['feature_range'] = df['feature_max'] - df['feature_min']

        # Station 24 specific aggregates (top predictive station)
        s24_cols = [c for c in numeric_cols if '_S24_' in c]
        if len(s24_cols) > 0:
            logger.info("Creating: L1_S24 aggregates...")
            df['S24_mean'] = df[s24_cols].mean(axis=1)
            df['S24_std'] = df[s24_cols].std(axis=1)
            df['S24_min'] = df[s24_cols].min(axis=1)
            df['S24_max'] = df[s24_cols].max(axis=1)
            df['S24_range'] = df['S24_max'] - df['S24_min']
            df['S24_non_null_count'] = df[s24_cols].notnull().sum(axis=1)

        # Station 26 aggregates
        s26_cols = [c for c in numeric_cols if '_S26_' in c]
        if len(s26_cols) > 0:
            logger.info("Creating: L2_S26 aggregates...")
            df['S26_mean'] = df[s26_cols].mean(axis=1)
            df['S26_std'] = df[s26_cols].std(axis=1)

        # Station 27 aggregates
        s27_cols = [c for c in numeric_cols if '_S27_' in c]
        if len(s27_cols) > 0:
            logger.info("Creating: L2_S27 aggregates...")
            df['S27_mean'] = df[s27_cols].mean(axis=1)
            df['S27_std'] = df[s27_cols].std(axis=1)

        logger.info(f"✅ Created {17} aggregation features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Why:
        - Tree models split on one feature at a time
        - Pre-computing interactions helps models find patterns faster
        - Focus on domain-meaningful interactions

        Examples:
        - Coverage × rare_station_visits (parts with low coverage but hit S32?)
        - L3_coverage × overall_sparsity (is main line sparse for this part?)
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREATING INTERACTION FEATURES")
        logger.info("=" * 60)

        # Interaction 1: Rare station visits despite low coverage
        if 'rare_station_visits' in df.columns and 'total_stations_visited' in df.columns:
            logger.info("Creating: rare_visits_per_total...")
            df['rare_visits_per_total'] = df['rare_station_visits'] / (df['total_stations_visited'] + 1)

        # Interaction 2: Critical station coverage × L3 coverage
        if 'critical_station_coverage' in df.columns and 'L3_coverage_ratio' in df.columns:
            logger.info("Creating: critical_L3_interaction...")
            df['critical_L3_interaction'] = df['critical_station_coverage'] * df['L3_coverage_ratio']

        # Interaction 3: Sparsity × feature variance
        if 'overall_sparsity' in df.columns and 'feature_std' in df.columns:
            logger.info("Creating: sparsity_variance_interaction...")
            df['sparsity_variance_interaction'] = df['overall_sparsity'] * df['feature_std']

        # Interaction 4: Station 24 visit × Station 24 mean value
        if 'visited_S24' in df.columns and 'S24_mean' in df.columns:
            logger.info("Creating: S24_visit_mean_interaction...")
            df['S24_visit_mean_interaction'] = df['visited_S24'] * df['S24_mean'].fillna(0)

        logger.info(f"✅ Created {4} interaction features")
        return df

    def engineer_all_features(self) -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        # Load data
        df = self.load_data()

        logger.info(f"\nBASELINE: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        # Create features
        df = self.create_routing_features(df)
        df = self.create_null_pattern_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_interaction_features(df)

        # Summary
        original_features = 425  # From Day 4 (427 - Id - Response)
        engineered_features = df.shape[1] - 427  # New columns created
        total_features = df.shape[1] - 2  # Exclude Id, Response

        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original features: {original_features}")
        logger.info(f"Engineered features: {engineered_features}")
        logger.info(f"Total features: {total_features}")
        logger.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        # Memory report
        memory_usage_report(df, "Engineered Dataset")

        # Save
        output_path = self.features_dir / 'train_engineered.parquet'
        logger.info(f"\nSaving to {output_path}...")
        df.to_parquet(output_path, compression='snappy', index=False)

        logger.info("✅ Feature engineering complete!")

        return df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Engineer features for Bosch data")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    args = parser.parse_args()

    config = Config(args.config)
    engineer = FeatureEngineer(config)
    df_engineered = engineer.engineer_all_features()

    logger.info("\n✅ Done! Next steps:")
    logger.info("  1. Validate engineered features (check for leakage)")
    logger.info("  2. Train baseline LightGBM model (Day 6)")
    logger.info("  3. Measure feature importance")


if __name__ == "__main__":
    main()
