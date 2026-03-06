"""
Time-based Feature Engineering

Winners' insights:
- 0.01 time granularity = 6 minutes
- Rolling mean of target with LARGE windows (1000, 5000) works best
- Time since/until failures is predictive
- Weekly patterns exist (part of week mod 1680)

Expected MCC gain: +0.05 to +0.08
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.logger import setup_logger

logger = setup_logger(__name__)


def create_rolling_target_features(df, target='Response'):
    """
    Rolling mean of target with large windows.
    Winners said big windows (1000, 5000) capture failure rate periods.
    """
    logger.info("=" * 60)
    logger.info("TIME #1: ROLLING TARGET MEAN")
    logger.info("=" * 60)

    logger.info(f"\nSorting by Id (proxy for time)...")
    df = df.sort_values('Id').reset_index(drop=True)

    # Out-of-fold rolling mean (shift to prevent leakage)
    windows = [5, 10, 20, 100, 1000, 5000]

    logger.info(f"\nCreating rolling features for {len(windows)} windows...")
    for window in tqdm(windows, desc="Rolling windows"):
        df[f'rolling_mean_target_{window}'] = (
            df[target]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)  # Shift to prevent leakage
        )

        # Fill first row (NaN after shift)
        df[f'rolling_mean_target_{window}'].fillna(df[target].mean(), inplace=True)

    logger.info(f"\n✅ Created {len(windows)} rolling mean features")

    return df


def create_failure_distance_features(df, target='Response'):
    """
    Distance to last/next N failures.
    Captures "how long since last failure" signal.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TIME #2: FAILURE DISTANCE")
    logger.info("=" * 60)

    logger.info(f"\nFinding failure indices...")
    failure_indices = df[df[target] == 1].index.tolist()
    logger.info(f"  Total failures: {len(failure_indices)}")

    # Distance to last N failures
    for n in tqdm([1, 5, 10], desc="Distance features"):
        df[f'distance_to_last_{n}_failures'] = 0.0
        df[f'distance_to_next_{n}_failures'] = 0.0

        for idx in range(len(df)):
            # Last N failures
            past_failures = [f for f in failure_indices if f < idx]
            if len(past_failures) >= n:
                recent = past_failures[-n:]
                df.loc[idx, f'distance_to_last_{n}_failures'] = idx - np.mean(recent)
            else:
                df.loc[idx, f'distance_to_last_{n}_failures'] = idx  # Distance to start

            # Next N failures
            future_failures = [f for f in failure_indices if f > idx]
            if len(future_failures) >= n:
                upcoming = future_failures[:n]
                df.loc[idx, f'distance_to_next_{n}_failures'] = np.mean(upcoming) - idx
            else:
                # Distance to end
                df.loc[idx, f'distance_to_next_{n}_failures'] = len(df) - idx

    logger.info(f"\n✅ Created 6 distance features")

    return df


def create_time_window_counts(df):
    """
    Count records in time windows.
    Winners used 2.5h, 24h, 168h windows.
    Time granularity: 0.01 = 6 mins → 2.5h = 25 units
    """
    logger.info("\n" + "=" * 60)
    logger.info("TIME #3: TIME WINDOW COUNTS")
    logger.info("=" * 60)

    # Assume Id is roughly chronological (winners sorted by station start times)
    windows = [25, 240, 1680]  # 2.5h, 24h, 168h in 6-min units

    logger.info(f"\nCounting records in windows...")
    for window in tqdm(windows, desc="Window counts"):
        # Records in past window
        df[f'records_last_{window}'] = (
            df['Id']
            .rolling(window=window, min_periods=1)
            .count()
        )

        # Records in next window (reverse rolling)
        df[f'records_next_{window}'] = (
            df['Id'][::-1]
            .rolling(window=window, min_periods=1)
            .count()
            .values[::-1]
        )

    logger.info(f"\n✅ Created {len(windows) * 2} window count features")

    return df


def create_weekly_patterns(df):
    """
    Weekly patterns in manufacturing.
    Winners used mod 1680 (1 week in 6-min units).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TIME #4: WEEKLY PATTERNS")
    logger.info("=" * 60)

    logger.info("\nCreating weekly cycle features...")

    # Part of week (0-1679)
    df['id_mod_1680'] = df['Id'] % 1680

    # Day of week proxy (0-6)
    df['day_of_week'] = (df['Id'] // 240) % 7

    # Hour of day proxy (0-23)
    df['hour_of_day'] = (df['Id'] // 10) % 24

    # Shift within day (morning/afternoon/night)
    df['shift'] = pd.cut(df['hour_of_day'], bins=[0, 8, 16, 24], labels=[0, 1, 2])
    df['shift'] = df['shift'].astype(float)

    logger.info(f"\n✅ Created 4 weekly pattern features")

    return df


def main():
    logger.info("=" * 60)
    logger.info("TIME FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info("\nExpected MCC gain: +0.05 to +0.08")

    # Load data with Response
    logger.info("\nLoading data...")
    df = pd.read_parquet("data/features/train_selected_top150.parquet")[['Id', 'Response']]
    logger.info(f"  Shape: {df.shape}")

    # Create time features
    df = create_rolling_target_features(df)
    df = create_failure_distance_features(df)
    df = create_time_window_counts(df)
    df = create_weekly_patterns(df)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    time_features = [c for c in df.columns if c not in ['Id', 'Response']]
    logger.info(f"\nTotal time features: {len(time_features)}")

    # Save
    output_path = "data/features/train_time_features.parquet"
    logger.info(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! ({file_size:.2f} MB)")

    logger.info(f"\n✅ Created {len(time_features)} time features:")
    logger.info(f"  - Rolling means: 6 features")
    logger.info(f"  - Failure distance: 6 features")
    logger.info(f"  - Window counts: 6 features")
    logger.info(f"  - Weekly patterns: 4 features")
    logger.info(f"\n🎯 NEXT: python notebooks/09_merge_time_features.py")


if __name__ == "__main__":
    main()
