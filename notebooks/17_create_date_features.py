"""
Phase 4 Script 17: Date Feature Engineering

Approach (from 1st place solution):
- Reduce 1,156 date cols to 52 station entry times (min per station)
- Derive part-level: start_time, end_time, duration
- Derive line-level: per L0/L1/L2/L3 start + duration
- Derive temporal: part_of_week, day_of_week, hour_of_day from real start_time
- Save ALL 52 station entry times as features

Output: data/features/train_date_features.parquet (~80 features)
"""
import gc
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

LINES = ['L0', 'L1', 'L2', 'L3']


def get_station_groups(cols):
    groups = {}
    for c in cols:
        parts = c.split('_')
        station_key = f"{parts[0]}_{parts[1]}"
        groups.setdefault(station_key, []).append(c)
    return groups


def main():
    logger.info("=" * 60)
    logger.info("PHASE 4 — SCRIPT 17: DATE FEATURE ENGINEERING")
    logger.info("=" * 60)

    # ── 1. Load date parquet ──────────────────────────────────────
    logger.info("\n1. Loading train_date.parquet...")
    df_date = pd.read_parquet(project_root / "data/processed/train_date.parquet")
    logger.info(f"   Shape: {df_date.shape}")
    memory_usage_report(df_date, "Raw Date")

    date_cols = [c for c in df_date.columns if c != 'Id']
    station_groups = get_station_groups(date_cols)
    logger.info(f"   Date cols: {len(date_cols)}")
    logger.info(f"   Unique stations: {len(station_groups)}")

    # ── 2. Compute min timestamp per station (entry time) ─────────
    logger.info("\n2. Computing station entry times (min per station)...")
    station_times = pd.DataFrame({'Id': df_date['Id']})

    for station, cols in tqdm(station_groups.items(), desc="Station entry times"):
        station_times[f'entry_{station}'] = df_date[cols].min(axis=1)

    logger.info(f"   Station entry time cols: {len(station_groups)}")

    # Free raw date memory — 5.1 GB no longer needed
    del df_date
    gc.collect()

    # ── 3. Part-level features ────────────────────────────────────
    logger.info("\n3. Computing part-level features...")
    entry_cols = [c for c in station_times.columns if c.startswith('entry_')]

    features = pd.DataFrame({'Id': station_times['Id']})

    features['start_time'] = station_times[entry_cols].min(axis=1)
    features['end_time']   = station_times[entry_cols].max(axis=1)
    features['duration']   = features['end_time'] - features['start_time']

    logger.info(f"   start_time range: {features['start_time'].min():.2f} – {features['start_time'].max():.2f}")
    logger.info(f"   duration   range: {features['duration'].min():.2f} – {features['duration'].max():.2f}")
    logger.info(f"   Parts with duration=0:  {(features['duration'] == 0).sum():,}")
    logger.info(f"   Parts with null start:  {features['start_time'].isna().sum():,}")

    # ── 4. Per-line features ──────────────────────────────────────
    logger.info("\n4. Computing per-line features...")
    for line in LINES:
        line_cols = [c for c in entry_cols if c.startswith(f'entry_{line}_')]
        if not line_cols:
            continue
        features[f'start_time_{line}'] = station_times[line_cols].min(axis=1)
        features[f'end_time_{line}']   = station_times[line_cols].max(axis=1)
        features[f'duration_{line}']   = (
            features[f'end_time_{line}'] - features[f'start_time_{line}']
        )
        features[f'n_stations_with_time_{line}'] = (
            station_times[line_cols].notna().sum(axis=1).astype('int16')
        )
        logger.info(f"   {line}: {len(line_cols)} stations, "
                    f"avg duration={features[f'duration_{line}'].mean():.4f}")

    features['n_stations_with_time'] = (
        station_times[entry_cols].notna().sum(axis=1).astype('int16')
    )

    # ── 5. Temporal cycle features ────────────────────────────────
    logger.info("\n5. Computing temporal cycle features from actual start_time...")
    # Verified: max start_time=1716.54, 1716/1680=1.02 weeks ✅
    # 1680 units = 1 week = 10,080 mins → 1 unit = 6 mins → confirmed
    features['part_of_week'] = features['start_time'] % 1680
    features['day_of_week']  = (features['start_time'] // 240).astype('float32') % 7
    features['hour_of_day']  = (features['start_time'] // 10).astype('float32') % 24

    # ── 6. Derived/scaled features ────────────────────────────────
    logger.info("\n6. Computing derived features...")
    features['log_duration'] = np.log1p(features['duration'])

    global_mean_dur = features['duration'].mean()
    global_std_dur  = features['duration'].std()
    features['duration_z']         = (features['duration'] - global_mean_dur) / global_std_dur
    features['is_long_duration']   = (features['duration'] > features['duration'].quantile(0.90)).astype('int8')
    features['is_short_duration']  = (features['duration'] < features['duration'].quantile(0.10)).astype('int8')

    # ── 6b. Merge 52 station entry times into features ────────────
    logger.info("\n6b. Merging 52 station entry times into features...")
    features = features.merge(station_times, on='Id', how='left')
    logger.info(f"   Shape after merge: {features.shape}")

    del station_times
    gc.collect()

    # ── 7. Fill nulls ─────────────────────────────────────────────
    logger.info("\n7. Filling nulls...")
    null_counts_before = features.isnull().sum()
    null_cols = null_counts_before[null_counts_before > 0]
    logger.info(f"   Cols with nulls: {len(null_cols)}")

    for col in features.columns:
        if col == 'Id':
            continue
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())

    logger.info(f"   Nulls after fill: {features.isnull().sum().sum()}")

    # ── 8. Summary ────────────────────────────────────────────────
    feature_cols = [c for c in features.columns if c != 'Id']
    logger.info("\n" + "=" * 60)
    logger.info("DATE FEATURES SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Station entry times (entry_*): 52")
    logger.info(f"  Part-level (start/end/dur):    3")
    logger.info(f"  Per-line features:             {len(LINES) * 4}")
    logger.info(f"  n_stations_with_time:          {len(LINES) + 1}")
    logger.info(f"  Temporal cycle features:       3")
    logger.info(f"  Derived/scaled:                5")
    logger.info(f"  Total features:                {len(feature_cols)}")

    memory_usage_report(features, "Date Features")

    # ── 9. Save ───────────────────────────────────────────────────
    output_path = project_root / "data/features/train_date_features.parquet"
    features.to_parquet(output_path, compression='snappy', index=False)
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"\n✅ Saved → {output_path} ({file_size:.2f} MB)")
    logger.info(f"\n🎯 NEXT: python notebooks/18_merge_phase4_features.py")


if __name__ == '__main__':
    main()
