"""
Phase 3: Path Feature Engineering

- Chunk features (chunk_id, chunk_size) only help the 7% of parts
that have identical sensor twins, 93% of the failures are singletons
with no chunk signal.
- Path features work for ALL 1.18M parts - every part has a path,
even if it's unique.

Each part travels through a subset of ~ 50 manufacturing stations.
Station presence is determined by the nullness in the raw numeric
 - All columns null for station X -> part never visited that station
 - Any col non-null -> part visited that station

FEATURES CREATED:
1. Station presence flags (~52 binary features, one per station)
2. Line-level counts (4 features: stations visited per line L0-L3)
3. Total stations visited (1 feature: breadth of manufacturing path)
4. Path signature (string: e.g. "L0_S0|L0_S1|L3_S29")
5. Path frequency (how many parts share exact same path)
6. Path failure rate (target encoding: % of failures on same path)
7. Path risk tier (binned: rare/common/very-common path)

Expected: +20-30 meaningful features, MCC jump from 0.29 -> 0.40+
"""
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

# Config
# Minimum parts that must share a path before we compute a failure rate.
# Paths with fewer than MIN_PATH_COUNT parts get the global mean instead.
# Prevents overfitting on rare paths seen only 1–2 times.
MIN_PATH_COUNT = 10

def extract_station_names(columns: list) -> dict:
    """
    Parse all column names to find unique station names and their columns
    Input columns: ['L0_S0_F0', 'L0_S0_F2', 'L0_S1_F4', ...]
    Output: {'L0_S0': ['L0_S0_F0', 'L0_S0_F2'], 'L0_S1': ['L0_S1_F4'], ...}
    """
    station_cols = {}
    for col in columns:
        if col in ['Id', 'Response']:
            continue
        parts = col.split('_')
        if len(parts) >= 3:
            # 'L0_S0_F0' → station = 'L0_S0'
            station = f"{parts[0]}_{parts[1]}"
            station_cols.setdefault(station, []).append(col)
    return station_cols

def compute_station_presence(df: pd.DataFrame,
                             station_cols: dict) -> pd.DataFrame:
    """
    For each station, create a binary flag: did this part visit it?
    A part "visited" a station if ANY of that station's sensor columns
    is non-null. We use .any(axis=1) which returns True if at least
    one value across all columns for that station is non-null.

    df[cols].notna().any(axis=1) → boolean Series of length N.
    Cast to int8 (0 or 1) to save memory — no need for float64 here.

    Returns df with cols like 'visited_L0_S0', 'visited_L3_S29', etc.
    """
    logger.info(f"\nComputing station presence flags for {len(station_cols)} stations...")

    presence_df = pd.DataFrame({'Id': df['Id']})

    for station, cols in tqdm(station_cols.items(), desc="Station presence"):
        presence_df[f'visited_{station}'] = (
            df[cols].notna().any(axis=1).astype(np.int8)
        )
    return presence_df

def compute_line_counts(presence_df: pd.DataFrame,
                        station_cols: dict) -> pd.DataFrame:
    """
    Count how many stations each part visited within each production line.

    Lines are L0, L1, L2, L3 — each contains multiple stations.
    e.g. n_stations_L3 = sum of visited_L3_S29 + visited_L3_S30 + ...

    group visited_ columns by their line prefix (first 2 chars
    of station name), sum across each group.
    """
    lines = sorted(set(s.split('_')[0] for s in station_cols.keys()))

    for line in lines:
        # Get all 'visited_' cols for this line
        line_visited_cols = [
            c for c in presence_df.columns if c.startswith(f'visited_{line}_')
        ]
        if line_visited_cols:
            # Sum of binary flags = count of stations visited in this line
            presence_df[f'n_stations_{line}'] = (
                presence_df[line_visited_cols].sum(axis=1).astype(np.int8)
            )
            logger.info(f"  {line}: {len(line_visited_cols)} stations, "
                        f"avg visited={presence_df[f'n_stations_{line}'].mean():.2f}")

    # Total stations visited across ALL lines
    total_cols = [c for c in presence_df.columns if c.startswith('n_stations_L')]
    presence_df['n_stations_total'] = presence_df[total_cols].sum(axis=1)

    return presence_df

def compute_path_signature(presence_df: pd.DataFrame) -> pd.Series:
    """
    Create a string signature representing each part's unique path.
    Concatenates the names of all visited stations in sorted order.
    e.g. "L0_S0|L0_S1|L0_S12|L3_S29|L3_S33"
    """
    logger.info("\nBuilding path signatures (string concat of visited stations)...")

    visited_cols = [c for c in presence_df.columns if c.startswith('visited_')]

    def row_to_path(row):
        # Extract station name from column name: 'visited_L0_S0' → 'L0_S0'
        visited = [col[8:] for col in visited_cols if row[col] == 1]
        return '|'.join(sorted(visited)) if visited else 'no_stations'

    path_sig = presence_df[visited_cols].apply(row_to_path, axis=1)

    n_unique = path_sig.nunique()
    logger.info(f"  Unique paths found: {n_unique:,}")
    logger.info(f"  Most common path (first 80 chars): {path_sig.value_counts().index[0][:80]}")

    return path_sig

def compute_path_statistics(presence_df: pd.DataFrame,
                            path_sig: pd.Series,
                            response: pd.Series) -> pd.DataFrame:
    """
    Compute per path stats using the path signature as a group key.
    Features created:
        path_count - how many parts share this exact same path (rare path = unusual mfg = higher risk)
        path_failure_rate - fraction of parts on this path that failed (direct target encoding of path group)
        path_risk_tier - binned version: 0=rare, 1=common, 2=very_common (helps model generalize on unseen paths)

    groupby(path_sig) computes statistics per group.
    map() broadcasts group-level stats back to individual row level.
    """
    logger.info("\nComputing path-level statistics...")

    path_count = path_sig.map(path_sig.value_counts())
    logger.info(f"  Path count range: {path_count.min()} – {path_count.max():,}")
    logger.info(f"  Paths with count >= {MIN_PATH_COUNT}: "
                f"{(path_count >= MIN_PATH_COUNT).sum():,} parts "
                f"({(path_count >= MIN_PATH_COUNT).mean():.1%})")

    # path_failure_rate: target encoding of path
    # Build a lookup table {path_string: failure_rate}
    path_stats = pd.DataFrame({
        'path': path_sig.values,
        'response': response.values
    })
    # For paths with enough data, use real rate; else use global mean
    global_mean = response.mean()
    path_rate_map = (
        path_stats.groupby('path')['response'].agg(['mean', 'count'])
        .rename(columns={'mean': 'rate', 'count': 'cnt'})
    )
    # Apply smoothing: rare paths pull toward global mean
    # Formula: (count * group_rate + MIN_PATH_COUNT * global_mean) / (count + MIN_PATH_COUNT)
    # If only 3 parts took a path and all 3 failed (100%), we don't-
    # -fully trust that — we blend it with the global 0.58% base rate.

    path_rate_map['smoothed_rate'] = (
    (path_rate_map['cnt'] * path_rate_map['rate'] +
     MIN_PATH_COUNT * global_mean) /
    (path_rate_map['cnt'] + MIN_PATH_COUNT)
    )

    path_failure_rate = path_sig.map(path_rate_map['smoothed_rate'])
    logger.info(f"  Path failure rate range: "
                f"{path_failure_rate.min():.4f} – {path_failure_rate.max():.4f}")
    logger.info(f"  Global mean: {global_mean:.4f}")

    # path_risk_tier: binned path_count into 3 tiers
    # Rare paths (few parts) are unusual manufacturing events
    # pd.qcut divides into roughly equal-population bins
    path_risk_tier = pd.cut(
        path_count,
        bins=[0, 10, 100, float('inf')],
        labels=[0,1,2]  # 0=rare, 1=common, 2=very_common
    ).astype(np.int8)

    stats_df = pd.DataFrame({
        'path_count': path_count.values,
        'path_failure_rate': path_failure_rate.values,
        'path_risk_tier': path_risk_tier.values
    })

    return stats_df

def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: PATH FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info("""
    Strategy: Determine which stations each part visited (from nullness),
    then compute path-level statistics. Works for ALL parts including
    the 93% of singleton failures that have no chunk signal.
    """)

    # Load raw numerical data
    # We load the FULL raw numeric data (970 columns), NOT the
    # top-150 selected features. Why? Because we need to check nullness
    # across ALL station columns to determine the complete path.

    logger.info("\n1. Loading full raw numeric data...")
    df = pd.read_parquet(project_root / "data/processed/train_numeric.parquet")
    logger.info(f"   Shape: {df.shape}")
    memory_usage_report(df, "Raw Numeric")

    # Load response labels - needed for path failure_rate target encoding
    logger.info("\n   Response already present in numeric parquet ✅")
    logger.info(f"   Failures: {df['Response'].sum():,} ({df['Response'].mean():.4%})")

    # Extract station names
    logger.info("\n2. Extracting station structure...")
    feature_cols = [c for c in df.columns if c not in ['Id', 'Response']]
    station_cols = extract_station_names(feature_cols)

    # Log station breakdown by line
    for line in sorted(set(s.split('_')[0] for s in station_cols)):
        line_stations = [s for s in station_cols if s.startswith(line + '_')]
        logger.info(f"   {line}: {len(line_stations)} stations, "
                    f"{sum(len(station_cols[s]) for s in line_stations)} sensor columns")

    # Station presence flags
    # This is core transformation: 970 numeric columns -> ~ 52 binary flags
    # Each flag answers: "did this part visit this station?"
    logger.info("\n3. Computing station presence flags...")
    presence_df = compute_station_presence(df, station_cols)

    n_visit_cols = len([c for c in presence_df.columns if c.startswith('visited_')])
    logger.info(f"   Station presence features: {n_visit_cols}")
    logger.info(f"   Avg stations visited per part: "
                f"{presence_df[[c for c in presence_df.columns if c.startswith('visited_')]].sum(axis=1).mean():.1f}")

    # Line-level counts
    logger.info("\n4. Computing line-level station counts...")
    presence_df = compute_line_counts(presence_df, station_cols)

    # Path signature
    logger.info("\n5. Computing path signatures...")
    path_sig = compute_path_signature(presence_df)
    presence_df['path_signature'] = path_sig # just for debugging, remove for training

    # Path stats
    logger.info("\n6. Computing path statistics (target encoding)...")
    stats_df = compute_path_statistics(presence_df, path_sig, df['Response'])
    for col in stats_df.columns:
        presence_df[col] = stats_df[col].values

    # Summary
    feature_cols_final = [c for c in presence_df.columns if c not in ['Id', 'path_signature']]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PATH FEATURES SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Station presence flags:  {len([c for c in feature_cols_final if c.startswith('visited_')])}")
    logger.info(f"  Line station counts:     {len([c for c in feature_cols_final if c.startswith('n_stations_')])}")
    logger.info(f"  Path statistics:         {len([c for c in feature_cols_final if c.startswith('path_')])}")
    logger.info(f"  Total path features:     {len(feature_cols_final)}")
    memory_usage_report(presence_df, "Path Features")

    # Save
    # Drop path_signature before saving — it's a raw string, not a model feature.
    # The model gets path_count, path_failure_rate, path_risk_tier instead.
    output_path = project_root / "data/features/train_path_features.parquet"
    presence_df.drop(columns=['path_signature']).to_parquet(
        output_path, compression='snappy', index=False
    )
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"\n✅ Saved → {output_path} ({file_size:.2f} MB)")
    logger.info(f"\n🎯 NEXT: python notebooks/15_merge_path_features.py")

if __name__ == '__main__':
    main()

















