"""
Leak Feature Engineering - Using FULL numeric data from parquet

Data sources:
- data/processed/train_numeric.parquet (ALL 465 features, fast!)
- data/processed/train_date.parquet (for time-based ordering)
- Response from train_selected_top150.parquet

"Consecutive rows with duplicate features have correlated Response"

Expected MCC gain: +0.12 to +0.15 (massive!)

Three types of leaks:
1. Mathias's Leak: Row-level duplication detection
2. Station Leak: Per-station duplication (works best on L3_S29, L3_S30)
3. String Concatenation: Unique pattern counting

Q: What is a "leak" in this context?
A: A data property where consecutive products with identical measurements
   have highly correlated failure outcomes. It's not a traditional "leak"
   (test data in train), but rather a manufacturing reality: products
   produced together with identical specs tend to fail together.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib

from src.logger import setup_logger

logger = setup_logger(__name__)

def compute_start_time(df_date):
    """
    Compute the earliest timestamp each part appeared on the production line.

    Takes all date columns (station timestamps) and finds the minimum non-null value per row.
    This represents when the first part entered the line.
    """
    date_cols = [c for c in df_date.columns if c not in ['Id']]

    # Convert all date cols to numeric (coz they are stored as floats in parquet)
    # Take row-wise min, ignoring NaN
    start_time = df_date[date_cols].min(axis=1)

    return pd.DataFrame({'Id': df_date['Id'], 'start_time': start_time})



def fast_hash(row):
    """Hash row values for duplicate detection."""
    row_str = '_'.join(row.astype(str).values)
    return hashlib.md5(row_str.encode()).hexdigest()


def create_mathias_leak(df_numeric):
    """
    LEAK #1: Mathias's Leak (Row Duplication Detection)

    Approach:
    - Hash each row's numeric features
    - Detect consecutive rows with identical hashes
    - Create "chunks" of consecutive duplicates
    - Extract chunk statistics

    Expected gain: +0.10 to +0.15 MCC

    Consecutive duplicate detection on FULL 465 numeric features.
    More accurate than using only top 150.

    """
    logger.info("=" * 60)
    logger.info("LEAK #1: CONSECUTIVE DUPLICATES (ALL 465 FEATURES)")
    logger.info("=" * 60)

    feature_cols = [c for c in df_numeric.columns if c not in ['Id', 'Response', 'start_time']]

    logger.info(f"\nProcessing {len(df_numeric):,} rows × {len(feature_cols)} features...")

    # Hash rows
    logger.info("  Hashing rows (this takes ~5 mins for 465 features)...")
    tqdm.pandas(desc="Hashing")
    df_numeric['row_hash'] = df_numeric[feature_cols].progress_apply(fast_hash, axis=1)

    # Detect consecutive duplicates
    logger.info("  Detecting consecutive duplicates...")
    df_numeric['is_duplicate'] = (
            df_numeric['row_hash'] == df_numeric['row_hash'].shift(1)
    ).astype(np.int8)

    # Create chunks
    logger.info("  Creating chunks...")
    df_numeric['chunk_id'] = (~df_numeric['is_duplicate'].astype(bool)).cumsum()

    # Chunk statistics
    logger.info("  Computing chunk stats...")
    chunk_stats = df_numeric.groupby('chunk_id').agg({
        'Id': 'count',
    }).rename(columns={'Id': 'chunk_size'})

    df_numeric = df_numeric.merge(chunk_stats, on='chunk_id', how='left')
    df_numeric['chunk_rank_asc'] = df_numeric.groupby('chunk_id').cumcount() + 1
    df_numeric['chunk_rank_desc'] = df_numeric['chunk_size'] - df_numeric['chunk_rank_asc'] + 1

    # Response correlation within chunks
    if 'Response' in df_numeric.columns:
        logger.info("  Computing response correlation within chunks...")
        df_numeric['prev_response_in_chunk'] = df_numeric.groupby('chunk_id')['Response'].shift(1)
        df_numeric['next_response_in_chunk'] = df_numeric.groupby('chunk_id')['Response'].shift(-1)
        df_numeric['prev_response_in_chunk'].fillna(-1, inplace=True)
        df_numeric['next_response_in_chunk'].fillna(-1, inplace=True)

    # Stats
    dup_count = df_numeric['is_duplicate'].sum()
    dup_pct = (dup_count / len(df_numeric)) * 100
    unique_chunks = df_numeric['chunk_id'].nunique()
    avg_chunk_size = df_numeric['chunk_size'].mean()
    max_chunk_size = df_numeric['chunk_size'].max()

    logger.info(f"\n✅ Chunk Distribution (TOP 10):")
    logger.info(df_numeric['chunk_size'].value_counts().head(10).to_string())

    logger.info(f"\n✅ Results:")
    logger.info(f"  Duplicates:    {dup_count:,} ({dup_pct:.2f}%)")
    logger.info(f"  Unique chunks: {unique_chunks:,}")
    logger.info(f"  Avg chunk size: {avg_chunk_size:.3f}")
    logger.info(f"  Max chunk size: {max_chunk_size:,}")

    # PASS/FAIL check — if still broken you'll know immediately
    # if dup_pct < 5.0:
    #     logger.warning("⚠️  WARNING: Less than 5% duplicates detected.")
    #     logger.warning("   Sort order may still be incorrect.")
    #     logger.warning("   Check that start_time values are non-null and varied.")
    # else:
    #     logger.info(f"  ✅ Duplicate rate looks healthy ({dup_pct:.1f}%)")

    leak_features = [
        'Id', 'is_duplicate', 'chunk_id', 'chunk_size',
        'chunk_rank_asc', 'chunk_rank_desc'
    ]

    if 'Response' in df_numeric.columns:
        leak_features.extend(['prev_response_in_chunk', 'next_response_in_chunk'])

    return df_numeric[leak_features]


def create_station_leaks(df_numeric):
    """
    LEAK #2: Per-Station Duplication Detection

    Approach:
    - For each station, hash its features
    - Count duplicate hashes within station
    - Works best on stations with few NAs (L3_S29, L3_S30)

    Expected gain: +0.05 to +0.08 MCC

    Features created (per station):
    - {station}_dup_count: Count of duplicate patterns in station
    - {station}_dup_rank: Rank of occurrence of this pattern

    Focus on key stations (S29, S30, S33, S12, S0).
    """
    logger.info("\n" + "=" * 60)
    logger.info("LEAK #2: STATION-LEVEL DUPLICATES")
    logger.info("=" * 60)

    # Extract station names
    all_cols = df_numeric.columns
    stations = set()
    for col in all_cols:
        if col in ['Id', 'Response']:
            continue
        parts = col.split('_')
        if len(parts) >= 3:
            station = f"{parts[0]}_{parts[1]}"
            stations.add(station)

    stations = sorted(stations)

    # Focus on key stations (winners said these work best)
    priority_stations = [s for s in stations if any(key in s for key in ['S29', 'S30', 'S33', 'S12', 'S0'])]

    logger.info(f"\nProcessing {len(priority_stations)} priority stations...")

    leak_df = df_numeric[['Id']].copy()

    for station in tqdm(priority_stations, desc="Stations"):
        station_cols = [c for c in all_cols if c.startswith(station + '_')]

        if len(station_cols) == 0:
            continue

        # Only consider rows where this station has real data
        # At least one col should be non NaN to be a valid row.
        has_data_mask = df_numeric[station_cols].notna().any(axis=1)

        if has_data_mask.sum() < 10:
            continue

        # Hash only valid rows
        station_hashes = df_numeric.loc[has_data_mask, station_cols].apply(fast_hash, axis=1)

        # Count duplicates only among valid rows
        hash_counts = station_hashes.map(station_hashes.value_counts())

        # Assign counts back - rows that never visited this station get 0
        leak_df[f'{station}_dup_count'] = 0
        leak_df.loc[has_data_mask, f'{station}_dup_count'] = hash_counts.values

        # Rank among valid rows only - non-visitors get rank 0
        leak_df[f'{station}_dup_rank'] = 0
        leak_df.loc[has_data_mask, f'{station}_dup_rank'] = (
            df_numeric.loc[has_data_mask].groupby(station_hashes).cumcount() + 1
        ).values


    # Sum across stations
    dup_cols = [c for c in leak_df.columns if '_dup_count' in c]
    leak_df['total_station_dups'] = leak_df[dup_cols].sum(axis=1)

    logger.info(f"\n✅ Results:")
    logger.info(f"  Features created: {len(dup_cols) * 2}")
    logger.info(f"  Avg station dups: {leak_df['total_station_dups'].mean():.2f}")

    return leak_df


def create_string_concat_leaks(df_numeric):
    """
    LEAK #3: String Concatenation + Unique Counting

    FIX: Only concatenate col_name=value for sensors that actually fired.
    NaN columns are completely excluded from the string.

    This means two rows only match if they measured the SAME sensors
    with the SAME values. NaN patterns no longer cause false matches.
    """

    logger.info("\n" + "=" * 60)
    logger.info("LEAK #3: STRING CONCATENATION")
    logger.info("=" * 60)

    # Extract stations
    all_cols = df_numeric.columns
    stations = set()
    for col in all_cols:
        if col in ['Id', 'Response', 'start_time']:
            continue
        parts = col.split('_')
        if len(parts) >= 3:
            station = f"{parts[0]}_{parts[1]}"
            stations.add(station)

    stations = sorted(stations)

    # Focus on priority stations (all stations would be too slow)
    priority_stations = [s for s in stations if
                         any(key in s for key in ['S29', 'S30', 'S33', 'S32', 'S35', 'S12', 'S0'])]

    leak_df = df_numeric[['Id']].copy()

    logger.info(f"\nProcessing {len(priority_stations)} priority stations...")

    for station in tqdm(priority_stations, desc="Concat"):
        station_cols = [c for c in all_cols if c.startswith(station + '_')]

        if len(station_cols) == 0:
            continue

        # Filter rows where this station has real data
        has_data_mask = df_numeric[station_cols].notna().any(axis=1)

        if has_data_mask.sum() < 10:
            continue

        station_data = df_numeric.loc[has_data_mask, station_cols]

        # Concat only valid rows
        # Only include col_name=value for sensors that actually fired
        # "nan" values are completely excluded from the string
        def concat_non_null(row):
            parts = [f"{col}={val:.6f}" for col, val in row.items() if pd.notna(val)]
            return '|'.join(parts) if parts else 'empty'

        station_concat = station_data.apply(concat_non_null, axis=1)

        n_valid = has_data_mask.sum()
        n_unique = station_concat.nunique()
        uniqueness_ratio = n_unique / n_valid

        if uniqueness_ratio < 0.01:
            logger.info(f"  Skipping {station}: {n_unique:,} unique / {n_valid:,} rows "
                        f"(ratio={uniqueness_ratio:.4f}) — constant/near-constant station, not discriminative")
            continue

        # Count occurences
        concat_counts = station_concat.map(station_concat.value_counts())

        # Back-fill - non-visitors get 0
        leak_df[f'{station}_concat_count'] = 0
        leak_df.loc[has_data_mask, f'{station}_concat_count'] = concat_counts.values

    # Sum
    concat_cols = [c for c in leak_df.columns if '_concat_count' in c]
    for col in concat_cols:
        leak_df[f'{col}_log'] = np.log1p(leak_df[col])

    log_cols = [f'{c}_log' for c in concat_cols]

    leak_df['total_concat_count'] = leak_df[log_cols].sum(axis=1)

    # Drop intermediate log columns — model only needs the final sum + per-station raw counts
    leak_df.drop(columns=log_cols, inplace=True)

    logger.info(f"\n✅ Results:")
    logger.info(f"  Features created: {len(concat_cols) + 1}")
    logger.info(f"  Avg concat count: {leak_df['total_concat_count'].mean():.2f}")

    # PASS/FAIL check
    avg = leak_df['total_concat_count'].mean()
    if avg > 20:
        logger.warning(f"⚠️  Avg total_concat_count {avg:.1f} still high — check station distributions")
    else:
        logger.info(f"  ✅ total_concat_count healthy (log-normalized avg={avg:.2f})")


    return leak_df


def main():
    logger.info("=" * 60)
    logger.info("LEAK FEATURE ENGINEERING - FULL NUMERIC DATA")
    logger.info("=" * 60)

    # Load FULL numeric from parquet (ALL 465 features!)
    logger.info("\nLoading full numeric data...")
    df_numeric = pd.read_parquet(project_root/"data/processed/train_numeric.parquet")
    logger.info(f"  Shape: {df_numeric.shape}")
    logger.info(f"  Memory: {df_numeric.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Load date data and compute start_time
    logger.info("\nLoading date data to compute manufacturing order...")
    df_date = pd.read_parquet(project_root/"data/processed/train_date.parquet")
    logger.info(f"  Date shape: {df_date.shape}")

    df_start_time = compute_start_time(df_date)
    del df_date # large data file - need to free RAM to accomodate more space
    logger.info(f"  start_time nulls: {df_start_time['start_time'].isna().sum():,}")

    # Merge start_time into numeric
    df_numeric = df_numeric.merge(df_start_time, on='Id', how='left')

    # Critical SORT - manufacturing sequence order
    logger.info("\nSorting by start_time (manufacturing sequence)...")
    df_numeric = df_numeric.sort_values(
        by=['start_time', 'Id'], # Start time first, Id as tiebreaker
        ascending=True,
        na_position='last'
    ).reset_index(drop=True)
    logger.info(" Sort complete.")

    # Verify: check a sample to confirm order makes sense
    logger.info(f"  First 3 start_times: {df_numeric['start_time'].head(3).values}")
    logger.info(f"  Last 3 start_times:  {df_numeric['start_time'].tail(3).values}")


    # Load and merge Response
    logger.info("\nLoading Response...")
    df_response = pd.read_parquet(project_root/"data/features/train_selected_top150.parquet")[['Id', 'Response']]

    df_numeric = df_numeric.merge(df_response, on='Id', how='left')
    logger.info(f"  Merged shape: {df_numeric.shape}")

    # Create leaks
    leak1 = create_mathias_leak(df_numeric)
    leak2 = create_station_leaks(df_numeric)
    leak3 = create_string_concat_leaks(df_numeric)

    # Merge all leaks
    logger.info("\n" + "=" * 60)
    logger.info("MERGING ALL LEAKS")
    logger.info("=" * 60)

    leak_all = leak1.merge(leak2, on='Id', how='left')
    leak_all = leak_all.merge(leak3, on='Id', how='left')

    logger.info(f"\nTotal leak features: {leak_all.shape[1] - 1}")

    # Save
    output_path = project_root/"data/features/train_leak_features.parquet"
    logger.info(f"\nSaving to {output_path}...")
    leak_all.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! ({file_size:.2f} MB)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nFeatures: {leak_all.shape[1] - 1}")
    logger.info(f"  Leak #1 (Mathias): 7 features")
    logger.info(f"  Leak #2 (Station): {len([c for c in leak_all.columns if '_dup_' in c])} features")
    logger.info(f"  Leak #3 (Concat): {len([c for c in leak_all.columns if '_concat_' in c])} features")
    logger.info(f"\n🎯 NEXT: python notebooks/08_merge_leak_features.py")


if __name__ == "__main__":
    main()
