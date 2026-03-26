"""
Phase 5 Script 20: Temporal Features

From 1st place winners:
  MeanTimeDiff since last 1/5/10 failures  → backward-looking, clean
  MeanTimeDiff till next 1/5/10 failures   → forward-looking, leaky (same as path_failure_rate)
  StationTimeDiff (inter-line gaps)        → L0→L1, L1→L2, L2→L3 transition times
  Records in same 6-min window             → simultaneous production batch size

ELI5:
  since_last_K: "how long ago did the last K failures happen?"
                If failures just happened 10 minutes ago, current part is at higher risk
                (same machine state, same batch of materials)
  till_next_K:  "how soon until the next K failures happen?"
                Backward-filled from future — leaky but very strong signal
  StationTimeDiff: "how long did this part sit waiting between production lines?"
                   Long waits between stations = held for inspection = high risk
  records_same_6min: "how many parts were processed at the exact same time?"
                     High batch sizes may correlate with quality issues

Computation strategy:
  All MeanTimeDiff uses vectorized numpy searchsorted + cumsum — O(n) no Python loops
  StationTimeDiff uses already-computed per-line start/end times from script 17

Output: data/features/train_temporal_features.parquet (~15 features)
"""
import gc
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)


def compute_meantimedf_features(end_times: np.ndarray,
                                 responses: np.ndarray,
                                 ks: list = [1, 5, 10]) -> pd.DataFrame:
    """
    Vectorized computation of MeanTimeDiff since/till last K failures.

    ELI5 of the math:
      Sort all parts by end_time.
      For each part at time T:
        - Find all failures that ended BEFORE T using binary search (searchsorted)
        - Take the last K of those failures
        - Feature = T - mean(those K failure times)
      Repeat in reverse for till_next.

    Key trick: use cumulative sum of failure times so mean of any slice
               is (cumsum[end] - cumsum[start]) / K  — no loops needed.

    Sample before (3 parts, 1 failure):
      end_time = [10, 20, 30], response = [0, 1, 0]

    Sample after (since_last_1):
      part at t=10: no failures before → NaN
      part at t=30: failure at t=20 → 30 - 20 = 10
    """
    # Sort by end_time — critical for binary search to work
    sort_idx = np.argsort(end_times, kind='stable')
    end_times_sorted = end_times[sort_idx]
    responses_sorted = responses[sort_idx]

    # Extract failure times in sorted order
    failure_mask = responses_sorted == 1
    failure_times = end_times_sorted[failure_mask]
    n_failures = len(failure_times)
    logger.info(f"  Total failures: {n_failures:,}")
    logger.info(f"  Failure time range: {failure_times.min():.2f} – {failure_times.max():.2f}")

    # Cumulative sums for vectorized mean computation
    # failure_cumsum[i] = sum of failure_times[0:i]
    failure_cumsum = np.concatenate([[0.0], np.cumsum(failure_times)])

    results = {}

    # ── SINCE LAST K (backward-looking, clean) ────────────────────
    logger.info("\n  Computing since_last_K features...")

    # For each part at time T: how many failures had end_time < T?
    # idx_before[i] = number of failures strictly before end_times_sorted[i]
    idx_before = np.searchsorted(failure_times, end_times_sorted, side='left')
    # Note: side='left' means: failures at exactly T are NOT counted as "before"

    for K in ks:
        col = f'mean_timediff_since_last_{K}'
        result = np.full(len(end_times_sorted), np.nan)

        has_k = idx_before >= K
        valid_idx = idx_before[has_k]

        # Mean of last K failures = (cumsum[idx] - cumsum[idx-K]) / K
        mean_last_k = (failure_cumsum[valid_idx] - failure_cumsum[valid_idx - K]) / K

        # TimeDiff = current_time - mean_failure_time (positive = failures were in the past)
        result[has_k] = end_times_sorted[has_k] - mean_last_k
        results[col] = result
        logger.info(f"    {col}: non-null={has_k.sum():,} "
                    f"| mean={np.nanmean(result):.1f} "
                    f"| median={np.nanmedian(result):.1f}")

    # ── TILL NEXT K (forward-looking, leaky) ─────────────────────
    logger.info("\n  Computing till_next_K features (leaky — uses future labels)...")

    # For each part at time T: how many failures had end_time > T?
    # idx_after[i] = index of first failure AFTER end_times_sorted[i]
    idx_after = np.searchsorted(failure_times, end_times_sorted, side='right')
    # failures_after[i] = n_failures - idx_after[i]

    for K in ks:
        col = f'mean_timediff_till_next_{K}'
        result = np.full(len(end_times_sorted), np.nan)

        has_k_next = (n_failures - idx_after) >= K
        valid_idx_after = idx_after[has_k_next]

        # Mean of next K failures = (cumsum[idx+K] - cumsum[idx]) / K
        mean_next_k = (failure_cumsum[valid_idx_after + K] -
                       failure_cumsum[valid_idx_after]) / K

        # TimeDiff = mean_failure_time - current_time (positive = failures are in the future)
        result[has_k_next] = mean_next_k - end_times_sorted[has_k_next]
        results[col] = result
        logger.info(f"    {col}: non-null={has_k_next.sum():,} "
                    f"| mean={np.nanmean(result):.1f} "
                    f"| median={np.nanmedian(result):.1f}")

    # Build result df in sorted order, then unsort back to original order
    df_results = pd.DataFrame(results)
    # Unsort: put rows back in original order
    unsort_idx = np.argsort(sort_idx, kind='stable')
    df_results = df_results.iloc[unsort_idx].reset_index(drop=True)

    return df_results


def compute_station_timediff_features(df_date: pd.DataFrame) -> pd.DataFrame:
    """
    Inter-line transition gaps: time a part waited between consecutive lines.

    ELI5: Part finishes Line 0 at time 100. It enters Line 1 at time 105.
          Gap = 5 units = 30 minutes. Long gaps suggest held for inspection.

    Uses already-computed per-line start/end times from script 17.
    No row-wise iteration needed — pure vectorized subtraction.

    Sample before:
      end_time_L0=100, start_time_L1=105 → gap_L0_to_L1 = 5

    Sample after cols: gap_L0_to_L1, gap_L1_to_L2, gap_L2_to_L3
    Plus: min/max/mean/std of available gaps per part
    """
    features = pd.DataFrame({'Id': df_date['Id']})

    # Inter-line gaps
    for from_line, to_line in [('L0', 'L1'), ('L1', 'L2'), ('L2', 'L3')]:
        col = f'gap_{from_line}_to_{to_line}'
        raw_gap = df_date[f'start_time_{to_line}'] - df_date[f'end_time_{from_line}']
        # Keep NaN where either line wasn't visited — don't force-compute invalid gaps
        features[col] = raw_gap.where(
            df_date[f'start_time_{to_line}'].notna() & df_date[f'end_time_{from_line}'].notna()
        )

    # Aggregate across the 3 gaps per part
    gap_cols = [c for c in features.columns if c.startswith('gap_')]
    gap_data = features[gap_cols].values

    features['station_timediff_min'] = np.nanmin(gap_data, axis=1)
    features['station_timediff_max'] = np.nanmax(gap_data, axis=1)
    features['station_timediff_mean'] = np.nanmean(gap_data, axis=1)
    features['station_timediff_std'] = np.nanstd(gap_data, axis=1)
    features['station_timediff_count'] = (~np.isnan(gap_data)).sum(axis=1).astype('int8')

    logger.info(f"  gap_L0_to_L1:  mean={features['gap_L0_to_L1'].mean():.4f}, "
                f"null={features['gap_L0_to_L1'].isna().sum():,}")
    logger.info(f"  gap_L1_to_L2:  mean={features['gap_L1_to_L2'].mean():.4f}, "
                f"null={features['gap_L1_to_L2'].isna().sum():,}")
    logger.info(f"  gap_L2_to_L3:  mean={features['gap_L2_to_L3'].mean():.4f}, "
                f"null={features['gap_L2_to_L3'].isna().sum():,}")

    return features


def compute_same_window_count(start_times: np.ndarray, window: float = 0.01) -> np.ndarray:
    sort_idx = np.argsort(start_times, kind='stable')
    times_sorted = start_times[sort_idx]
    half = window / 2.0

    # Binary search: for each part, find index of first part AFTER window end
    right_idx = np.searchsorted(times_sorted, times_sorted + half, side='right')
    # Binary search: for each part, find index of first part AT/AFTER window start
    left_idx  = np.searchsorted(times_sorted, times_sorted - half, side='left')
    # Count = parts in window, minus self
    counts = (right_idx - left_idx - 1).astype('int32')

    result = np.empty(len(start_times), dtype='int32')
    result[sort_idx] = counts
    return result



def main():
    logger.info("=" * 60)
    logger.info("PHASE 5 — SCRIPT 20: TEMPORAL FEATURES")
    logger.info("=" * 60)

    # ── 1. Load inputs ────────────────────────────────────────────
    logger.info("\n1. Loading inputs...")

    df_date = pd.read_parquet(
        project_root / "data/features/train_date_features.parquet",
        columns=['Id', 'start_time', 'end_time',
                 'start_time_L0', 'end_time_L0',
                 'start_time_L1', 'end_time_L1',
                 'start_time_L2', 'end_time_L2',
                 'start_time_L3', 'end_time_L3']
    )
    logger.info(f"  Date features shape: {df_date.shape}")

    df_labels = pd.read_parquet(
        project_root / "data/features/train_phase4_features.parquet",
        columns=['Id', 'Response']
    )
    logger.info(f"  Labels shape: {df_labels.shape}")
    logger.info(f"  Failures: {df_labels['Response'].sum():,}")

    # Merge end_time + Response for MeanTimeDiff computation
    df = df_labels.merge(df_date[['Id', 'start_time', 'end_time']], on='Id', how='left')
    logger.info(f"  Merged shape: {df.shape}")
    logger.info(f"  Null end_time: {df['end_time'].isna().sum()}")

    # ── 2. MeanTimeDiff features ──────────────────────────────────
    logger.info("\n2. Computing MeanTimeDiff features...")
    timediff_df = compute_meantimedf_features(
        end_times=df['end_time'].values,
        responses=df['Response'].values,
        ks=[1, 5, 10]
    )
    timediff_df.insert(0, 'Id', df['Id'].values)
    logger.info(f"  MeanTimeDiff shape: {timediff_df.shape}")

    # ── 4. Records in same 6-min window ──────────────────────────
    logger.info("\n4. Computing records in same 6-min window...")
    df['records_same_6min'] = compute_same_window_count(
        start_times=df['start_time'].values,
        window=0.01
    )
    logger.info(f"  records_same_6min: mean={df['records_same_6min'].mean():.1f} "
                f"| max={df['records_same_6min'].max()}")
    same_window_df = df[['Id', 'records_same_6min']].copy()

    # ── 5. Merge all temporal features ───────────────────────────
    logger.info("\n5. Merging all temporal features...")
    features = timediff_df.merge(same_window_df, on='Id', how='left')
    logger.info(f"  Final shape: {features.shape}")

    # ── 6. Fill nulls ─────────────────────────────────────────────
    logger.info("\n6. Filling nulls...")
    null_before = features.isnull().sum().sum()
    logger.info(f"  Total nulls before fill: {null_before:,}")
    for col in features.columns:
        if col == 'Id':
            continue
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())
    logger.info(f"  Nulls after fill: {features.isnull().sum().sum()}")

    # ── 7. Summary ────────────────────────────────────────────────
    feature_cols = [c for c in features.columns if c != 'Id']
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL FEATURES SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  MeanTimeDiff since_last:  3 features")
    logger.info(f"  MeanTimeDiff till_next:   3 features (leaky)")
    logger.info(f"  records_same_6min:        1 feature")
    logger.info(f"  Total features:           {len(feature_cols)}")
    logger.info(f"\n  Feature list:")
    for col in feature_cols:
        logger.info(f"    {col}")

    del df_labels, timediff_df, same_window_df
    gc.collect()

    # ── 8. Save ───────────────────────────────────────────────────
    output_path = project_root / "data/features/train_temporal_features.parquet"
    features.to_parquet(output_path, compression='snappy', index=False)
    file_size = output_path.stat().st_size / 1024 ** 2
    logger.info(f"\n✅ Saved → {output_path} ({file_size:.2f} MB)")
    logger.info(f"\n🎯 NEXT: python notebooks/21_merge_phase5_features.py")


if __name__ == '__main__':
    main()
