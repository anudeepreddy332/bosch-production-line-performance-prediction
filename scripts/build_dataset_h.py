from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.training.cv import ChunkCVConfig, assign_fold_ids, make_chunk_aware_splits

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"

BASELINE_COLUMNS = [
    "start_time",
    "duration",
    "feature_mean",
    "records_last_1hr",
    "records_last_24hr",
    "density_ratio",
    "chunk_id",
    "chunk_size",
]


def parse_signature(signature: str) -> tuple[str, ...]:
    if pd.isna(signature) or signature == "__none__":
        return tuple()
    return tuple(token for token in str(signature).split("|") if token)


def transitions_from_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if len(tokens) < 2:
        return tuple()
    return tuple(f"{tokens[i]}>{tokens[i + 1]}" for i in range(len(tokens) - 1))


def pairs_from_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if len(tokens) < 2:
        return tuple()
    return tuple(f"{a}&{b}" for a, b in combinations(tokens, 2))


def _mean_max_std(values: list[float], default_mean: float) -> tuple[float, float, float]:
    if not values:
        return float(default_mean), float(default_mean), 0.0
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.max()), float(arr.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dataset H with transition/station risk features.")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    baseline_path = FEATURES_DIR / "dataset_baseline.parquet"
    path_meta_path = FEATURES_DIR / "path_metadata.parquet"

    if not baseline_path.exists() or not path_meta_path.exists():
        raise FileNotFoundError("Missing baseline artifacts. Run scripts/build_dataset_baseline.py first.")

    baseline_df = pd.read_parquet(baseline_path)
    meta_df = pd.read_parquet(path_meta_path)

    df = baseline_df.merge(
        meta_df[["Id", "path_signature", "path_count"]],
        on="Id",
        how="inner",
        validate="one_to_one",
    )

    unique_signatures = pd.Index(df["path_signature"].fillna("__none__").astype(str).unique())
    sig_tokens = {sig: parse_signature(sig) for sig in unique_signatures}
    sig_transitions = {sig: transitions_from_tokens(tokens) for sig, tokens in sig_tokens.items()}
    sig_pairs = {sig: pairs_from_tokens(tokens) for sig, tokens in sig_tokens.items()}

    # Non-target pair co-occurrence counts can be computed once globally.
    pair_global_count: defaultdict[str, int] = defaultdict(int)
    signature_freq = df["path_signature"].value_counts(dropna=False)
    for sig, freq in signature_freq.items():
        sig_key = "__none__" if pd.isna(sig) else str(sig)
        for pair in sig_pairs.get(sig_key, tuple()):
            pair_global_count[pair] += int(freq)

    pair_stat_map: dict[str, tuple[float, float, float]] = {}
    for sig in unique_signatures:
        pair_values = [float(pair_global_count[p]) for p in sig_pairs[sig] if p in pair_global_count]
        pair_stat_map[sig] = _mean_max_std(pair_values, default_mean=0.0)

    n = len(df)
    global_mean = float(df["Response"].mean())

    transition_mean = np.full(n, global_mean, dtype=np.float32)
    transition_max = np.full(n, global_mean, dtype=np.float32)
    transition_std = np.zeros(n, dtype=np.float32)
    station_risk_mean = np.full(n, global_mean, dtype=np.float32)
    pair_cooccur_mean = np.zeros(n, dtype=np.float32)
    pair_cooccur_max = np.zeros(n, dtype=np.float32)
    pair_cooccur_std = np.zeros(n, dtype=np.float32)

    cv_cfg = ChunkCVConfig(n_splits=args.n_splits, random_state=42, shuffle=True)
    splits = make_chunk_aware_splits(df, target_col="Response", group_col="chunk_id", config=cv_cfg)
    fold_ids = assign_fold_ids(n_rows=n, splits=splits)

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        tr = df.iloc[train_idx]
        va = df.iloc[valid_idx]

        sig_stats = tr.groupby("path_signature")["Response"].agg(["sum", "count"])

        station_sum: defaultdict[str, float] = defaultdict(float)
        station_cnt: defaultdict[str, int] = defaultdict(int)
        trans_sum: defaultdict[str, float] = defaultdict(float)
        trans_cnt: defaultdict[str, int] = defaultdict(int)

        for sig, row in sig_stats.iterrows():
            sig_key = "__none__" if pd.isna(sig) else str(sig)
            y_sum = float(row["sum"])
            y_cnt = int(row["count"])

            for station in sig_tokens.get(sig_key, tuple()):
                station_sum[station] += y_sum
                station_cnt[station] += y_cnt

            for trans in sig_transitions.get(sig_key, tuple()):
                trans_sum[trans] += y_sum
                trans_cnt[trans] += y_cnt

        station_rate = {k: station_sum[k] / station_cnt[k] for k in station_sum if station_cnt[k] > 0}
        trans_rate = {k: trans_sum[k] / trans_cnt[k] for k in trans_sum if trans_cnt[k] > 0}

        valid_sig_features: dict[str, tuple[float, float, float, float]] = {}
        for sig in va["path_signature"].fillna("__none__").astype(str).unique():
            stations = sig_tokens.get(sig, tuple())
            transitions = sig_transitions.get(sig, tuple())

            station_values = [float(station_rate.get(s, global_mean)) for s in stations]
            trans_values = [float(trans_rate.get(t, global_mean)) for t in transitions]

            tr_mean, tr_max, tr_std = _mean_max_std(trans_values, default_mean=global_mean)
            st_mean, _, _ = _mean_max_std(station_values, default_mean=global_mean)

            valid_sig_features[sig] = (tr_mean, tr_max, tr_std, st_mean)

        valid_sig = va["path_signature"].fillna("__none__").astype(str)
        fold_transition_mean = valid_sig.map(lambda s: valid_sig_features[s][0]).to_numpy(dtype=np.float32)
        fold_transition_max = valid_sig.map(lambda s: valid_sig_features[s][1]).to_numpy(dtype=np.float32)
        fold_transition_std = valid_sig.map(lambda s: valid_sig_features[s][2]).to_numpy(dtype=np.float32)
        fold_station_mean = valid_sig.map(lambda s: valid_sig_features[s][3]).to_numpy(dtype=np.float32)

        transition_mean[valid_idx] = fold_transition_mean
        transition_max[valid_idx] = fold_transition_max
        transition_std[valid_idx] = fold_transition_std
        station_risk_mean[valid_idx] = fold_station_mean

        pair_cooccur_mean[valid_idx] = valid_sig.map(lambda s: pair_stat_map[s][0]).to_numpy(dtype=np.float32)
        pair_cooccur_max[valid_idx] = valid_sig.map(lambda s: pair_stat_map[s][1]).to_numpy(dtype=np.float32)
        pair_cooccur_std[valid_idx] = valid_sig.map(lambda s: pair_stat_map[s][2]).to_numpy(dtype=np.float32)

        logger.info(
            "Dataset H fold=%d train=%d valid=%d", fold_idx, len(train_idx), len(valid_idx)
        )

    out = df[["Id", "Response", *BASELINE_COLUMNS]].copy()
    out["transition_fail_rate_mean"] = transition_mean
    out["transition_fail_rate_max"] = transition_max
    out["transition_fail_rate_std"] = transition_std
    out["station_risk_mean"] = station_risk_mean
    out["path_count"] = pd.to_numeric(df["path_count"], errors="coerce").fillna(1).astype(np.int32)
    out["pair_cooccur_mean"] = pair_cooccur_mean
    out["pair_cooccur_max"] = pair_cooccur_max
    out["pair_cooccur_std"] = pair_cooccur_std
    out["cv_fold"] = fold_ids.astype(np.int16)

    output_path = FEATURES_DIR / "dataset_h.parquet"
    out.to_parquet(output_path, index=False)

    logger.info("Saved Dataset H: %s rows=%d", output_path, len(out))


if __name__ == "__main__":
    main()
