"""E2: Out-of-time durability of the presence signal.

Forward-chaining temporal split: train on chunks 0-82 (~70% by time, start_time <= ~1157),
validate on chunks 83-118 (~30% by time, 353,747 rows, 1,392 positives).

Three arms evaluated (identical features to E1a/b/c decomposition):
  dataset_h  — baseline (16 routing features)
  E1b        — presence-only (16 routing + 50 binary station flags, 66 total)
  E1c        — value-only (16 routing + 50 station means, median-filled, 66 total)

Pre-registered success bar (DR-007 §5):
  E1b retains >= 70-80% of in-CV uplift (+0.00804) out-of-time AND ordering holds.
Failure: gain collapses or sign-flips OOT.

Single train/test split per arm (no k-fold for OOT); n_estimators=700 fixed (no early
stopping — using the test set for early stopping would bias the OOT MCC).

Returns all evidence to Opus for interpretation. Sonnet does not interpret.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import matthews_corrcoef

from src.features.dataset_h_pipeline import DATASET_H_FEATURE_COLS
from src.logger import setup_logger
from src.training.modeling import compute_data_fingerprint, search_best_mcc_threshold

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"

# ── In-CV reference numbers (from E1a/b/c decomposition, DR-006) ──────────────
_INCV = {
    "dataset_h":  {"oof_mcc": 0.15337, "fold_mccs": [0.13813, 0.13536, 0.18114, 0.15277, 0.18498]},
    "dataset_e1b": {"oof_mcc": 0.16141, "fold_mccs": [0.15217, 0.14110, 0.19090, 0.15333, 0.18973]},
    "dataset_e1c": {"oof_mcc": 0.15977, "fold_mccs": [0.15184, 0.14223, 0.19375, 0.14763, 0.18572]},
}

# ── Temporal split boundary ──────────────────────────────────────────────────
# chunk_id <= 82: 830,000 rows, 5,487 pos (0.661%)
# chunk_id >= 83: 353,747 rows, 1,392 pos (0.394%)
# No chunk straddles the boundary (verified before run).
TRAIN_MAX_CHUNK = 82


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _station_mean_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("sensor_mean_"))


def _build_e1b_features(df: pd.DataFrame, station_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Presence-only: 50 binary station presence flags, no measurement values.
    Identical to E1b construction in train_dataset_e1_decomposition.py."""
    presence_names = [f"sensor_present_{c.removeprefix('sensor_mean_')}" for c in station_cols]
    out = df[["Id", "Response"] + DATASET_H_FEATURE_COLS].copy()
    for pname, mcol in zip(presence_names, station_cols):
        out[pname] = df[mcol].notna().astype(np.uint8)
    cols = DATASET_H_FEATURE_COLS + presence_names
    return cols, out


def _build_e1c_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame, station_cols: list[str]
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Value-only: station means with NaN filled by per-station median.

    Median is computed from TRAINING rows only (what production would have
    at scoring time). Same mechanism as E1c decomposition except fill stats
    are anchored to the train split — the global-vs-train distinction is
    negligible for a non-target statistic, but the train-only version is
    the more principled OOT estimate.
    """
    def _fill(df_src: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
        out = df_src[["Id", "Response"] + DATASET_H_FEATURE_COLS].copy()
        for col in station_cols:
            median_val = df_ref[col].median()
            fill_val = median_val if not np.isnan(median_val) else 0.0
            out[col] = df_src[col].fillna(fill_val).astype(np.float32)
        return out

    train_out = _fill(df_train, df_train)
    test_out = _fill(df_test, df_train)   # fill with train median
    cols = DATASET_H_FEATURE_COLS + station_cols
    return cols, train_out, test_out


def _train_and_eval(
    arm_name: str,
    feature_cols: list[str],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    """Train a single LightGBM on df_train, evaluate on df_test."""
    logger.info("=== ARM %s: %d features, train=%d test=%d ===",
                arm_name, len(feature_cols), len(df_train), len(df_test))
    mem_before = _memory_gb()
    t0 = time.perf_counter()

    X_train = df_train[feature_cols].copy()
    y_train = df_train["Response"].astype(np.int8).to_numpy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test["Response"].astype(np.int8).to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=50,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)
    best_thr, oot_mcc = search_best_mcc_threshold(y_test, y_pred)

    elapsed = time.perf_counter() - t0
    mem_peak = _memory_gb()

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_.astype(np.float64),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    importance_path = OUTPUTS_DIR / f"feature_importance_e2_{arm_name}.csv"
    importance_df.to_csv(importance_path, index=False)

    data_fp = compute_data_fingerprint(df_train, feature_cols, "Response")

    result = {
        "arm": arm_name,
        "features": len(feature_cols),
        "train_rows": len(df_train),
        "train_pos": int(y_train.sum()),
        "test_rows": len(df_test),
        "test_pos": int(y_test.sum()),
        "oot_mcc": float(oot_mcc),
        "oot_threshold": float(best_thr),
        "elapsed_s": elapsed,
        "mem_peak_gb": mem_peak,
        "mem_delta_gb": mem_peak - mem_before,
        "importance_path": str(importance_path),
        "data_fingerprint": data_fp,
    }

    incv = _INCV.get(arm_name, {})
    if incv:
        incv_mcc = incv["oof_mcc"]
        result["incv_mcc"] = incv_mcc
        # retained_pct: absolute MCC retained (for reference)
        result["retained_pct"] = round(oot_mcc / incv_mcc * 100, 1) if incv_mcc > 0 else None

    logger.info(
        "arm=%s oot_mcc=%.5f thr=%.2f elapsed=%.0fs mem_peak=%.2fGB",
        arm_name, oot_mcc, best_thr, elapsed, mem_peak,
    )
    return result, importance_df


def _importance_rank_table(
    arm_name: str,
    incv_importance_path: Path,
    oot_importance_df: pd.DataFrame,
    major_stations: list[str],
) -> pd.DataFrame:
    """Build rank-change table for major stations (descriptive only)."""
    if not incv_importance_path.exists():
        logger.warning("In-CV importance file not found: %s", incv_importance_path)
        return pd.DataFrame()

    incv_df = pd.read_csv(incv_importance_path).reset_index(drop=False)
    incv_df = incv_df.reset_index(drop=True)
    incv_df["incv_rank"] = range(1, len(incv_df) + 1)
    incv_rank_map = incv_df.set_index("feature")["incv_rank"].to_dict()

    oot_rank_map = oot_importance_df.set_index("feature")["rank"].to_dict()

    rows = []
    for station in major_stations:
        incv_rank = incv_rank_map.get(station, None)
        oot_rank = oot_rank_map.get(station, None)
        rank_change = (oot_rank - incv_rank) if (incv_rank and oot_rank) else None
        rows.append({
            "feature": station,
            "incv_rank": incv_rank,
            "oot_rank": oot_rank,
            "rank_change": rank_change,
        })
    return pd.DataFrame(rows)


def _report(results: list[dict], rank_tables: dict[str, pd.DataFrame]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("E2 OUT-OF-TIME VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "Temporal split: train=chunks 0-82 (830k rows, 5,487 pos @ 0.661%%),\n"
        "                test=chunks 83-118 (353,747 rows, 1,392 pos @ 0.394%%)"
    )
    logger.info("No early stopping; n_estimators=700 fixed; random_state=42\n")

    # Compute durability (pre-registered bar): retained fraction of additive gain over dataset_h
    h_incv = _INCV["dataset_h"]["oof_mcc"]
    h_oot = next(r["oot_mcc"] for r in results if r["arm"] == "dataset_h")

    header = (
        f"{'Arm':<16} {'Feats':>5} {'in-CV MCC':>10} {'OOT MCC':>9} "
        f"{'ABS ret%':>9} {'Δ_incv':>7} {'Δ_oot':>7} {'Durability%':>12} {'Runtime':>8} {'MemΔ':>6}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        incv_mcc = r.get("incv_mcc", float("nan"))
        retained = r.get("retained_pct", float("nan"))
        delta_incv = incv_mcc - h_incv
        delta_oot = r["oot_mcc"] - h_oot
        durability = (delta_oot / delta_incv * 100) if abs(delta_incv) > 1e-8 else float("nan")
        r["delta_incv"] = round(delta_incv, 5)
        r["delta_oot"] = round(delta_oot, 5)
        r["durability_pct"] = round(durability, 1)
        logger.info(
            "%-16s %5d  %10.5f  %9.5f  %8.1f%%  %+6.4f  %+6.4f  %11.1f%%  %6.0fs  %+5.2fGB",
            r["arm"], r["features"], incv_mcc, r["oot_mcc"],
            retained if retained is not None else float("nan"),
            delta_incv, delta_oot, durability,
            r["elapsed_s"], r["mem_delta_gb"],
        )

    logger.info("\nIn-CV fold MCCs (reference from E1a/b/c decomposition):")
    fold_header = f"{'Model':<16} {'F0':>8} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'OOF':>8}"
    logger.info(fold_header)
    for arm_key, arm_label in [
        ("dataset_h", "dataset_h"),
        ("dataset_e1b", "dataset_e1b"),
        ("dataset_e1c", "dataset_e1c"),
    ]:
        incv = _INCV[arm_key]
        folds = incv["fold_mccs"]
        logger.info(
            "%-16s %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f",
            arm_label, *folds, incv["oof_mcc"],
        )

    logger.info("\nFeature importance rank stability (major stations, in-CV vs OOT):")
    for arm_name, rank_df in rank_tables.items():
        if rank_df.empty:
            continue
        logger.info(f"  --- {arm_name} ---")
        logger.info(
            f"  {'Feature':<30} {'in-CV rank':>10} {'OOT rank':>9} {'Δrank':>7}"
        )
        for _, row in rank_df.iterrows():
            delta = f"{row['rank_change']:+.0f}" if row["rank_change"] is not None else "N/A"
            logger.info(
                "  %-30s %10s %9s %7s",
                row["feature"],
                str(int(row["incv_rank"])) if row["incv_rank"] else "—",
                str(int(row["oot_rank"])) if row["oot_rank"] else "—",
                delta,
            )


def main() -> None:
    dataset_path = FEATURES_DIR / "dataset_e1.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Missing dataset_e1.parquet. Run scripts/build_dataset_e1.py first."
        )

    logger.info("Loading dataset_e1 for E2 out-of-time evaluation")
    df = pd.read_parquet(dataset_path)
    logger.info("Loaded %d rows, %d positive (%.3f%%)", len(df),
                int(df["Response"].sum()), df["Response"].mean() * 100)

    # ── Temporal split at chunk boundary ────────────────────────────────────
    train_mask = df["chunk_id"] <= TRAIN_MAX_CHUNK
    test_mask = ~train_mask
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Verify no chunk spans the boundary
    overlap = set(df_train["chunk_id"].unique()) & set(df_test["chunk_id"].unique())
    if overlap:
        raise RuntimeError(f"Chunk boundary violation: {len(overlap)} chunks span train/test. {overlap}")
    logger.info(
        "Split: train=%d rows (%d pos @ %.3f%%), test=%d rows (%d pos @ %.3f%%)",
        len(df_train), int(df_train["Response"].sum()), df_train["Response"].mean() * 100,
        len(df_test), int(df_test["Response"].sum()), df_test["Response"].mean() * 100,
    )

    station_cols = _station_mean_cols(df)
    logger.info("Station columns: %d", len(station_cols))

    # ── Build feature matrices for each arm ─────────────────────────────────
    # dataset_h: use routing features directly (chunk_id is already in DATASET_H_FEATURE_COLS)
    h_cols = DATASET_H_FEATURE_COLS
    df_h_train = df_train[["Id", "Response"] + h_cols].copy()
    df_h_test = df_test[["Id", "Response"] + h_cols].copy()

    # E1b: presence flags (same construction as decomposition, applied per split)
    e1b_cols, df_e1b_train = _build_e1b_features(df_train, station_cols)
    _,        df_e1b_test  = _build_e1b_features(df_test,  station_cols)

    # E1c: station means, median filled from training rows only
    e1c_cols, df_e1c_train, df_e1c_test = _build_e1c_features(df_train, df_test, station_cols)

    # Sanity: E1c should have no NaN in station cols in either split
    for split_name, split_df in [("train", df_e1c_train), ("test", df_e1c_test)]:
        nans = split_df[station_cols].isna().sum().sum()
        assert nans == 0, f"E1c {split_name} has {nans} NaNs after imputation"
    logger.info("E1c imputation sanity check passed")

    # ── Train and evaluate each arm ─────────────────────────────────────────
    results = []
    oot_importance_dfs: dict[str, pd.DataFrame] = {}

    for arm_name, feature_cols, train_df, test_df in [
        ("dataset_h", h_cols, df_h_train, df_h_test),
        ("dataset_e1b", e1b_cols, df_e1b_train, df_e1b_test),
        ("dataset_e1c", e1c_cols, df_e1c_train, df_e1c_test),
    ]:
        r, imp_df = _train_and_eval(arm_name, feature_cols, train_df, test_df)
        results.append(r)
        oot_importance_dfs[arm_name] = imp_df

    # ── Feature importance rank tables for major stations ───────────────────
    # Major stations identified in E1 report (DR-004, DR-005, DR-006, DR-007):
    # E1c/value channel: L3_S33, L3_S30, L3_S29 (top by split-gain)
    # E1b/presence channel: top presence flags by split-gain (L0_S11, L3_S38, L0_S10)
    major_stations_e1c = [
        "sensor_mean_L3_S33", "sensor_mean_L3_S30", "sensor_mean_L3_S29",
        "sensor_mean_L3_S36", "sensor_mean_L3_S35",
        "sensor_mean_L0_S0", "sensor_mean_L0_S1",
    ]
    major_stations_e1b = [
        "sensor_present_L0_S11", "sensor_present_L3_S38", "sensor_present_L0_S10",
        "sensor_present_L0_S9", "sensor_present_L3_S35",
        "sensor_present_L3_S33", "sensor_present_L3_S34",  # the high-signal skips
        "sensor_present_L0_S4", "sensor_present_L0_S6",
    ]

    rank_tables = {
        "dataset_e1b": _importance_rank_table(
            "dataset_e1b",
            OUTPUTS_DIR / "feature_importance_dataset_e1b.csv",
            oot_importance_dfs["dataset_e1b"],
            major_stations_e1b,
        ),
        "dataset_e1c": _importance_rank_table(
            "dataset_e1c",
            OUTPUTS_DIR / "feature_importance_dataset_e1c.csv",
            oot_importance_dfs["dataset_e1c"],
            major_stations_e1c,
        ),
    }

    # ── Print full report ────────────────────────────────────────────────────
    _report(results, rank_tables)

    # ── Persist results JSON for decisions.md update ────────────────────────
    output_path = OUTPUTS_DIR / "e2_out_of_time_results.json"
    summary = {
        "experiment": "E2",
        "split": {
            "method": "forward-chaining temporal split at chunk boundary",
            "train_chunks": f"0-{TRAIN_MAX_CHUNK}",
            "test_chunks": f"{TRAIN_MAX_CHUNK + 1}+",
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "train_pos": int(df_train["Response"].sum()),
            "test_pos": int(df_test["Response"].sum()),
        },
        "arms": results,
        "rank_tables": {
            arm: df.to_dict(orient="records") for arm, df in rank_tables.items()
        },
        "reproduce": (
            "PYTHONPATH=. python scripts/train_e2_out_of_time.py "
            "(requires data/features/dataset_e1.parquet from build_dataset_e1.py)"
        ),
    }
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results written to %s", output_path)
    logger.info("Evidence returned to Opus for interpretation.")


if __name__ == "__main__":
    main()
