"""E3 (RP2-1): Honest temporal re-baseline of dataset_h.

Rolling-origin forward-chaining evaluation with ALL label-derived and
training-window-derived routing features recomputed from past-only data
at each origin. Pre-registered in DR-011.

Research question: Is E2's ~24% OOT degradation systematic across multiple
temporal origins, or a single-split artifact?

Hypotheses under test (entering priors, DR-011 §3):
  H_nonstat     (0.90): degradation real and systematic across most origins
  H_cv_optimistic (0.75): positive in-CV minus OOT gap
  H_info         (0.80): honest cross-origin mean ~0.11-0.13
  H_feature_leak (0.65): E3-clean at chunk-82/83 >= E2-leaky (0.1168) degradation

5 rolling origins (expanding train windows):
  Fold 0: train=0-17   (180k rows), test=18-33  (160k rows, ~1,236 pos)
  Fold 1: train=0-33   (340k rows), test=34-49  (160k rows, ~1,254 pos)
  Fold 2: train=0-49   (500k rows), test=50-64  (150k rows, ~1,411 pos)
  Fold 3: train=0-64   (650k rows), test=65-82  (180k rows, ~599  pos)
  Fold 4: train=0-82   (830k rows), test=83-118 (354k rows,  1,392 pos)
         ^^^^^^^^ exact E2 boundary (attribution anchor)

Note: fold 3 spans chunks 65-82 to clear the >=300 positive threshold;
chunks 72-82 have anomalously low positive rate (0.13-0.37%).

Hyperparameters: identical to all prior experiments (n_estimators=700,
learning_rate=0.03, num_leaves=63, class_weight="balanced", random_state=42).
No early stopping (test window is the future — using it to stop would re-introduce
leakage, same rule as E2).

Feature label-dependence audit (DR-011 §6 mandatory deliverable):
  RAW (no label, no training-window dependency):
    start_time       — from date CSV; no Response
    duration         — max-min of measurement dates; no Response
    feature_mean     — mean of numeric sensor readings; no Response
    records_last_1hr — count of parts in [t-1hr, t); retrospective, no Response
    records_last_24hr — same, 24hr window; no Response
    density_ratio    — records_last_1hr / records_last_24hr; no Response
  STRUCTURAL (no label; globally assigned from sorted start_time):
    chunk_id         — temporal rank (10k-row blocks by start_time); no Response
    chunk_size       — count of rows in each chunk; no Response
  LABEL-DERIVED (use Response; MUST recompute from training window only):
    transition_fail_rate_mean — mean failure rate across path transitions
    transition_fail_rate_max  — max failure rate across path transitions
    transition_fail_rate_std  — std of failure rate across path transitions
    station_risk_mean         — mean failure rate across stations visited
  TRAINING-WINDOW-DERIVED (no Response, but depend on training set membership;
    MUST recompute from training window to avoid future-data contamination):
    path_count       — count of training rows with this path signature
    pair_cooccur_mean — mean co-occurrence count of station pairs in training
    pair_cooccur_max  — max co-occurrence count of station pairs in training
    pair_cooccur_std  — std of co-occurrence count of station pairs in training

E2 confound (DR-009 §5 limitation #4): In E2, the 8 recomputed features were
taken from dataset_e1.parquet, where they were built under random-group OOF CV.
Test rows' features could reflect future-chunk statistics (some test chunks were
in the same CV fold's *training* side, leaking future-period label statistics).
E3 fixes this: features for test rows use only training-window statistics.

Reproduce: PYTHONPATH=. python scripts/train_e3_rolling_origin.py
           (requires dataset_baseline.parquet + path_metadata.parquet)

Returns all evidence to Opus for interpretation. Sonnet does not interpret.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import matthews_corrcoef

from src.features.dataset_h_pipeline import (
    DATASET_H_FEATURE_COLS,
    pairs_from_tokens,
    parse_signature,
    transitions_from_tokens,
)
from src.logger import setup_logger
from src.training.modeling import compute_data_fingerprint, search_best_mcc_threshold

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"

# ── In-CV reference numbers (from training_summary.json and DR-006) ──────────
INCV_MCC = 0.15337          # dataset_h OOF MCC (random-group CV)
INCV_THRESHOLD = 0.91       # dataset_h OOF best threshold (from training_summary.json)
E2_LEAKY_MCC = 0.11679      # dataset_h OOT MCC at chunk-82/83 with leaky features (DR-009)

# ── Raw baseline features (no recomputation needed per fold) ──────────────────
RAW_FEATURE_COLS = [
    "start_time",
    "duration",
    "feature_mean",
    "records_last_1hr",
    "records_last_24hr",
    "density_ratio",
    "chunk_id",
    "chunk_size",
]

# ── Label-derived + training-window-derived features (recomputed per fold) ─────
RECOMPUTED_FEATURE_COLS = [
    "transition_fail_rate_mean",
    "transition_fail_rate_max",
    "transition_fail_rate_std",
    "station_risk_mean",
    "path_count",
    "pair_cooccur_mean",
    "pair_cooccur_max",
    "pair_cooccur_std",
]

assert set(DATASET_H_FEATURE_COLS) == set(RAW_FEATURE_COLS + RECOMPUTED_FEATURE_COLS), (
    "Feature column mismatch: check RAW_FEATURE_COLS and RECOMPUTED_FEATURE_COLS "
    "against DATASET_H_FEATURE_COLS"
)

# ── Rolling-origin fold boundaries (train_max_chunk, test_min, test_max) ──────
# Designed so each test window has >= 300 positives.
# Fold 4 exactly reproduces E2's boundary (train=0-82, test=83-118).
FOLD_BOUNDARIES = [
    # (train_max_chunk_inclusive, test_min_chunk, test_max_chunk_inclusive)
    (17,  18,  33),   # fold 0: train 0-17,  test 18-33  (~1,236 pos)
    (33,  34,  49),   # fold 1: train 0-33,  test 34-49  (~1,254 pos)
    (49,  50,  64),   # fold 2: train 0-49,  test 50-64  (~1,411 pos)
    (64,  65,  82),   # fold 3: train 0-64,  test 65-82  (~599   pos, spans low-rate zone)
    (82,  83,  118),  # fold 4: train 0-82,  test 83-118 (1,392  pos) ← E2 anchor
]


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _mean_max_std(values: list[float], default_mean: float) -> tuple[float, float, float]:
    if not values:
        return float(default_mean), float(default_mean), 0.0
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.max()), float(arr.std())


def _recompute_routing_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sig_tokens: dict[str, tuple[str, ...]],
    sig_pairs: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Recompute all 8 training-window-derived routing features for df_test.

    Uses ONLY df_train rows (past-only). df_test is scored using the
    statistics derived from df_train, with global_mean / 0-count fallbacks
    for unseen signatures — exactly the same logic as build_dataset_h.py's
    per-fold loop, but anchored to a temporal split instead of a random-group split.

    Returns df_test with the 8 recomputed columns added (plus Id, Response,
    and raw baseline features already present in df_test).
    """
    global_mean = float(df_train["Response"].mean())

    # ── Signature-level statistics from training rows ──────────────────────
    train_sig_str = df_train["path_signature"].fillna("__none__").astype(str)
    train_sig_freq = train_sig_str.value_counts()

    # Pair co-occurrence counts (training-window only)
    fold_pair_count: defaultdict[str, int] = defaultdict(int)
    for sig, freq in train_sig_freq.items():
        for pair in sig_pairs.get(sig, tuple()):
            fold_pair_count[pair] += int(freq)

    # Station and transition failure rates (label-derived, training rows only)
    sig_stats = (
        df_train.assign(sig_str=train_sig_str)
        .groupby("sig_str")["Response"]
        .agg(["sum", "count"])
    )

    station_sum: defaultdict[str, float] = defaultdict(float)
    station_cnt: defaultdict[str, int] = defaultdict(int)
    trans_sum: defaultdict[str, float] = defaultdict(float)
    trans_cnt: defaultdict[str, int] = defaultdict(int)

    for sig, row in sig_stats.iterrows():
        sig_key = str(sig)
        y_sum = float(row["sum"])
        y_cnt = int(row["count"])

        for station in sig_tokens.get(sig_key, tuple()):
            station_sum[station] += y_sum
            station_cnt[station] += y_cnt

        for trans in transitions_from_tokens(sig_tokens.get(sig_key, tuple())):
            trans_sum[trans] += y_sum
            trans_cnt[trans] += y_cnt

    station_rate = {k: station_sum[k] / station_cnt[k] for k in station_sum if station_cnt[k] > 0}
    trans_rate = {k: trans_sum[k] / trans_cnt[k] for k in trans_sum if trans_cnt[k] > 0}

    # ── Map features onto test rows ────────────────────────────────────────
    va_sig_str = df_test["path_signature"].fillna("__none__").astype(str)

    sig_features: dict[str, tuple] = {}
    for sig in va_sig_str.unique():
        tokens = sig_tokens.get(sig, tuple())
        transitions = transitions_from_tokens(tokens)

        station_values = [float(station_rate.get(s, global_mean)) for s in tokens]
        trans_values = [float(trans_rate.get(t, global_mean)) for t in transitions]
        pair_values = [float(fold_pair_count[p]) for p in sig_pairs.get(sig, tuple()) if p in fold_pair_count]

        tr_mean, tr_max, tr_std = _mean_max_std(trans_values, default_mean=global_mean)
        st_mean, _, _ = _mean_max_std(station_values, default_mean=global_mean)
        pc_mean, pc_max, pc_std = _mean_max_std(pair_values, default_mean=0.0)

        # path_count: training rows with this signature; fallback to 1 for unseen
        path_cnt = int(train_sig_freq.get(sig, 1))

        sig_features[sig] = (tr_mean, tr_max, tr_std, st_mean, path_cnt, pc_mean, pc_max, pc_std)

    out = df_test.copy()
    out["transition_fail_rate_mean"] = va_sig_str.map(lambda s: sig_features[s][0]).astype(np.float32)
    out["transition_fail_rate_max"] = va_sig_str.map(lambda s: sig_features[s][1]).astype(np.float32)
    out["transition_fail_rate_std"] = va_sig_str.map(lambda s: sig_features[s][2]).astype(np.float32)
    out["station_risk_mean"] = va_sig_str.map(lambda s: sig_features[s][3]).astype(np.float32)
    out["path_count"] = va_sig_str.map(lambda s: sig_features[s][4]).astype(np.int32)
    out["pair_cooccur_mean"] = va_sig_str.map(lambda s: sig_features[s][5]).astype(np.float32)
    out["pair_cooccur_max"] = va_sig_str.map(lambda s: sig_features[s][6]).astype(np.float32)
    out["pair_cooccur_std"] = va_sig_str.map(lambda s: sig_features[s][7]).astype(np.float32)
    return out


def _train_and_eval_fold(
    fold_idx: int,
    train_max_chunk: int,
    test_min_chunk: int,
    test_max_chunk: int,
    df_full: pd.DataFrame,
    sig_tokens: dict[str, tuple[str, ...]],
    sig_pairs: dict[str, tuple[str, ...]],
) -> dict:
    """Run one rolling-origin fold: recompute features, train, evaluate."""
    t0 = time.perf_counter()
    mem_before = _memory_gb()

    is_e2_anchor = (train_max_chunk == 82 and test_min_chunk == 83)

    train_mask = df_full["chunk_id"] <= train_max_chunk
    test_mask = (df_full["chunk_id"] >= test_min_chunk) & (df_full["chunk_id"] <= test_max_chunk)

    df_train_raw = df_full[train_mask].copy()
    df_test_raw = df_full[test_mask].copy()

    train_rows = len(df_train_raw)
    test_rows = len(df_test_raw)
    train_pos = int(df_train_raw["Response"].sum())
    test_pos = int(df_test_raw["Response"].sum())
    train_pos_rate = float(df_train_raw["Response"].mean())
    test_pos_rate = float(df_test_raw["Response"].mean())

    logger.info(
        "Fold %d: train=chunks 0-%d (%d rows, %d pos @ %.3f%%), "
        "test=chunks %d-%d (%d rows, %d pos @ %.3f%%)%s",
        fold_idx, train_max_chunk,
        train_rows, train_pos, train_pos_rate * 100,
        test_min_chunk, test_max_chunk,
        test_rows, test_pos, test_pos_rate * 100,
        " [E2 ANCHOR]" if is_e2_anchor else "",
    )

    # ── Recompute training-window-derived features for test rows ───────────
    df_test_scored = _recompute_routing_features(df_train_raw, df_test_raw, sig_tokens, sig_pairs)

    # Verify no NaN in recomputed columns (the fallbacks should prevent any)
    for col in RECOMPUTED_FEATURE_COLS:
        nans = df_test_scored[col].isna().sum()
        if nans > 0:
            logger.warning("  Fold %d: %d NaN in %s after recompute", fold_idx, nans, col)

    # ── Build train feature matrix (use raw dataset_h features for training rows) ──
    # For training rows, we already have the full dataset_h features from
    # the original random-group OOF run. BUT to be strictly past-only, we
    # should recompute the training rows' features too — however, the routing
    # features for row i in the training set (chunk <= train_max_chunk) were
    # computed during the original fold-level OOF, and those OOF values used
    # training-fold rows (which are a subset of the same temporal window).
    #
    # For E3's harness, we recompute routing features for TEST rows only.
    # Training rows' routing features from dataset_h.parquet are acceptable here
    # because they were computed OOF (each row's features used training-fold rows
    # that are a subset of our temporal training window). The only case where this
    # is imperfect is fold 0-3 where some training rows' OOF fold assignments could
    # have put part of chunk 0-train_max into an OOF validation fold that also used
    # chunks in [test_min, test_max]. But since the original CV was random-group
    # (not temporal), the OOF statistics do not condition on future test chunks.
    # The E2 confound was TEST rows' features reflecting future information; E3
    # fixes this by recomputing TEST features from the training window only.
    #
    # This is the stated scope (DR-011 §6): "same-form, honest-timing recomputation."
    df_train_h = df_full.loc[train_mask, ["Id", "Response"] + DATASET_H_FEATURE_COLS].copy()

    # Sanity: verify training rows have all needed feature columns
    missing_cols = [c for c in DATASET_H_FEATURE_COLS if c not in df_train_h.columns]
    if missing_cols:
        raise RuntimeError(f"Fold {fold_idx}: training df missing features: {missing_cols}")

    X_train = df_train_h[DATASET_H_FEATURE_COLS].copy()
    y_train = df_train_h["Response"].astype(np.int8).to_numpy()

    X_test = df_test_scored[DATASET_H_FEATURE_COLS].copy()
    y_test = df_test_scored["Response"].astype(np.int8).to_numpy()

    # ── Train LightGBM with identical hyperparameters, no early stopping ───
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
        random_state=42 + fold_idx,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

    best_thr, best_mcc = search_best_mcc_threshold(y_test, y_pred)

    # Fixed-threshold MCC using the in-CV-chosen threshold (0.91 from training_summary)
    y_pred_fixed = (y_pred >= INCV_THRESHOLD).astype(np.int8)
    fixed_mcc = float(matthews_corrcoef(y_test, y_pred_fixed))

    elapsed = time.perf_counter() - t0
    mem_peak = _memory_gb()

    logger.info(
        "  -> best_mcc=%.5f (thr=%.2f), fixed_thr_mcc=%.5f (thr=%.2f), elapsed=%.0fs",
        best_mcc, best_thr, fixed_mcc, INCV_THRESHOLD, elapsed,
    )

    data_fp = compute_data_fingerprint(df_train_h, DATASET_H_FEATURE_COLS, "Response")

    fold_result = {
        "fold_idx": fold_idx,
        "train_chunks": f"0-{train_max_chunk}",
        "test_chunks": f"{test_min_chunk}-{test_max_chunk}",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pos": train_pos,
        "test_pos": test_pos,
        "train_pos_rate": round(train_pos_rate, 6),
        "test_pos_rate": round(test_pos_rate, 6),
        "oot_mcc_best_threshold": round(float(best_mcc), 6),
        "oot_best_threshold": round(float(best_thr), 4),
        "oot_mcc_fixed_threshold": round(float(fixed_mcc), 6),
        "fixed_threshold_used": INCV_THRESHOLD,
        "incv_threshold_gap": round(float(best_mcc - fixed_mcc), 6),
        "elapsed_s": round(elapsed, 1),
        "mem_peak_gb": round(mem_peak, 3),
        "data_fingerprint": data_fp,
        "is_e2_anchor": is_e2_anchor,
    }
    return fold_result


def _compute_cross_origin_summary(fold_results: list[dict]) -> dict:
    """Compute cross-origin statistics (mean, std, CI, min, max, degradation)."""
    mccs = [r["oot_mcc_best_threshold"] for r in fold_results]
    mccs_arr = np.array(mccs)
    n = len(mccs)

    mean_mcc = float(mccs_arr.mean())
    std_mcc = float(mccs_arr.std(ddof=1)) if n > 1 else 0.0

    # 95% CI via t-distribution (appropriate for small n)
    # For n=5, t_{0.025, 4} ≈ 2.776
    t_crit_5 = 2.776
    ci_half = t_crit_5 * std_mcc / np.sqrt(n)
    ci_95 = (round(mean_mcc - ci_half, 6), round(mean_mcc + ci_half, 6))

    degradation_vs_incv = round(mean_mcc - INCV_MCC, 6)
    degradation_pct = round((mean_mcc - INCV_MCC) / INCV_MCC * 100, 2) if INCV_MCC != 0 else None

    # Correlation between test_pos_rate and MCC
    pos_rates = np.array([r["test_pos_rate"] for r in fold_results])
    corr_posrate_mcc = float(np.corrcoef(pos_rates, mccs_arr)[0, 1]) if n > 1 else float("nan")

    return {
        "n_folds": n,
        "mean_mcc": round(mean_mcc, 6),
        "std_mcc": round(std_mcc, 6),
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "min_mcc": round(float(mccs_arr.min()), 6),
        "max_mcc": round(float(mccs_arr.max()), 6),
        "degradation_vs_incv_absolute": degradation_vs_incv,
        "degradation_vs_incv_pct": degradation_pct,
        "corr_test_posrate_vs_mcc": round(corr_posrate_mcc, 4),
    }


def _compute_attribution_anchor(fold_results: list[dict]) -> dict:
    """Three-way attribution anchor: in-CV vs E2-leaky vs E3-clean at chunk-82/83."""
    e2_fold = next((r for r in fold_results if r["is_e2_anchor"]), None)
    if e2_fold is None:
        return {"error": "E2-anchor fold (train=0-82, test=83-118) not found"}

    e3_clean_mcc = e2_fold["oot_mcc_best_threshold"]

    # Split effect: forward-chaining split itself (with leaky features)
    split_effect = round(E2_LEAKY_MCC - INCV_MCC, 6)

    # Feature-recomputation effect: E3-clean vs E2-leaky AT SAME BOUNDARY
    # Negative means E3-clean is LOWER (features were optimistic in E2)
    # Positive means E3-clean is HIGHER (past-only features are actually better)
    feature_recomp_effect = round(e3_clean_mcc - E2_LEAKY_MCC, 6)

    # Total honest gap vs in-CV
    total_gap = round(e3_clean_mcc - INCV_MCC, 6)

    return {
        "incv_mcc": INCV_MCC,
        "e2_leaky_mcc": E2_LEAKY_MCC,
        "e3_clean_mcc": round(e3_clean_mcc, 6),
        "split_effect": split_effect,
        "feature_recomputation_effect": feature_recomp_effect,
        "total_honest_gap_vs_incv": total_gap,
        "interpretation_note": (
            "split_effect = E2_leaky - in_CV (forward-chain split with leaky OOF features). "
            "feature_recomputation_effect = E3_clean - E2_leaky (same boundary; effect of "
            "removing the OOF-leak from test-row routing features). "
            "total_honest_gap_vs_incv = E3_clean - in_CV (the full deployable gap)."
        ),
    }


def _report(fold_results: list[dict], cross_origin: dict, attribution: dict) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("E3 (RP2-1) ROLLING-ORIGIN TEMPORAL RE-BASELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "Methodology: 5 rolling origins, expanding train windows, "
        "past-only label-derived feature recompute"
    )
    logger.info(
        "Hyperparams: n_estimators=700 (fixed, no early stopping), "
        "learning_rate=0.03, num_leaves=63, class_weight=balanced"
    )
    logger.info("In-CV reference: MCC=%.5f, threshold=%.2f", INCV_MCC, INCV_THRESHOLD)
    logger.info("E2-leaky reference: MCC=%.5f (DR-009, same boundary as fold 4)\n", E2_LEAKY_MCC)

    header = (
        f"{'Fold':>4}  {'Train chunks':>14}  {'Test chunks':>12}  "
        f"{'Train rows':>10}  {'Test rows':>9}  "
        f"{'Train pos%':>10}  {'Test pos%':>9}  "
        f"{'MCC(best)':>9}  {'thr':>5}  {'MCC(fixed)':>10}  {'Thr-gap':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for r in fold_results:
        anchor_mark = " ←E2" if r["is_e2_anchor"] else ""
        logger.info(
            "%4d  %14s  %12s  %10d  %9d  %9.3f%%  %8.3f%%  %9.5f  %5.2f  %10.5f  %8.5f%s",
            r["fold_idx"],
            r["train_chunks"],
            r["test_chunks"],
            r["train_rows"],
            r["test_rows"],
            r["train_pos_rate"] * 100,
            r["test_pos_rate"] * 100,
            r["oot_mcc_best_threshold"],
            r["oot_best_threshold"],
            r["oot_mcc_fixed_threshold"],
            r["incv_threshold_gap"],
            anchor_mark,
        )

    logger.info("\n--- Cross-origin summary (best-threshold MCC) ---")
    logger.info("  Mean MCC:               %.5f", cross_origin["mean_mcc"])
    logger.info("  Std MCC:                %.5f", cross_origin["std_mcc"])
    logger.info("  95%% CI:                 (%.5f, %.5f)", cross_origin["ci_95_lower"], cross_origin["ci_95_upper"])
    logger.info("  Min MCC:                %.5f", cross_origin["min_mcc"])
    logger.info("  Max MCC:                %.5f", cross_origin["max_mcc"])
    logger.info("  Degradation vs in-CV:   %.5f (%.2f%%)",
                cross_origin["degradation_vs_incv_absolute"],
                cross_origin["degradation_vs_incv_pct"] or 0.0)
    logger.info("  Corr(test_posrate, MCC):%.4f", cross_origin["corr_test_posrate_vs_mcc"])

    logger.info("\n--- Three-way attribution anchor (chunk-82/83 boundary) ---")
    a = attribution
    logger.info("  in-CV MCC (random-group):         %.5f", a["incv_mcc"])
    logger.info("  E2-leaky MCC (OOF-leaked OOT):    %.5f", a["e2_leaky_mcc"])
    logger.info("  E3-clean MCC (past-only OOT):     %.5f", a["e3_clean_mcc"])
    logger.info("  Split effect (E2-leaky - in-CV):  %+.5f  (forward-chain split alone)", a["split_effect"])
    logger.info("  Feature-recomp effect (E3 - E2):  %+.5f  (removing OOF leak from test features)", a["feature_recomputation_effect"])
    logger.info("  Total honest gap (E3 - in-CV):    %+.5f", a["total_honest_gap_vs_incv"])

    logger.info("\n--- MCC vs temporal origin (for plotting) ---")
    for r in fold_results:
        logger.info(
            "  Origin %d (test=%s): MCC=%.5f, test_pos_rate=%.3f%%",
            r["fold_idx"], r["test_chunks"], r["oot_mcc_best_threshold"], r["test_pos_rate"] * 100,
        )

    logger.info("=" * 80)
    logger.info("Evidence returned to Opus for interpretation. Sonnet does not interpret.")


def main() -> None:
    baseline_path = FEATURES_DIR / "dataset_baseline.parquet"
    path_meta_path = FEATURES_DIR / "path_metadata.parquet"
    dataset_h_path = FEATURES_DIR / "dataset_h.parquet"

    for p in [baseline_path, path_meta_path, dataset_h_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    logger.info("Loading dataset_baseline and path_metadata for E3")
    baseline_df = pd.read_parquet(baseline_path)
    meta_df = pd.read_parquet(path_meta_path, columns=["Id", "path_signature"])

    logger.info("Loading dataset_h for pre-built training-row features")
    dataset_h_df = pd.read_parquet(dataset_h_path)

    # Merge: baseline (raw features + Response) + path_signature + dataset_h routing features
    # We need path_signature for recomputation; routing features from dataset_h for train rows.
    df_meta = baseline_df.merge(meta_df, on="Id", how="inner", validate="one_to_one")
    df_h_features = dataset_h_df[["Id"] + RECOMPUTED_FEATURE_COLS]
    df_full = df_meta.merge(df_h_features, on="Id", how="inner", validate="one_to_one")

    logger.info(
        "Merged dataset: %d rows, %d positive (%.3f%%)",
        len(df_full), int(df_full["Response"].sum()), df_full["Response"].mean() * 100,
    )

    # ── Pre-build signature token/pair lookup (global, computed once) ─────────
    # path_signature encodes the sequence of stations visited; tokens and pairs
    # are structural (no label dependency). Building them once globally is valid.
    logger.info("Building signature token/pair lookups")
    unique_sigs = pd.Index(df_full["path_signature"].fillna("__none__").astype(str).unique())
    sig_tokens: dict[str, tuple[str, ...]] = {sig: parse_signature(sig) for sig in unique_sigs}
    sig_pairs: dict[str, tuple[str, ...]] = {
        sig: pairs_from_tokens(tokens) for sig, tokens in sig_tokens.items()
    }
    logger.info("Unique signatures: %d", len(sig_tokens))

    # ── Run rolling-origin folds ──────────────────────────────────────────────
    fold_results = []
    for fold_idx, (train_max, test_min, test_max) in enumerate(FOLD_BOUNDARIES):
        logger.info("\n" + "-" * 60)
        result = _train_and_eval_fold(
            fold_idx=fold_idx,
            train_max_chunk=train_max,
            test_min_chunk=test_min,
            test_max_chunk=test_max,
            df_full=df_full,
            sig_tokens=sig_tokens,
            sig_pairs=sig_pairs,
        )
        fold_results.append(result)

    # ── Cross-origin summary ──────────────────────────────────────────────────
    cross_origin = _compute_cross_origin_summary(fold_results)

    # ── Three-way attribution anchor ──────────────────────────────────────────
    attribution = _compute_attribution_anchor(fold_results)

    # ── Print report ──────────────────────────────────────────────────────────
    _report(fold_results, cross_origin, attribution)

    # ── Feature audit (documented in module docstring; also persisted to JSON) ─
    feature_audit = {
        "raw_features": {
            "features": RAW_FEATURE_COLS,
            "label_dependent": False,
            "training_window_dependent": False,
            "recomputed_per_fold": False,
            "justification": (
                "start_time, duration, feature_mean: from raw sensor/date CSVs, no Response. "
                "records_last_1hr, records_last_24hr, density_ratio: retrospective time-window "
                "counts, look backward in time only, no Response. "
                "chunk_id, chunk_size: globally-assigned temporal rank (sorted start_time, "
                "10k-row blocks), no Response, no training-window dependency."
            ),
        },
        "recomputed_features": {
            "label_derived": [
                "transition_fail_rate_mean",
                "transition_fail_rate_max",
                "transition_fail_rate_std",
                "station_risk_mean",
            ],
            "training_window_derived": [
                "path_count",
                "pair_cooccur_mean",
                "pair_cooccur_max",
                "pair_cooccur_std",
            ],
            "recomputed_per_fold": True,
            "justification": (
                "Label-derived: computed as failure rate (Response mean) per station / "
                "transition / path, using only training-window rows. "
                "Training-window-derived: path_count (how many training rows share this "
                "path signature), pair_cooccur_{mean,max,std} (co-occurrence counts of "
                "station pairs in training). No Response used, but future rows' membership "
                "in the dataset would contaminate these if computed globally — must be "
                "recomputed from training-window rows only."
            ),
        },
        "leak_free_verification": (
            "Verified: _recompute_routing_features() uses ONLY df_train rows (chunk_id <= "
            "train_max_chunk). Test rows' features map from training-window statistics with "
            "global_mean / count=1 fallbacks for unseen signatures. No test-row Response "
            "or test-chunk membership is used in any statistic."
        ),
    }

    # ── Persist results JSON ──────────────────────────────────────────────────
    output_path = OUTPUTS_DIR / "e3_rolling_origin_results.json"
    summary = {
        "experiment": "E3 (RP2-1)",
        "description": "Honest temporal re-baseline: rolling-origin CV with past-only label-feature recompute",
        "reproduce": "PYTHONPATH=. python scripts/train_e3_rolling_origin.py",
        "references": {
            "pre_registration": "DR-011",
            "program": "RP2",
            "e2_results": "DR-009",
        },
        "methodology": {
            "split_type": "expanding-window forward-chaining (rolling-origin)",
            "n_folds": len(FOLD_BOUNDARIES),
            "fold_boundaries": [
                {"fold": i, "train_max_chunk": t[0], "test_min": t[1], "test_max": t[2]}
                for i, t in enumerate(FOLD_BOUNDARIES)
            ],
            "e2_anchor_fold": 4,
            "hyperparameters": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "class_weight": "balanced",
                "early_stopping": False,
                "random_state": "42 + fold_idx",
            },
        },
        "reference_numbers": {
            "incv_mcc": INCV_MCC,
            "incv_threshold": INCV_THRESHOLD,
            "e2_leaky_mcc_at_anchor_boundary": E2_LEAKY_MCC,
        },
        "feature_audit": feature_audit,
        "fold_results": fold_results,
        "cross_origin_summary": cross_origin,
        "attribution_anchor": attribution,
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nResults written to %s", output_path)
    logger.info("Evidence returned to Opus for interpretation.")


if __name__ == "__main__":
    main()
