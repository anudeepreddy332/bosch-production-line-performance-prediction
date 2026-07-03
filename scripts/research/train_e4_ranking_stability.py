"""E4 (RP2 diagnostic): Ranking-stability & calibration decomposition.

Pre-registered in DR-013. Measurement-only experiment. Reuses E3 rolling-origin
harness verbatim (same 5 forward-chaining splits, same dataset_h model and
hyperparameters, same past-only label-derived feature recomputation). Zero new
modeling, no intervention, no tuning.

Research question (DR-013 §5):
    Across the 5 E3 rolling-origin regimes, is the model's ranking quality
    (threshold-free, prevalence-invariant) stable -- such that the low-prevalence
    regimes' poor thresholded performance is INTRINSIC PREVALENCE-HARDNESS
    (irreducible) -- or does ranking quality DEGRADE in those regimes
    (CONCEPT DRIFT, model-addressable)?

Hypotheses (DR-013 §5):
    H_intrinsic_hardness  (prior 0.50): ranking stable; low-prev loss is a prevalence cap.
    H_concept_drift       (prior 0.45): ranking quality degrades in low-prev regimes.
    H_threshold_sufficient (prior 0.15): decision-layer alone recovers the loss.

Branch classification (pre-registered, DR-013 §5):
    Branch A (prior 0.15): production hybrid/budget policy recovers the regime loss
        where ranking is intact.
    Branch B (prior 0.45): AUC and lift stable across regimes (CIs overlap);
        prevalence-matched control reproduces low-prev MCC -> loss is irreducible.
    Branch C (prior 0.40): AUC/lift (esp. top-region) degrade in low-prev regimes
        beyond prevalence-matched explanation -> concept drift, model-addressable.

Metrics computed per fold (all with bootstrap 95% CIs):
    Ranking:    ROC-AUC, PR-AUC (Average Precision), Lift = AP/prevalence
    Operational: Precision/Recall/Lift @ top 0.1%, 0.5%, 1%
    Decision ceiling: MCC@fixed-0.91, MCC@oracle, cost+MCC@production-hybrid-policy
    Calibration: Brier score, reliability (calibration curve data)

Prevalence-matched control:
    For each high-prev fold (f1, f2), subsample positives to match each low-prev
    fold (f3, f4) target prevalence; compare subsampled vs actual low-prev metrics
    to isolate concept drift from mechanical prevalence effects.

Stopping condition: run, record evidence, return to Opus. Do NOT implement
any intervention. RP1 remains frozen.

Reproduce:
    PYTHONPATH=. python scripts/train_e4_ranking_stability.py
    (requires data/features/dataset_baseline.parquet + path_metadata.parquet
                + data/features/dataset_h.parquet)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from src.features.dataset_h_pipeline import (
    DATASET_H_FEATURE_COLS,
    pairs_from_tokens,
    parse_signature,
    transitions_from_tokens,
)
from src.inference.decision_engine import (
    DecisionPolicy,
    apply_hybrid,
    metrics_from_labels,
)
from src.logger import setup_logger
from src.training.modeling import compute_data_fingerprint, search_best_mcc_threshold

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"

# ── In-CV reference numbers (from training_summary.json and DR-006) ──────────
INCV_MCC = 0.15337
INCV_THRESHOLD = 0.91
E2_LEAKY_MCC = 0.11679

# ── E3 oracle best-threshold MCC per fold (from DR-011 evidence) ─────────────
E3_ORACLE_MCC = {0: 0.07972, 1: 0.18164, 2: 0.17045, 3: 0.06110, 4: 0.10427}
E3_ORACLE_THR = {0: 0.14,    1: 0.40,    2: 0.72,    3: 0.40,    4: 0.62}

# ── Production policy (from configs/production.yaml) ─────────────────────────
PROD_POLICY = DecisionPolicy(threshold_high=0.60, inspection_budget_pct=10.0)
COST_FN = 100.0
COST_FP = 5.0

# ── Operational top-K budget percentages (from DR-013 §5 + user prompt) ──────
TOP_K_PCTS = [0.1, 0.5, 1.0]  # percent of test rows

# ── Bootstrap settings ────────────────────────────────────────────────────────
# 200 iterations gives stable 95% CI estimates; oracle_mcc excluded from
# bootstrap (threshold search over 99 grid points × 200 iterations × 160k rows
# is prohibitive; oracle MCC reported as full-sample point estimate only).
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 12345

# ── E3 verbatim: raw vs recomputed feature columns ──────────────────────────
RAW_FEATURE_COLS = [
    "start_time", "duration", "feature_mean",
    "records_last_1hr", "records_last_24hr", "density_ratio",
    "chunk_id", "chunk_size",
]
RECOMPUTED_FEATURE_COLS = [
    "transition_fail_rate_mean", "transition_fail_rate_max", "transition_fail_rate_std",
    "station_risk_mean",
    "path_count",
    "pair_cooccur_mean", "pair_cooccur_max", "pair_cooccur_std",
]
assert set(DATASET_H_FEATURE_COLS) == set(RAW_FEATURE_COLS + RECOMPUTED_FEATURE_COLS)

# ── E3 verbatim: fold boundaries ─────────────────────────────────────────────
FOLD_BOUNDARIES = [
    (17,  18,  33),
    (33,  34,  49),
    (49,  50,  64),
    (64,  65,  82),
    (82,  83, 118),
]

# ── Prevalence-matched control pairs: (high_prev_fold_idx, low_prev_fold_idx) ─
# High-prev folds: 1 (0.784%), 2 (0.941%). Low-prev targets: 3 (0.333%), 4 (0.394%).
PREVALENCE_MATCH_PAIRS = [(1, 3), (1, 4), (2, 3), (2, 4)]


# ─────────────────────────────────────────────────────────────────────────────
# E3 verbatim: routing-feature recomputation (copied exactly from E3 script)
# ─────────────────────────────────────────────────────────────────────────────

def _memory_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / (1024**3)


def _mean_max_std(values: list[float], default_mean: float) -> tuple[float, float, float]:
    if not values:
        return float(default_mean), float(default_mean), 0.0
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.max()), float(arr.std())


def _recompute_routing_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sig_tokens: dict[str, tuple],
    sig_pairs: dict[str, tuple],
) -> pd.DataFrame:
    """Recompute all 8 training-window-derived routing features for df_test.
    Copied verbatim from E3. Uses ONLY df_train rows (past-only).
    """
    global_mean = float(df_train["Response"].mean())

    train_sig_str = df_train["path_signature"].fillna("__none__").astype(str)
    train_sig_freq = train_sig_str.value_counts()

    fold_pair_count: defaultdict[str, int] = defaultdict(int)
    for sig, freq in train_sig_freq.items():
        for pair in sig_pairs.get(sig, tuple()):
            fold_pair_count[pair] += int(freq)

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
        path_cnt = int(train_sig_freq.get(sig, 1))
        sig_features[sig] = (tr_mean, tr_max, tr_std, st_mean, path_cnt, pc_mean, pc_max, pc_std)

    out = df_test.copy()
    out["transition_fail_rate_mean"] = va_sig_str.map(lambda s: sig_features[s][0]).astype(np.float32)
    out["transition_fail_rate_max"]  = va_sig_str.map(lambda s: sig_features[s][1]).astype(np.float32)
    out["transition_fail_rate_std"]  = va_sig_str.map(lambda s: sig_features[s][2]).astype(np.float32)
    out["station_risk_mean"]         = va_sig_str.map(lambda s: sig_features[s][3]).astype(np.float32)
    out["path_count"]                = va_sig_str.map(lambda s: sig_features[s][4]).astype(np.int32)
    out["pair_cooccur_mean"]         = va_sig_str.map(lambda s: sig_features[s][5]).astype(np.float32)
    out["pair_cooccur_max"]          = va_sig_str.map(lambda s: sig_features[s][6]).astype(np.float32)
    out["pair_cooccur_std"]          = va_sig_str.map(lambda s: sig_features[s][7]).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# E4-specific metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mcc(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred_binary))


def _compute_cost(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    return float(fn * COST_FN + fp * COST_FP)


def compute_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC (Average Precision), Lift = AP / prevalence."""
    prevalence = float(y_true.mean())
    roc_auc = float(roc_auc_score(y_true, y_pred))
    ap = float(average_precision_score(y_true, y_pred))
    lift = ap / prevalence if prevalence > 0 else float("nan")
    return {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(ap, 6),
        "average_precision": round(ap, 6),
        "lift_ap_over_prevalence": round(lift, 4),
        "prevalence": round(prevalence, 6),
    }


def compute_top_k_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, k_pcts: list[float]
) -> dict[str, Any]:
    """Precision, Recall, Lift @ top k% of predictions by score."""
    n = len(y_true)
    prevalence = float(y_true.mean())
    results: dict[str, Any] = {}
    sorted_idx = np.argsort(-y_pred, kind="mergesort")
    cumsum_pos = np.cumsum(y_true[sorted_idx])
    total_pos = int(y_true.sum())

    for k_pct in k_pcts:
        k = max(1, int(np.ceil(n * k_pct / 100.0)))
        tp_at_k = int(cumsum_pos[k - 1]) if k <= n else int(cumsum_pos[-1])
        precision_at_k = tp_at_k / k
        recall_at_k = tp_at_k / total_pos if total_pos > 0 else 0.0
        lift_at_k = precision_at_k / prevalence if prevalence > 0 else float("nan")
        key = f"top_{k_pct:.1f}pct"
        results[key] = {
            "k_pct": k_pct,
            "k_rows": k,
            "tp_at_k": tp_at_k,
            "precision": round(precision_at_k, 6),
            "recall": round(recall_at_k, 6),
            "lift": round(lift_at_k, 4),
        }
    return results


def compute_decision_ceiling(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    oracle_thr: float | None = None,
) -> dict[str, Any]:
    """Decision-layer ceiling: fixed-threshold, oracle, and production hybrid policy."""
    # Fixed in-CV threshold
    y_fixed = (y_pred >= INCV_THRESHOLD).astype(np.int8)
    mcc_fixed = _compute_mcc(y_true, y_fixed)
    cost_fixed = _compute_cost(y_true, y_fixed)
    m_fixed = metrics_from_labels(y_true, y_fixed)

    # Oracle single threshold (best MCC over grid)
    best_thr, best_mcc = search_best_mcc_threshold(y_true, y_pred)
    y_oracle = (y_pred >= best_thr).astype(np.int8)
    cost_oracle = _compute_cost(y_true, y_oracle)
    m_oracle = metrics_from_labels(y_true, y_oracle)

    # Production hybrid policy (threshold_high=0.60, inspection_budget_pct=10)
    y_hybrid, y_auto, y_manual = apply_hybrid(y_pred, PROD_POLICY)
    mcc_hybrid = _compute_mcc(y_true, y_hybrid)
    cost_hybrid = _compute_cost(y_true, y_hybrid)
    m_hybrid = metrics_from_labels(y_true, y_hybrid)

    return {
        "fixed_threshold": {
            "threshold": INCV_THRESHOLD,
            "mcc": round(mcc_fixed, 6),
            "cost": round(cost_fixed, 1),
            "precision": round(m_fixed["precision"], 6),
            "recall": round(m_fixed["recall"], 6),
            "tp": m_fixed["tp"], "fp": m_fixed["fp"],
            "fn": m_fixed["fn"], "tn": m_fixed["tn"],
        },
        "oracle_threshold": {
            "threshold": round(float(best_thr), 4),
            "mcc": round(float(best_mcc), 6),
            "cost": round(cost_oracle, 1),
            "precision": round(m_oracle["precision"], 6),
            "recall": round(m_oracle["recall"], 6),
            "tp": m_oracle["tp"], "fp": m_oracle["fp"],
            "fn": m_oracle["fn"], "tn": m_oracle["tn"],
        },
        "hybrid_policy": {
            "threshold_high": PROD_POLICY.threshold_high,
            "inspection_budget_pct": PROD_POLICY.inspection_budget_pct,
            "mcc": round(mcc_hybrid, 6),
            "cost": round(cost_hybrid, 1),
            "precision": round(m_hybrid["precision"], 6),
            "recall": round(m_hybrid["recall"], 6),
            "auto_flagged": int(y_auto.sum()),
            "manual_flagged": int(y_manual.sum()),
            "total_flagged": int(y_hybrid.sum()),
            "flagged_pct": round(float(y_hybrid.sum() / len(y_true) * 100), 4),
            "tp": m_hybrid["tp"], "fp": m_hybrid["fp"],
            "fn": m_hybrid["fn"], "tn": m_hybrid["tn"],
        },
    }


def compute_calibration_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Brier score + reliability curve data per fold."""
    brier = float(brier_score_loss(y_true, y_pred))

    # sklearn calibration_curve: fraction_of_positives vs mean_predicted_value
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        frac_pos, mean_pred = np.array([]), np.array([])

    # Score distribution summary
    return {
        "brier_score": round(brier, 6),
        "reliability_curve": {
            "mean_predicted": [round(v, 6) for v in mean_pred.tolist()],
            "fraction_positive": [round(v, 6) for v in frac_pos.tolist()],
        },
        "score_distribution": {
            "mean": round(float(y_pred.mean()), 6),
            "std": round(float(y_pred.std()), 6),
            "p10": round(float(np.percentile(y_pred, 10)), 6),
            "p25": round(float(np.percentile(y_pred, 25)), 6),
            "p50": round(float(np.percentile(y_pred, 50)), 6),
            "p75": round(float(np.percentile(y_pred, 75)), 6),
            "p90": round(float(np.percentile(y_pred, 90)), 6),
            "p95": round(float(np.percentile(y_pred, 95)), 6),
            "p99": round(float(np.percentile(y_pred, 99)), 6),
        },
    }


def _compute_all_metrics_for_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute a flat dict of scalar metrics for one bootstrap sample.

    oracle_mcc is intentionally excluded: searching 99 thresholds × 200
    bootstrap samples × up to 354k rows is prohibitive. oracle_mcc is
    reported as a full-sample point estimate in compute_decision_ceiling().
    """
    prevalence = float(y_true.mean())
    metrics: dict[str, float] = {}

    # Ranking (threshold-free; the primary crux metrics for Branch B vs C)
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        ap = float(average_precision_score(y_true, y_pred))
        metrics["pr_auc"] = ap
        metrics["lift"] = ap / prevalence if prevalence > 0 else float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["lift"] = float("nan")

    # Brier
    metrics["brier_score"] = float(brier_score_loss(y_true, y_pred))

    # Fixed threshold MCC (uses pre-set INCV_THRESHOLD; no search needed)
    y_fixed = (y_pred >= INCV_THRESHOLD).astype(np.int8)
    metrics["fixed_mcc"] = _compute_mcc(y_true, y_fixed)

    # Hybrid policy MCC (no search; fixed production policy)
    y_hybrid, _, _ = apply_hybrid(y_pred, PROD_POLICY)
    metrics["hybrid_mcc"] = _compute_mcc(y_true, y_hybrid)
    metrics["hybrid_cost"] = _compute_cost(y_true, y_hybrid)

    # Top-K metrics (key operational ranking metrics)
    n = len(y_true)
    total_pos = int(y_true.sum())
    if total_pos > 0:
        sorted_idx = np.argsort(-y_pred, kind="mergesort")
        cumsum = np.cumsum(y_true[sorted_idx])
        for k_pct in TOP_K_PCTS:
            k = max(1, int(np.ceil(n * k_pct / 100.0)))
            tp_k = int(cumsum[min(k, n) - 1])
            prec_k = tp_k / k
            rec_k = tp_k / total_pos
            lift_k = prec_k / prevalence if prevalence > 0 else float("nan")
            metrics[f"prec_top_{k_pct:.1f}pct"] = prec_k
            metrics[f"rec_top_{k_pct:.1f}pct"] = rec_k
            metrics[f"lift_top_{k_pct:.1f}pct"] = lift_k
    else:
        for k_pct in TOP_K_PCTS:
            metrics[f"prec_top_{k_pct:.1f}pct"] = float("nan")
            metrics[f"rec_top_{k_pct:.1f}pct"] = float("nan")
            metrics[f"lift_top_{k_pct:.1f}pct"] = float("nan")

    return metrics


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Bootstrap 95% CIs by resampling rows within the fold's test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    metric_samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        m = _compute_all_metrics_for_bootstrap(yt, yp)
        for k, v in m.items():
            if not (np.isnan(v) or np.isinf(v)):
                metric_samples[k].append(v)

    ci_out: dict[str, dict[str, float]] = {}
    for metric_name, samples in metric_samples.items():
        if len(samples) < 10:
            ci_out[metric_name] = {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_valid": len(samples)}
            continue
        arr = np.array(samples)
        ci_out[metric_name] = {
            "mean": round(float(arr.mean()), 6),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 6),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 6),
            "n_valid": len(samples),
        }
    return ci_out


# ─────────────────────────────────────────────────────────────────────────────
# Prevalence-matched control
# ─────────────────────────────────────────────────────────────────────────────

def compute_prevalence_matched_control(
    y_true_high: np.ndarray,
    y_pred_high: np.ndarray,
    target_prevalence: float,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Subsample positives from a high-prev fold to match target_prevalence.

    For each bootstrap iteration:
    - Keep all negatives from the high-prev fold
    - Randomly select n_pos_needed positives (without replacement if possible)
    - Compute metrics on the subsampled set

    Returns per-iteration distribution -> CI estimates.
    """
    pos_idx = np.where(y_true_high == 1)[0]
    neg_idx = np.where(y_true_high == 0)[0]
    n_neg = len(neg_idx)

    # n_pos_needed to achieve target_prevalence with all negatives kept
    n_pos_needed = int(np.round(n_neg * target_prevalence / (1.0 - target_prevalence)))

    if n_pos_needed == 0:
        return {
            "feasible": False,
            "reason": "target prevalence too low: 0 positives needed",
            "n_pos_available": len(pos_idx),
            "n_pos_needed": 0,
            "target_prevalence": target_prevalence,
        }
    if n_pos_needed > len(pos_idx):
        return {
            "feasible": False,
            "reason": f"insufficient positives: need {n_pos_needed} but only {len(pos_idx)} available",
            "n_pos_available": len(pos_idx),
            "n_pos_needed": n_pos_needed,
            "target_prevalence": target_prevalence,
        }

    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = defaultdict(list)
    achieved_prevalences: list[float] = []

    for _ in range(n_bootstrap):
        # Each bootstrap: fresh random subsample of positives (without replacement)
        chosen_pos = rng.choice(pos_idx, size=n_pos_needed, replace=False)
        # Bootstrap resample the combined set
        combined_idx = np.concatenate([neg_idx, chosen_pos])
        boot_idx = rng.integers(0, len(combined_idx), size=len(combined_idx))
        sub_idx = combined_idx[boot_idx]
        yt = y_true_high[sub_idx]
        yp = y_pred_high[sub_idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        achieved_prevalences.append(float(yt.mean()))
        m = _compute_all_metrics_for_bootstrap(yt, yp)
        for k, v in m.items():
            if not (np.isnan(v) or np.isinf(v)):
                metric_samples[k].append(v)

    ci_out: dict[str, dict[str, float]] = {}
    for metric_name, samples in metric_samples.items():
        if len(samples) < 10:
            ci_out[metric_name] = {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_valid": len(samples)}
            continue
        arr = np.array(samples)
        ci_out[metric_name] = {
            "mean": round(float(arr.mean()), 6),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 6),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 6),
            "n_valid": len(samples),
        }

    return {
        "feasible": True,
        "n_pos_available": int(len(pos_idx)),
        "n_neg": int(n_neg),
        "n_pos_needed": int(n_pos_needed),
        "target_prevalence": round(target_prevalence, 6),
        "achieved_prevalence_mean": round(float(np.mean(achieved_prevalences)), 6) if achieved_prevalences else float("nan"),
        "n_bootstrap_valid": len(achieved_prevalences),
        "metrics_ci": ci_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Branch classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_branch(fold_results: list[dict]) -> dict[str, Any]:
    """Classify into Branch A/B/C based on pre-registered rules (DR-013 §5).

    Branch A: hybrid policy recovers most loss where ranking is intact
    Branch B: AUC/lift stable across regimes; prevalence-matched control explains loss
    Branch C: AUC/lift degrade in low-prev regimes beyond prevalence-matching

    Uses bootstrap CIs to determine if low-prev lift/AUC overlaps high-prev values.
    """
    high_prev_folds = [r for r in fold_results if r["fold_idx"] in [1, 2]]
    low_prev_folds  = [r for r in fold_results if r["fold_idx"] in [3, 4]]

    def ci_overlaps(ci_a: dict, ci_b: dict) -> bool:
        return ci_a["ci_lower"] <= ci_b["ci_upper"] and ci_b["ci_lower"] <= ci_a["ci_upper"]

    # Assess ranking stability (AUC and lift)
    high_lift_cis = [r["bootstrap_ci"].get("lift", {}) for r in high_prev_folds]
    low_lift_cis  = [r["bootstrap_ci"].get("lift", {}) for r in low_prev_folds]
    high_auc_cis  = [r["bootstrap_ci"].get("roc_auc", {}) for r in high_prev_folds]
    low_auc_cis   = [r["bootstrap_ci"].get("roc_auc", {}) for r in low_prev_folds]

    # Overlap: any low-prev fold's CI overlap with any high-prev fold's CI
    lift_overlaps = [
        ci_overlaps(lo, hi)
        for lo in low_lift_cis for hi in high_lift_cis
        if lo.get("ci_lower") is not None and hi.get("ci_lower") is not None
    ]
    auc_overlaps = [
        ci_overlaps(lo, hi)
        for lo in low_auc_cis for hi in high_auc_cis
        if lo.get("ci_lower") is not None and hi.get("ci_lower") is not None
    ]

    lift_stable = all(lift_overlaps) if lift_overlaps else None
    auc_stable  = all(auc_overlaps) if auc_overlaps else None

    # Mean lift comparison
    high_lift_means = [r["bootstrap_ci"].get("lift", {}).get("mean", float("nan")) for r in high_prev_folds]
    low_lift_means  = [r["bootstrap_ci"].get("lift", {}).get("mean", float("nan")) for r in low_prev_folds]
    high_auc_means  = [r["bootstrap_ci"].get("roc_auc", {}).get("mean", float("nan")) for r in high_prev_folds]
    low_auc_means   = [r["bootstrap_ci"].get("roc_auc", {}).get("mean", float("nan")) for r in low_prev_folds]

    high_lift_mean = float(np.nanmean(high_lift_means))
    low_lift_mean  = float(np.nanmean(low_lift_means))
    high_auc_mean  = float(np.nanmean(high_auc_means))
    low_auc_mean   = float(np.nanmean(low_auc_means))

    lift_degradation = high_lift_mean - low_lift_mean
    auc_degradation  = high_auc_mean - low_auc_mean

    # Assess hybrid policy recovery (Branch A)
    # In each fold, does the hybrid policy MCC approach the oracle MCC?
    hybrid_vs_oracle = []
    for r in fold_results:
        oracle = r["decision_ceiling"]["oracle_threshold"]["mcc"]
        hybrid = r["decision_ceiling"]["hybrid_policy"]["mcc"]
        if oracle > 0:
            hybrid_vs_oracle.append(hybrid / oracle)

    hybrid_recovery_mean = float(np.mean(hybrid_vs_oracle)) if hybrid_vs_oracle else float("nan")

    # Classification (descriptive; Opus makes the final call)
    evidence_summary = {
        "lift_stable_all_ci_overlap": lift_stable,
        "auc_stable_all_ci_overlap": auc_stable,
        "high_prev_mean_lift": round(high_lift_mean, 4),
        "low_prev_mean_lift": round(low_lift_mean, 4),
        "lift_degradation_high_minus_low": round(lift_degradation, 4),
        "high_prev_mean_auc": round(high_auc_mean, 4),
        "low_prev_mean_auc": round(low_auc_mean, 4),
        "auc_degradation_high_minus_low": round(auc_degradation, 4),
        "hybrid_policy_recovery_vs_oracle_mean": round(hybrid_recovery_mean, 4),
        "n_hybrid_oracle_comparisons": len(hybrid_vs_oracle),
    }

    # Pre-registered criteria: Branch B if CIs overlap and control explains loss
    #                          Branch C if CIs don't overlap / degradation is material
    # These are evidence points for Opus to interpret — not a final automated verdict.
    if lift_stable and auc_stable:
        classification = "B_tentative"
        classification_note = (
            "All low-prev fold AUC/lift CIs overlap high-prev CIs. "
            "Consistent with Branch B (intrinsic hardness). "
            "Confirm with prevalence-matched control comparison."
        )
    elif not lift_stable or auc_degradation > 0.05:
        classification = "C_tentative"
        classification_note = (
            "Low-prev AUC/lift CIs do NOT overlap high-prev. "
            "Ranking degrades materially in low-prev regimes. "
            "Consistent with Branch C (concept drift)."
        )
    else:
        classification = "mixed_unresolved"
        classification_note = (
            "Partial overlaps: some metrics suggest stability (B), "
            "others suggest degradation (C). "
            "Inspect prevalence-matched control and per-fold CI tables. "
            "Return to Opus for classification."
        )

    return {
        "classification": classification,
        "classification_note": classification_note,
        "evidence": evidence_summary,
        "branch_priors": {
            "A": 0.15, "B": 0.45, "C": 0.40,
        },
        "note": "Tentative classification from evidence — Opus makes the final call against pre-registered branches.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fold training + full metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_score_fold(
    fold_idx: int,
    train_max_chunk: int,
    test_min_chunk: int,
    test_max_chunk: int,
    df_full: pd.DataFrame,
    sig_tokens: dict,
    sig_pairs: dict,
) -> tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    """Train E3-verbatim model per fold. Returns (y_true, y_pred, stats, test_df)."""
    t0 = time.perf_counter()

    train_mask = df_full["chunk_id"] <= train_max_chunk
    test_mask  = (df_full["chunk_id"] >= test_min_chunk) & (df_full["chunk_id"] <= test_max_chunk)

    df_train_raw = df_full[train_mask].copy()
    df_test_raw  = df_full[test_mask].copy()

    train_rows = len(df_train_raw)
    test_rows  = len(df_test_raw)
    train_pos  = int(df_train_raw["Response"].sum())
    test_pos   = int(df_test_raw["Response"].sum())
    train_pos_rate = float(df_train_raw["Response"].mean())
    test_pos_rate  = float(df_test_raw["Response"].mean())

    logger.info(
        "Fold %d: train=0-%d (%d rows, %d pos @ %.3f%%), test=%d-%d (%d rows, %d pos @ %.3f%%)",
        fold_idx, train_max_chunk,
        train_rows, train_pos, train_pos_rate * 100,
        test_min_chunk, test_max_chunk,
        test_rows, test_pos, test_pos_rate * 100,
    )

    df_test_scored = _recompute_routing_features(df_train_raw, df_test_raw, sig_tokens, sig_pairs)

    df_train_h = df_full.loc[train_mask, ["Id", "Response"] + DATASET_H_FEATURE_COLS].copy()
    X_train = df_train_h[DATASET_H_FEATURE_COLS].copy()
    y_train  = df_train_h["Response"].astype(np.int8).to_numpy()

    X_test = df_test_scored[DATASET_H_FEATURE_COLS].copy()
    y_test  = df_test_scored["Response"].astype(np.int8).to_numpy()

    # E3-verbatim LightGBM config; no early stopping
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

    elapsed = time.perf_counter() - t0
    logger.info("  Fold %d trained in %.0fs", fold_idx, elapsed)

    data_fp = compute_data_fingerprint(df_train_h, DATASET_H_FEATURE_COLS, "Response")

    stats = {
        "fold_idx": fold_idx,
        "train_chunks": f"0-{train_max_chunk}",
        "test_chunks": f"{test_min_chunk}-{test_max_chunk}",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pos": train_pos,
        "test_pos": test_pos,
        "train_pos_rate": round(train_pos_rate, 6),
        "test_pos_rate": round(test_pos_rate, 6),
        "elapsed_train_s": round(elapsed, 1),
        "data_fingerprint": data_fp,
        "is_e2_anchor": (train_max_chunk == 82 and test_min_chunk == 83),
    }

    # Save per-row scores for reproducibility
    score_df = pd.DataFrame({
        "Id": df_test_scored["Id"].values,
        "Response": y_test,
        "pred": y_pred,
        "fold_idx": fold_idx,
        "chunk_id": df_test_scored["chunk_id"].values,
    })

    return y_test, y_pred, stats, score_df


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ranking_metrics(fold_results: list[dict], out_dir: Path) -> None:
    """Bar chart: per-fold AUC, lift, oracle-MCC with bootstrap CIs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fold_ids = [r["fold_idx"] for r in fold_results]
    prev_rates = [r["test_pos_rate"] * 100 for r in fold_results]
    colors = ["#2196F3" if f in [1, 2] else "#F44336" for f in fold_ids]

    for ax, metric_key, title in zip(
        axes,
        ["roc_auc", "lift", "oracle_mcc"],
        ["ROC-AUC", "Lift (AP/prevalence)", "Oracle-MCC"],
    ):
        vals, ci_lo, ci_hi = [], [], []
        for r in fold_results:
            if metric_key == "oracle_mcc":
                ci = r["bootstrap_ci"].get("oracle_mcc", {})
            else:
                ci = r["bootstrap_ci"].get(metric_key, {})
            v = ci.get("mean", float("nan"))
            lo = ci.get("ci_lower", float("nan"))
            hi = ci.get("ci_upper", float("nan"))
            vals.append(v)
            ci_lo.append(v - lo if not np.isnan(v) and not np.isnan(lo) else 0)
            ci_hi.append(hi - v if not np.isnan(v) and not np.isnan(hi) else 0)

        bars = ax.bar(fold_ids, vals, color=colors, alpha=0.8, zorder=3)
        ax.errorbar(
            fold_ids, vals,
            yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=5, zorder=4,
        )
        ax.set_xlabel("Fold (prev%)")
        ax.set_title(title)
        ax.set_xticks(fold_ids)
        ax.set_xticklabels([f"f{f}\n({p:.2f}%)" for f, p in zip(fold_ids, prev_rates)])
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        if metric_key == "roc_auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
            ax.legend(fontsize=8)

    plt.suptitle("E4: Per-fold ranking metrics (bootstrap 95% CI)\nBlue=high-prev, Red=low-prev", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "e4_ranking_metrics.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / "e4_ranking_metrics.png")


def _plot_calibration(fold_results: list[dict], out_dir: Path) -> None:
    """Reliability (calibration) curves per fold."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for r in fold_results:
        ax = axes[r["fold_idx"]]
        rel = r["calibration"]["reliability_curve"]
        if rel["mean_predicted"]:
            ax.plot(rel["mean_predicted"], rel["fraction_positive"],
                    "o-", label="calibration", color="#2196F3")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"Fold {r['fold_idx']} (prev={r['test_pos_rate']*100:.3f}%)\nBrier={r['calibration']['brier_score']:.4f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[5].set_visible(False)
    plt.suptitle("E4: Calibration (reliability) curves per fold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "e4_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / "e4_calibration.png")


def _plot_prevalence_control(
    prev_control: dict[str, Any], fold_results: list[dict], out_dir: Path
) -> None:
    """Prevalence-matched control comparison: subsampled high-prev vs actual low-prev lift."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    low_fold_idxs = [3, 4]
    high_fold_idxs = [1, 2]

    for ax_idx, (low_fi, ax) in enumerate(zip(low_fold_idxs, axes)):
        low_r = next(r for r in fold_results if r["fold_idx"] == low_fi)
        actual_lift_ci = low_r["bootstrap_ci"].get("lift", {})
        actual_lift = actual_lift_ci.get("mean", float("nan"))
        actual_err_lo = actual_lift - actual_lift_ci.get("ci_lower", actual_lift)
        actual_err_hi = actual_lift_ci.get("ci_upper", actual_lift) - actual_lift

        labels = [f"Actual f{low_fi}"]
        vals = [actual_lift]
        err_lo = [actual_err_lo]
        err_hi = [actual_err_hi]
        colors = ["#F44336"]

        for high_fi in high_fold_idxs:
            key = f"f{high_fi}_to_f{low_fi}"
            ctrl = prev_control.get(key, {})
            if ctrl.get("feasible") and ctrl.get("metrics_ci"):
                ci = ctrl["metrics_ci"].get("lift", {})
                v = ci.get("mean", float("nan"))
                lo = v - ci.get("ci_lower", v)
                hi = ci.get("ci_upper", v) - v
                labels.append(f"f{high_fi}→f{low_fi} (subsampled)")
                vals.append(v)
                err_lo.append(lo)
                err_hi.append(hi)
                colors.append("#2196F3")

        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, alpha=0.8, zorder=3)
        ax.errorbar(x, vals, yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=5, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Lift (AP/prevalence)")
        ax.set_title(f"Prevalence-matched control → f{low_fi} target\nBlue=subsampled high-prev, Red=actual low-prev")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("E4: Prevalence-matched control\n(if subsampled-high ≈ actual-low → intrinsic; if actual-low << subsampled → concept drift)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "e4_prevalence_control.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / "e4_prevalence_control.png")


def _plot_topk_metrics(fold_results: list[dict], out_dir: Path) -> None:
    """Top-K operational metrics per fold."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fold_ids = [r["fold_idx"] for r in fold_results]
    colors = ["#2196F3" if f in [1, 2] else "#F44336" for f in fold_ids]
    prev_rates = [r["test_pos_rate"] * 100 for r in fold_results]

    for ax, k_pct, title in zip(
        axes,
        [0.1, 0.5, 1.0],
        ["Lift@Top 0.1%", "Lift@Top 0.5%", "Lift@Top 1.0%"],
    ):
        metric_key = f"lift_top_{k_pct:.1f}pct"
        vals, ci_lo, ci_hi = [], [], []
        for r in fold_results:
            ci = r["bootstrap_ci"].get(metric_key, {})
            v = ci.get("mean", float("nan"))
            lo = ci.get("ci_lower", float("nan"))
            hi = ci.get("ci_upper", float("nan"))
            vals.append(v)
            ci_lo.append(v - lo if not (np.isnan(v) or np.isnan(lo)) else 0)
            ci_hi.append(hi - v if not (np.isnan(v) or np.isnan(hi)) else 0)

        ax.bar(fold_ids, vals, color=colors, alpha=0.8, zorder=3)
        ax.errorbar(fold_ids, vals, yerr=[ci_lo, ci_hi],
                    fmt="none", color="black", capsize=5, zorder=4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="random=1")
        ax.set_title(title)
        ax.set_xticks(fold_ids)
        ax.set_xticklabels([f"f{f}\n({p:.2f}%)" for f, p in zip(fold_ids, prev_rates)])
        ax.set_xlabel("Fold (prev%)")
        ax.set_ylabel("Lift")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("E4: Operational lift at top-K% budgets\nBlue=high-prev, Red=low-prev", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "e4_topk_lift.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_dir / "e4_topk_lift.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    baseline_path = FEATURES_DIR / "dataset_baseline.parquet"
    path_meta_path = FEATURES_DIR / "path_metadata.parquet"
    dataset_h_path = FEATURES_DIR / "dataset_h.parquet"

    for p in [baseline_path, path_meta_path, dataset_h_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    logger.info("E4: Loading data (reusing E3 load logic)")
    baseline_df = pd.read_parquet(baseline_path)
    meta_df = pd.read_parquet(path_meta_path, columns=["Id", "path_signature"])
    dataset_h_df = pd.read_parquet(dataset_h_path)

    df_meta = baseline_df.merge(meta_df, on="Id", how="inner", validate="one_to_one")
    df_h_features = dataset_h_df[["Id"] + RECOMPUTED_FEATURE_COLS]
    df_full = df_meta.merge(df_h_features, on="Id", how="inner", validate="one_to_one")

    logger.info("Dataset: %d rows, %d pos (%.3f%%)",
                len(df_full), int(df_full["Response"].sum()), df_full["Response"].mean() * 100)

    logger.info("Building signature lookups")
    unique_sigs = pd.Index(df_full["path_signature"].fillna("__none__").astype(str).unique())
    sig_tokens = {sig: parse_signature(sig) for sig in unique_sigs}
    sig_pairs  = {sig: pairs_from_tokens(tokens) for sig, tokens in sig_tokens.items()}

    # ── Run all 5 folds: train + score ───────────────────────────────────────
    fold_y_true: dict[int, np.ndarray] = {}
    fold_y_pred: dict[int, np.ndarray] = {}
    fold_stats_list: list[dict] = []
    all_score_dfs: list[pd.DataFrame] = []

    for fold_idx, (train_max, test_min, test_max) in enumerate(FOLD_BOUNDARIES):
        logger.info("\n--- Fold %d ---", fold_idx)
        y_true, y_pred, stats, score_df = _train_and_score_fold(
            fold_idx, train_max, test_min, test_max,
            df_full, sig_tokens, sig_pairs,
        )
        fold_y_true[fold_idx] = y_true
        fold_y_pred[fold_idx] = y_pred
        fold_stats_list.append(stats)
        all_score_dfs.append(score_df)

    # ── Persist per-row scores ────────────────────────────────────────────────
    scores_dir = OUTPUTS_DIR / "e4_fold_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for score_df in all_score_dfs:
        fi = int(score_df["fold_idx"].iloc[0])
        score_df.to_parquet(scores_dir / f"fold_{fi}_scores.parquet", index=False)
    logger.info("Per-row scores saved to %s", scores_dir)

    # ── Compute all metrics per fold ─────────────────────────────────────────
    fold_results: list[dict] = []
    for fold_idx, stats in enumerate(fold_stats_list):
        logger.info("\n=== Computing metrics for fold %d ===", fold_idx)
        y_true = fold_y_true[fold_idx]
        y_pred = fold_y_pred[fold_idx]

        t_m = time.perf_counter()

        ranking = compute_ranking_metrics(y_true, y_pred)
        topk    = compute_top_k_metrics(y_true, y_pred, TOP_K_PCTS)
        ceiling = compute_decision_ceiling(y_true, y_pred)
        calib   = compute_calibration_metrics(y_true, y_pred)

        logger.info("  AUC=%.4f  AP=%.4f  Lift=%.3f  Brier=%.4f",
                    ranking["roc_auc"], ranking["average_precision"],
                    ranking["lift_ap_over_prevalence"], calib["brier_score"])

        logger.info("  Oracle MCC=%.4f  Hybrid MCC=%.4f  Fixed MCC=%.4f",
                    ceiling["oracle_threshold"]["mcc"],
                    ceiling["hybrid_policy"]["mcc"],
                    ceiling["fixed_threshold"]["mcc"])

        logger.info("  Running bootstrap (n=%d)...", N_BOOTSTRAP)
        bootstrap_ci = compute_bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP)

        logger.info("  Bootstrap done in %.1fs; AUC CI=[%.4f, %.4f] lift CI=[%.4f, %.4f]",
                    time.perf_counter() - t_m,
                    bootstrap_ci.get("roc_auc", {}).get("ci_lower", float("nan")),
                    bootstrap_ci.get("roc_auc", {}).get("ci_upper", float("nan")),
                    bootstrap_ci.get("lift", {}).get("ci_lower", float("nan")),
                    bootstrap_ci.get("lift", {}).get("ci_upper", float("nan")),
                    )

        fold_result = {
            **stats,
            "ranking_metrics": ranking,
            "topk_metrics": topk,
            "decision_ceiling": ceiling,
            "calibration": calib,
            "bootstrap_ci": bootstrap_ci,
        }
        fold_results.append(fold_result)

    # ── Prevalence-matched control ────────────────────────────────────────────
    logger.info("\n=== Prevalence-matched control ===")
    prev_control: dict[str, Any] = {}
    for (hi_fi, lo_fi) in PREVALENCE_MATCH_PAIRS:
        target_prev = fold_stats_list[lo_fi]["test_pos_rate"]
        key = f"f{hi_fi}_to_f{lo_fi}"
        logger.info("  Pair f%d -> f%d (target prevalence=%.4f%%)", hi_fi, lo_fi, target_prev * 100)

        ctrl = compute_prevalence_matched_control(
            y_true_high=fold_y_true[hi_fi],
            y_pred_high=fold_y_pred[hi_fi],
            target_prevalence=target_prev,
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED + hi_fi * 10 + lo_fi,
        )
        prev_control[key] = ctrl

        if ctrl["feasible"]:
            ctrl_lift_mean = ctrl["metrics_ci"].get("lift", {}).get("mean", float("nan"))
            actual_lift_mean = fold_results[lo_fi]["bootstrap_ci"].get("lift", {}).get("mean", float("nan"))
            logger.info(
                "    Subsampled lift=%.3f vs actual-f%d lift=%.3f (diff=%.3f)",
                ctrl_lift_mean, lo_fi, actual_lift_mean, ctrl_lift_mean - actual_lift_mean,
            )
        else:
            logger.info("    Infeasible: %s", ctrl.get("reason"))

    # ── Branch classification ─────────────────────────────────────────────────
    logger.info("\n=== Branch classification ===")
    branch = classify_branch(fold_results)
    logger.info("  Tentative: %s", branch["classification"])
    logger.info("  %s", branch["classification_note"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("\n=== Generating plots ===")
    _plot_ranking_metrics(fold_results, OUTPUTS_DIR)
    _plot_calibration(fold_results, OUTPUTS_DIR)
    _plot_topk_metrics(fold_results, OUTPUTS_DIR)
    _plot_prevalence_control(prev_control, fold_results, OUTPUTS_DIR)

    # ── Summary report to log ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("E4 RANKING-STABILITY & CALIBRATION DECOMPOSITION SUMMARY")
    logger.info("=" * 80)
    logger.info("Fold | prev%  | AUC    | AP     | Lift   | BrierSc | OracleMCC | HybridMCC | FixedMCC")
    logger.info("-" * 90)
    for r in fold_results:
        logger.info(
            "  %d  | %5.3f%% | %.4f | %.4f | %6.3f | %.4f  | %.5f  | %.5f  | %.5f",
            r["fold_idx"], r["test_pos_rate"] * 100,
            r["ranking_metrics"]["roc_auc"],
            r["ranking_metrics"]["average_precision"],
            r["ranking_metrics"]["lift_ap_over_prevalence"],
            r["calibration"]["brier_score"],
            r["decision_ceiling"]["oracle_threshold"]["mcc"],
            r["decision_ceiling"]["hybrid_policy"]["mcc"],
            r["decision_ceiling"]["fixed_threshold"]["mcc"],
        )
    logger.info("-" * 90)
    logger.info("Bootstrap 95%% CIs on AUC and Lift:")
    for r in fold_results:
        auc_ci  = r["bootstrap_ci"].get("roc_auc", {})
        lift_ci = r["bootstrap_ci"].get("lift", {})
        logger.info(
            "  Fold %d: AUC=[%.4f, %.4f]  Lift=[%.3f, %.3f]",
            r["fold_idx"],
            auc_ci.get("ci_lower", float("nan")), auc_ci.get("ci_upper", float("nan")),
            lift_ci.get("ci_lower", float("nan")), lift_ci.get("ci_upper", float("nan")),
        )
    logger.info("\nTop-K metrics (Lift@Top K%):")
    logger.info("Fold | prev%  | Lift@0.1% | Lift@0.5% | Lift@1.0%")
    for r in fold_results:
        tk = r["topk_metrics"]
        logger.info(
            "  %d  | %5.3f%% | %9.3f | %9.3f | %9.3f",
            r["fold_idx"], r["test_pos_rate"] * 100,
            tk.get("top_0.1pct", {}).get("lift", float("nan")),
            tk.get("top_0.5pct", {}).get("lift", float("nan")),
            tk.get("top_1.0pct", {}).get("lift", float("nan")),
        )
    logger.info("\nPrevalence-matched control (lift comparison):")
    for key, ctrl in prev_control.items():
        # key format: "f{hi}_to_f{lo}"
        parts = key.split("_to_f")
        hi_fi = int(parts[0][1:])
        lo_fi = int(parts[1])
        if ctrl.get("feasible"):
            ctrl_lift = ctrl["metrics_ci"].get("lift", {}).get("mean", float("nan"))
            actual_lift = fold_results[lo_fi]["bootstrap_ci"].get("lift", {}).get("mean", float("nan"))
            logger.info("  %s: subsampled_lift=%.3f  actual_low_prev_lift=%.3f  diff=%.3f",
                        key, ctrl_lift, actual_lift, ctrl_lift - actual_lift)
        else:
            logger.info("  %s: infeasible (%s)", key, ctrl.get("reason", ""))
    logger.info("\nBranch classification: %s", branch["classification"])
    logger.info("%s", branch["classification_note"])
    logger.info("=" * 80)
    logger.info("Evidence returned to Opus. Sonnet does not interpret.")

    # ── Persist results JSON ──────────────────────────────────────────────────
    output_path = OUTPUTS_DIR / "e4_ranking_stability_results.json"

    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if np.isnan(v) or np.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    summary = {
        "experiment": "E4 (RP2 diagnostic)",
        "description": "Ranking-stability & calibration decomposition",
        "reproduce": "PYTHONPATH=. python scripts/train_e4_ranking_stability.py",
        "references": {
            "pre_registration": "DR-013",
            "program": "RP2",
            "e3_results": "DR-011",
            "e3_results_file": "outputs/e3_rolling_origin_results.json",
        },
        "methodology": {
            "reuses_e3_verbatim": True,
            "split_type": "expanding-window forward-chaining (rolling-origin) — identical to E3",
            "n_folds": 5,
            "fold_boundaries": [
                {"fold": i, "train_max_chunk": t[0], "test_min": t[1], "test_max": t[2]}
                for i, t in enumerate(FOLD_BOUNDARIES)
            ],
            "hyperparameters": {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "class_weight": "balanced",
                "early_stopping": False,
                "random_state": "42 + fold_idx",
            },
            "production_policy": {
                "threshold_high": PROD_POLICY.threshold_high,
                "inspection_budget_pct": PROD_POLICY.inspection_budget_pct,
                "source": "configs/production.yaml",
            },
            "cost_config": {"cost_fn": COST_FN, "cost_fp": COST_FP},
            "bootstrap": {"n_iterations": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "method": "percentile"},
            "top_k_pcts": TOP_K_PCTS,
            "prevalence_match_pairs": PREVALENCE_MATCH_PAIRS,
        },
        "reference_numbers": {
            "incv_mcc": INCV_MCC,
            "incv_threshold": INCV_THRESHOLD,
            "e2_leaky_mcc_at_anchor": E2_LEAKY_MCC,
            "e3_oracle_mcc_per_fold": E3_ORACLE_MCC,
        },
        "fold_results": _jsonify(fold_results),
        "prevalence_matched_control": _jsonify(prev_control),
        "branch_classification": _jsonify(branch),
        "artifacts": {
            "per_row_scores": "outputs/e4_fold_scores/fold_{i}_scores.parquet",
            "ranking_plot": "outputs/e4_ranking_metrics.png",
            "calibration_plot": "outputs/e4_calibration.png",
            "topk_lift_plot": "outputs/e4_topk_lift.png",
            "prevalence_control_plot": "outputs/e4_prevalence_control.png",
            "results_json": "outputs/e4_ranking_stability_results.json",
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nResults written to %s", output_path)
    logger.info("Evidence returned to Opus for interpretation. Sonnet does not interpret.")


if __name__ == "__main__":
    main()
