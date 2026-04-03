from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionPolicy:
    threshold_high: float = 0.60
    inspection_budget_pct: float = 5.0


def apply_threshold(pred: np.ndarray, threshold: float) -> np.ndarray:
    return (pred >= threshold).astype(np.int8)


def apply_topk_budget(pred: np.ndarray, budget_pct: float) -> np.ndarray:
    n = len(pred)
    k = int(np.ceil(n * budget_pct / 100.0))
    out = np.zeros(n, dtype=np.int8)
    if k <= 0:
        return out
    idx = np.argsort(-pred, kind="mergesort")[:k]
    out[idx] = 1
    return out


def apply_hybrid(pred: np.ndarray, policy: DecisionPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pred)
    auto_reject = pred >= policy.threshold_high
    flagged_total = int(np.ceil(n * policy.inspection_budget_pct / 100.0))
    remaining = max(0, flagged_total - int(auto_reject.sum()))

    manual = np.zeros(n, dtype=bool)
    if remaining > 0:
        rest_idx = np.where(~auto_reject)[0]
        rest_sorted = rest_idx[np.argsort(-pred[rest_idx], kind="mergesort")]
        manual[rest_sorted[:remaining]] = True

    decisions = (auto_reject | manual).astype(np.int8)
    return decisions, auto_reject.astype(np.int8), manual.astype(np.int8)


def metrics_from_labels(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "flagged_pct": float((tp + fp) / len(y_true) * 100.0),
    }


def simulate_batches(df: pd.DataFrame, policy: DecisionPolicy, batch_size: int = 10_000) -> pd.DataFrame:
    rows = []
    for start in range(0, len(df), batch_size):
        end = min(len(df), start + batch_size)
        chunk = df.iloc[start:end]
        y = chunk["Response"].to_numpy(dtype=np.int8, copy=False)
        pred = chunk["pred"].to_numpy(dtype=np.float32, copy=False)

        y_hat, auto, manual = apply_hybrid(pred, policy)
        m = metrics_from_labels(y, y_hat)
        m.update(
            {
                "batch_start": start,
                "batch_end": end,
                "rows": int(end - start),
                "auto_reject_count": int(auto.sum()),
                "manual_inspect_count": int(manual.sum()),
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
            }
        )
        rows.append(m)

    return pd.DataFrame(rows)
