from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.decision_engine import DecisionPolicy, simulate_batches


OUT = ROOT / "outputs"
FEAT = ROOT / "data/features"
STATE_PATH = OUT / "batch_simulation_state.json"


def load_best_pred() -> pd.DataFrame:
    meta = pd.read_parquet(FEAT / "meta_dataset.parquet", columns=["Id", "Response"])
    candidates = [
        ("context_meta_v2_blend", FEAT / "oof_predictions_context_meta_v2_blend.parquet"),
        ("meta_v3_dataset_h_blend", FEAT / "oof_predictions_meta_v3_dataset_h_blend.parquet"),
    ]
    best_name = None
    best_mcc = -1.0
    best_df = None

    for name, path in candidates:
        if not path.exists():
            continue
        pred = pd.read_parquet(path)[["Id", "oof_pred"]]
        df = meta.merge(pred.rename(columns={"oof_pred": "pred"}), on="Id", how="left", validate="one_to_one")
        df["pred"] = df["pred"].fillna(0.0)

        y_hat = (df["pred"].to_numpy() >= 0.74).astype("int8")
        y = df["Response"].to_numpy().astype("int8")
        tp = int(((y_hat == 1) & (y == 1)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        fn = int(((y_hat == 0) & (y == 1)).sum())
        tn = int(((y_hat == 0) & (y == 0)).sum())
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = 0.0 if denom <= 0 else ((tp * tn) - (fp * fn)) / (denom ** 0.5)

        if mcc > best_mcc:
            best_mcc = mcc
            best_name = name
            best_df = df

    if best_df is None:
        raise FileNotFoundError("No OOF prediction files found for simulation.")

    print(f"Using source: {best_name} (proxy MCC@0.74={best_mcc:.4f})")
    return best_df.sort_values("Id").reset_index(drop=True)


def load_policy() -> DecisionPolicy:
    summary_path = OUT / "max_recall_system_summary.json"
    if not summary_path.exists():
        return DecisionPolicy(threshold_high=0.60, inspection_budget_pct=5.0)
    s = json.loads(summary_path.read_text())
    rec = s.get("final_recommendation", {})
    return DecisionPolicy(
        threshold_high=float(rec.get("threshold_high", 0.60)),
        inspection_budget_pct=float(rec.get("inspection_budget_pct", 5.0)),
    )


def run_full(df: pd.DataFrame, policy: DecisionPolicy, batch_size: int, schedule_frequency_seconds: int) -> None:
    batches = simulate_batches(df, policy=policy, batch_size=batch_size)

    summary = {
        "mode": "full",
        "policy": {
            "threshold_high": policy.threshold_high,
            "inspection_budget_pct": policy.inspection_budget_pct,
        },
        "batch_size": int(batch_size),
        "schedule_frequency_seconds": int(schedule_frequency_seconds),
        "n_batches": int(len(batches)),
        "overall": {
            "recall_mean": float(batches["recall"].mean()),
            "precision_mean": float(batches["precision"].mean()),
            "flagged_pct_mean": float(batches["flagged_pct"].mean()),
            "pred_mean_mean": float(batches["pred_mean"].mean()),
        },
        "drift_signals": {
            "pred_mean_std": float(batches["pred_mean"].std()),
            "pred_std_std": float(batches["pred_std"].std()),
            "recall_std": float(batches["recall"].std()),
            "precision_std": float(batches["precision"].std()),
        },
    }
    summary_path = OUT / "batch_simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Saved:", summary_path)


def run_sliding(
    df: pd.DataFrame,
    policy: DecisionPolicy,
    batch_size: int,
    schedule_frequency_seconds: int,
    state_path: Path,
) -> None:
    state = {"pointer": 0}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:  # noqa: BLE001
            state = {"pointer": 0}

    n = len(df)
    start = int(state.get("pointer", 0))
    if start < 0 or start >= n:
        start = 0
    end = min(n, start + batch_size)

    batch = df.iloc[start:end].copy()
    batch_metrics = simulate_batches(batch, policy=policy, batch_size=max(len(batch), 1))
    row = batch_metrics.iloc[0].to_dict()

    log_row = {
        "start": int(start),
        "end": int(end),
        "rows": int(end - start),
        "reset_triggered": bool(end >= n),
        "schedule_frequency_seconds": int(schedule_frequency_seconds),
        "batch_size": int(batch_size),
        "threshold_high": float(policy.threshold_high),
        "inspection_budget_pct": float(policy.inspection_budget_pct),
        "recall": float(row["recall"]),
        "precision": float(row["precision"]),
        "flagged_pct": float(row["flagged_pct"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tn": int(row["tn"]),
    }

    log_path = OUT / "batch_stream_log.csv"
    if log_path.exists():
        hist = pd.read_csv(log_path)
        hist = pd.concat([hist, pd.DataFrame([log_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([log_row])
    hist.to_csv(log_path, index=False)

    next_pointer = 0 if end >= n else end
    state_out = {
        "pointer": int(next_pointer),
        "last_start": int(start),
        "last_end": int(end),
        "batch_size": int(batch_size),
        "schedule_frequency_seconds": int(schedule_frequency_seconds),
        "dataset_rows": int(n),
        "reset_on_end": True,
    }
    state_path.write_text(json.dumps(state_out, indent=2))

    summary = {
        "mode": "sliding",
        "policy": {
            "threshold_high": policy.threshold_high,
            "inspection_budget_pct": policy.inspection_budget_pct,
        },
        "batch_size": int(batch_size),
        "schedule_frequency_seconds": int(schedule_frequency_seconds),
        "processed_window": {"start": int(start), "end": int(end), "rows": int(end - start)},
        "pointer_state_path": str(state_path),
        "next_pointer": int(next_pointer),
        "reset_triggered": bool(end >= n),
        "window_metrics": {
            "recall": float(row["recall"]),
            "precision": float(row["precision"]),
            "flagged_pct": float(row["flagged_pct"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "tn": int(row["tn"]),
        },
    }
    summary_path = OUT / "batch_simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Saved:", log_path)
    print("Saved:", state_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch simulation runner")
    parser.add_argument("--mode", choices=["full", "sliding"], default="full")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--schedule-frequency-seconds", type=int, default=300)
    parser.add_argument("--state-path", type=str, default=str(STATE_PATH))
    args = parser.parse_args()

    df = load_best_pred()
    policy = load_policy()
    state_path = Path(args.state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        run_full(
            df=df,
            policy=policy,
            batch_size=int(args.batch_size),
            schedule_frequency_seconds=int(args.schedule_frequency_seconds),
        )
    else:
        run_sliding(
            df=df,
            policy=policy,
            batch_size=int(args.batch_size),
            schedule_frequency_seconds=int(args.schedule_frequency_seconds),
            state_path=state_path,
        )
