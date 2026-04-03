from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    cost_false_negative: float = 100.0
    cost_false_positive: float = 5.0


def load_tables(outputs_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "max_recall_threshold_sweep": outputs_dir / "max_recall_threshold_sweep.csv",
        "inspection_budget_results": outputs_dir / "inspection_budget_results.csv",
        "production_threshold_sweep": outputs_dir / "production_threshold_sweep.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required decision artifacts: {missing}")
    return {name: pd.read_csv(path) for name, path in required.items()}


def build_decision_table(tables: dict[str, pd.DataFrame], dataset_size: int, cost_cfg: CostConfig) -> pd.DataFrame:
    rows = []

    # Threshold-based configs from two sweeps.
    for source_name in ["max_recall_threshold_sweep", "production_threshold_sweep"]:
        df = tables[source_name].copy()
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": source_name,
                    "mode": "threshold",
                    "configuration": f"thr={r['threshold']:.2f}",
                    "inspection_budget_pct": float(r.get("positives_pct", np.nan)),
                    "threshold": float(r["threshold"]),
                    "recall": float(r["recall"]),
                    "precision": float(r["precision"]),
                    "mcc": float(r.get("mcc", 0.0)),
                    "tp": int(r["tp"]),
                    "fp": int(r["fp"]),
                    "fn": int(r["fn"]),
                    "tn": int(r["tn"]),
                }
            )

    # Budget-based configs.
    bdf = tables["inspection_budget_results"].copy()
    for _, r in bdf.iterrows():
        rows.append(
            {
                "source": "inspection_budget_results",
                "mode": "budget",
                "configuration": f"budget={int(r['inspection_budget_pct'])}%",
                "inspection_budget_pct": float(r["inspection_budget_pct"]),
                "threshold": float(r.get("score_cutoff", np.nan)),
                "recall": float(r["recall"]),
                "precision": float(r["precision"]),
                "mcc": float(r.get("mcc", 0.0)),
                "tp": int(r["tp"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
                "tn": int(r["tn"]),
            }
        )

    out = pd.DataFrame(rows)
    out["total_flagged_rows"] = out["tp"] + out["fp"]
    out["failures_caught"] = out["tp"]
    out["failures_missed"] = out["fn"]
    out["total_cost"] = out["fn"] * cost_cfg.cost_false_negative + out["fp"] * cost_cfg.cost_false_positive
    out["cost_per_100k_rows"] = out["total_cost"] / float(dataset_size) * 100_000.0
    return out.sort_values(["total_cost", "recall"], ascending=[True, False]).reset_index(drop=True)


def summarize_operating_points(decision_df: pd.DataFrame) -> dict[str, object]:
    min_cost = decision_df.sort_values(["total_cost", "recall"], ascending=[True, False]).iloc[0].to_dict()

    def best_recall_under_budget(max_budget: float) -> dict[str, object]:
        eligible = decision_df[decision_df["inspection_budget_pct"] <= max_budget]
        if eligible.empty:
            return {"available": False, "inspection_budget_pct": max_budget}
        best = eligible.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0].to_dict()
        best["available"] = True
        return best

    # Balance point via harmonic mean of precision and recall (F1-like).
    eps = 1e-12
    balance = decision_df.copy()
    balance["balance_score"] = 2.0 * balance["precision"] * balance["recall"] / (
        balance["precision"] + balance["recall"] + eps
    )
    best_balance = balance.sort_values(["balance_score", "mcc"], ascending=[False, False]).iloc[0].to_dict()

    return {
        "minimum_cost_configuration": min_cost,
        "best_recall_under_1pct_inspection": best_recall_under_budget(1.0),
        "best_recall_under_5pct_inspection": best_recall_under_budget(5.0),
        "best_recall_under_10pct_inspection": best_recall_under_budget(10.0),
        "best_precision_recall_balance": best_balance,
    }


def run_decision_system_summary(project_root: Path, cost_cfg: CostConfig | None = None) -> dict[str, object]:
    cost_cfg = cost_cfg or CostConfig()
    outputs_dir = project_root / "outputs"
    feature_dir = project_root / "data/features"

    meta = pd.read_parquet(feature_dir / "meta_dataset.parquet", columns=["Id"])  # for normalization scale
    tables = load_tables(outputs_dir)
    decision_df = build_decision_table(tables=tables, dataset_size=len(meta), cost_cfg=cost_cfg)

    decision_table_path = outputs_dir / "production_decision_table.csv"
    decision_df.to_csv(decision_table_path, index=False)

    summary = {
        "cost_config": {
            "cost_false_negative": cost_cfg.cost_false_negative,
            "cost_false_positive": cost_cfg.cost_false_positive,
        },
        "decision_table_path": str(decision_table_path),
        "operating_points": summarize_operating_points(decision_df),
        "dataset_rows": int(len(meta)),
    }

    summary_path = outputs_dir / "production_decision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
