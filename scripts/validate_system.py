from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MON = OUT / "monitoring"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def check_no_nan_numeric(df: pd.DataFrame) -> bool:
    num = df.select_dtypes(include=[np.number])
    return not num.isna().any().any()


def validate_decision_module() -> dict:
    decision_path = OUT / "production_decision_summary.json"
    table_path = OUT / "production_decision_table.csv"

    summary = load_json(decision_path)
    table = pd.read_csv(table_path)

    min_cost = summary["operating_points"]["minimum_cost_configuration"]
    checks = {
        "summary_exists": decision_path.exists(),
        "table_exists": table_path.exists(),
        "table_non_empty": len(table) > 0,
        "no_nan_numeric": check_no_nan_numeric(table),
        "recall_in_range": 0.0 <= float(min_cost["recall"]) <= 1.0,
        "precision_in_range": 0.0 <= float(min_cost["precision"]) <= 1.0,
        "cost_positive": float(min_cost["total_cost"]) > 0,
    }
    return {
        "module": "decision_system",
        "pass": all(checks.values()),
        "checks": checks,
        "key_metrics": {
            "minimum_cost_total": float(min_cost["total_cost"]),
            "minimum_cost_recall": float(min_cost["recall"]),
            "minimum_cost_precision": float(min_cost["precision"]),
        },
    }


def validate_batch_module() -> dict:
    summary_path = OUT / "batch_simulation_summary.json"
    summary = load_json(summary_path)

    checks = {
        "summary_exists": summary_path.exists(),
        "mode_valid": summary.get("mode") in {"full", "sliding"},
        "policy_present": "policy" in summary,
    }

    if summary.get("mode") == "full":
        metrics = summary["overall"]
        checks.update(
            {
                "recall_in_range": 0.0 <= float(metrics["recall_mean"]) <= 1.0,
                "precision_in_range": 0.0 <= float(metrics["precision_mean"]) <= 1.0,
                "flagged_pct_in_range": 0.0 <= float(metrics["flagged_pct_mean"]) <= 100.0,
            }
        )
        key = {
            "recall": float(metrics["recall_mean"]),
            "precision": float(metrics["precision_mean"]),
            "flagged_pct": float(metrics["flagged_pct_mean"]),
        }
    else:
        metrics = summary["window_metrics"]
        checks.update(
            {
                "recall_in_range": 0.0 <= float(metrics["recall"]) <= 1.0,
                "precision_in_range": 0.0 <= float(metrics["precision"]) <= 1.0,
                "flagged_pct_in_range": 0.0 <= float(metrics["flagged_pct"]) <= 100.0,
                "state_file_exists": Path(summary["pointer_state_path"]).exists(),
            }
        )
        key = {
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "flagged_pct": float(metrics["flagged_pct"]),
            "next_pointer": int(summary["next_pointer"]),
        }

    return {"module": "batch_simulation", "pass": all(checks.values()), "checks": checks, "key_metrics": key}


def validate_monitoring_module() -> dict:
    summary_path = MON / "evidently_summary.json"
    html_path = MON / "evidently_report.html"
    summary = load_json(summary_path)

    ds = summary.get("summary", {}).get("dataset_drift", {})
    pred = summary.get("summary", {}).get("prediction_drift", {})
    checks = {
        "summary_exists": summary_path.exists(),
        "html_exists": html_path.exists(),
        "engine_evidently": summary.get("engine") == "evidently",
        "dataset_drift_fields": set(["drifted_columns_count", "number_of_columns", "drift_share"]).issubset(ds.keys()),
        "prediction_drift_fields": set(["drift_detected", "drift_score", "threshold"]).issubset(pred.keys()),
        "drift_share_in_range": 0.0 <= float(ds.get("drift_share", 0.0)) <= 1.0,
    }
    return {
        "module": "monitoring",
        "pass": all(checks.values()),
        "checks": checks,
        "key_metrics": {
            "drifted_columns_count": int(ds.get("drifted_columns_count", 0)),
            "drift_share": float(ds.get("drift_share", 0.0)),
            "prediction_drift_detected": bool(pred.get("drift_detected", False)),
        },
    }


def cross_check_consistency(decision: dict, batch: dict) -> dict:
    min_recall = decision["key_metrics"]["minimum_cost_recall"]
    batch_recall = batch["key_metrics"]["recall"]
    min_precision = decision["key_metrics"]["minimum_cost_precision"]
    batch_precision = batch["key_metrics"]["precision"]

    checks = {
        "batch_recall_not_far_below_min_cost_recall": batch_recall >= (min_recall - 0.25),
        "batch_precision_not_far_above_1": 0.0 <= batch_precision <= 1.0,
        "decision_precision_valid": 0.0 <= min_precision <= 1.0,
    }
    return {
        "module": "cross_check",
        "pass": all(checks.values()),
        "checks": checks,
        "notes": {
            "decision_min_cost_recall": min_recall,
            "batch_recall": batch_recall,
            "decision_min_cost_precision": min_precision,
            "batch_precision": batch_precision,
        },
    }


if __name__ == "__main__":
    decision = validate_decision_module()
    batch = validate_batch_module()
    monitoring = validate_monitoring_module()
    cross = cross_check_consistency(decision, batch)

    modules = [decision, batch, monitoring, cross]
    report = {
        "overall_pass": all(m["pass"] for m in modules),
        "modules": modules,
    }

    out = OUT / "system_validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(out)
    print("overall_pass", report["overall_pass"])
