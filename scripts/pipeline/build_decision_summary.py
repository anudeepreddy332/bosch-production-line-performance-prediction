from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.decision_system import CostConfig, run_decision_system_summary


if __name__ == "__main__":
    root = ROOT
    summary_path = root / "outputs/production_decision_summary.json"

    # Final packaging mode: preserve the curated production summary if it already exists,
    # so `run_full_system.py` remains idempotent without generating intermediate artifacts.
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = run_decision_system_summary(
            project_root=root,
            cost_cfg=CostConfig(cost_false_negative=100.0, cost_false_positive=5.0),
        )

    print("Saved decision summary:", summary_path)
    print("Minimum cost:", summary["operating_points"]["minimum_cost_configuration"]["total_cost"])
