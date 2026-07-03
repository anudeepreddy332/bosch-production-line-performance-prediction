"""Track 3 label-free drift monitoring.

Reads all cycle/batch-partitioned prediction parquets produced by
scripts/run_production_inference.py, strips structural counter columns, renames
risk_score -> pred (required by drift_detection.generate_evidently_report which has
ValueDrift(column='pred') hardcoded), and runs Evidently drift detection on a
temporal 70/30 split: earlier rows (sorted by run_seq) serve as the reference
baseline; later rows are the current window.

This script NEVER reads Response, meta_dataset.parquet, or any labeled dataset.
All inputs come exclusively from outputs/production/dataset_h/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_detection import generate_evidently_report


OUT = ROOT / "outputs"
MON = OUT / "monitoring"
DEFAULT_PRODUCTION_DIR = OUT / "production" / "dataset_h"

# Columns excluded before drift analysis:
#   - batch_id / cycle_id / run_seq: monotonically increasing counters — always
#     "drift" between any two windows regardless of score-distribution shift.
#   - scored_at_utc: timestamp string with no predictive drift signal.
#   - decision / auto_reject / manual_inspect: binary flags deterministically
#     derived from risk_score via a fixed DecisionPolicy; carry no independent
#     signal beyond risk_score itself.
#   - Id: excluded by drift_detection._clean_columns (strips 'id' case-insensitively).
_EXCLUDE_COLS: frozenset[str] = frozenset(
    {"batch_id", "cycle_id", "run_seq", "scored_at_utc",
     "decision", "auto_reject", "manual_inspect"}
)


def load_production_batches(production_dir: Path) -> pd.DataFrame:
    """Concatenate all Track 3 prediction parquets, ordered by run_seq."""
    batch_files = list(production_dir.glob("cycle=*/batch=*/predictions.parquet"))
    if not batch_files:
        raise FileNotFoundError(
            f"No production batch parquets found under {production_dir}. "
            "Run scripts/run_production_inference.py to generate at least one batch "
            "before running drift monitoring."
        )
    df = pd.concat([pd.read_parquet(p) for p in batch_files], ignore_index=True)
    if "Response" in df.columns:
        raise ValueError(
            f"Production batch data at {production_dir} unexpectedly contains a "
            "'Response' column. Track 3 output must be label-free by construction — "
            "check scripts/build_test_dataset_h.py and scripts/run_production_inference.py."
        )
    return df.sort_values("run_seq").reset_index(drop=True)


def build_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal 70/30 split: earlier rows = reference baseline, later = current window.

    Rows are already sorted by run_seq (temporal order). This is
    production-meaningful: the reference captures the historical score distribution;
    the current window captures recent batches. With a single batch the split falls
    within that batch (rows 0..6999 vs 7000..9999), which degrades gracefully.

    risk_score is renamed to pred so generate_evidently_report's hardcoded
    ValueDrift(column='pred') resolves correctly without modifying drift_detection.py.
    """
    monitoring_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    df_mon = df[monitoring_cols].rename(columns={"risk_score": "pred"})

    n = len(df_mon)
    split = max(1, min(int(n * 0.7), n - 1))
    ref = df_mon.iloc[:split].reset_index(drop=True)
    cur = df_mon.iloc[split:].reset_index(drop=True)
    return ref, cur


def run(production_dir: Path) -> dict:
    MON.mkdir(parents=True, exist_ok=True)
    df = load_production_batches(production_dir)
    ref, cur = build_reference_current(df)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path = tmp_dir / "reference.parquet"
        cur_path = tmp_dir / "current.parquet"
        ref.to_parquet(ref_path, index=False)
        cur.to_parquet(cur_path, index=False)
        report = generate_evidently_report(
            reference_path=ref_path,
            current_path=cur_path,
            output_json=MON / "evidently_summary.json",
            output_html=MON / "evidently_report.html",
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track 3 label-free drift monitoring. "
                    "Reads production batch parquets; never reads labeled data."
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_DIR,
        help="Directory containing cycle=*/batch=*/predictions.parquet files "
             f"(default: {DEFAULT_PRODUCTION_DIR})",
    )
    args = parser.parse_args()

    report = run(args.production_dir)
    print("Drift engine:", report.get("engine"))
    print("Reference rows:", report.get("reference_rows"))
    print("Current rows:", report.get("current_rows"))
    print("Saved:", MON / "evidently_summary.json")
