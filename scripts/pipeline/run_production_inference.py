"""Track 3: true label-free production batch inference for dataset_h.

Consumes the unlabeled dataset_h feature contract built by
`scripts/build_test_dataset_h.py` (`data/features/test_dataset_h.parquet` by default --
Id plus every column in `src.features.dataset_h_pipeline.DATASET_H_FEATURE_COLS`, no
`Response` column at all, by construction) and scores it batch-by-batch against
`models/dataset_h_model.pkl`, applying the shared `DecisionPolicy`/`apply_hybrid` policy
from `src.inference.decision_engine` (the same label-free primitives Track 1's
`scripts/run_offline_batch_eval.py` already reuses for its policy, just without any of
that script's labeled-metric computation).

This script NEVER reads or requires `Response` and NEVER computes a supervised metric
(no MCC, precision, recall, accuracy, confusion matrix, TP/FP/TN/FN) -- it only emits:
  - per-row risk scores and policy decisions (predictions, not metrics)
  - batch_id / cycle_id / run_seq / a UTC timestamp
  - batch-level counts and score-distribution statistics (of OUR OWN predictions,
    never compared against ground truth)

There is no live stream in this repo, so each invocation treats the already-built,
real, unlabeled Bosch test feature table as the "incoming batch source" and advances a
small persisted pointer (mirroring `scripts/run_offline_batch_eval.py`'s existing
sliding-mode state pattern) -- one call processes one batch and wraps to a new cycle
when it reaches the end of the dataset. Output is append-only and partitioned by
cycle/batch (`outputs/production/dataset_h/cycle={n}/batch={n}/predictions.parquet`);
an existing batch path is never overwritten.

batch_id/cycle_id semantics follow `tasks.md`'s documented state-machine contract
(TASK 5 / "LOOP LOGIC": "if all batches complete -> reset batch_id -> increment
cycle_id"; example state `{"cycle_id": 1, "last_batch_id": 3}`; example paths
`cycle=1/batch=001`, `cycle=1/batch=002`): `batch_id` resets to 0 at the start of each
new cycle, `cycle_id` increments on wraparound. A separate `run_seq` field is a
lifetime-global monotonic counter (never resets) for unambiguous ordering across the
whole run history -- it is intentionally a distinct field rather than overloading
batch_id with two meanings.

After the local parquet is written, the same batch is uploaded to S3 (per `tasks.md`
TASK 4) at the partitioned, append-only key
`predictions/cycle={cycle_id}/batch={batch_id}/predictions.parquet`, mirroring the local
path. The upload is upload-then-advance: state (`dataset_h_batch_state.json`) only
advances after the upload succeeds (or is explicitly skipped via `--no-s3`), so a failed
upload leaves state untouched and the same batch is retried on the next invocation.
Upload failures (existing key, credentials, network, client errors) raise and exit the
process non-zero -- never print-and-continue.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from scripts...`/`from src...` resolve when run via
# `python scripts/run_production_inference.py` without PYTHONPATH set externally --
# matches scripts/run_offline_batch_eval.py and scripts/build_decision_summary.py, both
# invoked the same way by scripts/run_full_system.py.

from scripts.generate_submission import load_validated_payload, predict_proba_ensemble
from src.inference.decision_engine import apply_hybrid, load_policy
from src.logger import setup_logger
from src.utils.s3_utils import BUCKET_NAME, upload_file_append_only

logger = setup_logger(__name__)

FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
PRODUCTION_DIR = OUTPUTS_DIR / "production" / "dataset_h"

DEFAULT_FEATURES_PATH = FEATURES_DIR / "test_dataset_h.parquet"
DEFAULT_MODEL_PATH = MODELS_DIR / "dataset_h_model.pkl"
DEFAULT_STATE_PATH = PRODUCTION_DIR / "dataset_h_batch_state.json"
DEFAULT_STATS_LOG_PATH = PRODUCTION_DIR / "dataset_h_batch_stats_log.csv"


def _load_state(state_path: Path, dataset_rows: int) -> dict:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if int(state.get("dataset_rows", -1)) == dataset_rows:
                state.setdefault("run_seq", 0)  # tolerate state files predating this field
                return state
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {"pointer": 0, "cycle_id": 0, "batch_id": 0, "run_seq": 0, "dataset_rows": dataset_rows}


def _score_distribution(pred: np.ndarray) -> dict:
    return {
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "pred_min": float(pred.min()),
        "pred_p50": float(np.quantile(pred, 0.50)),
        "pred_p90": float(np.quantile(pred, 0.90)),
        "pred_p99": float(np.quantile(pred, 0.99)),
        "pred_max": float(pred.max()),
    }


def run_one_batch(
    features_path: Path,
    model_path: Path,
    batch_size: int,
    state_path: Path,
    output_dir: Path,
    stats_log_path: Path,
    policy_summary_path: Path | None,
    no_s3: bool = False,
) -> dict:
    features_df = pd.read_parquet(features_path)
    if "Response" in features_df.columns:
        raise ValueError(
            f"{features_path} unexpectedly has a Response column. This is a label-free "
            f"production inference path and must never read or score against labels -- "
            f"refusing to proceed. Rebuild with scripts/build_test_dataset_h.py, which never "
            f"emits Response by construction."
        )

    n = len(features_df)
    state = _load_state(state_path, dataset_rows=n)

    start = int(state["pointer"])
    if start < 0 or start >= n:
        start = 0
    end = min(n, start + batch_size)
    reset_triggered = end >= n

    batch = features_df.iloc[start:end].reset_index(drop=True)

    payload = load_validated_payload(model_path)
    pred = predict_proba_ensemble(payload, batch)
    policy = load_policy(policy_summary_path)

    decisions, auto_reject, manual = apply_hybrid(pred, policy)

    scored_at_utc = datetime.now(timezone.utc).isoformat()
    cycle_id = int(state["cycle_id"])
    batch_id = int(state["batch_id"])
    run_seq = int(state["run_seq"])

    rows_out = pd.DataFrame(
        {
            "Id": batch["Id"].to_numpy(dtype=np.int64),
            "risk_score": pred.astype(np.float32),
            "decision": decisions,
            "auto_reject": auto_reject,
            "manual_inspect": manual,
            "batch_id": np.full(len(batch), batch_id, dtype=np.int64),
            "cycle_id": np.full(len(batch), cycle_id, dtype=np.int64),
            "run_seq": np.full(len(batch), run_seq, dtype=np.int64),
            "scored_at_utc": scored_at_utc,
        }
    )

    batch_dir = output_dir / f"cycle={cycle_id}" / f"batch={batch_id}"
    batch_path = batch_dir / "predictions.parquet"
    if batch_path.exists():
        raise FileExistsError(
            f"{batch_path} already exists -- production output is append-only and must "
            f"never overwrite a previously written batch. Remove it manually first if this "
            f"is intentional reprocessing, or check the state file at {state_path}."
        )
    batch_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = batch_dir / f"{batch_path.name}.tmp"
    rows_out.to_parquet(tmp_path, index=False)

    # Upload-then-advance: batch_path (the append-only-guarded final path) is only
    # created by the rename below, which only runs once the upload has succeeded or
    # been explicitly skipped. If upload_file_append_only raises, tmp_path is left
    # behind but batch_path never exists, so the FileExistsError guard above does not
    # block retrying this exact same pending batch on the next invocation -- and state
    # (written further below) is never reached, so it stays unchanged too.
    s3_key = f"predictions/cycle={cycle_id}/batch={batch_id}/predictions.parquet"
    if no_s3:
        s3_uploaded = False
        logger.info("S3 upload skipped (--no-s3): local-only batch at %s", tmp_path)
    else:
        upload_file_append_only(str(tmp_path), s3_key)
        s3_uploaded = True
        logger.info("Uploaded batch to s3://%s/%s", BUCKET_NAME, s3_key)

    tmp_path.replace(batch_path)

    flagged = int(decisions.sum())
    stats_row = {
        "scored_at_utc": scored_at_utc,
        "cycle_id": cycle_id,
        "batch_id": batch_id,
        "run_seq": run_seq,
        "batch_start": start,
        "batch_end": end,
        "rows": int(end - start),
        "auto_reject_count": int(auto_reject.sum()),
        "manual_inspect_count": int(manual.sum()),
        "flagged_count": flagged,
        "flagged_pct": float(flagged / len(batch) * 100.0) if len(batch) else 0.0,
        "threshold_high": policy.threshold_high,
        "inspection_budget_pct": policy.inspection_budget_pct,
        "model_threshold": float(payload["threshold"]),
        "reset_triggered": reset_triggered,
        "output_path": str(batch_path),
        "s3_key": s3_key,
        "s3_uploaded": s3_uploaded,
        **_score_distribution(pred),
    }

    next_pointer = 0 if reset_triggered else end
    next_cycle_id = cycle_id + 1 if reset_triggered else cycle_id
    # batch_id resets per cycle (tasks.md TASK 5: "reset batch_id -> increment cycle_id");
    # run_seq is the separate, never-resetting lifetime counter.
    next_batch_id = 0 if reset_triggered else batch_id + 1
    state_out = {
        "pointer": next_pointer,
        "cycle_id": next_cycle_id,
        "batch_id": next_batch_id,
        "run_seq": run_seq + 1,
        "dataset_rows": n,
        "last_batch_start": start,
        "last_batch_end": end,
        "last_scored_at_utc": scored_at_utc,
    }
    # State (the critical resume pointer) advances before the stats log (observability
    # only) is written, so a stats-write failure after a successful upload + local
    # finalization cannot leave this batch stuck unable to advance.
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state_out, indent=2))

    stats_log_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_log_path.exists():
        hist = pd.read_csv(stats_log_path)
        hist = pd.concat([hist, pd.DataFrame([stats_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([stats_row])
    hist.to_csv(stats_log_path, index=False)

    logger.info(
        "Scored batch cycle=%d batch=%d run_seq=%d rows=%d flagged=%d (%.4f%%) -> %s",
        cycle_id,
        batch_id,
        run_seq,
        stats_row["rows"],
        flagged,
        stats_row["flagged_pct"],
        batch_path,
    )
    return stats_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track 3: label-free production batch inference for dataset_h."
    )
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--stats-log-path", type=Path, default=DEFAULT_STATS_LOG_PATH)
    parser.add_argument(
        "--policy-summary-path",
        type=Path,
        default=None,
        help="Defaults to src.inference.decision_engine.DEFAULT_POLICY_SUMMARY_PATH if not set.",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help=(
            "Skip the S3 upload and write the partitioned batch locally only. Default is "
            "S3-enabled (upload-then-advance), matching the rest of the production pipeline's "
            "existing S3 behavior. Use this for local-only smoke testing."
        ),
    )
    args = parser.parse_args()

    if not args.features_path.exists():
        raise FileNotFoundError(
            f"{args.features_path} does not exist. Run scripts/build_test_dataset_h.py first "
            f"to build the unlabeled dataset_h feature contract."
        )

    if args.no_s3:
        print("[S3] --no-s3 set: upload will be SKIPPED, local-only run.")

    stats_row = run_one_batch(
        features_path=args.features_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        state_path=args.state_path,
        output_dir=args.output_dir,
        stats_log_path=args.stats_log_path,
        policy_summary_path=args.policy_summary_path,
        no_s3=args.no_s3,
    )

    print(f"cycle_id={stats_row['cycle_id']}")
    print(f"batch_id={stats_row['batch_id']}")
    print(f"run_seq={stats_row['run_seq']}")
    print(f"rows_scored={stats_row['rows']}")
    print(f"flagged_count={stats_row['flagged_count']} ({stats_row['flagged_pct']:.4f}%)")
    print(f"pred_mean={stats_row['pred_mean']:.6f}")
    print(f"output_path={stats_row['output_path']}")
    if stats_row["s3_uploaded"]:
        print(f"s3_key={stats_row['s3_key']}")
    else:
        print(f"s3_upload=skipped (--no-s3); would-be s3_key={stats_row['s3_key']}")


if __name__ == "__main__":
    main()
