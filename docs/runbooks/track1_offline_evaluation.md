# Track 1: Offline Training + Evaluation

## What it is

Track 1 is the only place in this project where supervised metrics (MCC, precision, recall,
accuracy, confusion matrix, TP/FP/TN/FN) are computed. It operates entirely on **labeled** data:
training data with `Response`, or out-of-fold (OOF) predictions generated during cross-validation
on that labeled data. See [`docs/ml_system_tracks.md`](../ml_system_tracks.md) for the full
three-track rationale.

## Why labels and supervised metrics are allowed here

Production/test data is unlabeled by definition (Bosch never gives you the true failure outcome
for parts you haven't inspected yet) — so Track 3 (production inference) and Track 2 (Kaggle
submission) structurally cannot compute these metrics; there's no ground truth available to them.
Track 1 is different: it operates on data where the label **is** known (training data, or
held-out folds within it), specifically so that model quality and decision thresholds can be
measured before anything is approved for production use. This is standard offline
evaluation/model-selection work, not a violation of the "production = inference only" rule in
`CLAUDE.md`/`execution_rules.md` — that rule is about the production *inference* path, which
Track 1 is not part of.

## How to run the existing offline evaluation workflows

These operate on **already-generated** OOF/meta prediction parquet files — they do not retrain
anything. The training pipeline that produces those files
(`prepare_data.py` → `build_dataset_*.py` → `train_*.py` → `train_meta_model.py`) is documented in
the top-level `CLAUDE.md` and is out of scope for this runbook.

```bash
# Decision summary: threshold sweep + inspection-budget sweep -> ranked operating points
python scripts/pipeline/build_decision_summary.py

# Labeled threshold/budget replay, batch-by-batch (the "offline batch eval" -- this is the
# script that was formerly misnamed run_batch_simulation.py; it computes real
# precision/recall/tp/fp/fn/tn because its input (meta_dataset.parquet) IS labeled)
python scripts/pipeline/run_offline_batch_eval.py --mode full
python scripts/pipeline/run_offline_batch_eval.py --mode sliding --batch-size 10000

# Drift monitoring (Evidently) -- also Track-1-shaped today: reads meta_dataset.parquet
# (labeled) plus a historical, non-reproducible full-scale blend file, not Track 3's S3 output
python scripts/pipeline/run_drift_monitoring.py

# Run all three in sequence + upload the two static JSON summaries to S3
python scripts/pipeline/run_full_system.py
```

The dashboard's labeled-data pages (everything except "Production Monitoring (Track 3)") are also
Track 1 — see [`dashboard.md`](dashboard.md) for how to run those.

## What outputs to expect

| Command | Output | What it contains |
|---|---|---|
| `build_decision_summary.py` | `outputs/production_decision_summary.json` | Threshold sweep, inspection-budget sweep, and ranked operating points (min-cost, best-recall-under-budget, best precision/recall balance) |
| `run_offline_batch_eval.py --mode full` | `outputs/batch_simulation_summary.json` | Aggregate recall/precision/tp/fp/fn/tn across the full labeled dataset replayed in batches |
| `run_offline_batch_eval.py --mode sliding` | same file + `outputs/batch_simulation_state.json` (gitignored) | Same metrics, but resumable batch-by-batch like Track 3's state pattern |
| `run_drift_monitoring.py` | `outputs/monitoring/evidently_summary.json` + `.html` | Evidently drift report between a stable 70/30 reference/current split |
| `run_full_system.py` | all of the above + S3 upload of the two JSON summaries | End-to-end orchestration |
| `python scripts/pipeline/validate_system.py` | `outputs/system_validation_report.json` | Sanity-checks ranges/invariants across the three JSONs above and cross-checks consistency |

## Warnings: World A vs. World B, reproducibility

The numbers you'll see quoted in `README.md` ("Best MCC: ~0.317", "Recall @ 10% inspection:
~0.63") come from a historical blend file
(`data/features/oof_predictions_context_meta_v2_blend.parquet`) whose generating
raw/intermediate/model artifacts were deliberately deleted from this repo. **No training script
in this repository's history reproduces that file** — treat those numbers as a record of a past
experiment, not a guarantee about the current pipeline.

The metrics that **are** reproducible today come from the committed dataset and the current
`train_*.py` code. Before trusting any MCC/recall number from this track, read
[`docs/reproducible_metrics_report.md`](../reproducible_metrics_report.md) for:
- the exact distinction between "World A" (the committed, reproducible artifacts) and "World B"
  (the historical, unreproducible blend file),
- the exact regeneration command sequence for both the dev-sample and full-scale runs,
- the current honest OOF MCC values per model.

`data/README.md` has the underlying provenance explanation for why World B cannot currently be
regenerated from committed code.
