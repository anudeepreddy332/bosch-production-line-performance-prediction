# Track 3 — Production Inference (label-free)

**State:** Frozen (`track3-frozen`, commit `f743da3`). **Governing log:** [Research Log](research/README.md),
entries DR-011–DR-015 (RP2) in `docs/research/decisions.md`.

## Objective

Simulate what happens when the approved, frozen `dataset_h` model scores a live, unlabeled
factory stream: ongoing batch inference plus drift monitoring, with zero supervised metrics —
because production/test data has no labels, by definition.

## Input data

Unlabeled batches (`data/features/test_dataset_h.parquet`, no `Response` column), consumed
batch-by-batch with persisted `cycle_id`/`batch_id`/`run_seq` state.

## Approach

`scripts/run_production_inference.py` scores each batch with `dataset_h` + `DecisionPolicy`
(hybrid threshold + inspection-budget policy) and writes append-only, partitioned Parquet output
(`outputs/production/dataset_h/cycle={n}/batch={n}/predictions.parquet`), uploaded to S3 with the
same partitioning (never overwriting an existing key). `scripts/run_drift_monitoring.py` then reads
only those prediction Parquets — never labeled data — and runs Evidently's `DataDriftPreset` +
`ValueDrift` on the single `risk_score` column (renamed `pred` for Evidently's API), on a stable
70/30 random split.

## Results

- 5 batches scored, 50,000 rows, at the time of freeze.
- Drift summary: `dataset_drift` and `prediction_drift` are both KS tests on the single
  `risk_score` column — they are the same signal measured two ways, not two independent checks,
  and they say nothing about input-feature drift.
- No MCC/precision/recall/confusion matrix anywhere in this track's output — grep-verified, not
  just asserted.

## Known limitations

- **Score-distribution drift only.** Feature-level drift (are the *inputs* changing, not just the
  model's output distribution) is not monitored by this track today.
- **No re-threshold-on-drift loop.** Drift is detected and surfaced in the dashboard; nothing
  automatically retunes the threshold in response. That decision stays with a human operator.
- **Batch source is a real unlabeled feature table treated as an incoming stream, not a live
  factory feed** — the "production" framing describes the data contract this code satisfies (label
  -free, append-only, state-tracked), not a live deployment. See [Model Card](model_card.md) for
  the gap between this and the always-on hosted tier (`PF7`, optional).

## Where the evidence lives

- Case study: `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md` §9 (monitoring), §12 (production
  readiness status).
- Validation: `scripts/validate_system.py` → `validate_production_inference()` — asserts batch
  existence, required output columns, `Response` absence, state-file validity, Evidently schema.
- Runbook: [Track 3 production inference](runbooks/track3_production_inference.md).
- Tag: `track3-frozen` at commit `f743da3`.
