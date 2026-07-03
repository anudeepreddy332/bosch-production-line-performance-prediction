# Track 3: Production Inference (Label-Free Batch Scoring)

> **Track:** 3 · **Status:** Verified

## What it is

Track 3 simulates a live, unlabeled production scoring stream: it treats the real, unlabeled
Bosch test feature table as an "incoming batch source," scores one batch per invocation against
the approved `dataset_h` model, and emits only predictions/risk scores/decisions and batch
statistics — never a label, never a supervised metric. This is implemented in
`scripts/pipeline/run_production_inference.py`. See [`docs/ml_system_tracks.md`](../ml_system_tracks.md)
for why this is a separate track from Track 1 (labeled offline evaluation) and Track 2 (Kaggle
submission).

There is no real live stream in this repo — each invocation advances a persisted pointer over the
already-built `data/features/test_dataset_h.parquet` and treats the next slice as "the new
batch."

## The label-free contract

This script never reads or requires `Response`, and computes no supervised metric (no MCC,
precision, recall, accuracy, confusion matrix, TP/FP/TN/FN) anywhere. It hard-refuses to run if
its input unexpectedly has a `Response` column:

```python
if "Response" in features_df.columns:
    raise ValueError(...)
```

It emits only: per-row `risk_score` + policy decision flags, `batch_id`/`cycle_id`/`run_seq` +
timestamp, and batch-level counts/score-distribution statistics of its own predictions (never
compared against ground truth).

## `batch_id` / `cycle_id` / `run_seq` semantics

State is a small JSON file (`outputs/production/dataset_h/dataset_h_batch_state.json` by default):

```json
{
  "pointer": 50000,
  "cycle_id": 0,
  "batch_id": 5,
  "run_seq": 5,
  "dataset_rows": 1183748,
  "last_batch_start": 40000,
  "last_batch_end": 50000,
  "last_scored_at_utc": "2026-06-25T09:47:38.357826+00:00"
}
```

- **`pointer`** — row offset into `test_dataset_h.parquet` for the next batch.
- **`batch_id`** — resets to `0` at the start of each new cycle (per `tasks.md` TASK 5/"LOOP
  LOGIC": "if all batches complete → reset batch_id → increment cycle_id").
- **`cycle_id`** — increments by 1 each time the dataset wraps around (the pointer reaches the
  end and restarts at 0).
- **`run_seq`** — a separate, lifetime-global monotonic counter that **never resets**, for
  unambiguous whole-history ordering even across cycle wraparounds. This is intentionally distinct
  from `batch_id` rather than overloading one field with two meanings.

Example: with `dataset_rows=1,183,748` and `--batch-size 10000`, after ~119 batches the dataset
wraps: `batch_id` goes `...118, 0` and `cycle_id` goes `0, 1`, while `run_seq` just keeps
incrementing (`...118, 119`).

## Local output

```
outputs/production/dataset_h/cycle={cycle_id}/batch={batch_id}/predictions.parquet
outputs/production/dataset_h/dataset_h_batch_stats_log.csv      # one row appended per batch
outputs/production/dataset_h/dataset_h_batch_state.json         # current pointer/cycle/batch/run_seq
```

Per-row parquet columns: `Id, risk_score, decision, auto_reject, manual_inspect, batch_id,
cycle_id, run_seq, scored_at_utc`. No `Response`.

This whole directory is gitignored (`outputs/production/`) — it's generated, append-only, runtime
state, not something to commit.

## S3 output

Same partition scheme, mirrored to S3:

```
predictions/cycle={cycle_id}/batch={batch_id}/predictions.parquet
```

See [`aws_s3.md`](aws_s3.md) for the bucket/credentials this depends on, and how to list/inspect
what's there.

## `--no-s3` local smoke mode

```bash
python scripts/pipeline/run_production_inference.py --no-s3
```

Writes the local parquet, advances state, but skips the S3 upload entirely (prints
`[S3] --no-s3 set: upload will be SKIPPED, local-only run.` and
`s3_upload=skipped (--no-s3); would-be s3_key=...`). Default behavior (no flag) is **S3-enabled**,
matching the rest of the production pipeline. Use `--no-s3` for local testing without touching S3,
or when you don't have AWS credentials configured yet.

## Append-only behavior and retry/failure semantics

Both the local write and the S3 upload are append-only — neither will ever overwrite existing
data:

- **Local:** if `outputs/production/dataset_h/cycle={n}/batch={n}/predictions.parquet` already
  exists, the script raises `FileExistsError` immediately and does no work.
- **S3:** `src/utils/s3_utils.upload_file_append_only()` does a `head_object` check before
  uploading; if the key already exists, it raises `FileExistsError` instead of overwriting.

**Upload-then-advance ordering:** the batch is first written to a `.tmp` file, then uploaded to
S3 (or skipped via `--no-s3`), and only *then* renamed to the final, guarded local path. State
(`dataset_h_batch_state.json`) is written immediately after that rename, and the stats-log CSV is
appended last. This means:

- If the S3 upload fails (existing key, credentials, network, client error), it **raises and the
  process exits non-zero** — never prints-and-continues. The final local path was never created,
  so state was never advanced, so **the exact same batch is safely retried on the next
  invocation** — nothing needs to be cleaned up manually in the common case.
- The only manual-recovery scenario is if a prior run uploaded to S3 successfully but then crashed
  *before* finishing local finalization (a narrow window — see the error message from
  `upload_file_append_only` for what to check: delete the stray S3 key or local `.tmp` file before
  retrying). See [`aws_s3.md`](aws_s3.md) "manual recovery notes" and
  [`troubleshooting.md`](troubleshooting.md).

## Commands

**One batch** (default paths, S3-enabled):

```bash
python scripts/pipeline/run_production_inference.py
```

Prints `cycle_id=`, `batch_id=`, `run_seq=`, `rows_scored=`, `flagged_count=`, `pred_mean=`,
`output_path=`, and either `s3_key=...` or `s3_upload=skipped (--no-s3); ...`.

**Multiple batches.** The script intentionally processes exactly one batch per invocation and
exits (per `tasks.md` TASK 6: "DO NOT process all batches in one run — simulate real-time
system"). To advance multiple batches, invoke it multiple times — there is no built-in loop flag,
and adding one would contradict that constraint:

```bash
for i in 1 2 3; do python scripts/pipeline/run_production_inference.py; done
```

For a continuously-running simulation, schedule this command externally (cron, a systemd timer,
or similar) — see [`ec2_deployment.md`](ec2_deployment.md).

**Custom batch size / paths** (useful for testing without touching the real state file):

```bash
python scripts/pipeline/run_production_inference.py \
  --batch-size 500 \
  --state-path /tmp/state.json \
  --output-dir /tmp/out \
  --stats-log-path /tmp/stats.csv \
  --no-s3
```

## Inspecting state/stats/output

```bash
# Current pointer/cycle/batch/run_seq
cat outputs/production/dataset_h/dataset_h_batch_state.json

# Per-batch history (one row per invocation): includes batch bounds, flagged counts,
# score-distribution stats, the S3 key attempted, and whether the upload succeeded
column -s, -t outputs/production/dataset_h/dataset_h_batch_stats_log.csv | less -S

# Read a specific batch's predictions
python -c "
import pandas as pd
df = pd.read_parquet('outputs/production/dataset_h/cycle=0/batch=2/predictions.parquet')
print(df.head())
print('Response' in df.columns)  # always False
"
```
