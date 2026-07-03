# Dashboard

> **Track:** 1 (View B) + 3 (View A) · **Status:** Verified

## Running Streamlit locally

```bash
streamlit run apps/streamlit_dashboard/app.py
```

Opens at `http://localhost:8501`. `DATA_SOURCE` (env var, default `local`) picks the data source:

- **`local` (default, credential-free):** both views read directly from disk —
  `data/features/{meta_dataset,oof_predictions_final}.parquet` for View B, `outputs/production/*/
  cycle=*/batch=*/predictions.parquet` for View A. `src.utils.s3_utils` (which builds a boto3
  client at import time) is never imported in this mode, so no AWS credentials are required.
- **`s3`:** both views read the same files from the bucket named by `AWS_BUCKET_NAME` in `.env` —
  see "What S3 data this dashboard expects" below and [`aws_s3.md`](aws_s3.md).

Set `DATA_SOURCE=s3` in `.env` (or the shell environment) to switch modes; everything below that
references "S3" applies only when running in that mode.

## View A: Production Monitoring (Track 3)

The sidebar page **"Production Monitoring (Track 3)"**. Label-free by contract: it lists every
`predictions/cycle=*/batch=*/predictions.parquet` object in S3, concatenates them, and
runtime-raises if the result ever contains a `Response` column — this view refuses to render data
that could be labeled. It shows only:

- Total predictions, latest cycle/batch/run_seq, latest `scored_at_utc`
- Flagged / auto-reject / manual-inspect counts
- A risk-score histogram
- Batch growth (rows per batch + cumulative predictions over `run_seq`)
- Top 100 risky parts by `risk_score`

**No MCC, precision, recall, accuracy, or confusion matrix anywhere on this page** — there is no
ground truth to compute them against.

This page is genuinely new and was the explicit "View A" gap documented in
[`docs/ml_system_tracks.md`](../ml_system_tracks.md) as "not yet built" — it now exists, see that
document's current state notes for full detail.

## View B: Offline Evaluation (labeled OOF data)

Every other sidebar page (Overview, Threshold Explorer, Inspection Budget Simulator, Recall at
Fixed Precision, Cost Simulator, Model Insights, Failure Analysis). These load
`data/features/meta_dataset.parquet` joined to `oof_predictions_final.parquet` from S3 — both
**labeled** Track 1 data — and compute real MCC/precision/recall/confusion-matrix figures. This is
legitimate offline model-evaluation work, not a production-inference view; see
[`track1_offline_evaluation.md`](track1_offline_evaluation.md) for why supervised metrics are
allowed here.

A one-line caption at the top of every page now states that everything except "Production
Monitoring (Track 3)" uses labeled OOF data. The underlying code in these pages
(`load_scoring_data()`, the `live_df` variable name throughout) still implies "live" data more
than it should — a known, deliberately-deferred naming cleanup, not a correctness issue (the
*data* loaded is correctly labeled OOF data; only the *names* are misleading).

## What S3 data View A expects

```
predictions/cycle={cycle_id}/batch={batch_id}/predictions.parquet
```

written by `scripts/pipeline/run_production_inference.py` — see
[`track3_production_inference.md`](track3_production_inference.md). If this prefix is empty (no
batches scored yet), the page shows a clean warning, not a crash:

> No production batches found yet under s3://.../predictions/cycle=\*/batch=\*/predictions.parquet.
> Run scripts/pipeline/run_production_inference.py to generate the first batch.

## How to refresh the cache

Both the production-batch loader and View B's loaders use `@st.cache_data`. View A's cache has a
60-second TTL (`ttl=60`) and a manual **"🔄 Refresh from S3"** button on the page that clears the
cache and reruns immediately — use this after running a new `run_production_inference.py` batch if
you don't want to wait up to 60 seconds. View B's caches have no TTL (the underlying OOF/meta
parquet files don't change during a session) — restart the Streamlit process to pick up changes
there.

## Current limitations

- **No drift/data-quality rendering.** `scripts/pipeline/run_drift_monitoring.py` already produces
  `outputs/monitoring/evidently_summary.json` + `.html`, but neither view renders it. A `grep` for
  `drift`/`evidently` in `apps/streamlit_dashboard/app.py` returns zero matches. Out of scope for
  the Docker/S3 hardening phase that fixed the items below — still open.
- **Resolved: bucket/region duplication.** The dashboard previously hardcoded its own
  `AWS_BUCKET`/`AWS_REGION` constants and built a separate boto3 client on the default credential
  chain. It now imports `BUCKET_NAME`/`s3` directly from `src.utils.s3_utils` — one client, one
  `.env`-driven source of truth, shared with `scripts/pipeline/run_production_inference.py`. See
  [`aws_s3.md`](aws_s3.md).
- **`AWS_REGION` is now a required `.env` variable for the dashboard**, not optional. The old
  hardcoded `"ap-south-2"` fallback no longer exists — if `.env` doesn't set `AWS_REGION` (and
  no other AWS config provides one), S3 calls will fail with a region-resolution error rather
  than silently using `ap-south-2`.
- **Resolved: Docker container missing `boto3`.** `Dockerfile.dashboard` now installs `boto3` and
  `python-dotenv`; the import chain that previously crashed the container immediately on start has
  been verified working inside the built image. See [`docker.md`](docker.md) for what was and
  wasn't verified.
- View B naming cleanup (`live_df` → something like `oof_eval_df`, per-page "Offline Evaluation"
  labels) is still open — only the single top-of-page caption exists today. Out of scope for this
  phase.

## Troubleshooting common S3/dashboard failures

See [`troubleshooting.md`](troubleshooting.md) for the full table. Quick pointers:

- **`S3 Load Failed: ...` on any page (View A or View B)** → both views now share one S3 client
  (`src.utils.s3_utils`), so this is the same `.env` credentials/bucket/region issue regardless of
  which page triggered it. Check `AWS_ACCESS_KEY`/`AWS_SECRET_KEY`/`AWS_REGION`/`AWS_BUCKET_NAME`
  in `.env`, or (if relying on an IAM role/default chain instead) confirm `.env` doesn't set those
  keys at all and that a region is resolvable some other way. See [`aws_s3.md`](aws_s3.md).
- **Production Monitoring page shows the empty-state warning even though you ran batches** →
  confirm you didn't use `--no-s3` when running `scripts/pipeline/run_production_inference.py` (local-only
  batches never reach S3, so View A can't see them); click "🔄 Refresh from S3" in case the
  60-second cache is stale.
- **`ModuleNotFoundError: No module named 'boto3'`** → see [`local_setup.md`](local_setup.md) §1
  for the host environment, or [`docker.md`](docker.md) if running in a container that predates
  this phase's `Dockerfile.dashboard` change.
