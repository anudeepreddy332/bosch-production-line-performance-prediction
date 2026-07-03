# Local Setup

> **Track:** All · **Status:** Verified

How to get this repo runnable on your own machine: environment, required local artifacts, `.env`
variables, and sanity checks. Everything below was verified against the current branch
(`feature/dataset-h-production-batch-inference`) using a conda env named `bosch`.

## 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

(`environment.yml` was removed in PF2 of the portfolio master plan — `requirements.txt` is now the
single, pinned dependency source for both local setup and Docker.)

`boto3` is included in `requirements.txt` (added during the Docker/S3 hardening phase), so the
install above already gives you everything Track 3's upload, `src/utils/s3_utils.py`, and the
dashboard's Production Monitoring page need for S3 access — no separate install step required.

## 2. Required local files/artifacts

Most of these are already committed in this repo snapshot. Verify before assuming a script will
work:

| Path | Used by | Present in this repo? |
|---|---|---|
| `models/{baseline,dataset_g,dataset_h,meta_model}_model.pkl` | Track 1/2/3 inference | Yes — committed despite `models/*.pkl` being gitignored (intentionally force-added) |
| `data/features/dataset_h_lookup.json` | Track 2/3 dataset_h feature building | Gitignored, regenerate via `scripts/pipeline/build_dataset_h.py` if missing |
| `data/features/test_dataset_h.parquet` | Track 2 (Kaggle) and Track 3 (production inference) input | Yes — regenerate via `scripts/pipeline/build_test_dataset_h.py` if missing |
| `data/features/meta_dataset.parquet`, `oof_predictions_final.parquet` | Track 1 dashboard pages, drift monitoring | Yes |
| `data/processed/sample_submission.parquet` | Track 2 row-count/Id sanity check | Yes (1,183,748 rows — full Kaggle scale, not the 50k dev sample) |
| `outputs/max_recall_system_summary.json` | Default decision policy (`threshold_high`, `inspection_budget_pct`) for the API and Track 3 | **Does not exist in this repo.** Both `apps/api/main.py` and `src/inference/decision_engine.load_policy()` fall back to `DecisionPolicy(threshold_high=0.60, inspection_budget_pct=5.0)` when it's missing — this is expected, not an error. |
| `.env` | All S3 access via `src/utils/s3_utils.py` | You must create this yourself (gitignored, never commit it) |

## 3. `.env` variables

`src/utils/s3_utils.py` reads these via `python-dotenv`:

```
AWS_ACCESS_KEY=<your access key id>
AWS_SECRET_KEY=<your secret access key>
AWS_REGION=<bucket region, e.g. ap-south-2>
AWS_BUCKET_NAME=<your bucket name>
```

`apps/streamlit_dashboard/app.py` now imports its S3 client and bucket name directly from
`src.utils.s3_utils` (`from src.utils.s3_utils import BUCKET_NAME, s3`), so these four `.env`
variables are the single source of truth for S3 access across scripts and the dashboard alike —
there is no separate hardcoded bucket/region or credential path to worry about. See
[`aws_s3.md`](aws_s3.md) and [`troubleshooting.md`](troubleshooting.md) if S3 calls still fail.

Never commit `.env`. It is already in `.gitignore`.

## 4. Sanity checks

Run these after setup, before touching any of the track-specific runbooks.

**Imports:**

```bash
python -c "import pandas, numpy, pyarrow, sklearn, lightgbm, streamlit, plotly, fastapi, boto3, dotenv; print('imports OK')"
```

**Model payload validity** (checks all four committed `models/*.pkl` are the Phase-2 payload
format the inference scripts expect, plus the `dataset_h` lookup-table/model fingerprint match):

```bash
PYTHONPATH=. python scripts/ops/validate_model_payload.py
```

Expected tail of output (verified on this branch):

```
=== Best-effort check of committed models/*.pkl (if present) ===
  baseline_model.pkl: valid payload (model_name='baseline')
  dataset_g_model.pkl: valid payload (model_name='dataset_g')
  dataset_h_model.pkl: valid payload (model_name='dataset_h')
    checking dataset_h lookup dependency: .../data/features/dataset_h_lookup.json
    PASS: lookup present and data_fingerprint matches (...) -- dataset_h inference is runnable.
  meta_model.pkl: valid payload (model_name='meta_model')
```

Exit code `0` means all payloads and the `dataset_h` lookup cross-check passed. Exit code `1`
means at least one is broken — read the printed `FAIL`/`INVALID` lines for which one.

**Production system output validation** (after running `scripts/pipeline/run_full_system.py` at least
once — see [`track3_production_inference.md`](track3_production_inference.md)):

```bash
python scripts/pipeline/validate_system.py
```

This sanity-checks `outputs/production_decision_summary.json`,
`outputs/batch_simulation_summary.json`, and `outputs/monitoring/evidently_summary.json` for
range/invariant violations and cross-consistency. There is no other test suite in this repo (no
`tests/`, no pytest config) — this script is the closest thing to one.

## 5. Quick per-track smoke commands

Once the above passes, each track has its own runbook with full detail. The fastest checks:

```bash
# Track 1 (labeled, offline) -- should run and print threshold/budget output
python scripts/pipeline/build_decision_summary.py

# Track 3 (label-free, production) -- writes one local batch, skips S3
python scripts/pipeline/run_production_inference.py --no-s3

# Dashboard (both views)
streamlit run apps/streamlit_dashboard/app.py
```
