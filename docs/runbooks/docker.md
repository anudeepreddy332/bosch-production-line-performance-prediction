# Docker

## Current files and services

| File | Builds | Exposes | Installs (pip, inside the image) |
|---|---|---|---|
| `Dockerfile.api` | `bosch_api` container | `8000` | `fastapi`, `uvicorn[standard]`, `pandas`, `numpy`, `pyarrow`, `pydantic` |
| `Dockerfile.dashboard` | `bosch_dashboard` container | `8501` | `streamlit`, `plotly`, `pandas`, `numpy`, `pyarrow`, `boto3`, `python-dotenv` |
| `docker-compose.yml` | both, with `dashboard` depending on `api` | `8000`, `8501` | — |

Both containers mount the full repo (`./:/app`) plus `./data` and `./outputs` as volumes, and set
`PYTHONPATH=/app` / `BOSCH_PROJECT_ROOT=/app`. The `dashboard` service additionally loads `.env`
via `env_file` (optional — see "S3 credentials in Docker" below).

## How to build/run with docker compose today

```bash
docker compose up --build
```

The API container has no S3/boto3 dependency — it only reads
`outputs/max_recall_system_summary.json` (or falls back to defaults) and serves `/health`,
`/predict`, `/batch_predict`.

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## API container: `joblib` import fix (historical) and PF2 cleanup

`bosch_api` previously crashed on startup with `ModuleNotFoundError: No module named 'joblib'`.
Cause: `apps/api/main.py` does `from src.inference.decision_engine import ...`, and importing any
submodule of a package always runs that package's `__init__.py` first — `src/inference/__init__.py`
unconditionally did `from .predictor import BoschPredictor` and
`from .two_stage_predictor import TwoStagePredictor`, both of which import `joblib`, which was
never installed in `Dockerfile.api` (the API doesn't use either class — it only needs
`decision_engine`).

At the time, this was fixed by making those two re-exports lazy in `src/inference/__init__.py` (a
module-level `__getattr__`, PEP 562) instead of adding `joblib` to `Dockerfile.api`.

**As of PF2** (`docs/implementation/portfolio_master_plan.md`), `BoschPredictor` and
`TwoStagePredictor` (`src/inference/predictor.py`, `src/inference/two_stage_predictor.py`) were
deleted outright — they had zero real call sites anywhere in the codebase (verified by grep before
removal) and depended on artifacts (`selected_features_top150.txt`, `train_selected.parquet`) that
were never present in this repo. `src/features/pipeline.py` (`FeaturePipeline`, their only other
consumer) was deleted with them for the same reason. The lazy-`__getattr__` workaround this section
originally described is gone too — there is nothing left to lazily import, so
`src/inference/__init__.py` is now a plain docstring-only package init. The lightweight API image's
"no joblib/FeaturePipeline dependency" property still holds, now simply because the eager import
path this fix worked around no longer exists.

## Dashboard container: what changed

The dashboard now imports its S3 client from `src.utils.s3_utils` (`from src.utils.s3_utils
import BUCKET_NAME, s3`) instead of hardcoding its own bucket/region and building a separate
boto3 client. This means:

- **`Dockerfile.dashboard` now installs `boto3` and `python-dotenv`** (previously missing,
  causing an immediate `ModuleNotFoundError: No module named 'boto3'` on container start). This
  was the actual blocker — confirmed fixed by running the exact failing import chain inside a
  built image (see "Validation performed" below).
- **The dashboard's S3 config now comes from the same `.env` variables as everything else**:
  `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `AWS_REGION`, `AWS_BUCKET_NAME`. There is one source of
  truth instead of two. See [`aws_s3.md`](aws_s3.md).

## S3 credentials in Docker

`docker-compose.yml`'s `dashboard` service now has:

```yaml
env_file:
  - path: .env
    required: false
```

`required: false` means `docker compose up` still works with no `.env` present (e.g. on an EC2
instance relying on an IAM instance role — see [`ec2_deployment.md`](ec2_deployment.md)). If
`.env` exists, its variables are loaded into the container's environment, where
`src/utils/s3_utils.py`'s `load_dotenv()` call (and boto3's own env-var fallback) can use them.

This is the **only** credential-wiring mechanism implemented in this phase. Two other options
were considered and are documented, not implemented, here:
- **Mount `~/.aws:/root/.aws:ro`** as a volume to use your host's AWS CLI profile instead of
  `.env`. Not added to `docker-compose.yml` — add it yourself if you prefer this over `.env`.
  Compose's `env_file` and an `~/.aws` mount are not mutually exclusive; whichever ends up
  resolving to real credentials wins.
- **IAM instance role with no credentials wired into Compose at all** — the recommended approach
  once this is actually deployed on EC2 (see [`aws_s3.md`](aws_s3.md) credential guidance). Works
  today with this `env_file` change as-is: if `.env` doesn't exist, `required: false` is a no-op
  and boto3 falls through to the instance role automatically.

**Docker Compose version note:** `env_file` with the `path`/`required` sub-key form requires a
reasonably recent Compose. Verified working on Docker Compose v5.1.4
(`docker compose -f <file> config` resolves a `required: false` entry pointing at a nonexistent
file with no error). If your installed Compose is much older and rejects this syntax, fall back to
the simpler single-string form (`env_file: .env`) — but then `.env` must exist, or `up` will fail.

## What's still a gap

- **`AWS_REGION` is now required for the dashboard, not just convenient.** The previous hardcoded
  `"ap-south-2"` fallback is gone — if `.env`/the container environment doesn't set `AWS_REGION`,
  `src/utils/s3_utils.py`'s `boto3.client("s3", region_name=os.getenv("AWS_REGION"))` will pass
  `region_name=None`, and boto3 will only resolve a region if one is configured elsewhere (e.g.
  `~/.aws/config` or an EC2 instance's default region). See [`aws_s3.md`](aws_s3.md).
- **If `.env` has stale or wrong explicit keys, they take precedence over a working default
  credential chain** (e.g. an EC2 instance role) — `s3_utils.py` always passes whatever
  `AWS_ACCESS_KEY`/`AWS_SECRET_KEY` it finds (even empty/wrong ones) explicitly to `boto3.client(...)`,
  it does not conditionally omit them. Make sure `.env` is absent entirely (not present-but-wrong)
  if you want the instance role to be used.
- **`Dockerfile.api` was intentionally left unchanged** — it has no S3 dependency, so it's out of
  scope for this phase.
- Drift monitoring (`scripts/run_drift_monitoring.py`) is still not part of either dashboard view
  and was out of scope for this phase too — see `docs/ml_system_tracks.md`.

## Validation performed

Run on this branch, Docker Desktop (Docker 29.5.3, Compose v5.1.4), across both the dashboard S3
consolidation and the API `joblib` fix described above. This phase is **not yet committed** — the
results below are from direct local validation against the working tree, not from a commit/PR.

```bash
docker compose build
```

```bash
docker compose run --rm dashboard python -c \
  "import boto3, dotenv; from src.utils.s3_utils import s3, BUCKET_NAME; print('ok', BUCKET_NAME)"
```

```bash
docker compose up -d
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8501
```

Both images build successfully. The dashboard import chain that previously failed
(`ModuleNotFoundError: No module named 'boto3'`) now succeeds inside the container.
`bosch_api` previously crashed on startup with `ModuleNotFoundError: No module named 'joblib'`
(see "API container: `joblib` import fix" above); after that fix, `/health` returns
`{"status":"ok"}` with HTTP 200 and clean logs. The dashboard container independently returns
HTTP 200 (real Streamlit HTML, not just the import-chain check) with clean logs.

Both containers (`bosch_api`, `bosch_dashboard`) were left running intentionally after this
validation rather than torn down — run `docker compose down` whenever you're done inspecting
them.
