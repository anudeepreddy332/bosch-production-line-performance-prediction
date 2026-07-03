# EC2 Deployment

> **Track:** All · **Status:** Planned, not verified

No EC2 instance was provisioned or tested while writing this
runbook. Every command below is derived from the verified local/Docker behavior documented in the
other runbooks ([`local_setup.md`](local_setup.md), [`docker.md`](docker.md),
[`aws_s3.md`](aws_s3.md)) — treat it as "should work" guidance, not a confirmed procedure. Where a
step depends on a gap from another runbook, that dependency is called out explicitly rather than
hidden. As of the Docker/S3 hardening phase, the dashboard's `boto3`/credential gaps described
below are resolved — see [`docker.md`](docker.md) and [`aws_s3.md`](aws_s3.md) for what changed
and what's still unverified (a real EC2 instance, specifically).

## Instance prerequisites

- Amazon Linux 2023 or Ubuntu 22.04+, `x86_64` (no GPU needed — this is a LightGBM/pandas
  workload, not deep learning).
- At least 4 vCPU / 16 GB RAM for comfortable batch scoring of the ~1.18M-row test set; the raw
  Bosch CSVs are several GB each, so provision disk accordingly (50+ GB) if you intend to
  regenerate `data/processed/*` from `data/raw/` rather than relying on already-committed
  artifacts.
- Python 3.11, git.

## Cloning the repo

```bash
sudo yum install -y git python3.11 python3.11-pip   # Amazon Linux 2023
# or: sudo apt-get install -y git python3.11 python3.11-venv   # Ubuntu

git clone <this repo's URL> bosch-production-line-performance
cd bosch-production-line-performance
```

## Setting env vars / secrets

**Prefer an IAM instance role over static `.env` credentials on EC2** — see
[`aws_s3.md`](aws_s3.md)'s credential guidance. If you attach an instance role with the right S3
permissions, `src/utils/s3_utils.py`'s `boto3.client(...)` call will still need `AWS_ACCESS_KEY`/
`AWS_SECRET_KEY` to be unset (not empty strings) for boto3 to fall through to the instance role —
as currently written, `s3_utils.py` always passes `aws_access_key_id=os.getenv("AWS_ACCESS_KEY")`
explicitly; passing `None` (the env var simply absent) is fine and boto3 will fall back to the
instance role correctly, but you should verify this directly rather than assume it, since it
hasn't been tested on this branch.

If you do use static keys instead (e.g. for a quick demo, not recommended for anything
longer-lived):

```bash
cat > .env <<'EOF'
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
AWS_REGION=...
AWS_BUCKET_NAME=...
EOF
chmod 600 .env
```

Never commit this file; `.gitignore` already excludes `.env`.

**Recommended IAM role policy scope** (attach to the instance role, not as static keys):
`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, `s3:HeadObject` on the specific bucket and the
`predictions/`, `data/`, `outputs/` prefixes this project uses — not account-wide S3 access.

## Installing dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(`environment.yml` was removed in PF2 of the portfolio master plan — `requirements.txt` is now the
single, pinned dependency source for both local setup and Docker.)

`boto3` is included in `requirements.txt` as of the Docker/S3 hardening phase — no separate
install step needed.

## Running batch inference

One batch per invocation, exactly as documented in
[`track3_production_inference.md`](track3_production_inference.md):

```bash
python scripts/pipeline/run_production_inference.py
```

`tasks.md` is explicit that this should **not** loop indefinitely in-process ("simulate real-time
system"). For a continuously-running simulation on EC2, schedule it externally rather than writing
a new looping wrapper:

- **systemd timer** (preferred for a long-lived instance): a `.timer` unit firing on whatever
  interval simulates your desired batch cadence, triggering a `.service` unit that runs
  `python scripts/pipeline/run_production_inference.py` once.
- **cron**: `*/5 * * * * cd /path/to/repo && /path/to/.venv/bin/python scripts/pipeline/run_production_inference.py >> /var/log/bosch_batch.log 2>&1`
- **tmux**, only for interactive/manual testing (run a few batches by hand, watch the output) —
  not a substitute for a real scheduler in a deployed environment.

## Running dashboard/API

**Directly (without Docker):**

```bash
# API
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# Dashboard (separate process/terminal — both need the same env active)
streamlit run apps/streamlit_dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

Run each under `tmux` for a quick manual deployment, or wrap each in its own `systemd` service for
anything persistent (restart-on-crash, log capture via journald).

**Via Docker:** `docker compose up --build` now builds both containers successfully and the
dashboard's `boto3 import → src.utils.s3_utils` chain has been verified working inside the built
image (host-side verification — see [`docker.md`](docker.md) for exact commands/output). This was
**not** re-verified specifically on an EC2 instance; the underlying Docker behavior should be
identical, but EC2-specific factors (instance role propagation into the container, network
reachability to S3 from inside Docker's bridge network) have not been tested there.

## Security group ports

| Port | Purpose | Recommended source |
|---|---|---|
| `22` | SSH | Your IP / bastion only — never `0.0.0.0/0` |
| `8000` | FastAPI | Trusted IPs, or behind a load balancer / reverse proxy with its own auth |
| `8501` | Streamlit dashboard | Trusted IPs, or behind a load balancer; Streamlit has no built-in auth |

Neither the API nor the dashboard implement authentication as currently written — do not expose
either port to `0.0.0.0/0` in any environment that isn't a fully disposable demo, since the
dashboard's View B pages and the API's `/predict` endpoint require no credentials to reach.

## IAM role recommendation

One instance role, attached to the EC2 instance (not baked into AMI, not static keys), scoped to:
- the project's S3 bucket only, for the prefixes this project writes/reads
  (`predictions/`, `data/`, `outputs/`)
- no other AWS services

This is a recommendation based on standard least-privilege practice, not a policy that has been
written and tested against this specific bucket's actual usage on this branch.

## What is verified vs. planned (summary)

| Step | Status |
|---|---|
| Local Python env setup (`boto3` included in `requirements.txt`) | Verified (see [`local_setup.md`](local_setup.md)) |
| `scripts/pipeline/run_production_inference.py` batch scoring, append-only S3 upload | Verified locally (see [`track3_production_inference.md`](track3_production_inference.md)) |
| FastAPI `/health`/`/predict`/`/batch_predict` | Verified locally, not specifically on EC2 |
| Streamlit dashboard, both views | Verified locally, not specifically on EC2 |
| `docker compose build` + `docker compose up api` | Verified locally (see [`docker.md`](docker.md)) |
| Dashboard container's `boto3`/S3-import chain (`docker compose run --rm dashboard python -c "..."`) | Verified locally (see [`docker.md`](docker.md)) |
| Full `docker compose up dashboard` + loading the UI in a real browser | **Not verified** — see [`docker.md`](docker.md) "Validation performed" for the exact scope of what was and wasn't checked |
| Everything EC2-specific in this document (instance setup, IAM role, systemd/cron scheduling, security groups, an instance role's actual S3 access from inside a container) | **Planned only — not provisioned or tested** |
