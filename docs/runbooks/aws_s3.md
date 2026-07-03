# AWS S3

> **Track:** 3 (+ dashboard `DATA_SOURCE=s3` mode) · **Status:** Verified

## Required bucket / env vars

`src/utils/s3_utils.py` is the **single, canonical S3 client** for this entire project —
`scripts/pipeline/run_production_inference.py`, `scripts/pipeline/run_full_system.py`, and (as of this phase)
`apps/streamlit_dashboard/app.py` all `import` its `s3` client and `BUCKET_NAME` constant rather
than building their own. It reads from `.env` via `python-dotenv`:

```
AWS_ACCESS_KEY=<access key id>
AWS_SECRET_KEY=<secret access key>
AWS_REGION=<bucket region>
AWS_BUCKET_NAME=<bucket name>
```

These four variables are now the **one source of truth** for every S3-touching part of this
project, including the dashboard. The previous duplication — the dashboard hardcoding its own
`AWS_BUCKET`/`AWS_REGION` constants and building a separate `boto3.client(...)` on the default
credential chain — has been removed; `app.py` now does
`from src.utils.s3_utils import BUCKET_NAME, s3` instead. See [`docker.md`](docker.md) and
[`dashboard.md`](dashboard.md) for what this changes in practice.

**Consequences worth knowing:**
- **`AWS_REGION` is now required for the dashboard too.** The old hardcoded `"ap-south-2"`
  fallback is gone. If `.env`/the environment doesn't set `AWS_REGION`, the shared client passes
  `region_name=None` to boto3, which only resolves a region from elsewhere (e.g.
  `~/.aws/config` or an EC2 instance's configured default) if one is available — otherwise S3
  calls will fail with a region-resolution error.
- **If `.env` has stale or wrong explicit keys, they take precedence over a working default
  credential chain** (e.g. an EC2 IAM instance role). `s3_utils.py` always passes whatever
  `AWS_ACCESS_KEY`/`AWS_SECRET_KEY` it finds explicitly to `boto3.client(...)` — it does not
  conditionally omit bad values. If you intend to rely on an instance role, `.env` must not set
  these keys at all (not even to empty strings); simply not having the file is the safest way to
  guarantee fallback to the default chain.
- **IAM role / default credential chain remains fully viable** when `AWS_ACCESS_KEY`/
  `AWS_SECRET_KEY` are absent from the environment: `os.getenv("AWS_ACCESS_KEY")` returns `None`
  in that case, and boto3 treats `aws_access_key_id=None` as "not provided," falling through to
  its default chain (env vars, `~/.aws/credentials`, or an EC2 instance role) — provided
  `AWS_REGION` is still resolvable some other way. Nothing about this phase's change weakens that
  fallback; it now also benefits the dashboard, not just the production scripts.

## Track 3 key layout

```
predictions/cycle={cycle_id}/batch={batch_id}/predictions.parquet
```

e.g. `predictions/cycle=0/batch=2/predictions.parquet`. This mirrors the local output layout
exactly (`outputs/production/dataset_h/cycle={cycle_id}/batch={batch_id}/predictions.parquet`).
See [`track3_production_inference.md`](track3_production_inference.md) for what's in each file
and how `cycle_id`/`batch_id` advance.

## Append-only expectations

**Never overwritten, by construction:**

```python
def upload_file_append_only(local_path, s3_key):
    if key_exists(s3_key):
        raise FileExistsError(...)
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
```

`key_exists()` does a `head_object` call before every upload. If the key already exists, the
upload is refused (raises `FileExistsError`) rather than silently overwriting. This is
**failure-loud**: credential errors, network errors, and bucket errors all raise rather than being
caught and printed — the calling script (`run_production_inference.py`) exits non-zero on any of
these, never prints-and-continues. This is deliberately different from the older, separate
`upload_file()` helper in the same module (used by `run_full_system.py` for the two static JSON
summaries), which does catch and print on failure — `upload_file_append_only` is the
stricter, newer helper and is not a drop-in replacement for the older one.

## How to list/inspect prediction objects

Via the AWS CLI:

```bash
aws s3 ls s3://<bucket>/predictions/ --recursive
```

Via boto3 (uses the same client/credentials as `scripts/pipeline/run_production_inference.py`):

```bash
python -c "
from src.utils.s3_utils import s3, BUCKET_NAME
r = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='predictions/', MaxKeys=50)
print('KeyCount:', r.get('KeyCount'))
for o in r.get('Contents', []):
    print(' -', o['Key'], o['Size'], o['LastModified'])
"
```

To read one batch's predictions:

```bash
python -c "
from src.utils.s3_utils import s3, BUCKET_NAME
import pandas as pd
from io import BytesIO
obj = s3.get_object(Bucket=BUCKET_NAME, Key='predictions/cycle=0/batch=2/predictions.parquet')
df = pd.read_parquet(BytesIO(obj['Body'].read()))
print(df.head())
"
```

## How the dashboard reads S3

`apps/streamlit_dashboard/app.py`'s `list_production_batch_keys()` paginates
`list_objects_v2(Prefix="predictions/")`, keeps only keys matching
`^predictions/cycle=\d+/batch=\d+/predictions\.parquet$`, then `load_production_batches()`
downloads and concatenates every matching object via `get_object` + `pd.read_parquet(BytesIO(...))`,
cached for 60 seconds (`@st.cache_data(ttl=60)`). It runtime-raises if the concatenated frame ever
contains a `Response` column. See [`dashboard.md`](dashboard.md).

## Credential / security guidance

- **Never commit `.env` or any AWS key.** It's already gitignored; keep it that way.
- **Prefer an IAM role over static access keys** wherever the code actually runs unattended (EC2
  instance profile, ECS task role, etc.) — see [`ec2_deployment.md`](ec2_deployment.md). Static
  keys in `.env` are acceptable for local development only.
- **Scope the IAM policy to the specific bucket and `predictions/`/`data/`/`outputs/` prefixes
  this project actually uses** (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, `s3:HeadObject`)
  rather than `AmazonS3FullAccess` or account-wide access.
- The bucket name itself (`bosch-ml-production-anudeep-193116635897-ap-south-2-an`) is not a
  secret — it has been visible in this repo's source/docs throughout this project's history. The
  access key/secret pair is the only thing that must never be committed.

## Manual recovery notes: collision / finalization edge cases

Two distinct failure shapes, both raise loudly rather than corrupting data:

1. **Most common: upload itself fails** (credentials, network, the key already existing from an
   earlier, completed run). Local state (`dataset_h_batch_state.json`) was never advanced because
   the upload happens *before* the local rename that finalizes a batch — so simply **re-run the
   exact same command**. No manual cleanup needed; the retry naturally targets the same
   cycle/batch coordinates.
2. **Rare: upload succeeded, but the process crashed before local finalization finished** (the
   rename, the state write, or the stats-log append). On the next run, the script will recompute
   the same `cycle_id`/`batch_id` and attempt the same S3 key — `upload_file_append_only` will
   raise `FileExistsError` because that key genuinely already exists from the prior successful
   upload, but local state still thinks the batch is pending. The error message tells you what to
   do:
   > S3 key 's3://.../predictions/cycle={n}/batch={n}/predictions.parquet' already exists ...
   > If state still points at this batch, a prior run likely uploaded successfully but failed
   > during local finalization afterward -- inspect/delete this S3 key (or any leftover local
   > .tmp file for this batch) before retrying.

   Concretely: confirm via `aws s3 ls`/the boto3 snippet above that the S3 object exists, check
   `outputs/production/dataset_h/cycle={n}/batch={n}/predictions.parquet.tmp` for a leftover local
   temp file, delete whichever side is the stale partial artifact (usually you'd delete the S3
   object if you want a clean retry, since the local `.tmp` content and the uploaded content
   should be identical — the model and input batch are deterministic), then re-run.

See [`troubleshooting.md`](troubleshooting.md) for the symptom-first version of this.
