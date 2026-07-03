# Track 2: Kaggle Submission

## What it is

Track 2 generates a Kaggle leaderboard submission: unlabeled Kaggle test data in, `Id,Response`
`submission.csv` out, using an already-approved model and its training-time threshold. It is a
one-shot batch job, distinct from Track 3's ongoing simulated-stream inference — see
[`docs/ml_system_tracks.md`](../ml_system_tracks.md) for why these are split into separate tracks
rather than one "production" bucket.

Full design notes, what the script does and does not do, and the history of why this track was
blocked are in [`docs/kaggle_submission.md`](../kaggle_submission.md) — **this runbook does not
duplicate that document**. Read it for the "why"; read this runbook for the current "how."

## Current state: `dataset_h` works end-to-end, the other three models do not

`docs/kaggle_submission.md` documents two historical blockers: no test-side feature-engineering
script, and committed `models/*.pkl` being pre-Phase-2 bare estimators. As of this branch, **both
gaps are resolved specifically for `dataset_h`**:

- `scripts/pipeline/build_test_dataset_h.py` exists and has already been run —
  `data/features/test_dataset_h.parquet` is committed (1,183,748 rows, real Kaggle scale, no
  `Response` column).
- `models/dataset_h_model.pkl` is a valid Phase-2 payload (verified via
  `scripts/ops/validate_model_payload.py` — see [`local_setup.md`](local_setup.md)).

**This was verified by actually running the command below**, not just by inspecting the code.

The other three models (`baseline`, `dataset_g`, `meta_model`) still have no test-side
feature-engineering script — there is no `build_test_dataset_baseline.py`,
`build_test_dataset_g.py`, or test-side meta-feature builder in this repo. Running
`generate_submission.py` against those models will fail at the missing-feature-columns check (see
"Validation checks" below). This is a real, unresolved gap, not something this runbook works
around.

## How `dataset_h` Kaggle submission generation works

`scripts/generate_submission.py`:
1. Loads `models/dataset_h_model.pkl` and validates it's a Phase-2 payload (`{"models",
   "feature_cols", "threshold", ...}`).
2. Loads `data/features/test_dataset_h.parquet`, checks it has `Id` plus every column in the
   payload's `feature_cols`.
3. Averages `predict_proba` across every CV-fold model in the payload.
4. Applies the payload's stored threshold (or `--threshold` if you override it).
5. Writes `Id,Response` to `--output`.

It never reads or scores against a `Response` column even if one is present in the input (it
isn't, here, by construction of `build_test_dataset_h.py`).

## Input/output contract

| | Contract |
|---|---|
| Input | `data/features/test_dataset_h.parquet`: `Id` + every column in `src.features.dataset_h_pipeline.DATASET_H_FEATURE_COLS`. No `Response`. |
| Model | `models/dataset_h_model.pkl`: Phase-2 payload dict (`models`, `feature_cols`, `threshold`, ...) |
| Output | `outputs/submission.csv` (or `--output`): exactly `Id,Response`, `Response` is `0`/`1` |

## No local supervised metrics on test data

Kaggle test labels are hidden — there is no ground truth to compute MCC/precision/recall/accuracy
against locally for this track, and the script has no code path that would do so. The only
"checks" performed are structural (row count, Id-set membership) against
`data/processed/sample_submission.parquet`, never against the predicted `Response` values.

## Commands and validation checks

```bash
PYTHONPATH=. python scripts/generate_submission.py \
  --model-path models/dataset_h_model.pkl \
  --test-features data/features/test_dataset_h.parquet \
  --output outputs/submission.csv
```

Verified output on this branch:

```
model_name='dataset_h'
threshold_used=0.91
output_path=outputs/submission.csv
row_count=1183748
positive_prediction_count=2993
```

No `WARNING:` lines were printed — `data/processed/sample_submission.parquet` also has 1,183,748
rows, so the row-count/Id check against it passed silently. If you ever see
`WARNING: row count ... != sample_submission row count ...` or an Id-mismatch warning, your
`--test-features` doesn't match the real Kaggle test set scale (e.g. you pointed it at a dev-sample
artifact) — re-check which `test_dataset_h.parquet` you built.

`outputs/submission.csv` is gitignored (`outputs/*.csv`) — a generated submission is never
accidentally committed.

### Trying the other three models (to see the current failure mode)

```bash
PYTHONPATH=. python scripts/generate_submission.py \
  --model-path models/baseline_model.pkl \
  --test-features data/features/test_dataset_h.parquet \
  --output outputs/submission.csv
```

This will print one `ERROR: ... is missing N feature column(s) ...` line and exit 1 — no
traceback, no partial output file — because `dataset_h`'s feature table doesn't match
`baseline`'s `feature_cols`, and there is no `baseline`-shaped test feature table to point it at
instead. This is the expected, documented gap, not a bug in this runbook's instructions.
