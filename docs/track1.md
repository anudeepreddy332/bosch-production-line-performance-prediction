# Track 1 — Offline Training + Evaluation

**State:** Frozen (`track1-frozen`). **Governing log:** [Research Log](research/README.md), entries
DR-001–DR-015 in `docs/research/decisions.md`.

This page answers a fixed set of questions for every track landing page in this site: objective,
input data, approach, measured results (with caveats), known limitations, and where the evidence
lives. Track 2 and Track 3 mirror this structure.

## Objective

Turn labeled Bosch training data into an approved, frozen model plus a decision threshold — the
only place in this repository where supervised metrics (MCC, precision, recall, confusion matrix)
are computed, and the only track allowed to compute them.

## Input data

`data/processed/train_*.parquet` — labeled (`Response` present). Chunk-aware, leakage-safe CV
(`StratifiedGroupKFold` grouped by `chunk_id`, derived from sorted `start_time`) so no chunk's rows
appear in both train and validation.

## Approach

Four models, stacked:

- **baseline** — start_time/duration/feature_mean/rolling counts/density/chunk columns only.
- **dataset_g** — adds OOF-safe failure-rate target features (chunk/signature/path/rolling),
  computed fold-by-fold from training-fold statistics only.
- **dataset_h** — adds path-transition and station-pair co-occurrence risk features. **This is the
  approved production candidate.**
- **meta_model** — stacks the three OOF predictions plus mean/std/max/agreement-count into a final
  LightGBM model.

## Results (honest, with caveats)

Full-scale run, 1,183,747 rows:

| Model | OOF MCC | Best threshold |
|---|---|---|
| baseline | 0.02254 | 0.36 |
| dataset_g | 0.13662 | 0.90 |
| **dataset_h** | **0.15337** | **0.91** |
| meta_model | 0.14942 | 0.96 |

**Caveat 1 — stacking is regressive here.** The meta-model (0.14942) scores *below* `dataset_h`
alone (0.15337). Stacking three OOF predictions and adding agreement features does not help on
this data; `dataset_h` is deployed, not the meta-model.

**Caveat 2 — a single in-CV number overstates deployed performance.** A rolling-origin evaluation
(`train_e3_rolling_origin.py`) scoring `dataset_h` at its fixed threshold (0.91) against 5
out-of-time folds gives a **0.06–0.18 MCC range**, not one number (mean 0.11944, 95% CI
[0.05235, 0.18653]). See [Results](results.md#temporal-robustness-rp2-rolling-origin-evaluation)
for the full per-fold table.

## Known limitations

- Threshold is static; it is not re-tuned as the score distribution drifts over time (Track 3's
  drift monitoring exists to detect when this matters, not to correct it automatically).
- `chunk_id` is itself a model feature in some variants — a documented, not hidden, design choice
  (see `docs/production_readiness_audit.md` §7.5 for the critique that motivated re-examining it).

## Where the evidence lives

- Decision log: `docs/research/decisions.md`, DR-001–DR-015.
- Machine-readable metrics: `outputs/training_summary.json`.
- Reproducibility status (what's regenerable vs. historical): [Data Card](data_card.md).
- Runbook: [Track 1 offline evaluation](runbooks/track1_offline_evaluation.md).
- Tag: `track1-frozen`.

## Why not just one "production" track?

`bosch_agent.md` and `system_design.md` originally described a two-way training/production split,
with Kaggle either out of scope or absent. That framing collapses two genuinely different
consumers of the trained model — a one-shot Kaggle leaderboard submission and an ongoing
label-free monitored inference stream — into one bucket, which is what previously let Kaggle
submission go unbuilt while "production" absorbed supervised-metric logic that belongs here in
Track 1. Splitting into three tracks makes each one's data contract (labeled vs. unlabeled,
one-shot vs. ongoing) independently checkable. Full audit: `docs/ml_system_tracks.md`.
