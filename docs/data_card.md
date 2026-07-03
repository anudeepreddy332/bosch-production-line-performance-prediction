# Data Card

## Source and license

Data: **Bosch Production Line Performance**, a Kaggle competition dataset. It is governed by
Kaggle's competition-specific data terms, not a general open-data license:

- The dataset may be used **only** for the competition and directly related research/portfolio
  work referencing this repository — it is **not** freely redistributable.
- Access requires a Kaggle account, accepting the competition rules on kaggle.com, and downloading
  through Kaggle's own mechanism (the competition page's "Download" or the `kaggle competitions
  download` CLI) — not a link in this repository.
- **This repository never commits the raw or processed data.** `data/raw/`, `data/processed/*`
  (except one provenance file), and `data/features/*.{parquet,pkl,json}` are all gitignored (see
  `.gitignore`). What ships in git is code, model pickles, and metadata — never the Bosch CSVs
  themselves, in any form.
- If you fork or clone this repository, you must obtain the data yourself from Kaggle under your
  own account; nothing here bypasses that requirement.

## Dataset shape

Anonymized manufacturing measurements from Bosch production lines: numeric sensor readings, date
(timing) features, and categorical features per part, with a binary `Response` (failure) label on
the training split only.

| Split | Rows | Positive rate |
|---|---|---|
| Train (full) | 1,183,747 | ~0.58% (6,879 positives) |
| Test (full, unlabeled) | 1,183,748 | unknown (Kaggle holds out) |

The severe class imbalance (~0.58% failures) is the central modeling difficulty this whole
repository is built around — see [Track 1](track1.md) for how the training pipeline handles it
(chunk-aware CV, cost-weighted threshold selection).

## How to obtain and prepare the data

1. Download the competition ZIP from Kaggle (requires an account + accepted competition rules).
2. Convert to Parquet — this is the **only** repository entrypoint that touches the raw CSVs:

   ```bash
   # Full data, no sampling (the default — omitting --sample-rows means full data)
   python scripts/pipeline/prepare_data.py --zip-path <path-to-bosch-zip>
   ```

   Pass `--sample-rows N --sample-tag <label>` instead for a smaller, explicit dev sample; the
   resulting `data/processed/PROVENANCE.json` records which mode produced the files on disk, so
   the committed metadata is self-documenting about what a given run actually contains.
3. Proceed with `build_dataset_{baseline,g,h}.py` → `train_*.py` per the root `README.md`
   quickstart.

## Fingerprint verification

Every trained model records a **data fingerprint** — a 16-character hex digest
(`src/training/modeling.py::compute_data_fingerprint`) computed from the sorted feature-column
list, row count, a hash of the sorted `Id` column, and the sum of the target column. Two runs
produce the same fingerprint if and only if they trained on the same rows, the same feature set,
and the same label distribution — it is how this repository verifies "my local rerun matches the
recorded result" without redistributing the data itself.

| Model / experiment | Data fingerprint | Recorded in |
|---|---|---|
| `dataset_h` (production) | `a5bb652f2b20aca6` | `outputs/training_summary.json` |
| `baseline` (production) | `b2b69d3289bd69f0` | `outputs/training_summary.json` |
| `dataset_g` (production) | `0487c5e79a3a0445` | `outputs/training_summary.json` |
| `meta_model` (production) | `ebfd40c6c929d7af` | `outputs/training_summary.json` |
| K1 (Kaggle, reproduces `dataset_h`) | `a5bb652f2b20aca6` | `results/leaderboard.json` — matches `dataset_h` exactly, confirming K1 is a byte-for-byte reproduction |
| K5-A (Kaggle) | `e9df7ffff186b6fa` | `results/leaderboard.json` — reused as the fixed regression-anchor target for every subsequent wide-modeling experiment |
| P0 (Kaggle) | `f4a25438ad901355` | `results/leaderboard.json` |
| P1 (Kaggle) | `c602cd26810cf626` | `results/leaderboard.json` |

If your local rerun of `train_dataset_h.py` produces a different fingerprint than
`a5bb652f2b20aca6`, the input data, feature set, or row/label counts differ from the recorded
run — treat any metric comparison as invalid until the fingerprint matches.

## Model artifact distribution

As of `v1.0.0`, the four production model pickles (`models/{baseline,dataset_g,dataset_h,
meta_model}_model.pkl`, ~93 MB total) are **no longer tracked in git** — `git rm --cached` removed
them from `HEAD` without rewriting history (they remain retrievable from any commit or tag before
this release, e.g. `git show track1-frozen:models/dataset_h_model.pkl`). They're distributed as
GitHub Release attachments instead:

```bash
gh release download v1.0.0 --dir models --pattern '*.pkl'
```

Each downloaded pickle should reproduce the data fingerprint in the table above when re-scored;
if it doesn't, re-download rather than trust a partial/corrupted file. Regenerating from scratch
(`python scripts/pipeline/train_dataset_h.py`, etc. — see the root `README.md` quickstart)
produces a fingerprint-identical model without needing the Release at all.

## Known data-provenance caveats

- The committed `data/processed/PROVENANCE.json` (the one data file tracked in git) records the
  full-scale run (1,183,747 train rows) that produced every number on this site. An earlier
  revision of `docs/reproducible_metrics_report.md` described a stale 50,000-row "World A" dev
  sample as the current committed artifact set; that description no longer matches
  `PROVENANCE.json` or `outputs/training_summary.json` and is tracked as a documentation-freshness
  item in the portfolio engineering backlog (`docs/implementation/portfolio_master_plan.md` §11),
  not restated as current fact here.
- A separate historical file, `data/features/oof_predictions_context_meta_v2_blend.parquet`
  ("World B"), predates this fingerprinting practice and cannot be regenerated from any script in
  this repository — see `docs/reproducible_metrics_report.md` §2 for the full account. No number
  on this site is sourced from that file.
