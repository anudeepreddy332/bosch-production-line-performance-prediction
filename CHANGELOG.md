# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from `v1.0.0` onward.

## [Unreleased]

## [1.0.0] — 2026-07-03

The portfolio-launch release. Summarizes the transition of this repository from a frozen
research program (three ML tracks: production, Kaggle, monitoring) into a public engineering
portfolio, executed against `docs/implementation/portfolio_master_plan.md` (PF0–PF6).

### Added

- `results/leaderboard.json` — the single machine-readable source of truth for every Kaggle
  leaderboard number quoted anywhere in this repository (PF0).
- Rewritten `README.md` and new `SYSTEM_OVERVIEW.md`, results-first and traceable to the
  leaderboard registry (PF1).
- `pyproject.toml` project metadata, `ruff`/`pytest` config, `Makefile`, hardened Dockerfiles,
  `.dockerignore`, a credential-free local dashboard mode (`DATA_SOURCE=local`) (PF2).
- CI (`.github/workflows/ci.yml`): lint, test, Docker build, `leaderboard.json` schema
  validation; `tests/` suite trains synthetic in-memory models rather than depending on any
  committed pickle (PF3).
- Recruiter dashboard at `bosch.themachinist.org` — Vite + React + TypeScript, four pages
  (Story, Decision Explorer, Model Internals, Governance & Reproducibility), reading a static
  JSON bundle exported by `scripts/ops/export_dashboard_data.py`; deployed via
  `.github/workflows/deploy-pages.yml` (PF4).
- MkDocs Material documentation site at `bosch.themachinist.org/docs/` — architecture, per-track
  landing pages, research summary, model and data cards, ADR log, runbooks (PF5).
- `CHANGELOG.md` (this file) and `.github/workflows/release.yml` / `weekly-health.yml` (PF6).

### Changed

- Track 2 (Kaggle leaderboard research) closed at `KDR-009`, frozen at private MCC **0.41917**
  (tag `track2-frozen`) after nine sealed, pre-registered experiments (K1–K5, P0, P1).
- Track 1 (offline training/evaluation) and Track 3 (production inference) frozen at
  `track1-frozen` / `track3-frozen` respectively, ahead of the portfolio transition.

### Removed

- Tracked model pickles (`models/baseline_model.pkl`, `models/dataset_g_model.pkl`,
  `models/dataset_h_model.pkl`, `models/meta_model.pkl` — ~93 MB total) untracked from `HEAD`
  in PF6. **Git history is not rewritten** — these files remain retrievable from any commit or
  tag before this release (e.g. `git show track1-frozen:models/dataset_h_model.pkl`). Going
  forward, trained models are distributed as attachments on GitHub Releases; see
  `docs/data_card.md` for the exact `gh release download` command. Rationale: a 93 MB binary
  payload in every clone provided no reproducibility benefit the training pipeline itself
  doesn't already provide (`python scripts/pipeline/train_dataset_h.py` regenerates a
  fingerprint-identical artifact), and a Release is the more conventional distribution point
  for a versioned binary artifact than a git-tracked file.

## Prehistory

Before this CHANGELOG existed, project milestones were tracked via annotated git tags rather
than semantic versions. The full tag series remains in the repository and is not superseded by
semantic versioning — it is the detailed research/engineering audit trail; `v1.0.0` and later
tags mark portfolio-facing releases specifically. Notable tags, oldest to newest:
`baseline-v1`, `phase-architecture-clarity`, `phase-feature-methodology-cleanup`,
`phase-full-scale-training-plan`, `phase-full-scale-training-results`,
`phase-dataset-h-train-serve-contract`, `phase-2-model-contract`,
`phase-3-kaggle-submission-wrapper`, `production-rp1`, `E1-result`…`E4-result` (Track 1 RP2),
`track1-frozen`, `K1-result`…`K5-result`, `P0-result`, `P1-result` (Track 2 Kaggle ladder),
`track2-frozen`, `track3-frozen`. See `docs/research/decisions.md` and
`docs/research/kaggle_decisions.md` for what each research tag closed out.
