# Track 2 — Kaggle Research (frozen)

**State:** Frozen (`track2-frozen`, `KDR-009`). **Governing log:** [Research Log](research/README.md),
entries KDR-001–KDR-009 in `docs/research/kaggle_decisions.md`.

## Objective

Maximize private Kaggle leaderboard MCC via any Kaggle-legal mechanism, and attribute *why* the
leaderboard score is so much higher than the honest production ceiling — as a fully quarantined
research program that never informs production decisions.

## Input data

Kaggle's unlabeled `test_numeric.csv` / `test_date.csv` / `test_categorical.csv`. Kaggle scores
`submission.csv` off-platform; this repository never sees test labels.

## Approach

Nine sealed, LB-scored experiments (K1–K5, two variants each for K3/K5, plus P0/P1), executed as a
single continuous, pre-registered research program:

1. **K1** — reproduce the frozen `dataset_h` production model as the honest Track 2 baseline.
2. **K2–K5** — attribute a leakage-family decomposition: record-adjacency, timing-cohort geometry,
   and duplicate-identity effects, each split into an honest (label-free) and a contaminated
   (label-touching) variant to isolate which part of each gain is real vs. leaked.
3. **P0** — test whether Kaggle-legal but production-forbidden *width* (the full ~968-column raw
   numeric matrix) beats the leakage families outright.
4. **P1** — layer per-station timing and station/line aggregate features on P0, plus LightGBM
   capacity tuning, for the program-best result.

## Results (full ladder on the [Results](results.md#kaggle-track-track-2-frozen-leaderboard-only) page)

Private MCC progression: K1 0.16160 → K2 0.32702 → K3-A 0.33161 → K4 0.33447 → K5-A 0.33711 →
K5-B 0.33989 → P0 0.40391 → **P1 0.41917** (program best, `H_station_temporal_moderate` confirmed).

**Six of six pre-registered hypothesis classifications confirmed** across the program; two
(`H_label_contributes`, `H_position_optimistic`) explicitly rejected — neighbor-label leakage
conditioned on record *adjacency* does not generalize, while conditioned on record *identity*
(K5-B) it does. Full mechanism attribution: [Research Summary](RESEARCH_SUMMARY.md).

## Known limitations / non-goals

- **Nothing here informs production.** No metric or conclusion from this log may appear in the
  production decision log (`docs/research/decisions.md`) or gate any `DR`/`E` decision — a
  leaderboard score is a *lead*, never *evidence*, for anything outside this track.
- **Several winning experiments use contaminated OOF** (K2, K3-B, K5-B) — leakage that would never
  survive contact with a real, one-at-a-time factory scoring stream. The Kaggle LB is the only
  valid measurement of those models; their OOF numbers must never be compared to an honest MCC.
- **Frozen, not abandoned.** The revised realistic target for further work is ~0.435–0.445 private
  MCC (down from an original ~0.52 aspiration) — see `KDR-009` §3 for the full postmortem behind
  that revision. No `K<N>`/`P<N>` experiment is authorized without a new pre-registered KDR and
  explicit user authorization (`KDR-009` §5, unfreeze criteria).

## Where the evidence lives

- Decision log: `docs/research/kaggle_decisions.md`, KDR-001–KDR-009.
- Machine-readable ladder: `results/leaderboard.json` (repo root) — the single source of truth
  every number on this page traces to.
- Runbook: [Track 2 Kaggle submission](runbooks/track2_kaggle_submission.md).
- Code (quarantined): `src/kaggle/`, `scripts/kaggle/` — firewall-grepped at every merge; nothing
  outside those two trees may import them.
- Tags: `K1-result` … `K5-result`, `P0-result`, `P1-result`, `track2-frozen`.
