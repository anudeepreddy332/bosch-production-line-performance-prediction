# Results

This page elaborates on the results summary in the
[README](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis#readme) — full
tables, not just the headline numbers. Every value below is transcribed mechanically from
`results/leaderboard.json` and `outputs/training_summary.json` (repo root); neither table is
hand-typed. If you spot a discrepancy, those two files are the source of truth, not this page.

## Production track (Track 1 — honest, deployable)

Full-scale run, 1,183,747 rows, chunk-aware `StratifiedGroupKFold` CV (5 folds), leakage-safe OOF
predictions. `dataset_h` is the approved production candidate.

| Model | Rows | OOF MCC | Best threshold | Data fingerprint |
|---|---|---|---|---|
| baseline | 1,183,747 | 0.02254 | 0.36 | `b2b69d3289bd69f0` |
| dataset_g | 1,183,747 | 0.13662 | 0.9 | `0487c5e79a3a0445` |
| dataset_h | 1,183,747 | 0.15337 | 0.91 | `a5bb652f2b20aca6` |
| meta_model | 1,183,747 | 0.14942 | 0.96 | `ebfd40c6c929d7af` |

Note the meta-model stacks the three base OOF predictions and ends up **below** `dataset_h` alone
(0.14942 vs 0.15337) — stacking is regressive here, not a free win. See
[Model Card](model_card.md) for why `dataset_h`, not the meta-model, is the deployed candidate.

### Temporal robustness (RP2 rolling-origin evaluation)

A single in-CV MCC overstates what a *static* threshold does when deployed against a later time
window. `scripts/train_e3_rolling_origin.py` measures `dataset_h` at a fixed threshold (0.91)
across 5 out-of-time folds:

| Fold | Test chunks | Test positive rate | MCC (best per-fold threshold) | MCC (fixed 0.91 threshold) |
|---|---|---|---|---|
| 0 | 18–33 | 0.799% | 0.07972 | −0.00022 |
| 1 | 34–49 | 0.784% | 0.18164 | 0.03274 |
| 2 | 50–64 | 0.941% | 0.17045 | 0.04737 |
| 3 | 65–82 | 0.333% | 0.06110 | 0.05513 |
| 4 | 83–118 | 0.394% | 0.10427 | 0.04164 |

Mean MCC 0.11944, 95% CI [0.05235, 0.18653] across folds (best-per-fold threshold) — a
**0.06–0.18 range**, not a single number. This is what "Production (deployable, honest): MCC
0.06–0.18" in the README and Home page refers to. Full derivation: Case Study §7,
`outputs/e3_rolling_origin_results.json`.

## Kaggle track (Track 2 — frozen, leaderboard-only)

Nine sealed, LB-scored experiments, `track2-frozen` at `P1-result`. See
[Track 2](track2.md) and the [Research Summary](RESEARCH_SUMMARY.md) for the mechanism behind
each row; this table is the full ladder.

| Exp | KDR | OOF MCC | OOF status | Public LB | Private LB | Threshold | Hypothesis | Verdict | Tag |
|---|---|---|---|---|---|---|---|---|---|
| K1 | KDR-002 | 0.15337 | honest | 0.14389 | 0.16160 | 0.91 | — | PASS — reproducibility check | `K1-result` |
| K2 | KDR-003 | 0.37530 | contaminated | 0.31699 | 0.32702 | 0.98 | H_adjacency_dominant | CONFIRMED | `K2-result` |
| K3-A | KDR-004 | 0.31761 | honest | 0.31791 | 0.33161 | 0.98 | H_position_dominant | CONFIRMED | `K3-result` |
| K3-B | KDR-004 | 0.21171 | contaminated | 0.10065 | 0.10530 | 0.95 | H_label_contributes / H_position_optimistic | REJECTED (both) | `K3-result` |
| K4 | KDR-005 | 0.32192 | honest | 0.31697 | 0.33447 | 0.98 | H_cohort_modest | CONFIRMED (low end of the pre-registered band) | `K4-result` |
| K5-A | KDR-006 | 0.32506 | honest | 0.32330 | 0.33711 | 0.98 | H_duplicate_material | CONFIRMED (label-free component) | `K5-result` |
| K5-B | KDR-006 | 0.57828 | contaminated | 0.33571 | 0.33989 | 0.95 | H_duplicate_material | CONFIRMED (via this variant's clause) | `K5-result` |
| P0 | KDR-007 | 0.37892 | honest | 0.39226 | 0.40391 | 0.96 | H_raw_dominant | CONFIRMED | `P0-result` |
| **P1** | KDR-008 | **0.39484** | honest | **0.40447** | **0.41917** | 0.89 | H_station_temporal_moderate | CONFIRMED — program best | `P1-result` |

`oof_status: contaminated` means the OOF measurement leaks label information across CV folds (e.g.
a train-neighbor `Response` lookup); it is a valid *leaderboard* measurement but must never be
compared to an honest OOF MCC. See the Research Log for the full explanation.

## Why two numbers, not one

Production MCC (0.06–0.18) and Kaggle private MCC (0.16160 → 0.41917) answer different questions
against different rules. Kaggle rewards any signal the test set exposes, including record
order/adjacency and duplicate-identity effects that a real, one-at-a-time factory scoring stream
could never provide. The gap between the two ladders — not either number alone — is the point of
running both tracks side by side. See the [Research Summary](RESEARCH_SUMMARY.md) for the full
mechanism attribution behind that gap.
