# Research Summary — the Kaggle Ladder, End to End

This page supersedes `KDR-009`'s condensed in-log postmortem (`docs/research/kaggle_decisions.md`
§3) as the full narrative version — the two agree; this one has room to explain *why*, not just
*what*. `KDR-009` itself, and every KDR anchor below, remain the authoritative append-only record;
nothing here amends them.

## The ladder

Nine sealed, LB-scored experiments took private MCC from **0.16160 to 0.41917** — a 2.6x
improvement over the honest production baseline, entirely through mechanisms Kaggle's rules permit
and a real factory scoring stream cannot provide.

| Exp | Mechanism | Private MCC | Verdict |
|---|---|---|---|
| K1 | Frozen `dataset_h` production baseline, no new features | 0.16160 | PASS (reproducibility) |
| K2 | Record-adjacency magic (position + train-neighbor `Response` lookup) | 0.32702 | `H_adjacency_dominant` CONFIRMED |
| K3-A | Position-only ablation of K2 | 0.33161 | `H_position_dominant` CONFIRMED |
| K3-B | Label-only ablation of K2 | 0.10530 | REJECTED — falls *below* K1 |
| K4 | Label-free timing-cohort geometry on K3-A | 0.33447 | `H_cohort_modest` CONFIRMED (low end) |
| K5-A | Raw-signature duplicate/identity keys, label-free, on K3-A | 0.33711 | `H_duplicate_material` CONFIRMED |
| K5-B | Identity-conditioned label lookup on K3-A | 0.33989 | `H_duplicate_material` CONFIRMED (via this clause) |
| P0 | Raw ~968-column numeric matrix + high-capacity LightGBM | 0.40391 | `H_raw_dominant` CONFIRMED |
| **P1** | Station-temporal features + LightGBM capacity tuning | **0.41917** | `H_station_temporal_moderate` CONFIRMED — program best |

Full table with OOF MCC, thresholds, and tags: [Results](results.md#kaggle-track-track-2-frozen-leaderboard-only).

## Mechanism attribution — what actually moved the number

1. **Record proximity, not neighbor labels, drove K2's gain.** K3 split K2's 40-column "magic"
   stack into a position-only variant (K3-A) and a label-only variant (K3-B). K3-A *matched or
   exceeded* K2's full score on both LB splits despite dropping every label-touching column; K3-B
   fell *below* the honest K1 baseline. Conclusion: adjacency-conditioned neighbor-label lookup is
   actively harmful in isolation, and the entire K2 gain is explained by record proximity alone.
   This eliminated an entire planned follow-up direction (deeper neighbor-label engineering) before
   any further engineering hours were spent on it.
2. **The record-order/timing family saturated fast.** K2 (0.32702) → K3-A (0.33161) → K4
   (0.33447): a private-LB spread of under 0.008 across three experiments with materially different
   feature engineering. Declared saturated after K4; no further timing-family work was pursued.
3. **Identity-conditioned label lookup is a real, distinct mechanism from adjacency-conditioned
   lookup.** K5-B (identity-conditioned) generalizes; K3-B (adjacency-conditioned) does not. Same
   underlying idea — "look up a similar row's label" — but keyed differently, with opposite
   outcomes. Neither result contradicts the other; they isolate different mechanisms.
4. **Raw width was the single largest lever in the program, by an order of magnitude.** P0 added no
   new engineered features — just the full ~968-column raw numeric sensor matrix at high LightGBM
   capacity — and gained more private MCC (+0.06680 over K5-A) than every K1–K5 leakage-family
   experiment combined. This falsified an imported assumption (carried over from the frozen
   production track, RP1/RP2) that the honest feature/model space was near-exhausted: that
   pessimism had been calibrated on a 16–52-column feature set, not a 968-column one.
5. **Capacity tuning outweighed further feature engineering on top of P0.** P1 added Family
   D/S/L station-temporal features (57 + 108 columns) *and* relieved LightGBM's `num_leaves`
   (63→255) and `min_child_samples` (50→20). The features alone contributed +0.00306 OOF; capacity
   tuning on top of them recovered +0.01286 — more than 4x the features' own contribution. The
   takeaway the program closed on: at this stage, hyperparameter capacity was the higher-value
   lever, not more feature engineering.

## What was confirmed vs. rejected

**Confirmed (6 of 6 pre-registered hypotheses):** `H_adjacency_dominant` (K2), `H_position_dominant`
(K3), `H_cohort_modest` (K4), `H_duplicate_material` (K5), `H_raw_dominant` (P0),
`H_station_temporal_moderate` (P1) — no inconclusive result anywhere in the program.

**Rejected:** `H_label_contributes` and `H_position_optimistic` (K3) — the only negative result in
the ladder, and a governance-relevant one: it repaired confidence that the production track's
chunk-aware CV was not itself under-blocked for label leakage, since the honest, label-free K4/K5/
P0/P1 results that followed were built on the same CV harness.

## Freeze rationale and what's left on the table

`KDR-007` amended the program's objective mid-course (from "explain the leaderboard-vs-production
gap" to "maximize private MCC and quantify what it costs"). Against that amended objective, the
program is complete: nine sealed experiments, six-for-six confirmed hypotheses, a fully attributed
leakage-family decomposition, and a raw-signal-plus-capacity-tuning result that is the program best.

The original ~0.52 private-MCC aspiration (`KDR-001`) is not defended by the evidence gathered. The
revised, closing estimate is **~0.435–0.445 private MCC** from a further bounded tuning/blending
pass (0.45 a good result, 0.46–0.47 a stretch) — with further LightGBM tuning and a multi-seed blend
of the P1 winner ranked as the highest-confidence remaining levers, ahead of a fresh
categorical-feature program (P1 itself demonstrated tuning outweighing feature engineering by
>4x). None of this authorizes further work on its own — Track 2 is frozen (`track2-frozen`) and
stays frozen unless a new experiment is pre-registered as a KDR and explicitly authorized by the
user, per `KDR-009` §5.

## KDR anchors (for direct citation)

| KDR | Title | Link |
|---|---|---|
| KDR-001 | Open the Kaggle (Track 2) leaderboard-optimization track | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-001--open-the-kaggle-track-2-leaderboard-optimization-track) |
| KDR-002 | Pre-register K1: baseline reproduction from frozen production candidate | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-002--pre-register-k1-baseline-reproduction-from-frozen-production-candidate) |
| KDR-003 | Pre-register K2: quantify the leakage gap via record-adjacency "magic" features | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-003--pre-register-k2-quantify-the-leakage-gap-via-record-adjacency-magic-features) |
| KDR-004 | Pre-register K3: attribute K2's gain between record-proximity and neighbor-label leakage | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-004--pre-register-k3-attribute-k2s-gain-between-record-proximity-and-neighbor-label-leakage) |
| KDR-005 | Pre-register K4: label-free timing-cohort geometry | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-005--pre-register-k4-label-free-timing-cohort-geometry) |
| KDR-006 | Pre-register K5: duplicate-group (feature-identity) leakage attribution | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-006--pre-register-k5-duplicate-group-feature-identity-leakage-attribution) |
| KDR-007 | Amend Track 2 objective; pre-register P0: raw-numeric signal probe | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-007--amend-track-2-objective-pre-register-p0-raw-numeric-signal-probe) |
| KDR-008 | Pre-register P1: station-temporal feature engineering + capacity tuning | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-008--pre-register-p1-station-temporal-feature-engineering--capacity-tuning) |
| KDR-009 | Freeze Track 2 (Kaggle); transition to portfolio engineering | [GitHub](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/docs/research/kaggle_decisions.md#kdr-009--freeze-track-2-kaggle-transition-to-portfolio-engineering) |

Links point at GitHub's rendered anchors (verified byte-exact via `github-slugger`, the same
library the dashboard's Governance page uses) rather than this site's own anchors, since
`kaggle_decisions.md` is an append-only historical log meant to be read in its native GitHub
rendering, not restructured for site navigation.
