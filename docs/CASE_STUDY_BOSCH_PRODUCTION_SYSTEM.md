# Bosch Production Line Failure Decision System

> **Note on data provenance:** the RP2 numbers in this document (Executive Summary and §7's
> "authoritative, reproducible" table) are reproducible today from committed code. Section 7 also
> preserves an older set of operating points from a since-deleted artifact, kept only as a worked
> example of the decision-system *methodology* — see the sidebar directly above that table for the
> full provenance note before citing any number from that subsection specifically.

## Executive Summary
We converted a competition-style ML workflow into a production decision system for failure prevention on a highly imbalanced manufacturing problem (about **0.58% failures**). Instead of optimizing only MCC, we optimized business outcomes: failures caught, inspection load, and cost.

RP2 (research complete, DR-015) establishes the **honest deployable operating distribution** via 5-fold rolling-origin forward-chaining (`outputs/e3_rolling_origin_results.json`):
- Deployable MCC across operating windows: **0.06–0.18** (mean ≈ 0.12, 95% CI [0.05, 0.19])
- Non-stationarity (prevalence shift) dominates deployment behavior
- Static threshold transfer fails: optimal threshold ranged **0.14–0.72** across windows; a fixed threshold is not deployable
- Model ranking (AUC ≈ 0.55) is stable across regimes — the binding problem is the operating point, not the ranking

Track 3 label-free production inference and Evidently drift monitoring are now live. Historical World-B operating points (unverifiable; see warning above) are preserved in Section 7 as worked examples of the decision framework methodology.

This creates a practical control system where operations can choose policy based on staffing and quality risk appetite, subject to periodic threshold recalibration as the operating regime shifts.

---

## 1. Business Problem
A missed failure (false negative) can propagate quality escapes downstream. A false alarm (false positive) consumes inspection capacity. The objective is not “best offline score” but **best operational decision policy**.

---

## 2. Data and Risk Context
From project outputs:
- Total rows: **1,183,747**
- Failures are rare (roughly **0.5% to 0.6%**)
- Severe class imbalance means naive accuracy is misleading

Implication: We need explicit trade-off control, not a single static model threshold.

---

## 3. Why MCC Was Useful but Not Sufficient
MCC is a strong modeling metric for imbalance and was used in Kaggle evaluation. However, in production, leaders ask:
- How many failures do we catch?
- How many parts do we inspect?
- What is the total quality-control cost?

These are not answered by MCC alone.

---

## 4. Why We Did Not Chase 0.52 MCC
Some offline gains in this problem class can come from brittle patterns that do not generalize in live operations (especially leakage-adjacent temporal/group artifacts). We intentionally prioritized:
- leakage-safe decisioning,
- stable policy behavior,
- explainable trade-offs.

Result: lower headline MCC than leaderboard targets, and a measured deployability — not an assumed one. RP2 (DR-011 through DR-015) quantified this via rolling-origin forward-chaining: honest MCC is 0.06–0.18 across operating windows (mean ≈ 0.12), with non-stationarity identified as the binding limiter rather than model capacity or feature representation. The correct production claim is a regime distribution, not a single deployable number.

---

## 5. Solution Architecture
Production stack includes:
- **Decision analytics** (`src/evaluation/decision_system.py`)
- **Decision engine** (`src/inference/decision_engine.py`)
- **FastAPI service** (`apps/api/main.py`)
- **Streamlit dashboard** (`apps/streamlit_dashboard/app.py`)
- **Offline batch eval, Track 1 labeled replay** (`scripts/pipeline/run_offline_batch_eval.py`)
- **Production batch inference, Track 3 label-free** (`scripts/pipeline/run_production_inference.py`)
- **Evidently monitoring** (`src/monitoring/drift_detection.py`)

The model outputs risk scores; the decision engine converts those into operational actions.

---

## 6. Decision Framework
We support three operational modes:
1. **Threshold mode**: flag score >= threshold
2. **Inspection budget mode**: top-K by risk under capacity
3. **Hybrid mode**: auto-reject high risk + inspect next top-K

This lets operations tune between recall and workload.

---

## 7. Measured Operating Points (From Real Outputs)

### RP2 Honest Deployable Distribution (authoritative, reproducible)

From `outputs/e3_rolling_origin_results.json` (E3, DR-011/DR-012/DR-015). These numbers are reproducible by running `scripts/research/train_e3_rolling_origin.py`.

| Metric | Value |
|---|---|
| Rolling-origin MCC range (5 windows) | **0.061–0.182** |
| Mean MCC | **0.119** (std 0.054) |
| 95% CI | [0.052, 0.187] |
| In-CV MCC (random-group, interpolation-optimistic) | 0.153 |
| Mean degradation vs. in-CV | −22% |
| Prevalence ↔ MCC correlation | 0.68 |

Per-window detail:

| Fold | Chunks | Test prevalence | Oracle threshold | MCC at oracle | MCC at fixed 0.91 |
|---|---|---|---|---|---|
| 0 | 18–33 | 0.80% | 0.14 | **0.080** | ~0 |
| 1 | 34–49 | 0.78% | 0.40 | **0.182** | 0.033 |
| 2 | 50–64 | 0.94% | 0.72 | **0.170** | 0.047 |
| 3 | 65–82 | 0.33% | 0.40 | **0.061** | 0.055 |
| 4 | 83–118 | 0.39% | 0.62 | **0.104** | 0.042 |

Key deployment findings (DR-011 through DR-015):
- **Non-stationarity dominates**: failure-rate prevalence shifts from 0.33% to 0.94% across temporal windows, which drives most of the MCC variation
- **Static threshold not deployable**: fixed 0.91 yields MCC ≈ 0–0.06 across all windows; optimal threshold varied 0.14–0.72; H_threshold_nontransfer settled at confidence 0.92
- **Ranking is stable**: AUC ≈ 0.55 across all regimes (degradation = −0.001); the model's ranking quality is not the bottleneck
- **Hard-regime difficulty is substantially intrinsic**: prevalence-matched control confirms fold-3's extra difficulty persists even after controlling for prevalence; the hard pocket was transient (evaporated in fold 4), not a predictable stable regime
- **Production monitoring is label-free**: Evidently drift monitor alerts to regime entry without requiring `Response` labels; expected response is threshold recalibration, not model replacement

---

### Historical World-B Operating Points (unverifiable; for methodology illustration only)

> **⚠️ Provenance sidebar — read before citing any number in this subsection.** Every operating
> point below (recall, precision, cost, thresholds, the 1,183,747-row dataset size) was computed
> from `data/features/oof_predictions_context_meta_v2_blend.parquet` and related `outputs/*.json`
> files generated from it. The raw, intermediate, and model artifacts that produced that blend file
> were **deliberately deleted** from this repository for packaging size and cleanliness (see
> `data/README.md`), and **no training script in this repo's git history reproduces it**. **These
> numbers are NOT reproducible from the code currently committed to this repository** — they are
> preserved as a historical record and a worked example of the decision-system methodology (how a
> cost model turns scores into an operating point), not as a current, verifiable performance claim.
> The reproducible numbers are the RP2 table in §7 above and the dev-sample OOF MCC values in
> `outputs/training_summary.json`, tabulated with regeneration commands in
> [`docs/reproducible_metrics_report.md`](reproducible_metrics_report.md).

From `outputs/production_decision_summary.json` and related CSVs (see the provenance sidebar
immediately above — artifacts deleted, not reproducible from current code):

### Minimum Cost (FN=100, FP=5)
- Threshold: `0.23`
- Recall: **0.4505**
- Precision: **0.1401**
- TP/FP/FN/TN: `3099 / 19019 / 3780 / 1157849`
- Cost per 100k rows: **39,965.89**

### Best Recall by Inspection Budget
- **1% budget**: recall **0.3579**, precision **0.2080**
- **5% budget**: recall **0.5674**, precision **0.0659**
- **10% budget**: recall **0.6321**, precision **0.0367**

### Best Precision-Recall Balance (F1-like criterion)
- Threshold: `0.54`
- Recall: **0.2423**
- Precision: **0.3827**

---

## 8. Simulation Results

**Track 1 / Offline Evaluation, not production inference.** These numbers come from
`scripts/pipeline/run_offline_batch_eval.py` (formerly `run_batch_simulation.py`) replaying
labeled OOF data batch-by-batch -- recall/precision are only ever valid against
labeled data, never against the unlabeled stream Track 3 actually scores. See
`docs/ml_system_tracks.md` for the three-track split and `scripts/pipeline/run_production_inference.py`
for the genuinely label-free production batch inference path (no recall/precision
anywhere in its output, by construction).

From `outputs/batch_simulation_summary.json`:
- Policy: `threshold_high=0.6`, `inspection_budget=10%`
- Mean recall across simulated batches: **0.6320**
- Mean precision: **0.0368**
- Mean flagged rate: **10.0001%**

Interpretation: batch behavior is consistent with operating-point analysis.

---

## 9. Monitoring and Drift

Track 3 label-free drift monitoring is live (`scripts/pipeline/run_drift_monitoring.py`). The Evidently monitor reads exclusively from the production batch prediction parquets (`outputs/production/dataset_h/cycle=*/batch=*/predictions.parquet`) — no `Response` column is present or used. A temporal 70/30 split (earlier rows = reference baseline; later rows = current window) is applied before drift detection.

**Scope — score-distribution drift only.** After structural columns are excluded (batch_id, cycle_id, run_seq, scored_at_utc, decision, auto_reject, manual_inspect) and Evidently's own ID-column filter removes `Id`, the monitor operates on a single column: `risk_score` (renamed to `pred` for Evidently's API). Both the "dataset drift" and "prediction drift" metrics in the summary are computed on this one column and reflect the same underlying KS test on the score distribution. They are not independent signals and do not cover input-feature drift. This is a deliberate design choice: monitoring input features requires the test feature table, which is not always available at monitoring time; score-distribution drift is a sufficient first-alert proxy and requires only the label-free prediction output.

From `outputs/monitoring/evidently_summary.json` (current run, 50,000 production rows):
- Engine: **Evidently**
- Reference rows: **35,000** / Current rows: **15,000**
- Monitored columns: **1** (`risk_score`)
- Drifted columns count: **0**
- Drift share: **0.0**
- Drift score (KS): **0.024** (threshold: 0.1) — not detected

The Streamlit dashboard "Production Monitoring" view (`apps/streamlit_dashboard/app.py`) reads this file and displays drift status live. `scripts/pipeline/validate_system.py` asserts the monitoring schema in its `production_inference` module.

**Interpreting alerts in the RP2 context**: given that honest MCC varies 0.06–0.18 across regimes, a drift alert indicates regime entry (prevalence or score-distribution shift). The expected operational response is threshold recalibration against recent data, not model replacement — the model's ranking (AUC ≈ 0.55) is stable across regimes.

---

## 10. Business Trade-Offs (Plain Language)
- Lower threshold / higher inspection budget catches more bad parts.
- But it also sends many more parts to manual handling.
- If staffing is fixed, aggressive recall policies can overload inspection.

So the right policy depends on real plant constraints.

---

## 11. Recommended 3-Tier Policy
Recommended default:
1. **Auto reject** high-risk parts (`threshold_high` policy)
2. **Manual inspect** additional top-risk parts up to budget
3. **Pass** the rest

Start point options:
- Cost-efficient: threshold `0.23` (recall 0.4505)
- Detection-heavy: 5% budget (recall 0.5674)
- Maximum capture: 10% budget (recall 0.6321)

---

## 12. Production Readiness Status

### Achieved — Offline / Decision Layer (Track 1)
- Reproducible end-to-end runner (`scripts/pipeline/run_full_system.py`)
- API + dashboard + Docker definitions
- Offline batch simulation (labeled OOF replay, `scripts/pipeline/run_offline_batch_eval.py`)
- RP2 research phase complete (DR-015): honest deployable performance distribution measured and documented

### Achieved — Production Layer (Track 3)
- Label-free production batch inference (`scripts/pipeline/run_production_inference.py`): 5 batches scored (50,000 rows), append-only partitioned output under `outputs/production/dataset_h/`
- Persistent cycle/batch state (`dataset_h_batch_state.json`): pointer, cycle_id, batch_id, run_seq
- Evidently drift monitoring reading from production batches, no labels required (`scripts/pipeline/run_drift_monitoring.py`)
- `validate_system.py` extended with `validate_production_inference()`: asserts batch existence, required columns, `Response` absence, state file validity, and monitoring schema
- Streamlit dashboard "Production Monitoring" view wired to `evidently_summary.json` and production batch stats

### Remaining Work
- **Threshold recalibration policy**: the static default threshold is not deployable across regimes (H_threshold_nontransfer, confidence 0.92). The system needs a periodic recalibration step that updates the operating threshold against recent labeled outcomes when label feedback becomes available.
- **Automated drift alerting**: trigger an operator notification when the Evidently drift score exceeds the configured threshold; current monitoring is passive (requires manual dashboard check).
- **Label feedback integration**: when manual inspection outcomes are recorded, enable calibration-drift detection (supervised signal) in addition to the current unsupervised score-distribution monitoring.

---

## 13. Related Research: the Kaggle Leaderboard Track (separate, frozen)

**This section is informational context, not a production claim.** Everything above this section
describes the deployable decision system; this project's charter (`bosch_agent.md`,
`system_design.md`) forbids leaderboard-driven leakage tricks from ever entering that system, and
none of the numbers below informed any decision above.

Separately, this repository also contains a rigorously governed Kaggle leaderboard-optimization
track (`docs/research/kaggle_decisions.md`, KDR-001–KDR-009), run specifically to *measure* how
much of this competition's famous ~0.50-MCC public leaderboard ceiling comes from mechanisms that
are Kaggle-legal but not deployable — record-adjacency leakage, duplicate/identity signature
lookups, and eventually just far more raw signal plus model capacity than a production feature
contract would carry. Nine experiments, six pre-registered hypothesis classifications, all
confident, none inconclusive. Canonical numbers: [`results/leaderboard.json`](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis/blob/main/results/leaderboard.json).

| Experiment | Mechanism | Private LB MCC |
|---|---|---|
| K1 | Frozen production model, reproducibility baseline | 0.16160 |
| K2 | Record-adjacency magic (full) | 0.32702 |
| K5-B | Identity-conditioned label lookup | 0.33989 |
| P0 | Raw ~968-column numeric matrix, high-capacity LightGBM | 0.40391 |
| **P1** | Station-temporal features + capacity tuning | **0.41917** (program best) |

The track is now **frozen** (KDR-009, tag `track2-frozen`): the original ~0.52 target is not
defended by the evidence gathered, and the revised realistic forward estimate is ~0.435–0.445 via
further tuning/blending, not further leakage engineering. Full attribution, the complete 9-row
ladder, and the postmortem are in `docs/research/kaggle_decisions.md` (KDR-009) and summarized in
the [README](https://github.com/anudeepreddy332/bosch-production-line-defect-analysis#results).

---

## 14. Impact Summary
This system moves the project from model experimentation to operational decision support:
- quantifies recall-vs-cost explicitly,
- makes inspection capacity a first-class control,
- provides deployable interfaces and monitoring,
- supports explainable stakeholder decisions.
