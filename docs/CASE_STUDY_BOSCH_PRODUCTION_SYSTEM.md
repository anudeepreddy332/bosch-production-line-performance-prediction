# Bosch Production Line Failure Decision System

## Executive Summary
We converted a competition-style ML workflow into a production decision system for failure prevention on a highly imbalanced manufacturing problem (about **0.58% failures**). Instead of optimizing only MCC, we optimized business outcomes: failures caught, inspection load, and cost.

On the final decision layer:
- **Minimum-cost operating point**: threshold `0.23`
  - Recall: **0.4505**
  - Precision: **0.1401**
  - Total cost: **473,095** (FN=100, FP=5)
- **High-recall operating point (10% inspection budget)**:
  - Recall: **0.6321**
  - Precision: **0.0367**

This creates a practical control system where operations can choose policy based on staffing and quality risk appetite.

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

Result: lower headline MCC than leaderboard targets, but far stronger deployability.

---

## 5. Solution Architecture
Production stack includes:
- **Decision analytics** (`src/evaluation/decision_system.py`)
- **Decision engine** (`src/inference/decision_engine.py`)
- **FastAPI service** (`apps/api/main.py`)
- **Streamlit dashboard** (`apps/streamlit_dashboard/app.py`)
- **Batch simulator** (`scripts/run_batch_simulation.py`)
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
From `outputs/production_decision_summary.json` and related CSVs:

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
From `outputs/batch_simulation_summary.json`:
- Policy: `threshold_high=0.6`, `inspection_budget=10%`
- Mean recall across simulated batches: **0.6320**
- Mean precision: **0.0368**
- Mean flagged rate: **10.0001%**

Interpretation: batch behavior is consistent with operating-point analysis.

---

## 9. Monitoring and Drift
From `outputs/monitoring/evidently_summary.json`:
- Engine: **Evidently**
- Drifted columns count: **0 / 4**
- Drift share: **0.0**
- Prediction drift: **not detected**

Monitoring excludes identifier columns and uses stable reference/current split.

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
Achieved:
- Reproducible end-to-end runner (`scripts/run_full_system.py`)
- API + dashboard + Docker definitions
- Evidently HTML + JSON drift reports
- Cleanup and archive of experimental notebooks

Remaining practical next steps:
- Integrate real-time feedback loop from manual inspection outcomes
- Add alerting around drift and KPI thresholds
- Add policy auto-tuning by line/station context

---

## 13. Impact Summary
This system moves the project from model experimentation to operational decision support:
- quantifies recall-vs-cost explicitly,
- makes inspection capacity a first-class control,
- provides deployable interfaces and monitoring,
- supports explainable stakeholder decisions.
