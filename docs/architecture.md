# System Architecture

Two tracks run in parallel on `main`. Track 1 uses labeled OOF data and produces the approved
model and decision policy. Track 3 uses that frozen model to score label-free batches and monitor
score-distribution drift.

## Track 1 — Offline / Decision Layer (labeled OOF)

```mermaid
flowchart TD
    A["OOF Predictions + Labels\nmeta_dataset.parquet"] --> B["Decision Analytics\nThreshold/Budget/Cost"]
    B --> C["Policy Selection\nthreshold_high + inspection_budget"]
    C --> D["FastAPI Service\n/predict /batch_predict"]
    C --> E["Offline Batch Eval\nrun_offline_batch_eval.py"]
    E --> G["Batch Logs\nrecall/precision/flagged"]
    B --> J["Dashboard View B\nDecision Analysis"]
    A --> J
```

**Data flow:** OOF predictions with `Response` labels → threshold/budget sweep → business cost
model (`FN=100`, `FP=5`) → approved operating point → policy object (`DecisionPolicy`).

Scripts: `scripts/run_offline_batch_eval.py` (labeled replay), `scripts/build_decision_summary.py`.

## Track 3 — Production Inference (label-free)

```mermaid
flowchart TD
    P["Unlabeled Batches\ntest_dataset_h.parquet"] --> Q["run_production_inference.py\nDataset-H model + DecisionPolicy"]
    Q --> R["Partitioned Parquets\ncycle=n/batch=n/predictions.parquet"]
    Q --> S["Batch State\ndataset_h_batch_state.json"]
    R --> T["run_drift_monitoring.py\nScore-distribution drift\nrisk_score only — 1 column"]
    T --> U["Evidently Report\nHTML + JSON"]
    R --> V["Dashboard View A\nProduction Monitoring"]
    U --> V
```

**Data flow:** unlabeled rows (`test_dataset_h.parquet`, no `Response`) → dataset_h model
`predict_proba` → `DecisionPolicy` hybrid policy → append-only, cycle/batch-partitioned
output. Drift monitoring reads the prediction parquets (not labeled data): after structural-
column exclusion, Evidently sees exactly one column (`risk_score` → `pred`). Both
`dataset_drift` and `prediction_drift` in the summary are KS tests on this single column
(not independent signals). No recall/precision/MCC anywhere in Track 3 output.

Scripts: `scripts/run_production_inference.py`, `scripts/run_drift_monitoring.py`.
Validation: `scripts/validate_system.py` → `validate_production_inference()`.

## Track 2 — Kaggle Research (quarantined, frozen)

```mermaid
flowchart TD
    W["Kaggle Test CSVs\ntest_numeric/test_date/test_categorical"] --> X["src/kaggle/\nleaderboard-only feature engineering\n(record-adjacency, timing-cohort, duplicate-identity, raw-wide)"]
    X --> Y["scripts/kaggle/generate_submission_K2.py"]
    Y --> Z["submission.csv\nId,Response"]
    Z -.->|"manual upload,\nnever automated"| KL["Kaggle Leaderboard\npublic/private MCC"]
```

**Data flow:** unlabeled Kaggle test data → feature engineering that is deliberately allowed to
include leakage families the production charter forbids (record-adjacency, neighbor/identity label
lookups, timing-cohort geometry, raw ~968-column numeric width) → `submission.csv` → scored by
Kaggle itself, off-platform. No leaderboard number is computed by, or fed back into, this
repository's own code — Kaggle's servers are the only source of `public_mcc`/`private_mcc`.

**Quarantine, not just convention:** `src/kaggle/` and `scripts/kaggle/` are the only trees allowed
to contain this logic. A firewall grep (`import.*kaggle` outside those two trees) runs at every
merge and must be empty — nothing in Track 1 or Track 3 imports Track 2 code, and no Kaggle metric
or conclusion may appear in the production decision log. See
[`docs/research/kaggle_decisions.md`](research/README.md) for governance and
[Track 2](track2.md) for the full ladder. **Frozen** as of `KDR-009` (tag `track2-frozen`) — no
further experiment without a new pre-registered KDR and explicit user authorization.

Scripts: `scripts/kaggle/generate_submission_K2.py` (imports `scripts/generate_submission.py` by
exact module path — the one file the PF2 `scripts/` regroup deliberately left at the top level).

## Runtime Components

| Component | Track | File |
|---|---|---|
| Decision analytics | T1 | `src/evaluation/decision_system.py` |
| Decision engine (policy) | T1 + T3 | `src/inference/decision_engine.py` |
| Offline batch eval (labeled replay) | T1 | `scripts/run_offline_batch_eval.py` |
| Production batch inference (label-free) | T3 | `scripts/run_production_inference.py` |
| Score-distribution drift monitoring | T3 | `src/monitoring/drift_detection.py` |
| Kaggle feature engineering (quarantined) | T2 | `src/kaggle/` |
| Kaggle submission generation (quarantined) | T2 | `scripts/kaggle/generate_submission_K2.py` |
| API | T1 | `apps/api/main.py` |
| Dashboard View B (decision analysis) | T1 | `apps/streamlit_dashboard/app.py` |
| Dashboard View A (production monitoring) | T3 | `apps/streamlit_dashboard/app.py` |

## Entrypoints
- Full system: `python scripts/run_full_system.py`
- Validation: `python scripts/validate_system.py`
- Track 3 drift report: `python scripts/run_drift_monitoring.py`
- API server: `uvicorn apps.api.main:app --host 0.0.0.0 --port 8000`
- Dashboard: `streamlit run apps/streamlit_dashboard/app.py`

## Deployability
- `Dockerfile.api`
- `Dockerfile.dashboard`
- `docker-compose.yml`
