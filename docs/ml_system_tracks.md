# ML System Tracks: Three-Track Architecture

## Status

| Track | Description | Progress | State |
|-------|-------------|----------|-------|
| Track 1 | Offline Research | 100% | Frozen (DR-015, branch `research/rp2-temporal-robustness` merged) |
| Track 2 | Kaggle | ~90% | **Open** (KDR-001–KDR-008, 2026-06-28 to 2026-07-03; `kaggle-main` synced with origin). K1–K5 complete and merged; leakage-family decomposition complete; KDR-007 amended the objective (explain-the-gap → maximize private MCC, target ~0.52) and closed P0 (raw-numeric signal probe, `H_raw_dominant` confirmed, public 0.39226/private 0.40391); KDR-008 closed P1 (station-temporal features + capacity tuning, `H_station_temporal_moderate` confirmed, best-to-date public 0.40447/private 0.41917); a future KDR-009 (P2, categorical features) not yet pre-registered |
| Track 3 | Production | 100% | Frozen (tag: `track3-frozen`, commit: `f743da3`) |

**Current Active Track:** Track 2 (Kaggle) — K1 (honest baseline, public 0.14389/private 0.16160), K2 (record-adjacency magic, public 0.31699/private 0.32702), K3 (adjacency attribution: position-only public 0.31791/private 0.33161, label-only public 0.10065/private 0.10530 — neighbor-label leakage confirmed dead), K4 (label-free timing-cohort geometry: public 0.31697/private 0.33447, `H_cohort_modest`, record-order/timing family saturated), and K5 (duplicate-group identity attribution: label-free public 0.32330/private 0.33711; identity-conditioned label lookup public 0.33571/private 0.33989 — best model to date pre-P0, `H_duplicate_material` confirmed) are all complete, tagged (`K1-result`–`K5-result`), and merged to `kaggle-main`. K1–K5 closed the leakage-family decomposition objective; KDR-007 then amended Track 2's objective to maximizing private-leaderboard MCC (target ~0.52) and pre-registered P0, a raw-numeric (968-column) signal probe layered on K5-A's honest feature stack. P0 is complete: regression anchor reproduced K5-A's OOF exactly (0.32506), Cell C (raw + K5-A stack) scored honest OOF 0.37598 (default capacity) / 0.37892 (high-capacity), both classifying `H_raw_dominant` (Δ ≥ +0.05 over K5-A); the high-capacity model's Kaggle submission scored public 0.39226 / private 0.40391, tagged `P0-result` and merged to `kaggle-main`. KDR-008 then pre-registered and closed P1 (per-station timing Family D + station/line numeric aggregates Family S/L, layered on P0's Cell C stack, plus LightGBM `num_leaves`/`min_child_samples` tuning): the winning configuration (`num_leaves=255, min_child_samples=20`) scored honest OOF 0.39484 (Δ+0.01592 over P0), classifying `H_station_temporal_moderate`; its Kaggle submission scored public 0.40447 / private 0.41917 — the best result in the program to date, tagged `P1-result` and merged to `kaggle-main`. P1's key finding: the new features contributed only +0.00306 OOF directly, while relieving LightGBM's leaf/capacity constraints on top of them recovered +0.01286 — capacity tuning, not feature engineering, was P1's dominant lever. Governance: `docs/research/kaggle_decisions.md` (`KDR-001`–`KDR-008`). Next: a future `KDR-009` for P2 (categorical features) is not yet pre-registered.

---

This document is the canonical statement of the project's scope split. It **clarifies** the
existing two-flow framing in `system_design.md` and the two-track framing in `bosch_agent.md`
into three explicit tracks, and resolves an ambiguity neither of those docs addressed: the
Streamlit dashboard is not one thing, it must be (eventually) two separate views with different
data contracts. This document only records the target architecture and an audit of where current
docs/code already match or diverge from it. **No code was changed to produce this document.**

## Why three tracks, not two

`bosch_agent.md` and `system_design.md` both describe a two-way split: "training" vs
"production," with Kaggle either out of scope ("IGNORE FOR NOW" in `bosch_agent.md`) or absent
entirely (`system_design.md` never mentions Kaggle/submission). That framing collapses two
genuinely different consumers of the trained model into one "production" bucket:

- Scoring **Kaggle's unlabeled test set** to produce a leaderboard submission (a one-shot batch
  job with a fixed output contract: `Id,Response`).
- Scoring a **simulated live stream** of unlabeled batches for an internal decision/monitoring
  system (an ongoing job with state, drift checks, and a risk-score output, no CSV submission).

Both consume the same frozen, approved model and both operate on unlabeled data, but they have
different inputs, different outputs, and different success criteria. Treating them as one
"production" track is what previously let Kaggle submission go unbuilt while "production"
absorbed supervised-metric logic that belongs to offline evaluation (see the audit below).
Splitting them into Track 2 and Track 3 makes each one's contract checkable on its own.

---

## Track 1: Offline Training + Evaluation

- **Input:** labeled training data (`data/processed/train_*.parquet`, with `Response`).
- **Purpose:** EDA, feature engineering, model training, OOF/holdout validation, threshold
  tuning, MCC/precision/recall/accuracy, confusion matrix, feature importance, model approval.
- **Output:** an approved, frozen model artifact, its feature schema, a selected decision
  threshold, and a metrics report.
- **Maps to existing code:** `scripts/prepare_data.py` → `build_dataset_{baseline,g,h}.py` →
  `train_{baseline,dataset_g,dataset_h}.py` → `train_meta_model.py`, plus
  `src/evaluation/decision_system.py` and `docs/reproducible_metrics_report.md` for honest
  metrics reporting.
- **Dashboard view allowed:** Offline Evaluation / Decision Analysis (View B below) — and only
  this view.

## Track 2: Kaggle Submission

- **Input:** unlabeled Kaggle test files (`test_numeric.csv`, `test_categorical.csv`,
  `test_date.csv` / their parquet equivalents).
- **Purpose:** generate a competition submission.
- **Process:** load the Track 1 approved model artifact → apply the identical feature
  transformation used in training → predict probabilities → apply the selected threshold →
  write `submission.csv`.
- **Output:** `submission.csv` with exactly `Id` and `Response` columns, row count matching
  Kaggle's `sample_submission`.
- **Constraint:** no local supervised metrics — Kaggle test labels are hidden, so MCC/precision/
  recall cannot be computed locally for this track.
- **Current state: scripts/generate_submission.py exists, but cannot yet produce a real
  full-size submission.** It loads a Phase-2 model payload, applies it to an already
  feature-engineered unlabeled test table, and writes `Id,Response` — see
  `docs/kaggle_submission.md` for the full design and validation evidence. Two pre-existing gaps
  block an actual end-to-end run today: (1) no test-side feature-engineering script exists (only
  `train_*` has a `build_dataset_*.py`), so there is no engineered test parquet to point the
  script at; (2) the committed `models/*.pkl` are still bare `LGBMClassifier` objects (pre-Phase-2
  format), not the payload dict the script requires. Both are documented as known limitations in
  `docs/kaggle_submission.md`, not fixed in that change. `tasks.md` describes a related but
  distinct batch-inference spec that was also never built.

## Track 3: Production Inference Simulation

- **Input:** unlabeled incoming batches, simulated from test-data chunks.
- **Purpose:** mimic a Bosch factory scoring flow — i.e., simulate what would happen if the
  approved model scored a live, unlabeled stream.
- **Process:** load the frozen approved model → score each batch → append predictions/risk
  scores → compute batch statistics (counts, score distributions, throughput) → monitor
  drift/data quality → update the dashboard.
- **Output:** predictions, risk scores, batch stats, drift/data-quality summaries.
- **Constraint:** no MCC/precision/recall/accuracy/TP/FP/TN/FN/confusion matrix anywhere in this
  track's output, because by definition its input is unlabeled.
- **Current state: resolved for `dataset_h` via a SPLIT, not a rewrite-in-place.**
  `scripts/run_batch_simulation.py` (the script formerly positioned as this track, despite
  actually being Track 1 logic — see below) has been renamed to `scripts/run_offline_batch_eval.py`
  and re-labeled as what it actually is: a labeled OOF threshold/budget replay
  (`data/features/meta_dataset.parquet` **with `Response`**, `metrics_from_labels`/`simulate_batches`
  computing real `recall`/`precision`/`tp`/`fp`/`fn`/`tn`). It is no longer wired into
  `scripts/run_full_system.py`'s "production" stage and is no longer described as part of the
  "Production Pipeline" in `README.md`. It remains useful as a standalone Track 1 tool — a
  threshold/budget sweep replayed batch-by-batch — and its output
  (`outputs/batch_simulation_summary.json`, quoted in `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md`
  §8) is now explicitly labeled there as Track 1 output, not production output.

  The genuinely label-free replacement is `scripts/run_production_inference.py` (new,
  `dataset_h`-only — the approved candidate model): it consumes
  `data/features/test_dataset_h.parquet` (built by `scripts/build_test_dataset_h.py`, the
  fingerprint-validated, `Response`-free feature contract), scores it with
  `models/dataset_h_model.pkl`, and emits exactly this track's spec — predictions/risk scores,
  batch stats, score distributions, `batch_id`/`cycle_id`/timestamp — with zero MCC/precision/
  recall/accuracy/confusion-matrix/TP/FP/TN/FN anywhere, by construction (grep-verified). It is
  now wired into `scripts/run_full_system.py`'s "production" stage in place of the old script.
  `batch_id`/`cycle_id` state-machine semantics follow `tasks.md`'s documented contract (batch_id
  resets per cycle, cycle_id increments on wraparound; a separate `run_seq` field is the
  lifetime-monotonic counter).

  **S3 upload of Track 3's partitioned output is wired** (`src/utils/s3_utils.py`'s
  `upload_file_append_only`/`key_exists`, called from `scripts/run_production_inference.py`
  after the local parquet write, before state advances — upload-then-advance, failure-loud,
  never overwrites an existing key; `--no-s3` skips it for local-only runs).

  **`scripts/run_drift_monitoring.py` is genuinely label-free** (T3-1): it reads only the
  production batch parquets under `outputs/production/dataset_h/`, never `meta_dataset.parquet`
  or any file containing `Response`. It renames `risk_score` → `pred` for Evidently's API and
  runs a 70/30 temporal split. After structural-column exclusion, Evidently sees exactly one
  column. **Score-distribution drift only** — both `dataset_drift` and `prediction_drift` in
  the Evidently summary are KS tests on the single `risk_score` column; they are not
  independent signals and do not cover input-feature drift.

  **Dashboard View A (Production Monitoring)** is fully wired (T3-2): the "Production
  Monitoring (Track 3)" Streamlit page reads `outputs/monitoring/evidently_summary.json` and
  renders prediction drift detected/score, dataset drift, drifted-columns count, drift share,
  and last-run timestamp. All panels are label-free (no supervised metric).

  **Validator extended** (T3-3): `scripts/validate_system.py` includes
  `validate_production_inference()` — asserts batch existence, required output columns,
  `Response` absence, state file validity, and Evidently schema. Runs as part of the
  canonical `python scripts/validate_system.py` entry point.

---

## Dashboard: two views, not one

### View A — Production Monitoring View

- Uses unlabeled production-like batches.
- Shows: predictions, risk scores, batch counts, score distributions, data quality, drift,
  throughput, latest batch/cycle, top risky parts.
- **Must NOT show** MCC, precision, recall, accuracy, TP, FP, TN, FN, or a confusion matrix,
  because production/test batches are unlabeled.
- **Current state: complete (T3-1 + T3-2).**
  `apps/streamlit_dashboard/app.py`'s "Production Monitoring (Track 3)" page:
  - Lists and concatenates every `predictions/cycle=*/batch=*/predictions.parquet` object in S3.
  - Runtime-asserts the result has no `Response` column.
  - Renders label-free panels: total predictions, latest cycle/batch/run_seq,
    flagged/auto-reject/manual-inspect counts, risk-score histogram, batch growth,
    cumulative-predictions chart, top-100-by-risk-score table.
  - Renders Evidently drift section (T3-2): prediction drift detected/score, dataset drift
    detected, drifted-columns count, drift share, last-run timestamp. Handles missing
    `evidently_summary.json` gracefully with a warning banner.
  - No supervised metric (MCC/precision/recall/accuracy/TP/FP/TN/FN) anywhere on the page.

### View B — Offline Evaluation / Decision Analysis View (this is what exists today)

- Uses labeled training validation / OOF data only.
- May show threshold sliders and how MCC, precision, recall, accuracy, TP, FP, TN, FN,
  confusion matrix, inspection budget, and cost trade-offs change.
- This is allowed because it is not production inference — it is a model-evaluation and
  decision-policy analysis tool.
- **Current state: still every other page in the dashboard**, mostly not labeled as such.
  Every section in `apps/streamlit_dashboard/app.py` (Threshold Explorer, Inspection Budget
  Simulator, Recall at Fixed Precision, Cost Simulator, Model Insights, Failure Analysis) loads
  `meta_dataset.parquet` joined to `oof_predictions_final.parquet` — both labeled, both
  Track-1-derived. The data source is correctly labeled data (so the *math* is legitimate
  View-B work), but:
  - The loader function is named `load_scoring_data()` and its result is assigned to a variable
    named `live_df` throughout (e.g. `apps/streamlit_dashboard/app.py:274,427,475,497`) —
    `live_df` is a misleading name for labeled OOF data.
  - These pages still aren't individually labeled "Offline Evaluation" or "Decision Analysis" in
    the UI; a one-line top-of-page caption now states that every page except "Production
    Monitoring (Track 3)" uses labeled OOF data, but the per-page naming/labeling cleanup below is
    still open.

**Per the user's explicit instruction for the original change: do not refactor the dashboard
beyond what's needed to add View A.** Remaining work: (1) fully relabel existing sections as the
Offline Evaluation / Decision Analysis view (today there's only the one top-level caption), (2)
rename `live_df` to something like `oof_eval_df`, (3) **done** — a new, separate Production
Monitoring (Track 3) page now exists, backed by Track 3's real label-free S3 output, (4) wire the
existing Evidently HTML/JSON into that new view. Only (3) is done; (1), (2), (4) remain open.

---

## Current state vs. target state (summary table)

| Track / View | Target | Current state |
|---|---|---|
| Track 1: Offline Training + Evaluation | Labeled data in, approved model + metrics out | **Exists**, with the World A/B reproducibility caveats already documented in `docs/reproducible_metrics_report.md` |
| Track 2: Kaggle Submission | Unlabeled Kaggle test in, `submission.csv` out | **Open** (`KDR-001`–`KDR-008`, 2026-06-28 to 2026-07-03). K1 (honest baseline), K2 (record-adjacency magic), K3 (adjacency attribution), K4 (timing-cohort geometry), K5 (duplicate-identity attribution) complete, tagged, merged to `kaggle-main`. `KDR-007` amended the objective to maximizing private LB MCC and closed P0 (raw-numeric signal probe, `H_raw_dominant`, public 0.39226/private 0.40391). `KDR-008` closed P1 (station-temporal features + capacity tuning, `H_station_temporal_moderate`), now the best model to date (public 0.40447/private 0.41917). A future `KDR-009` (P2, categorical features) is not yet pre-registered. |
| Track 3: Production Inference Simulation | Unlabeled simulated batches in, label-free predictions/drift out | **Frozen** (tag: `track3-frozen`, commit: `f743da3`). 5 batches scored (50,000 rows). All DoD items met and validated (`validate_system.py` → `overall_pass: True`). |
| Dashboard View A: Production Monitoring | Label-free batch/drift/data-quality view | **Complete** in `apps/streamlit_dashboard/app.py`'s "Production Monitoring (Track 3)" page. Renders predictions/risk scores (label-free, `Response`-absent verified) AND Evidently drift section (score, detected flag, drifted-column count, drift share, timestamp). Handles missing monitoring output gracefully. |
| Dashboard View B: Offline Evaluation / Decision Analysis | Labeled OOF data, supervised metrics, threshold/cost tuning | **Exists and is correct on data**, but unlabeled as such and uses misleading naming (`live_df`) |

---

## Definition of Done (per track)

Each track transitions to "done" on **objective, checkable criteria**, not judgment. A track is
Done only when every box below is true. The roadmap (next section) gates transitions on these.

### Track 1 — Offline Training & Evaluation — Definition of Done

- [ ] Approved, frozen model artifact (`dataset_h` is the v1 candidate) + its feature schema +
      selected decision threshold/policy, persisted under version control.
- [ ] Honest metrics reported on labeled OOF/CV only (chunk-aware group-safe harness); no supervised
      metric on unlabeled data anywhere in this track.
- [ ] The research program (RP1 + RP2) record is complete and **landed on `main`**: `decisions.md`
      DR-001 → DR-015, plus the E1–E4 scripts and `outputs/e*` evidence.
- [ ] Retroactive result tags placed: `E1-result` (exists), `E2-result`, `E3-result`, `E4-result`,
      and a program/freeze tag (e.g. `track1-frozen`).
- [ ] **Closure convention for a continuous research program:** RP2 (E2–E4) was executed as one
      continuous program on a single research branch; it closes via **one program-level PR +
      per-experiment `E*-result` tags + the decisions log**. Retroactive **per-experiment PRs are NOT
      required** — this preserves historical integrity. (See `docs/research/git_workflow.md`.)
- [ ] RP1 frozen (DR-010); RP2 research phase closed (DR-015); no open research gate.

*Status: science complete; pending record-landing on `main` + result tags (roadmap Phase 0).*

### Track 2 — Kaggle Submission — Definition of Done

- [x] `kaggle-main` branch created (@ `13ab858`) + `KDR-001` opened in `kaggle_decisions.md`
      (lazy scaffold per DR-008). `src/kaggle/` / `scripts/kaggle/` are created at the first `K`
      experiment, not at opening (nothing to quarantine yet).
- [ ] Test-side feature pipeline producing engineered test tables for every model in the path
      (baseline, g, h, meta). `dataset_h` test features exist; baseline/g/meta pending.
- [ ] Persisted OOF-safe rate-lookup tables so g/h/meta features compute on test rows without leakage.
- [ ] `models/*.pkl` in the Phase-2 payload format `generate_submission.py` requires. `dataset_h`
      is already a valid payload (verified, `docs/dataset_h_submission_run.md`); baseline/g/meta pending.
- [ ] One real end-to-end `submission.csv` produced — exactly `Id,Response`, row count matching
      `sample_submission`. Done for `dataset_h` locally (2,993 positives @ thr 0.91); not yet
      submitted to Kaggle.
- [ ] Firewall intact: no `src/kaggle/` import outside it; no leaderboard number in `decisions.md`
      or any `DR`/`E`. **Verified at `KDR-001` opening** (code-valve grep empty).

*Status: track **open** (`KDR-001`). Governance + branch scaffolding complete; submission pipeline
mature for `dataset_h` only; full multi-model path and a live Kaggle submission remain.*

### Track 3 — Production Inference Simulation — Definition of Done

- [x] Frozen approved model scores unlabeled simulated batches; output carries predictions/risk
      scores + batch/cycle state + timestamps, with **zero** MCC/precision/recall/accuracy/
      TP/FP/TN/FN/confusion matrix (grep-verified). *(T3-1: `run_production_inference.py`)*
- [x] Append-only S3 partitioned output (upload-then-advance, never overwrite an existing key).
- [x] **Label-free score-distribution drift:** `run_drift_monitoring.py` runs on Track 3's
      label-free output (production batch parquets only, no `Response`), and its Evidently
      summary is rendered in Dashboard View A. Drift is computed on the single `risk_score`
      column — not input-feature drift. *(T3-1 + T3-2)*
- [x] Dashboard split clean: View A (Production Monitoring, label-free) exists and renders
      both batch stats and Evidently drift section; no supervised metric on View A.
      View B (Offline Evaluation, labeled OOF) is separated by the top-of-page caption.
      *(T3-2; remaining View-B per-page relabel / `live_df` rename tracked as post-freeze cleanup)*
- [x] Case study reports deployable performance as the **measured regime distribution** with the
      static-threshold-non-transfer caveat (the RP2 handoff), not a single in-CV number.
      *(T3-4: `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md` updated)*

*Status: all DoD items complete. **Frozen** — tag `track3-frozen` placed at commit `f743da3`.*

---

## Roadmap & transition gates

Transitions are gated on the Definition of Done above, not on judgment.

1. **Phase 0 — Freeze Track 1.** Commit DR-014/DR-015 → land the research record on `main` → place
   retroactive `E*-result` tags + a program/freeze tag. **Gate to Phase 1:** Track 1 DoD fully checked.
2. **Phase 1 — Production v1 (Track 3).** RP2 case-study handoff, label-free drift wiring, View-A
   Evidently rendering, freeze v1 model + policy. **Gate to the cleanup milestone:** Track 3 DoD
   fully checked.
3. **Milestone — Repository cleanup / documentation polish.** Clear documentation and architectural
   debt **before** branching Kaggle, so it does not leak into the Kaggle track: View-B relabel +
   `live_df`/`load_scoring_data` rename, `architecture.md` Track 3 node, README / CASE_STUDY
   overstatement fixes, CLAUDE.md "already enforce" revisit, `kaggle_decisions.md` branch-anchor
   line, the track-terminology glossary, and stray `_blend` artifact provenance/cleanup. **Gate to
   Phase 2:** no known doc/architecture inconsistency outstanding on `main`.
4. **Phase 2 — Track 2 (Kaggle). ← ACTIVE.** Track opened at `KDR-001` (2026-06-28); `kaggle-main`
   at `b058e58` (KDR-001 merged). `K1` pre-registered in `KDR-002`; branch
   `kaggle/K1-baseline-reproduction` created. Next: run K1 baseline, tag `K1-result`, design K2.

---

## Misleading-language audit

Searched the repo for `MCC`, `precision`, `recall`, `accuracy`, `confusion matrix`, `TP`, `FP`,
`TN`, `FN`, `Response` in docs and dashboard-related code, and classified every occurrence in a
production/dashboard-adjacent context into one of three buckets:

- **Valid** — offline-evaluation language, correctly scoped (Track 1, or explicit Kaggle-leaderboard discussion).
- **Invalid** — misleading production language: a "production"/"batch simulation"/"live" framing
  applied to what is actually labeled-data evaluation, without disclosing that.
- **Code-level issue** — the doc language is reporting actual code behavior accurately, and the
  *code* is what needs to change (not just the words).

| Location | Language | Classification |
|---|---|---|
| `execution_rules.md:38-43`, `tasks.md:17-22`, `bosch_agent.md:38-39` | "Test data is UNLABELED → NEVER compute MCC/precision/recall" | **Valid** — this is the rule statement itself |
| `system_design.md:48,55` | "Training Flow... MCC/precision/recall" / "Production flow MUST NOT compute MCC/precision/recall" | **Valid** — correct rule, but file is silent on Track 2 and the two dashboard views (updated in this change, see below) |
| `docs/reproducible_metrics_report.md` (all MCC/recall mentions) | OOF MCC per model, World A vs World B reproducibility | **Valid** — explicitly scoped to Track 1, already the most precise doc in the repo on this topic |
| `docs/architecture.md:1,10` | Diagram titled "Production Architecture" containing `Batch Logs\nrecall/precision/flagged` | **Invalid** — recall/precision attributed to a diagram titled "Production," with no disclosure that the underlying batches are labeled OOF data, not live unlabeled data |
| `README.md:34-39` ("🔵 Production Pipeline") | Lists "Batch simulation (streaming-like behavior)" under Production | **Invalid** — `run_batch_simulation.py` is Track 1 logic (see Track 3 section above); calling it "production" is the same mislabeling baked into a top-level doc |
| `README.md:81-84` ("📊 Key Results") | "Best MCC: ~0.317", "Recall @ 10% inspection: ~0.63", "Fully production-safe pipeline" | **Invalid** — already flagged as World-B/unverified for *reproducibility*, but the disclaimer never addresses that these are labeled-data evaluation numbers being presented under a doc titled "Production ML System"; "Fully production-safe pipeline" is an unsupported claim given Track 3's current state |
| `README.md:122-128` ("📈 Dashboard Features") | "Recall vs precision trade-offs" etc., listed without a view label | **Invalid** — these are View B (Offline Evaluation) features, listed under a README that frames the dashboard as part of the "Production Pipeline" with no A/B split |
| `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md` §7 "Measured Operating Points" | Recall/Precision/TP/FP/FN/TN from `production_decision_summary.json` | **Valid math, invalid placement** — the numbers are legitimately computed on labeled data (View B work), but the section sits inside a document framed entirely as a "production decision system" case study with no Track/View label on the section itself |
| `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md` §8 "Simulation Results" | "Mean recall across simulated batches: 0.6320" | **Invalid** + **code-level issue** — this is the doc faithfully reporting what `run_batch_simulation.py` actually does (supervised metrics on labeled "simulated" batches); the doc language is accurate to the code, but the code is the Track 3 violation described above |
| `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md` §12 "Production Readiness Status" | "Achieved: Reproducible end-to-end runner..." | **Invalid** — overstates readiness; contradicts the doc's own top World-B disclaimer and `docs/reproducible_metrics_report.md` |
| `apps/streamlit_dashboard/app.py` (`load_scoring_data`, `live_df` everywhere) | Function/variable naming implies live/production data | **Code-level issue** — the data loaded (`meta_dataset.parquet` + `oof_predictions_final.parquet`) is labeled OOF data; naming it `live_df` is misleading at the code level, not just in docs |
| `CLAUDE.md` ("Production / decision pipeline" section) | "...must never compute MCC/precision/recall against unlabeled data" / later: "The dashboard and decision-system code already enforce this split" | **Partially invalid** — the first clause is technically true (the data `run_batch_simulation.py` touches is labeled, not unlabeled, so it isn't violating *that* literal sentence), but the second clause ("already enforce this split") overstates the current state: there is no separate label-free Production Monitoring view, and `run_batch_simulation.py` is presented elsewhere (README, case study) as production behavior while running Track 1 logic. Flagged here for visibility; **not edited in this change** since it's the user's own active instructions file — worth a follow-up edit once Track 3/View A actually exist. |

---

## Known code-level issues to fix later (not fixed in this change)

These are implied by the audit above and by `docs/production_readiness_audit.md` — listed for
follow-up, not actioned here:

1. **RESOLVED** (option (a): relabeled, plus a real Track 3 was also built — option (b) for a
   new script, not a rewrite of the old one). `run_batch_simulation.py` is now
   `scripts/run_offline_batch_eval.py`, honestly Track 1. `scripts/run_production_inference.py`
   is the new, genuinely label-free Track 3 (dataset_h only). See the Track 3 section above.
2. **Resolved** (T3-2). `apps/streamlit_dashboard/app.py`'s "Production Monitoring (Track 3)"
   view now renders Evidently drift output (score, detected flag, drifted-column count, drift
   share, timestamp). **Still open (post-freeze cleanup):** View B per-page relabel and
   `live_df`/`load_scoring_data` rename to eliminate misleading live-data naming for labeled OOF.
3. **Resolved** (freeze cleanup batch). `docs/architecture.md` redrawn with two separate
   Mermaid diagrams — one for Track 1 (labeled OOF flow) and one for Track 3 (label-free
   production flow). The stale single diagram with `Batch Logs\nrecall/precision/flagged` under
   "Production Architecture" is gone.
4. **Resolved** (T3-4 + freeze cleanup batch). `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md`
   updated with RP2 honest distribution (§7), label-free monitoring status (§9), and Track 3
   Achieved items (§12). `README.md` updated with RP2 distribution and Production Monitoring
   dashboard feature. Per-page View B relabel/rename remains open (see item 2).
5. **Resolved for `dataset_h`** (see `docs/dataset_h_submission_run.md` and
   `docs/runbooks/track2_kaggle_submission.md`). `scripts/generate_submission.py` +
   `scripts/build_test_dataset_h.py` + `models/dataset_h_model.pkl` (Phase-2 payload) produce a
   validated 1,183,748-row submission locally. Other models (`baseline`, `dataset_g`, `meta_model`)
   still lack test-side feature tables and are not in scope for Track 2 opening. Track 2 opens on
   `kaggle-main`.
6. `CLAUDE.md`'s claim that "the dashboard and decision-system code already enforce this split"
   should be revisited once 1–2 are addressed, since it currently overstates the present state.

Items 1 and 2 have since been addressed (in part or fully) in follow-up changes described above;
3–6 remain open, per explicit scope limits on each of those changes.

---

## Relationship to other docs

- `system_design.md` — updated alongside this doc to point here and to stop being silent on
  Track 2 and the dashboard A/B split (see its diff). Note: `system_design.md` is listed in
  `.gitignore`, so this edit is local-only and will not appear in `git diff`/the eventual commit
  for this branch unless force-added.
- `bosch_agent.md` — its "two tracks, Kaggle ignored for now" framing is superseded by this doc's
  three-track framing. Not edited in this change (untracked file, not in scope of this task).
- `docs/reproducible_metrics_report.md` — remains the source of truth for which Track 1 metrics
  are actually reproducible (World A vs World B); unaffected by this change.
- `docs/production_readiness_audit.md` — merged onto `main` alongside this doc (both were
  consolidated from their respective feature branches in the same documentation-consolidation
  pass). It independently identified most of the same Track 3 / dashboard mislabeling issues
  documented here, from a read-only audit of the repo at commit `b96cd0d`.
