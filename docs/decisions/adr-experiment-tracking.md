# ADR: Files + Git Over MLflow for Experiment Tracking

**Status:** Accepted (in force since the project's earliest research decisions; formalized here in
PF5 of `docs/implementation/portfolio_master_plan.md`).

## Context

This project ran two disjoint, pre-registered research programs (Track 1/3's `decisions.md`,
DR-001–DR-015, and Track 2's `kaggle_decisions.md`, KDR-001–KDR-009) totaling 19 sealed decisions
and 14 sealed experiments, each requiring a hypothesis fixed before results, evidence, an outcome,
and a decision — the standard shape of a tracked ML experiment. A tool like MLflow (or Weights &
Biases, Neptune, etc.) is the conventional answer to "how do I track experiments" at this scale.

## Decision

Track every experiment as a **prose entry in an append-only Markdown log, committed to git,**
rather than adopting a dedicated experiment-tracking tool. Metrics that need to be
machine-readable (`results/leaderboard.json`) are a small, hand-authored, git-tracked JSON file
validated against the prose log, not a database or tracking-server export.

## Why this, not MLflow

- **Pre-registration is the actual discipline, not metric logging.** The valuable part of this
  project's research protocol is writing the hypothesis, evidence bar, and decision rule *before*
  seeing results — a narrative commitment device. MLflow tracks runs and metrics well; it has no
  native concept of "here is what I predicted before I ran this," which is the load-bearing
  artifact here, not the metric itself.
- **Git is already the audit trail.** Every experiment's evidence cites a commit SHA, a branch, and
  (for sealed results) an annotated tag (`K1-result` … `P1-result`, `track1-frozen`,
  `track2-frozen`, `track3-frozen`). Bringing in a second system of record for the same experiments
  would mean either duplicating that trail or reconciling two sources of truth — git history and an
  MLflow tracking store — every time someone asks "what actually happened."
- **No server, no infra, no credential surface.** A tracking server (or a hosted MLflow instance)
  is one more thing to run, secure, and keep available for a project whose current audience is a
  recruiter clicking through a public dashboard and a single researcher iterating locally, not a
  team running hundreds of concurrent parameterized jobs.
- **The log doubles as the documentation.** `decisions.md` and `kaggle_decisions.md` are readable
  narrative — a hiring manager or collaborator can read them start to finish and understand the
  full scientific story. An MLflow run table is a much harder artifact to read cover to cover; it
  answers "what were the metrics" well and "why did I do this, and what did I conclude" poorly.
- **Scale fits.** 14 sealed experiments over roughly a week of work is comfortably within what a
  disciplined human-maintained log can carry without becoming unnavigable. This decision would be
  revisited at meaningfully larger experiment counts or multiple concurrent contributors — neither
  applies here.

## Consequences

- **Positive:** zero additional infrastructure; the full research history is `git clone`-able,
  greppable, and diffable like any other artifact; pre-registration discipline is enforced by the
  log's own required-fields convention (see [Research Log → How to read a KDR](../research/README.md#how-to-read-a-kdr)),
  not by a tool feature.
- **Negative:** no built-in metric visualization, no automatic run comparison UI, no parameter
  sweep tracking beyond what's written into a `§4 Experimental matrix` table by hand (see
  `KDR-008` §4 for an example). The recruiter dashboard's Model Internals and Decision Explorer
  pages exist in part to cover the visualization gap for the results that matter most.
- **Reversible:** nothing about this decision blocks adopting a tracking tool later if experiment
  volume grows — the underlying data (`results/leaderboard.json`, `outputs/training_summary.json`)
  is already structured enough to backfill into one.
