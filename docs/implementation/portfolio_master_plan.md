# Portfolio Master Plan — Execution Ledger

**Status: FROZEN (ratified 2026-07-03).** This is the only execution planning document for the
portfolio transition. It supersedes all prior roadmap drafts and audit action lists. It is the
portfolio-track analog of the Kaggle KDR governance log: statuses change, scope does not.

- Source inputs: repository engineering audit (2026-07-01) + approved PF0–PF8 roadmap + 8 ratified
  amendments (external review, 2026-07-03).
- Kaggle research is frozen at `P1-result` (private MCC 0.41917); KDR-009 records the freeze
  (`docs/research/kaggle_decisions.md`, tag `track2-frozen`).
- This file is committed as the first commit of PF0 and updated at every phase transition.

---

## 1. Executive summary

The repository transitions from a frozen research program into a public engineering portfolio in
7 core phases (PF0–PF6) plus 2 optional phases (PF7–PF8). Two milestones gate release:

- **M1 — "safe to share"** (after PF3): all ten audit must-fixes verified; repo URL usable on a
  resume. ~30–43 h serial (~4.5–5.5 focused days; ~4–4.5 with PF1∥PF2).
- **M2 — "portfolio launch, v1.0.0"** (after PF6): dashboard live at `bosch.themachinist.org`,
  docs site live at `/docs/`, `v1.0.0` released. ~60–89 h serial (~9–11 days; ~8–9 with PF4∥PF5).

`results/leaderboard.json` (created in PF0) is the single source of truth for every result number
in every downstream artifact: README, SYSTEM_OVERVIEW, case study, dashboard, docs site, release.

## 2. Frozen scope statement

Scope is fixed by the audit plus exactly eight ratified amendments:

1. GitHub repository rename removed — deferred indefinitely; repo slug never changes.
2. `results/leaderboard.json` moved into PF0 as the single source of truth.
3. Recruiter dashboard uses a React-based stack (Vite + React + TypeScript primary; Next.js
   static export acceptable), still exported static, still Cloudflare Pages, client-side only,
   same four pages, same JSON export pipeline.
4. `results/leaderboard.json` is a PF0 deliverable, created before README work begins.
5. `CHANGELOG.md` added to PF6 (simple semantic version history).
6. `SYSTEM_OVERVIEW.md` added to PF1 (one-page architecture map, everything linked from one doc).
7. This ledger created; it replaces further planning.
8. Scope freeze: **no further redesign, no new roadmap, no scope expansion, no reprioritization.**
   Anything discovered during implementation is logged to the PF8 backlog (§11) — unless it is a
   correctness bug, which is fixed in the phase that finds it and noted here.

## 3. Standing rules (apply to every phase)

1. **No git-history rewrite, ever.** Evidence SHAs in the KDR logs and 22+ tags are the audit trail.
2. **Historical logs are append-only** (`kaggle_decisions.md`, `decisions.md`). Path renames touch
   living docs only (README, runbooks, SYSTEM_OVERVIEW, local CLAUDE.md).
3. **Firewall grep at every phase merge** (`import.*kaggle` outside `src/kaggle`/`scripts/kaggle`
   must be empty). `src/kaggle/` and `scripts/kaggle/` never move.
4. **No test-set prediction artifacts** published on the site or in Releases.
5. **Honesty semantics survive all reframing** — unverified numbers stay labeled unverified; any
   sampled/demo data ships behind an explicit "DEMO SAMPLE" banner.
6. **Branch per phase** (`portfolio/PF<n>-<slug>`), `--no-ff` merge to `main`, commit prefixes per
   §6, PRs once CI exists (PF3+). Pushes/merges run under per-phase authorization.
7. **Scope is frozen** (§2.8). Discoveries → PF8 backlog unless correctness bugs.
8. **`results/leaderboard.json` is the single source of truth** for result numbers; every number in
   any artifact must trace to it (and it, in turn, was validated against `kaggle_decisions.md`).
9. **Repository rename is deferred indefinitely.** Never rename; title consistency lives in docs.
10. **This ledger is updated at every phase transition** (§12) and is the only planning document.

## 4. Phase status table

| Phase | Name                                   | Status      | Checkpoint | Effort    |
|-------|-----------------------------------------|-------------|------------|-----------|
| PF0   | Research freeze & unification + registry | COMPLETE    | CP0 ✓ approved 2026-07-03 | 4–7 h |
| PF1   | Headline documents                     | COMPLETE    | CP1 ✓ approved 2026-07-03 | 8–12 h |
| PF2   | Code hygiene                           | IN PROGRESS | CP2        | 11–15 h   |
| PF3   | Tests + CI (M1 gate)                   | NOT STARTED | CP3        | 7–9 h     |
| PF4   | Recruiter dashboard + hosting          | NOT STARTED | CP4        | 18–28 h   |
| PF5   | Documentation site                     | NOT STARTED | CP5        | 8–12 h    |
| PF6   | Artifacts & v1.0.0 (M2 gate)           | NOT STARTED | CP6        | 4–6 h     |
| PF7   | Live tier (OPTIONAL, gated at CP6)     | NOT STARTED | —          | 5–8 h     |
| PF8   | Polish + backlog (OPTIONAL, elective)  | NOT STARTED | —          | 4–8 h+    |

Status lifecycle: `NOT STARTED → IN PROGRESS → AWAITING REVIEW (CPn) → COMPLETE`
(PF7 may become `SKIPPED` by CP6 decision.)

## 5. Dependencies and parallelization

```
PF0 ─┬─► PF1 (headline docs) ──────┐
     └─► PF2 (code hygiene) ──► PF3 ──► M1: safe to share
          PF1 ∥ PF2 (land PF2's move-commit first)
                                   │
                    ┌─► PF4 (dashboard + hosting) ─┐
        M1 ────────┤          PF4 ∥ PF5            ├─► PF6 (v1.0.0) ─► M2: launch
                    └─► PF5 (docs site) ───────────┘
                                                   │
        M2 ──► PF7 (go/no-go at CP6) ──► PF8 (elective)
```

- Strictly serial: PF0 → everything; PF2 → PF3; (PF4+PF5) → PF6; PF6 → PF7.
- PF1 ∥ PF2: disjoint files; PF2's commit-1 (pure `git mv`) lands before PF1's README merges so
  paths are real. PF4 ∥ PF5: disjoint except the shared Pages workflow (second-to-land rebases).
- User-side prep (parallel, non-blocking until their phase's last step): Cloudflare account + DNS
  for themachinist.org (end of PF4); uptime-monitor account (PF6); live-tier decision (CP6).

## 6. Commit prefixes, branches, tags

- Branches: `portfolio/PF<n>-<slug>` (e.g. `portfolio/PF2-code-hygiene`).
- Commits: `PF<n> <type>: <subject>` with `type ∈ {docs, chore, refactor, build, test, ci, feat, fix}`.
  Ledger status updates use the active phase's prefix (e.g. `PF2 docs: ledger status`).
- Merges to `main`: always `--no-ff`.
- Tags: `track2-frozen` (PF0 merge commit, annotated), `v1.0.0` (PF6, annotated). No other tags.

## 7. Review checkpoints

| CP  | After | User reviews                                                              | Time   |
|-----|-------|---------------------------------------------------------------------------|--------|
| CP0 | PF0   | Freeze record wording; LICENSE name line; leaderboard.json spot-check; this ledger | 10 min |
| CP1 | PF1   | README + case study + SYSTEM_OVERVIEW read-through — voice + number accuracy sign-off | 30 min |
| CP2 | PF2   | Smoke evidence: compose up, credential-free dashboard, validate_system output | 10 min |
| CP3 | PF3   | **M1 gate**: ten must-fixes vs evidence; authorize sharing the repo        | 15 min |
| CP4 | PF4   | Dashboard on Pages preview URL before DNS attach                           | 30 min |
| CP5 | PF5   | Docs-site nav click-through                                                | 15 min |
| CP6 | PF6   | **M2 launch**: full walkthrough; v1.0.0 authorization; **PF7 go/no-go**    | 30 min |

## 8. Completion criteria

**M1 — safe to share (exit of PF3, CP3):** unified `main` with the full K1→P1 program; rewritten
README; LICENSE; tests + CI green (3 consecutive runs); dormant predictor path removed; S3 bucket
parameterized + dashboard runs credential-free; Dockerfiles fixed (+ `.dockerignore`); single
dependency source (`pyproject.toml` + `requirements.txt`); case study + README refreshed from
`leaderboard.json`; screenshots present; secrets scan clean; CP1 sign-off recorded.

**M2 — portfolio launch (exit of PF6, CP6):** M1 + `bosch.themachinist.org` live via CI deploy +
`/docs/` live + `CHANGELOG.md` + `v1.0.0` tagged with GitHub Release + weekly health workflow and
uptime monitors green.

**Project complete** when M2 is reached and the CP6 PF7 decision is executed (built or SKIPPED).
PF8 remains an elective backlog thereafter.

## 9. Phase specifications

### PF0 — Research freeze, unification, registry, ledger

**Status: COMPLETE (CP0 approved 2026-07-03)**

- **Objective:** `main` becomes the single public source of truth; Track 2 formally frozen; the
  results registry and this ledger established; metadata table stakes in place.
- **Depends on:** nothing (root phase).
- **Deliverables:**
  1. Fast-forward merge `kaggle-main` → `main` (verified strict descendant), pushed.
  2. This ledger committed at `docs/implementation/portfolio_master_plan.md` (first commit).
  3. `results/leaderboard.json` — hand-authored registry, one row per sealed LB-scored experiment
     (9 rows: K1, K2, K3-A, K3-B, K4, K5-A, K5-B, P0, P1). Fields: experiment_id, mechanism
     one-liner, kdr ref + anchor, hypothesis band result, oof_mcc, public_mcc, private_mcc,
     threshold, data fingerprint, git tag, date. Values transcribed from `kaggle_decisions.md`
     under validation — never from memory.
  4. KDR-009 freeze entry appended to `docs/research/kaggle_decisions.md` (final ladder, postmortem
     link, revised target statement 0.52 → ~0.44, unfreeze criteria).
  5. `docs/ml_system_tracks.md`: Track 2 → Frozen / 100%.
  6. Annotated tag `track2-frozen` on the PF0 merge commit.
  7. `LICENSE` (MIT), `.env.example`, `__pycache__` untracked + `.gitignore` coverage.
  8. Branches `training-pipeline` and `kaggle-main` deleted (local + remote) after verification.
  9. GitHub description + topics set. **No repository rename — deferred indefinitely.**
- **Files:** `docs/implementation/portfolio_master_plan.md` (new), `results/leaderboard.json` (new),
  `docs/research/kaggle_decisions.md` (append-only), `docs/ml_system_tracks.md`, `LICENSE` (new),
  `.env.example` (new), `.gitignore`; index removal of `apps/api/__pycache__/*`. Local-only:
  `docs/agent_memory/claude_state.md`. External: GitHub description/topics.
- **Risks:** deleting branches before verifying tag reachability; editing existing KDR sections
  instead of appending; registry transcription errors.
- **Validation checklist (all verified — see CP0 report):**
  - [x] `git merge-base --is-ancestor main kaggle-main` true before FF; `git rev-parse` identical after
  - [x] All tags resolve and are reachable from `main`; `git branch --merged main` shows both
        branches merged before deletion
  - [x] Firewall grep clean on `main`
  - [x] KDR-009 diff touches no existing lines in the log
  - [x] `leaderboard.json` parses; exactly 9 rows; every OOF/public/private value grep-matched
        against `kaggle_decisions.md` evidence sections; fingerprints and tags present
  - [x] `git ls-files | grep __pycache__` empty; `git status` clean
- **Git workflow:** FF merge + push; then `portfolio/PF0-freeze-metadata` with commits —
  `PF0 docs:` ledger; `PF0 docs:` leaderboard.json; `PF0 docs:` KDR-009 + tracks; `PF0 chore:`
  LICENSE/.env.example/gitignore/pycache — `--no-ff` merge, tag `track2-frozen`, push with tags,
  delete obsolete branches.
- **Stopping point:** CP0. **Effort: 4–7 h.**
- **Execution record (2026-07-03):** FF merge `main` `b058e58`→`2644487` (identical to
  `kaggle-main`) pushed before this branch was cut; all 22 tags confirmed reachable from `main`
  pre-merge; firewall grep empty; `leaderboard.json` validated (9 rows, every MCC value and 8/9
  fingerprints grep-matched against `kaggle_decisions.md`, P1's fingerprint provenance caveat
  documented inline); KDR-009 diff confirmed append-only (adds lines only). No pycache files were
  tracked and `.gitignore` already covered `__pycache__/` — that sub-item required no change.
  Full command-level evidence and final branch-deletion/GitHub-metadata confirmation are recorded
  in the CP0 report delivered alongside this commit, not duplicated here.
- **CP0 outcome (2026-07-03): APPROVED.** Merge commit `31cdc10` (tag `track2-frozen` ->
  `895094c`) pushed to `origin/main`; `kaggle-main` and `training-pipeline` verified merged and
  deleted (local + remote); GitHub description/topics set. No corrections requested at CP0.

### PF1 — Headline documents

**Status: COMPLETE (CP1 approved 2026-07-03)**

- **Objective:** the two most-read documents lead with the verified results story; one-page system
  map exists.
- **Depends on:** PF0 (registry). PF2's commit-1 (file moves) lands before PF1's README merges.
- **Deliverables:** README rewritten per audit §9 spec (hero → results table sourced from
  `leaderboard.json` → architecture visual → three links → quickstart (plain commands; `make`
  swap happens in PF3) → honest-vs-leaderboard paragraph → governance blurb → stack → license);
  `SYSTEM_OVERVIEW.md` at repo root (one-page architecture map linking tracks, architecture,
  dashboard, deployment, research, case study — everything reachable from one document; README
  links to it); case study refreshed (P-line numbers from the registry; UNVERIFIED disclaimer
  demoted to a provenance sidebar with semantics intact); 3 interim visuals (existing committed
  PNGs + Evidently capture) under `docs/assets/`.
- **Files:** `README.md` (rewrite), `SYSTEM_OVERVIEW.md` (new), `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md`,
  `docs/assets/` (new), possibly `docs/architecture.md` (only if README embeds its Mermaid).
- **Risks:** number drift (all numbers must trace to `leaderboard.json`); overcorrection in the
  reframe (honesty semantics must survive); tone (subjective — CP1 gate).
- **Validation checklist (all verified — see CP1 report):**
  - [x] Every metric in README/case study/SYSTEM_OVERVIEW traces to `leaderboard.json`
  - [x] README renders correctly on GitHub; all links resolve, incl. every SYSTEM_OVERVIEW link
  - [x] Disclaimer content preserved verbatim within one click; no World-B number stated as verified
- **Git workflow:** `portfolio/PF1-headline-docs` (`PF1 docs:`), `--no-ff`.
- **Stopping point:** CP1 (voice + accuracy sign-off). **Effort: 8–12 h.**
- **Execution record (2026-07-03):** README rewritten (results-first: honest RP2 production range
  side-by-side with the frozen, clearly-labeled Kaggle ladder); `SYSTEM_OVERVIEW.md` added at repo
  root; case study's top-of-document disclaimer relocated verbatim to a sidebar directly above the
  §7 table it describes (all four key factual claims confirmed present via script), plus a new
  §13 covering the Kaggle track. Every 5-decimal MCC value across all three documents (27 in
  README, 2 in SYSTEM_OVERVIEW, 5 in the case study) was extracted by script and confirmed to
  match a `leaderboard.json` value exactly — zero unexplained numbers. Both README architecture
  diagrams are byte-identical copies of `docs/architecture.md`'s Mermaid blocks (diffed, not
  redrawn); `docs/architecture.md` itself was not modified. Two existing charts
  (`outputs/e4_ranking_metrics.png`, `outputs/e4_topk_lift.png`) were copied into `docs/assets/` and
  embedded, not regenerated. Deviation: only 2 of the spec'd "3 interim visuals" were used — no
  Evidently HTML screenshot was taken, since capturing one would itself be creating a new visual
  asset (no browser tooling was invoked); see the CP1 report for the full rationale.
- **CP1 outcome (2026-07-03): APPROVED.** Merge commit `5823568` pushed to `origin/main`. No
  corrections requested at CP1.

### PF2 — Code hygiene

**Status: IN PROGRESS**

- **Objective:** every file a reviewer opens works, builds, and belongs; one dependency source of
  truth; dashboard runs credential-free.
- **Depends on:** PF0.
- **Deliverables:** delete `src/inference/predictor.py` + `two_stage_predictor.py` (+ prune
  `src/features/pipeline.py` code serving only that path if cleanly separable); extract
  `load_validated_payload`/`predict_proba_ensemble` → `src/inference/payload.py`; regroup
  `scripts/` → `scripts/{pipeline,research,ops}/` (**`scripts/kaggle/` untouched**);
  `pyproject.toml` (metadata + ruff/pytest stubs); delete `environment.yml`; `Makefile`
  (`setup/test/lint/dashboard-data/docker-up`); Dockerfiles fixed (requirements-first layers,
  `COPY . .` last, non-root user, HEALTHCHECK); `.dockerignore` (new); compose fixed (dev bind
  mounts → override file, healthchecks, env passthrough); Streamlit app: `DATA_SOURCE=local|s3`
  with local default, bucket from env only (no real-bucket default), optional committed demo slice
  under `data/demo/` behind a mandatory "DEMO SAMPLE" banner; living docs swept for new paths.
  Historical logs untouched.
- **Files:** deletions in `src/inference/`; `src/inference/payload.py` (new); ~20 script moves;
  `scripts/generate_submission.py` + `scripts/run_production_inference.py` (imports);
  `pyproject.toml`, `Makefile`, `.dockerignore` (new); `environment.yml` (deleted);
  `Dockerfile.api`, `Dockerfile.dashboard`, `docker-compose.yml` (+ override file);
  `apps/streamlit_dashboard/app.py`, `src/utils/s3_utils.py`; `docs/runbooks/*.md`; `.gitignore`.
- **Risks:** highest blast radius — import breakage; doc-command drift; Docker regressions;
  dashboard refactor breaking S3 mode; scope temptation to refactor beyond data-loading (resist).
- **Validation checklist:**
  - [ ] `python -m compileall src scripts apps` clean; import sweep passes
  - [ ] Each pipeline script answers `--help` from its new path; `run_full_system.py` +
        `validate_system.py` smoke-run green on existing local artifacts
  - [ ] Both images build; `docker compose up` healthchecks pass; build context < 100 MB
  - [ ] Dashboard launches with zero AWS credentials (local mode); S3 mode still works with `.env`
  - [ ] `grep -rn "scripts/" README.md docs/runbooks/` — no stale paths; zero edits to historical logs
  - [ ] Firewall grep clean; `grep -rn "bosch-ml-production" --include="*.py"` empty
- **Git workflow:** `portfolio/PF2-code-hygiene`; commit-1 = pure `git mv`; then extraction,
  packaging, Docker, dashboard (`PF2 chore:/refactor:/build:`), `--no-ff`.
- **Stopping point:** CP2 (smoke evidence). **Effort: 11–15 h.**

### PF3 — Tests + CI (M1 gate)

**Status: NOT STARTED**

- **Objective:** production-discipline claims become machine-checked.
- **Depends on:** PF2 (import layout, Dockerfiles, pyproject).
- **Deliverables:** `tests/` — decision-engine pure functions; CV guards (leak-injection must
  raise); synthetic-fixture feature tests; submission validator on fixtures; API via TestClient
  with fixture policy JSON; in-test generated tiny LGBM fixture model (no committed pickle);
  schema + values test locking `results/leaderboard.json` against expected constants; ruff
  configured, codebase passing (autofixes + per-rule ignores, **no refactoring to satisfy lint**);
  `.github/workflows/ci.yml` (lint → pytest < 2 min → both docker builds, no push → leaderboard
  schema check); badges (CI, license, Python) into README; README quickstart swapped to `make`
  targets; `@pytest.mark.slow` documented as the manually-run regression-anchor tier.
- **Files:** `tests/` (new, ~6 files + fixtures), `pyproject.toml`, `.github/workflows/ci.yml`
  (new), `README.md`, mechanical lint fixes across `src/`/`scripts/`/`apps/`.
- **Risks:** lint explosion on legacy code (config-ignore, don't rewrite); CI flake (deterministic
  tests only); runtime creep past 2 min; docker-build context (tests PF2's `.dockerignore`).
- **Validation checklist:**
  - [ ] `pytest` green < 2 min locally; `ruff check .` clean
  - [ ] CI green on PR and on `main`; 3 consecutive re-runs green
  - [ ] Leak-injection test fails when the guard is disabled (test the test)
  - [ ] Docker build job green with no local data; badges render
- **Git workflow:** `portfolio/PF3-tests-ci` (`PF3 test:/ci:`), first PR-gated merge, `--no-ff`.
- **Stopping point:** CP3 = **M1 gate**. **Effort: 7–9 h.**

### PF4 — Recruiter dashboard + hosting

**Status: NOT STARTED**

- **Objective:** one URL — `bosch.themachinist.org` — that tells the whole story in 90 seconds.
- **Depends on:** M1; `results/leaderboard.json`. User-side: Cloudflare account + DNS (needed only
  at the end).
- **Stack (frozen by amendment 3):** **React + TypeScript, built with Vite** (Next.js static
  export is the sanctioned alternative); fully static export; Cloudflare Pages; client-side
  interactivity only; Plotly via npm (basic/partial bundle, code-split per page); pinned deps,
  committed lockfile, `.nvmrc`; bundle budget ≤ 1.5 MB gzipped total.
- **Deliverables:** `scripts/ops/export_dashboard_data.py` reading local OOF/summary artifacts →
  committed JSON bundle (leaderboard, per-model threshold sweeps ~200 grid points, PR/ROC/
  calibration points, top-25 importances with family tags, fold spreads, metadata cards — tens of
  KB each; never the parquets; export asserts consistency against `results/leaderboard.json`);
  four pages exactly per audit §11 spec — Story / Decision Explorer / Model Internals /
  Governance & Reproducibility; Cloudflare Pages project + DNS + TLS; CI deploy job
  (`npm ci && npm run build` → Pages) on main-merge; README hero links + dashboard screenshot
  replacing interim visuals; no test-set prediction CSVs anywhere.
- **Files:** `dashboard/` (new Vite app: `package.json`, lockfile, `tsconfig`, `vite.config`,
  `src/`, `public/data/*.json`, `.nvmrc`), `scripts/ops/export_dashboard_data.py` (new),
  `.github/workflows/deploy-pages.yml` (new or ci.yml job), `README.md`, short deploy runbook.
- **Risks:** scope creep (four pages, fixed content, done); data drift (export asserts against the
  registry, fails loudly); npm supply chain (minimal pinned dep set + lockfile); CI node build;
  bundle weight (Plotly partial bundle, code-split); DNS/user dependency; mobile QA.
- **Validation checklist:**
  - [ ] Export script deterministic (two runs, identical JSONs) and consistent with `leaderboard.json`
  - [ ] Slider threshold reproduces precomputed MCC/precision/recall at 5 spot-checked thresholds
        against the Python side
  - [ ] `npm ci && npm run build` reproducible in CI; bundle ≤ 1.5 MB gzipped; Lighthouse ≥ 90;
        renders on mobile
  - [ ] Every KDR/tag/repo deep link resolves; loads < 1.5 s on Pages preview
  - [ ] DNS + TLS live; production deploy comes from CI, not manual upload
- **Git workflow:** `portfolio/PF4-dashboard` (`PF4 feat:/ci:`); PR with Pages preview URL for CP4;
  `--no-ff` merge triggers production deploy; DNS attached only after CP4 approval.
- **Stopping point:** CP4 (preview-URL tour). **Effort: 18–28 h.**

### PF5 — Documentation site

**Status: NOT STARTED**

- **Objective:** 8,300 lines of existing docs become a navigable asset; every track independently
  understandable.
- **Depends on:** M1. Runs **parallel with PF4** (disjoint files; shared deploy workflow —
  second-to-land rebases).
- **Deliverables:** MkDocs Material site (nav: Home / Results / Architecture / Tracks 1–3 /
  Research Log / Runbooks / Model & Data Cards) reusing `SYSTEM_OVERVIEW.md` as home material;
  `docs/RESEARCH_SUMMARY.md` (2 pages: ladder + mechanism attribution + KDR anchors — the full
  research postmortem this document supersedes KDR-009's condensed inline version); `docs/research/README.md`
  (index + "how to read a KDR"); model card (P1 competition model **and** production dataset_h
  model, deployability distinction as centerpiece); data card (Kaggle license/no-redistribution,
  download + `prepare_data.py` + expected-fingerprint verification table); `docs/decisions/adr-experiment-tracking.md`
  (files+git over MLflow, one page); per-track runbook front-matter alignment; Track 2 lane in
  `architecture.md`; docs build merged into the Pages artifact under `/docs/`.
- **Files:** `mkdocs.yml` (new), `docs/index.md` (new), `docs/RESEARCH_SUMMARY.md` (new),
  `docs/research/README.md` (new), `docs/model_card.md` + `docs/data_card.md` (new),
  `docs/decisions/adr-experiment-tracking.md` (new), `docs/architecture.md`, runbook front-matter,
  deploy workflow, dev-deps in `pyproject.toml`.
- **Risks:** nav sprawl (curate; link into the 2,120-line log, never restructure it); broken links
  under strict mode (fix links, never historical content); README duplication (README summarizes,
  site elaborates).
- **Validation checklist:**
  - [ ] `mkdocs build --strict` clean
  - [ ] Each track landing answers the audit §9 question list, checked explicitly
  - [ ] Model card states deployable vs leaderboard numbers with the measured gap;
        data card fingerprint table matches recorded fingerprints
  - [ ] `/docs/` serves correctly inside the Pages deploy alongside the dashboard
- **Git workflow:** `portfolio/PF5-docs-site` (`PF5 docs:`), PR + `--no-ff`.
- **Stopping point:** CP5 (nav click-through). **Effort: 8–12 h.**

### PF6 — Artifacts & v1.0.0 (M2 gate)

**Status: NOT STARTED**

- **Objective:** artifact hygiene finished, release automation live, portfolio launched.
- **Depends on:** PF4 + PF5.
- **Deliverables:** 93 MB model pickles removed from HEAD (`git rm --cached`; **no history
  rewrite** — rationale documented); model download documented (`gh release download` in
  README/data card); `CHANGELOG.md` (new — Keep-a-Changelog style, semantic versions; `v1.0.0`
  entry summarizing the portfolio transition; prehistory pointer to the existing tag series);
  `.github/workflows/release.yml` (tag `v*` → GitHub Release: leaderboard.json,
  training_summary.json, Evidently HTML, model card, model pickles; release body links
  CHANGELOG; no test-prediction CSVs); weekly health workflow (link check + public-URL probes);
  uptime monitor on public URLs (user account); final sweeps (firewall, secrets, tag audit);
  annotated tag **`v1.0.0`** + published Release.
- **Files:** `models/*.pkl` (untracked from HEAD), `CHANGELOG.md` (new),
  `.github/workflows/release.yml` + `weekly-health.yml` (new), `README.md`, `docs/data_card.md`.
  External: uptime monitor.
- **Risks:** unnoticed model consumer (grep first — known consumer was deleted in PF2);
  release workflow artifact availability (repo-tracked files + documented manual pickle upload);
  filter-repo temptation (**forbidden**).
- **Validation checklist:**
  - [ ] `grep -rn "models/" src scripts apps --include="*.py"` — only training writers or
        documented download consumers remain
  - [ ] Fresh `git clone` + `make setup && make test` green in a clean environment
  - [ ] `v1.0.0-rc` rehearsal produces a correct draft Release; then real tag on `main`
  - [ ] `CHANGELOG.md` has the `v1.0.0` entry; release body links it
  - [ ] Weekly health workflow passes on manual dispatch; uptime monitors green
  - [ ] Final audit walkthrough: all must-fixes + freeze-checklist items ✓
- **Git workflow:** `portfolio/PF6-release` (`PF6 chore:/ci:/docs:`), PR + `--no-ff`, annotated
  `v1.0.0` on `main`, push with tags, publish Release.
- **Stopping point:** CP6 = **M2 launch review + PF7 go/no-go**. **Effort: 4–6 h.**

### PF7 — Live tier (OPTIONAL — gated on explicit CP6 go/no-go)

**Status: NOT STARTED**

- **Gate:** user commits to ~monthly VPS maintenance and ~$6/mo. If no → status `SKIPPED`
  permanently; fallback stands ("deployable, demonstrated in CI").
- **Objective:** `api.` + `console.bosch.themachinist.org` on an always-on VPS.
- **Deliverables:** Hetzner/DO VPS (Docker, SSH-key-only, firewall, unattended upgrades);
  `compose.prod.yaml` (GHCR images, no bind mounts, restart policies, healthchecks) + `Caddyfile`
  (auto-TLS); Cloudflare-proxied DNS; CI extended: GHCR push on main-merge + SSH deploy
  (`compose pull && up -d`) + `/health` gate; uptime monitors; dashboard runs local/demo mode on
  the server (no AWS credentials on the VPS).
- **Files:** `compose.prod.yaml`, `Caddyfile`, deploy workflow, ops runbook. External: VPS, DNS,
  Actions secrets.
- **Risks:** server-ops burden (boring stack mitigates); Actions secret handling.
- **Validation checklist:** cold-reboot self-heal; valid TLS; external `/health` 200; CI deploy
  observed end-to-end; monitors green 48 h before linking from the site.
- **Git workflow:** `portfolio/PF7-live-tier`, PR + `--no-ff`.
- **Stopping point:** both subdomains live and linked from the Governance page. **Effort: 5–8 h + $6/mo.**

### PF8 — Polish + backlog (OPTIONAL, elective, post-M2)

**Status: NOT STARTED**

- **Fixed items:** Decision-Explorer GIF; tag-timeline graphic; `CITATION.cff`;
  `.pre-commit-config.yaml` (ruff + whitespace); blog post on themachinist.org. Each independent,
  1–2 h; branch per item or one `portfolio/PF8-polish`; no checkpoint beyond tone review of the
  blog post.
- **Backlog intake:** see §11.

## 10. Frozen technical decisions

1. No git-history rewrite (evidence SHAs + tags are the governance trail).
2. `results/leaderboard.json` is hand-authored + validated against the KDR log, then locked by a
   PF3 test; it is the single source of truth for all result numbers.
3. Dashboard JSON bundle is committed, not CI-built (CI has no access to gitignored parquets).
4. Test fixture model is generated in-test (no committed pickle).
5. Dashboard stack: **Vite + React + TypeScript**, static export (Next.js static export is the
   sanctioned alternative); Plotly via npm partial bundle; client-side only. (Amendment 3 —
   replaces the earlier framework-free decision.)
6. One domain, one Pages project: dashboard at root, MkDocs output at `/docs/`.
7. `environment.yml` deleted; `pyproject.toml` = metadata + tool config; `requirements.txt` =
   pinned install set for Docker/CI.
8. `.dockerignore` added as completion of the Docker fixes (build context excludes data/models/git).
9. `scripts/kaggle/` and `src/kaggle/` never move (firewall is path-anchored).
10. GHCR image push deferred to PF7 (no dead infrastructure if the live tier is skipped).
11. Historical decision logs are immutable; only living docs get path updates.
12. Repository rename deferred indefinitely (amendment 1); title consistency in docs only.

## 11. PF8 backlog

Intake rule (frozen): anything discovered during implementation that is not a correctness bug is
appended here with one line (date, phase discovered, description) and is **not** acted on before
PF8. Correctness bugs are fixed in the phase that finds them and noted in that phase's section.

- (empty)

## 12. Ledger protocol

- Status transitions (§4 lifecycle) are updated in this file as part of the active phase's branch:
  set `IN PROGRESS` in the phase's first commit, `AWAITING REVIEW (CPn)` in its last, `COMPLETE`
  (with date + checkpoint outcome one-liner) in a follow-up commit on `main` after user sign-off.
- Checkpoint outcomes and any in-phase deviations are recorded in the phase's section, append-style,
  mirroring KDR Evidence discipline.
- This document's scope sections (§2, §3, §9 specifications, §10) are frozen. Only statuses,
  checkpoint records, deviation notes, and the §11 backlog may change.
