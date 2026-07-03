# Research Log — Index

Two disjoint, append-only decision logs, one per governance domain. Neither log is restructured by
this docs site — this page is a reading guide that links into them, not a replacement for either.

| Log | Domain | Entry numbering | Rule |
|---|---|---|---|
| `docs/research/decisions.md` | Track 1 (offline training/eval) + Track 3 (production inference) | `DR-NNN`, experiments `E<N>` | MCC/precision/recall only on labeled OOF/CV; chunk-aware CV is the default harness |
| `docs/research/kaggle_decisions.md` | Track 2 (Kaggle leaderboard research) | `KDR-NNN`, experiments `K<N>`/`P<N>` | No metric or conclusion from this log may ever appear in `decisions.md` or inform a `DR`/`E` decision |

Both logs share the same discipline: pre-registration before results (hypothesis, evidence bar,
and decision rule fixed *before* seeing outcomes), a mandatory Bayesian belief-update block per
entry (prior → evidence → posterior → confidence → why it moved), and negative results treated as
first-class (a null is an upper bound on an effect, never proof of absence).

## How to read a KDR

Each `KDR-NNN` entry in `docs/research/kaggle_decisions.md` follows the same shape — once you know
it, every entry reads the same way:

1. **Header** — date, decision type (pre-registration, amendment, or freeze), one-line trigger.
2. **§1 Trigger** — why this decision is being made now, referencing the prior KDR it builds on.
3. **§2 Imported priors** — what's carried in from earlier KDRs or the production track, cited by
   number, never silently assumed.
4. **§3 Hypotheses** — fixed *before* results, each named (`H_<name>`) so later entries can cite
   "confirmed" or "rejected" against a specific, checkable claim.
5. **§4 Feature/model specification** — exhaustive for that experiment; no post-hoc feature
   additions once results are seen.
6. **§5 Memory/model/validation design** — compute budget, determinism protocol (independent reruns
   must produce byte-identical OOF output), regression-anchor requirement where applicable.
7. **§6 Contamination safeguards** — the specific firewall checks for *that* experiment (which
   columns are label-touching, what the leakage risk is).
8. **§7–§8 Git strategy + required doc updates.**
9. **§9 Decision, confidence, next action** — closes the pre-registration; evidence sections
   (Implementation status, Evidence, Outcome, Hypothesis classification, Decision) are appended
   once results land, never rewriting the pre-registered sections above them.

The fastest way to verify a specific leaderboard number: look it up in
`results/leaderboard.json` (repo root) → follow its `kdr` field → jump to that KDR's Evidence
section using the anchors on the [Research Summary](../RESEARCH_SUMMARY.md) page.

## Where to start

- **Full mechanism attribution across the whole Kaggle ladder:** [Research Summary](../RESEARCH_SUMMARY.md)
  — start here if you want the narrative, not nine individual entries.
- **A specific experiment's exact evidence:** jump directly into `kaggle_decisions.md` via the
  anchors on the Research Summary page.
- **The production track's research record:** `docs/research/decisions.md`, DR-001–DR-015 — see
  [Track 1](../track1.md) and [Track 3](../track3.md) for the results those entries produced.
- **Git workflow used to land this research on `main`:** `docs/research/git_workflow.md`.
