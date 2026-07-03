# Dashboard UX v2 — Implementation Contract

**For:** the engineer implementing this (Sonnet). **This document is the single source of truth for
the v2 dashboard.** It supersedes the earlier PF8 "polish" direction (PR #7). Do not preserve any v1
decision because it exists — rebuild against this contract.

Implement exactly what is written here. Where this document says "keep," keep verbatim. Where it says
"rewrite," follow the spec. **Do not make new product or UX decisions.** If something is
genuinely ambiguous, stop and ask; do not improvise.

---

## 0. Frozen — do not touch

Presentation only. These stay byte-for-byte identical:

- `dashboard/public/data/*.json` (the whole bundle) and `scripts/ops/export_dashboard_data.py`.
- All metrics, models, governance records, the pipeline, CI, reproducibility.
- Routes: `/`, `/decision-explorer`, `/model-internals`, `/governance`.
- `src/lib/data.ts`, `src/lib/types.ts`, `src/lib/plotly.ts`, `src/lib/plotly-basic-dist.d.ts`,
  `src/lib/slug.ts` — reuse as-is (add to `types.ts` only if a new *view-model* type is needed;
  never change a shape that mirrors a JSON file).

**Honesty gate (blocks the PR):** no visual may make a number look better than the evidence.
Contaminated results stay marked. The low deployable MCC stays visible and framed, never softened.
Any element whose *individual* pieces are illustrative must carry the exact label specified below;
its *totals* must equal the real JSON values.

**Budget gate:** ≤1.5 MB gzip JS+CSS (enforced in CI). No new animation/charting dependency — build
motion with CSS/SVG/Canvas and the existing Plotly chunk. Keep per-page `React.lazy` code-splitting.

---

## 1. Design-system foundation (Phase 1 — build before any page)

Today every page inlines its own markup against class names in `styles.css`. v2 introduces a small
shared component layer under `src/components/` and a token layer in CSS. Build this first; it unblocks
every page.

### 1.1 CSS tokens (extend `styles.css`, keep existing custom-properties)

Add/confirm: a real type scale (`--fs-display`, `--fs-h1`, `--fs-h2`, `--fs-body`, `--fs-caption`)
with large jumps; spacing scale (`--space-1..8`); motion tokens (`--motion-fast: 120ms`,
`--motion-med: 320ms`, `--ease-standard`, `--ease-physics`); elevation (`--elev-1`, `--elev-2`).
Keep navy `#1e3a5f` and teal (`#2f9e8f` / `#146a58`). Colour **semantics are a teaching tool and must
be global constants**: green = caught/honest/good, red = missed/defective/danger, amber =
false-alarm/wasted, neutral = healthy/passed.

### 1.2 New shared components (`src/components/`)

| Component | Purpose | Complexity |
|---|---|---|
| `SectionHeading` | eyebrow + H2 + optional one-line intro | S |
| `StatCard` | value + label, tone variants (good/warn/danger), optional count-up | S |
| `InsightCard` | icon + title + ≤2-sentence body | S |
| `Callout` | short highlighted statement, `live` variant | S |
| `Accordion` | native `<details>` wrapper: summary + hint + body; controlled-open prop | S |
| `CountUpNumber` | animates a number once on scroll-into-view; static under reduced-motion | M |
| `ProvenancePopover` | wraps a value; on hover/focus shows source→KDR→tag→commit chain | M |
| `usePrefersReducedMotion` (hook, `src/lib/`) | boolean; every animated component branches on it | S |

### 1.3 Page-specific hero components

| Component | Page | Complexity |
|---|---|---|
| `GapHero` | Story | M |
| `RarityField` | Story | M |
| `ResearchLadder` | Story | L |
| `PipelineFlow` | Story | M |
| `StackVerdict` | Model Internals | M |
| `InspectionLine` (+ children, §4) | Decision Explorer | XL |

### 1.4 Components to remove

The v1 page bodies are rewritten, so their inline blocks disappear. Nothing in `src/lib/` is removed.
No standalone v1 component files exist to delete (v1 inlined everything) — deletion happens as part of
each page rewrite.

---

## 2. Language & copy rules (enforced in review)

First person, solo engineer, spare and concrete. Every sentence must justify itself. No paragraph
over two sentences outside an opened accordion.

**Banned (auto-reject):** "we tested / we discovered / we built," "the methodology," "the framework,"
"the approach," "leverages," "utilizes," "in order to," "it is worth noting," "seamless," "robust"
(as filler), "cutting-edge," "delve," and anything that reads like a paper abstract.

Calibration:
- ✗ "The system utilizes a chunk-aware methodology to ensure leakage-safe evaluation."
  → ✓ "No time-chunk sits in both train and test. A test proves the guard fires."
- ✗ "We intentionally prioritized leakage-safe decisioning."
  → ✓ "I left the leaky features out. The research track shows what that cost."

---

## 3. Per-page contracts

Each page answers the 14 required points. "Reuse" = existing `lib` + Phase-1 components.

### 3.1 Story — *the 5-second hook*

Emotional goal: intrigue → confidence. Understood in under 30 seconds without scrolling.

1. **Stays unchanged:** data loads (`loadRp2Summary`, `loadModelsMeta`, `loadGovernance`); every
   number still sourced from JSON.
2. **Deleted completely:** the lede paragraph; the per-page export-provenance note; two of the three
   "why it's hard" text cards; the list-style research timeline (replaced by a chart).
3. **Rewritten:** hero → `GapHero`; "why hard" → `RarityField` + ≤3 labels; "what I built" →
   `PipelineFlow`; research ladder → `ResearchLadder`.
4. **Simplified:** "what honestly ships" keeps 3 `StatCard`s + one sentence (detail → accordion);
   "decisions I'd defend" → 3 one-line teasers, full text in an `Accordion`.
5. **Visual instead of textual:** rarity (dot field), the climb (drawn chart), the pipeline (staged
   diagram). Replace ~4 paragraphs with 3 visuals.
6. **Interactive:** `GapHero` gap animates apart on load; `CountUpNumber` on the honest stats;
   `ResearchLadder` draws on scroll-in.
7. **Accordions:** rolling-origin detail table (already), "decisions I'd defend" full text.
8. **Hero treatment:** `GapHero` owns the first screen alone — two numbers (0.06–0.18 vs the
   `governance.json` best private MCC), the gap between them drawn.
9. **New components:** `GapHero`, `RarityField`, `PipelineFlow`, `ResearchLadder`.
10. **Reuse:** `SectionHeading`, `StatCard`, `Accordion`, `CountUpNumber`, CTA cards.
11. **Remove:** v1 Story inline card grids and timeline markup.
12. **A11y:** `RarityField`/`ResearchLadder` need text-equivalent summaries; count-up respects
    reduced-motion (renders final value); heading order H1→H2 only.
13. **Complexity:** L (ladder is the hard part).
14. **Risks:** scroll-triggered draw can jank on mobile — use `IntersectionObserver` + CSS/SVG, not
    JS-per-frame; ensure the ladder is fully legible as a static image under reduced-motion.

`ResearchLadder` spec: step/line chart of private MCC K1→P1 from `governance.json.experiments`, drawn
left→right on scroll-in. Honest points solid, contaminated points hollow/ringed (teaches which gains
were real). Annotate only P0 ("raw signal — the biggest jump") and K3-B ("a rejected idea — it fell
below the baseline"). Static fully-drawn fallback under reduced-motion.

### 3.2 Decision Explorer — *the centerpiece* (full spec in §4)

1. **Stays unchanged:** `loadModelsMeta`, `loadSweep`; PR/ROC Plotly charts (demoted, not changed);
   exact per-threshold values from `sweep_<model>.json`.
2. **Deleted completely:** the raise/lower comparison block; the bare TP/FP/FN/TN `<table>`; the
   "no free lunch" text callout as a standalone paragraph.
3. **Rewritten:** the whole primary surface → `InspectionLine`. The threshold slider becomes the
   line's dial. The cost inputs become the "what matters to you?" control.
4. **Simplified:** intro to one line; one dynamic consequence sentence retained (as a readout).
5. **Visual instead of textual:** the four outcomes become regions of the picture with live counts;
   precision/recall/MCC become small readouts beside the picture.
6. **Interactive:** the dial drives the line in real time; model chips swap the "brain"; the
   what-matters dial marks the cost-optimal gate.
7. **Accordions:** the metric glossary (keep, opened contextually from "what's precision?" links);
   the formal PR/ROC charts live below the fold (a section, not necessarily an accordion).
8. **Hero treatment:** `InspectionLine` is the page and the portfolio centerpiece.
9. **New components:** `InspectionLine`, `Dial`, `PartField`, `OutcomeLedger`, `WhatMattersDial`,
   `WaffleFallback`, `MetricReadout`.
10. **Reuse:** `Accordion` (glossary), `Callout` (the one live line), Plotly wrapper (formal charts),
    `usePrefersReducedMotion`.
11. **Remove:** v1 DecisionExplorer stat-card grid, confusion table, comparison block markup.
12. **A11y:** see §4 — keyboard dial, `aria-live` outcome summary, text equivalents for every visual
    state.
13. **Complexity:** XL.
14. **Risks:** honesty of the illustrative flow (must match totals exactly + label it); performance
    (canvas/transform, not React-per-frame); mobile layout of a horizontal flow; reduced-motion must
    teach equally well.

### 3.3 Model Internals — *the honesty flex*

1. **Stays unchanged:** `loadModelsMeta`, `loadImportances`, `loadCalibration`; all three Plotly
   charts (importances, fold spread, calibration) unchanged in data.
2. **Deleted completely:** nothing of substance — this page is re-ranked, not gutted. Delete only
   redundant captions.
3. **Rewritten:** the "why dataset_h ships" callout → `StackVerdict` hero visual at the top.
4. **Simplified:** intro to one line; fingerprint explanation → tooltip on the value, not a full
   label; each chart gets exactly one takeaway line.
5. **Visual instead of textual:** `StackVerdict` = a small comparison graphic (meta vs best base,
   loser marked) replacing the paragraph.
6. **Interactive:** model selector (keep); tooltips on fingerprint and family swatches.
7. **Accordions:** feature-family definitions (keep); fold-level MCC spread → accordion with a
   one-line summary visible.
8. **Hero treatment:** `StackVerdict` — "I stacked four models. The stack scored worse. I shipped the
   simpler one." with the two real numbers from `models.json`.
9. **New components:** `StackVerdict`.
10. **Reuse:** `SectionHeading`, `StatCard`, `Accordion`, Plotly wrapper.
11. **Remove:** v1 callout paragraph and dense captions.
12. **A11y:** `StackVerdict` needs a text equivalent; tooltips keyboard-focusable; chart titles
    remain in the Plotly layout for screen-reader context.
13. **Complexity:** M.
14. **Risks:** low. Keep the honest framing exact (meta 0.14942 < dataset_h 0.15337 — pull live, do
    not hardcode).

### 3.4 Governance — *the receipts* (progressive-disclosure spec below)

**Evidence removal is forbidden.** Every row, note, tag, and link in v1 survives verbatim. The change
is hierarchy + one interaction.

1. **Stays unchanged:** `loadGovernance`, `loadRepoLinks`; the 9-row ladder content; KDR list; notes;
   reproducibility text; tag grid; client-side GitHub anchor slugs (`githubHeadingSlug`).
2. **Deleted completely:** nothing. (Only the *instructional* "verify in 60 seconds" paragraph is
   replaced by the actual interaction — the words go, the capability becomes real.)
3. **Rewritten:** the verify-path paragraph → `ProvenancePopover` on the headline numbers.
4. **Simplified:** intro → one confident claim line.
5. **Visual instead of textual:** summary `StatCard`s stay as the 60-second read.
6. **Interactive:** provenance-on-hover/focus.
7. **Accordions (this is the core move):** see below.
8. **Hero treatment:** the claim + the four summary stats + the provenance interaction.
9. **New components:** none page-specific beyond reusing `ProvenancePopover`, `Accordion`.
10. **Reuse:** `Accordion`, `StatCard`, `Callout`, `ProvenancePopover`, `githubHeadingSlug`.
11. **Remove:** v1 always-open table markup (content moves into an accordion, unchanged).
12. **A11y:** every accordion keyboard-operable (native `<details>`); popover focusable and
    dismissible; contrast AA on badges.
13. **Complexity:** M.
14. **Risks:** must not drop a single evidence row in the reshuffle — diff the rendered text against
    v1 to confirm parity.

**Progressive-disclosure order (top → bottom):**
1. Claim line + 4 summary `StatCard`s (always visible).
2. Frozen-program banner (always visible).
3. `Accordion` "The full experiment ladder" — **open by default**, table verbatim.
4. `Accordion` "Decision record log (KDR-001 – KDR-009)" — closed.
5. `Accordion` "Metric definitions & contamination rules" — closed.
6. `Accordion` "Reproducibility" — closed.
7. `Accordion` "Repository & evidence tags" — closed.

`ProvenancePopover` chain per number: `source_of_truth` file → the row's `kdr` (link via
`githubHeadingSlug`) → `git_tag` (link to `/tree/<tag>`) → commit is implied by the tag. Use only
fields already in `governance.json` / `repo_links.json`.

---

## 4. The Inspection Line — complete specification

The centerpiece. A visitor with zero ML knowledge must understand the precision/recall trade-off by
playing, before reading any definition. Think interaction designer, not ML engineer.

### 4.1 Concept & honesty model

Parts stream toward an inspection **gate** whose height is the threshold (the **Dial**). Parts the
model would flag get **pulled** into an inspection bin; the rest **pass**. Defects are red and rare;
healthy parts neutral. Four fates, global colours: caught defect (red, pulled, green glow), missed
defect (red, passed → "ships", sober red flash), false alarm (neutral, pulled, amber), healthy pass
(neutral, passed, quiet).

**Honesty (mandatory).** `sweep_<model>.json` gives exact `tp/fp/fn/tn/precision/recall/mcc/
flagged_pct` per threshold for the full dataset — but no per-part scores. Therefore:
- The **Dial sets the real threshold.** The **share of parts pulled** and the **four outcome counts
  and rates** shown in the `OutcomeLedger` equal the real sweep values exactly.
- The streaming parts are **illustrative of flow and proportion only** — individual dots are not real
  parts. Assign each visible part a fate probabilistically so the on-screen mix matches the current
  confusion proportions; do not position parts by a fabricated score.
- Persistent label near the visual: **"Illustrative flow — the four counts and rates are exact."**
- The `OutcomeLedger` (exact TP/FP/FN/TN + precision/recall/MCC/inspection-load, straight from the
  sweep point) is the source of truth on the page.

### 4.2 Layout

Desktop (≥900px): full-width stage. Left→right flowing lane occupies the top ~60% of the stage; the
inspection **bin** collects pulled parts (top edge) and the **"ships" exit** is the right edge for
missed defects. The `Dial` sits at the gate (vertical), draggable. Below the stage: the
`OutcomeLedger` (four exact counts + rates) and `MetricReadout`s (precision/recall/MCC/load). Model
chips above the stage. `WhatMattersDial` beside the ledger. Formal PR/ROC charts far below the fold.

### 4.3 Animations

- Parts drift across the lane at a calm constant speed (`--ease-physics`, no bounce). Pool of reused
  DOM/canvas nodes (cap ~120 concurrent visible); recycle, do not mount-per-part.
- On dial change: the gate height eases to the new position; parts already in flight re-resolve fate
  to keep proportions matching; the bin fill height eases; the ledger numbers tween to the new exact
  values.
- Caught defect: brief green ring as it enters the bin. Missed defect: sober red flash at the ships
  exit (never celebratory). False alarm: amber tint as it enters the bin.
- Nothing loops idly beyond the ambient flow; the flow itself is the only continuous motion and pauses
  when the tab is hidden (`visibilitychange`).

### 4.4 Interactions

- **Dial (threshold):** drag vertically (pointer) or focus + arrow keys (±0.005, Home/End to
  extremes). Exact value always shown. "Reset to tuned" affordance returns to
  `models[model].best_threshold`.
- **Model chips:** switch model → reload sweep, reset dial to that model's tuned threshold, re-seed
  the pool. Framed as swapping the line's "brain."
- **WhatMattersDial (cost model):** a single slider from "catch every failure" ↔ "never waste an
  inspection." Maps to the FN:FP cost ratio; marks the cost-optimal gate on the lane and shows the
  exact total cost + the min-cost threshold (from the sweep) with a "jump to it" action. Keep an
  "advanced: exact cost weights" disclosure for the two numeric inputs (current behaviour preserved
  for technical users).

### 4.5 Micro-interactions

Ghosted one-time nudge on first load: "Drag the dial. Try to catch every red part." Dismisses on first
interaction (persist in `sessionStorage`). Hovering a fate region highlights matching parts + shows a
one-line plain-English definition ("Precision — of everything pulled, how much was really defective").
Bin "settles" as parts land.

### 4.6 Hover behaviour

Hover/focus any `MetricReadout` → short definition + which region of the picture it measures. Hover a
part → its fate label. No hover-only information without a focus/tap equivalent.

### 4.7 Transitions

Dial→ledger updates are tweened (`--motion-med`), never instant-jump, so causality reads. Model swap
cross-fades the lane (`--motion-med`).

### 4.8 Responsive

- ≥900px: horizontal lane as above.
- 600–900px: shorter lane, ledger stacks under the stage, chips wrap.
- <600px (mobile): **the `WaffleFallback` becomes primary** (a 10×10 "per-100 parts" grid that
  recolours live to exact rounded proportions as the dial moves) — a horizontal particle stream does
  not fit a phone. The dial becomes a bottom-anchored horizontal control; the ledger sits above it.
  Same honesty label, same exact ledger.

### 4.9 Reduced-motion fallback

Under `prefers-reduced-motion`: no flow. Render `WaffleFallback` (10×10 grid = per-100 parts) that
recolours to the exact proportions at the current threshold; dial changes recolour instantly (no
tween). Same teaching (you see the mix shift), zero motion. This is also the mobile primary — build it
once, use it in both cases.

### 4.10 Accessibility & keyboard

- The dial is a real `<input type="range">` (or ARIA slider) — full keyboard control, visible focus,
  `aria-valuenow/min/max`.
- An `aria-live="polite"` summary announces the state in words on each settled change: e.g. "Threshold
  0.91. Caught 811 of 6,879 failures. 3,066 false alarms. 811 of 3,877 flagged parts were real."
- Every colour-coded fate also carries a text label in the ledger; never rely on colour alone
  (colour-blind safe: pair red/green/amber with distinct shapes or labels).
- The honesty label and every definition are real text, not tooltip-only.
- `WaffleFallback` cells have an off-screen text summary of the proportions.

### 4.11 Mobile behaviour (summary)

Waffle-primary, bottom-anchored dial, stacked ledger, chips wrap, formal charts collapsed into an
accordion. Touch drag on the dial; tap a fate region for its definition.

### 4.12 Performance

Canvas or CSS-transform pool, capped node count, `requestAnimationFrame` loop that pauses on hidden
tab and on reduced-motion. No React state update per frame — drive the animation imperatively, sync
React only on settled dial changes. No new dependency.

---

## 5. Component architecture (target tree)

```
src/
  components/
    SectionHeading.tsx  StatCard.tsx  InsightCard.tsx  Callout.tsx  Accordion.tsx
    CountUpNumber.tsx   ProvenancePopover.tsx
    story/     GapHero.tsx  RarityField.tsx  PipelineFlow.tsx  ResearchLadder.tsx
    internals/ StackVerdict.tsx
    explorer/  InspectionLine.tsx  Dial.tsx  PartField.tsx  OutcomeLedger.tsx
               WhatMattersDial.tsx  WaffleFallback.tsx  MetricReadout.tsx
  lib/  usePrefersReducedMotion.ts  (+ existing data.ts, types.ts, plotly.ts, slug.ts unchanged)
  pages/ Story.tsx  DecisionExplorer.tsx  ModelInternals.tsx  Governance.tsx  (rewritten bodies)
  App.tsx  (nav/footer: keep v1 footer links; no structural change)
  styles.css  (extend with tokens; keep existing classes still in use)
```

---

## 6. Implementation roadmap (phases)

Each phase is one PR-reviewable unit and one logical commit. The site stays shippable after every
phase (pages not yet migrated keep working on the shared components).

**Phase 0 — Blueprint commit (this document).**
- Objective: land the contract before code. Files: `docs/design/dashboard_ux_blueprint_v2.md`.
- Complexity: trivial. Effort: done. Validation: doc committed; `mkdocs build --strict` still clean.
- Commit: `PF8 docs: v2 dashboard UX implementation contract`. Stop: after commit.

**Phase 1 — Tokens + shared component library.**
- Objective: CSS tokens, `SectionHeading/StatCard/InsightCard/Callout/Accordion/CountUpNumber/
  ProvenancePopover`, `usePrefersReducedMotion`. No page rewrites yet.
- Files: `styles.css`, `src/components/*`, `src/lib/usePrefersReducedMotion.ts`.
- Complexity: M. Effort: ~0.5–1 day.
- Validation: `tsc` + build clean; bundle within budget; components render in isolation; reduced-motion
  hook verified.
- Commit: `PF8 feat: v2 design tokens + shared component library`. Stop: after build passes.

**Phase 2 — Decision Explorer / Inspection Line (centerpiece; do first, it's riskiest).**
- Objective: full §4 spec. Files: `pages/DecisionExplorer.tsx`, `src/components/explorer/*`.
- Complexity: XL. Effort: ~1.5–2 days.
- Validation: outcome ledger equals `sweep_<model>.json` exactly at ≥3 spot thresholds; honesty label
  present; keyboard dial + `aria-live` work; reduced-motion → waffle; mobile → waffle; no per-frame
  React; Playwright: 0 console/network errors, dial changes ledger, glossary opens; bundle within
  budget.
- Commit: `PF8 feat: Decision Explorer -- Inspection Line (honest, playable)`. Stop: after Playwright
  + honesty checks pass.

**Phase 3 — Story.**
- Objective: §3.1. Files: `pages/Story.tsx`, `src/components/story/*`.
- Complexity: L. Effort: ~1 day.
- Validation: first screen = `GapHero` alone; ladder draws on scroll and is legible static under
  reduced-motion; all numbers from JSON; Playwright clean; heading order valid.
- Commit: `PF8 feat: Story -- gap hero, rarity, self-drawing research ladder`. Stop: after checks.

**Phase 4 — Model Internals.**
- Objective: §3.3. Files: `pages/ModelInternals.tsx`, `src/components/internals/StackVerdict.tsx`.
- Complexity: M. Effort: ~0.5 day.
- Validation: `StackVerdict` numbers pulled live (0.14942 vs 0.15337); charts unchanged in data;
  accordions work; Playwright clean.
- Commit: `PF8 feat: Model Internals -- lead with the stacking-honesty verdict`. Stop: after checks.

**Phase 5 — Governance.**
- Objective: §3.4 + progressive disclosure + `ProvenancePopover`. Files: `pages/Governance.tsx`.
- Complexity: M. Effort: ~0.5–1 day.
- Validation: **rendered evidence text diffed against v1 — zero rows lost**; accordions keyboard-
  operable; popover focusable; anchors resolve (spot-check one KDR link); Playwright clean.
- Commit: `PF8 feat: Governance -- confidence on top, all evidence one click down`. Stop: after parity
  diff.

**Phase 6 — Global copy pass + cleanup + final validation.**
- Objective: enforce §2 banned-list across all pages; single `⌁ traceable` affordance replacing
  per-page provenance notes; remove dead v1 CSS classes; first-person sweep of any remaining strings.
- Files: all `pages/*`, `styles.css`, `App.tsx`.
- Complexity: S–M. Effort: ~0.5 day.
- Validation: grep for banned phrases → none; no paragraph >2 sentences outside accordions;
  `tsc`/build/`mkdocs --strict` clean; full Playwright desktop+mobile; bundle within budget;
  `git diff` empty on `dashboard/public/data/`, `scripts/`, `src/` pipeline; screenshots for CP review.
- Commit: `PF8 chore: v2 copy pass + cleanup`. Stop: **CP review** — do not merge without approval.

---

## 7. Global acceptance criteria (CP gate)

- A non-technical viewer can explain the precision/recall trade-off after ~30s on Decision Explorer
  without reading a definition.
- Each page's first screen lands one idea with minimal reading.
- Every headline number still traces to the JSON bundle; `OutcomeLedger` matches the sweep exactly;
  illustrative elements carry the exact honesty label.
- `git diff` empty on data/pipeline; bundle ≤1.5 MB gzip; `tsc`/build/`mkdocs --strict` clean;
  Playwright desktop+mobile zero console/network errors; all interactions keyboard-operable;
  reduced-motion verified to teach equally; no banned phrase survives; Governance evidence parity
  confirmed against v1.
- Nothing merged to `main` without explicit CP approval.
