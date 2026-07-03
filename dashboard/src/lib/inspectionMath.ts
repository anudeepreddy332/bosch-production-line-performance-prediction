import type { SweepPoint } from "./types";

/** Nearest precomputed grid point to an arbitrary slider value -- same approach the v1 Decision
 * Explorer used: no interpolation, always read a real row from sweep_<model>.json. */
export function nearestSweepPoint(sweep: SweepPoint[], threshold: number): SweepPoint {
  let best = sweep[0];
  let bestDist = Math.abs(best.threshold - threshold);
  for (const p of sweep) {
    const d = Math.abs(p.threshold - threshold);
    if (d < bestDist) {
      best = p;
      bestDist = d;
    }
  }
  return best;
}

export interface DatasetTotals {
  totalDefects: number; // tp + fn -- invariant across every threshold for a given model
  totalHealthy: number; // fp + tn -- invariant across every threshold for a given model
  total: number;
}

/** tp+fn (real positives) and fp+tn (real negatives) are constant across the whole sweep for a
 * model -- they describe the dataset, not the threshold. Reading them off any single point
 * (here, the first) is exact, not an approximation. */
export function datasetTotals(sweep: SweepPoint[]): DatasetTotals {
  const p = sweep[0];
  const totalDefects = p.tp + p.fn;
  const totalHealthy = p.fp + p.tn;
  return { totalDefects, totalHealthy, total: totalDefects + totalHealthy };
}

export interface MinCostResult {
  point: SweepPoint;
  cost: number;
}

export function findMinCostPoint(sweep: SweepPoint[], costFn: number, costFp: number): MinCostResult {
  let best = sweep[0];
  let bestCost = Infinity;
  for (const p of sweep) {
    const cost = p.fn * costFn + p.fp * costFp;
    if (cost < bestCost) {
      bestCost = cost;
      best = p;
    }
  }
  return { point: best, cost: bestCost };
}

export type Fate = "caught" | "missed" | "alarm" | "pass";

/** The four fates, honest by construction: p(caught)+p(missed)+p(alarm)+p(pass) sum to the exact
 * tp/total, fn/total, fp/total, tn/total shares of the current sweep point -- this is the same
 * math the OutcomeLedger reads, just expressed as shares instead of counts. Used both to drive
 * PartField's illustrative particle stream (probabilistically) and WaffleFallback's exact
 * rounded 100-cell grid (deterministically). */
export function fateShares(point: SweepPoint, totals: DatasetTotals): Record<Fate, number> {
  return {
    caught: point.tp / totals.total,
    missed: point.fn / totals.total,
    alarm: point.fp / totals.total,
    pass: point.tn / totals.total,
  };
}

/** Largest-remainder rounding of the four shares onto exactly 100 integer cells, so a 10x10
 * waffle grid's cell counts always sum to 100 with no drift -- never let 4 independently-rounded
 * percentages silently sum to 99 or 101. */
export function waffleCellCounts(shares: Record<Fate, number>): Record<Fate, number> {
  const order: Fate[] = ["caught", "missed", "alarm", "pass"];
  const raw = order.map((fate) => shares[fate] * 100);
  const floors = raw.map(Math.floor);
  let remainder = 100 - floors.reduce((a, b) => a + b, 0);
  const fractionalOrder = order
    .map((fate, i) => ({ fate, frac: raw[i] - floors[i], i }))
    .sort((a, b) => b.frac - a.frac);
  const counts: Record<Fate, number> = { caught: floors[0], missed: floors[1], alarm: floors[2], pass: floors[3] };
  for (const { fate } of fractionalOrder) {
    if (remainder <= 0) break;
    counts[fate] += 1;
    remainder -= 1;
  }
  return counts;
}

/** Given a part's fixed true label (assigned once, at spawn, from the dataset's real
 * prevalence), decide whether the CURRENT threshold would flag it -- this is what lets parts
 * already mid-flight honestly "re-resolve" when the dial moves before they reach the gate: they
 * pick up whatever recall/fpr is current at the moment they cross it, not a value frozen at
 * spawn time. In expectation this reproduces the exact tp/fp/fn/tn shares. */
export function resolveFateAtGate(label: "defect" | "healthy", point: SweepPoint, rng: () => number): Fate {
  if (label === "defect") {
    return rng() < point.recall ? "caught" : "missed";
  }
  return rng() < point.fpr ? "alarm" : "pass";
}
