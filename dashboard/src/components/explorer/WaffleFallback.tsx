import { useMemo } from "react";
import type { SweepPoint } from "../../lib/types";
import { datasetTotals, fateShares, waffleCellCounts, type Fate } from "../../lib/inspectionMath";

interface WaffleFallbackProps {
  sweep: SweepPoint[];
  point: SweepPoint;
}

const FATE_LABEL: Record<Fate, string> = {
  caught: "caught failure",
  missed: "missed failure",
  alarm: "false alarm",
  pass: "healthy, passed",
};

const FATE_CLASS: Record<Fate, string> = {
  caught: "waffle-cell-caught",
  missed: "waffle-cell-missed",
  alarm: "waffle-cell-alarm",
  pass: "waffle-cell-pass",
};

/**
 * Same honest proportions as the flowing Inspection Line, rendered with zero motion: a 10x10
 * grid recolours instantly to the exact, largest-remainder-rounded share of each of the four
 * outcomes among 100 parts. This is the primary view under prefers-reduced-motion, and on
 * screens under 600px where a horizontal particle stream doesn't fit.
 */
export default function WaffleFallback({ sweep, point }: WaffleFallbackProps) {
  const cells = useMemo(() => {
    const totals = datasetTotals(sweep);
    const shares = fateShares(point, totals);
    const counts = waffleCellCounts(shares);
    const order: Fate[] = ["caught", "missed", "alarm", "pass"];
    const flat: Fate[] = [];
    for (const fate of order) {
      for (let i = 0; i < counts[fate]; i++) flat.push(fate);
    }
    return { flat, counts };
  }, [sweep, point]);

  return (
    <div className="waffle-wrap">
      <div className="waffle-grid" role="img" aria-hidden="true">
        {cells.flat.map((fate, i) => (
          <span key={i} className={`waffle-cell ${FATE_CLASS[fate]}`} title={FATE_LABEL[fate]} />
        ))}
      </div>
      <p className="visually-hidden">
        Out of every 100 parts: {cells.counts.caught} caught failures, {cells.counts.missed} missed
        failures, {cells.counts.alarm} false alarms, {cells.counts.pass} healthy parts passed clean.
        (Exact counts at this threshold are in the ledger above; this grid rounds to whole cells.)
      </p>
    </div>
  );
}
