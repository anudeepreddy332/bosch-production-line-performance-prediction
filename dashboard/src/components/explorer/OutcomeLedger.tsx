import StatCard from "../StatCard";
import MetricReadout from "./MetricReadout";
import type { SweepPoint } from "../../lib/types";

interface OutcomeLedgerProps {
  point: SweepPoint;
}

/**
 * The exact numbers -- straight off the current sweep point, no rounding beyond display
 * precision. This is the source of truth on the page; the Inspection Line's flowing parts are
 * illustrative of these same four shares, never the other way around.
 */
export default function OutcomeLedger({ point }: OutcomeLedgerProps) {
  const total = point.tp + point.fp + point.fn + point.tn;
  const liveKey = `${point.threshold}`;

  return (
    <>
      <div className="card-grid card-grid-tight" key={liveKey}>
        <StatCard value={point.tp} label="Caught -- flagged and really failing" tone="good" live />
        <StatCard value={point.fn} label="Missed -- shipped as good, actually bad" tone="danger" live />
        <StatCard value={point.fp} label="False alarm -- healthy, pulled anyway" tone="warn" live />
        <StatCard value={point.tn} label="Passed clean -- healthy, left alone" live />
      </div>

      <div className="card-grid card-grid-tight" key={`${liveKey}-rates`}>
        <MetricReadout
          value={point.recall.toFixed(3)}
          label="Recall"
          definition="Of every real failure, the share this threshold catches. Flag everything and recall hits 1 -- which is why recall alone never proves anything."
        />
        <MetricReadout
          value={point.precision.toFixed(3)}
          label="Precision"
          definition="Of everything pulled for inspection, the share that's really defective. Low precision means inspectors start ignoring the flags."
        />
        <MetricReadout
          value={point.mcc.toFixed(3)}
          label="MCC"
          definition="One number for all four outcomes at once. +1 is perfect, 0 is random guessing -- and predicting 'no failure' for everything scores exactly 0, not the 99.4% accuracy it looks like."
        />
        <MetricReadout
          value={`${point.flagged_pct.toFixed(2)}%`}
          label="Inspection load"
          definition="The share of all parts this threshold pulls for a human to look at."
        />
      </div>

      <p className="visually-hidden" aria-live="polite">
        Threshold {point.threshold.toFixed(3)}. Caught {point.tp.toLocaleString()} of{" "}
        {(point.tp + point.fn).toLocaleString()} real failures, missing {point.fn.toLocaleString()}.{" "}
        {point.fp.toLocaleString()} false alarms out of {total.toLocaleString()} parts, {point.flagged_pct.toFixed(2)}
        % flagged for inspection.
      </p>
    </>
  );
}
