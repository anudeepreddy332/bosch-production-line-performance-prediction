import { useState } from "react";
import type { SweepPoint } from "../../lib/types";
import { findMinCostPoint } from "../../lib/inspectionMath";

interface WhatMattersDialProps {
  sweep: SweepPoint[];
  costFn: number;
  costFp: number;
  onChangeCosts: (costFn: number, costFp: number) => void;
  currentPoint: SweepPoint;
  onJumpToThreshold: (threshold: number) => void;
}

// Maps a 0..1 slider position to a cost ratio on a log scale (0.1x .. 100x), so the same control
// reaches both extremes smoothly. costFp is held at the value already set (default 5); moving
// the slider only rewrites costFn -- ratio is the only thing that determines the optimal
// threshold, matching the "only the ratio matters" framing.
const RATIO_MIN_EXP = -1; // 10^-1 = 0.1x
const RATIO_MAX_EXP = 2; // 10^2 = 100x

function ratioToSlider(ratio: number): number {
  const exp = Math.log10(ratio);
  return (exp - RATIO_MIN_EXP) / (RATIO_MAX_EXP - RATIO_MIN_EXP);
}

function sliderToRatio(slider: number): number {
  const exp = RATIO_MIN_EXP + slider * (RATIO_MAX_EXP - RATIO_MIN_EXP);
  return Math.pow(10, exp);
}

export default function WhatMattersDial({
  sweep,
  costFn,
  costFp,
  onChangeCosts,
  currentPoint,
  onJumpToThreshold,
}: WhatMattersDialProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const ratio = costFn / costFp;
  const sliderValue = Math.min(1, Math.max(0, ratioToSlider(ratio)));

  const minCost = findMinCostPoint(sweep, costFn, costFp);
  const currentCost = currentPoint.fn * costFn + currentPoint.fp * costFp;
  const atOptimum = Math.abs(currentPoint.threshold - minCost.point.threshold) < 0.0025;

  return (
    <div className="what-matters">
      <label htmlFor="what-matters-slider" className="what-matters-label">
        What matters more to you?
      </label>
      <div className="what-matters-row">
        <span className="what-matters-pole">Never waste an inspection</span>
        <input
          id="what-matters-slider"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={sliderValue}
          onChange={(e) => {
            const newRatio = sliderToRatio(Number(e.target.value));
            onChangeCosts(Math.round(costFp * newRatio), costFp);
          }}
          aria-valuetext={`${ratio.toFixed(1)} times more costly to miss a failure than raise a false alarm`}
        />
        <span className="what-matters-pole">Catch every failure</span>
      </div>

      <div className="callout callout-live" aria-live="polite">
        <p>
          At this balance ({ratio.toFixed(1)}x), the cheapest threshold is{" "}
          <strong>{minCost.point.threshold.toFixed(3)}</strong> (cost {minCost.cost.toLocaleString()}).{" "}
          {atOptimum ? (
            "You're there."
          ) : (
            <>
              Right now you're paying{" "}
              <strong>{(currentCost - minCost.cost).toLocaleString()}</strong> above that.{" "}
              <button type="button" onClick={() => onJumpToThreshold(minCost.point.threshold)}>
                Jump to it
              </button>
            </>
          )}
        </p>
      </div>

      <details className="accordion" open={advancedOpen} onToggle={(e) => setAdvancedOpen((e.target as HTMLDetailsElement).open)}>
        <summary>
          Advanced: exact cost weights
          <span className="summary-hint">edit the raw numbers</span>
        </summary>
        <div className="accordion-body">
          <p className="chart-hint">
            Only the ratio between these two matters -- the units are relative. Defaults (100 vs
            5 = 20x) match <code>CostConfig</code> in <code>src/evaluation/decision_system.py</code>.
          </p>
          <div className="controls-row">
            <label>
              Cost per missed failure:{" "}
              <input
                type="number"
                min={0}
                value={costFn}
                onChange={(e) => onChangeCosts(Number(e.target.value), costFp)}
                style={{ width: "5rem" }}
              />
            </label>
            <label>
              Cost per false alarm:{" "}
              <input
                type="number"
                min={0}
                value={costFp}
                onChange={(e) => onChangeCosts(costFn, Number(e.target.value))}
                style={{ width: "5rem" }}
              />
            </label>
          </div>
        </div>
      </details>
    </div>
  );
}
