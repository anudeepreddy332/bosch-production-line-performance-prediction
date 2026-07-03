import { useMemo } from "react";
import { useRevealOnScroll } from "../../lib/useRevealOnScroll";
import type { KaggleExperiment } from "../../lib/types";

interface ResearchLadderProps {
  experiments: KaggleExperiment[];
}

const WIDTH = 640;
const HEIGHT = 220;
const MARGIN = { top: 24, right: 16, bottom: 28, left: 16 };
const ANNOTATED: Record<string, string> = {
  P0: "raw signal — the biggest jump",
  "K3-B": "a rejected idea — fell below the baseline",
};
// Larger than any real path length at this scale -- precision doesn't matter for a draw-on
// reveal effect, only that the dash covers the whole line.
const DASH_LENGTH = 3000;

/**
 * The K1->P1 private-MCC ladder, drawn left-to-right the first time it scrolls into view (a CSS
 * stroke-dashoffset transition, not a per-frame animation loop). Honest points are filled solid;
 * contaminated points are hollow rings, so the shape of the line itself teaches which gains were
 * real. Reduced motion renders the line fully drawn immediately.
 */
export default function ResearchLadder({ experiments }: ResearchLadderProps) {
  const { ref, revealed } = useRevealOnScroll<HTMLDivElement>();

  const { points, pathD, minMcc, maxMcc } = useMemo(() => {
    const values = experiments.map((e) => e.private_mcc);
    const minMcc = Math.min(...values);
    const maxMcc = Math.max(...values);
    const n = experiments.length;
    const innerW = WIDTH - MARGIN.left - MARGIN.right;
    const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

    const points = experiments.map((e, i) => {
      const x = MARGIN.left + (n === 1 ? innerW / 2 : (i * innerW) / (n - 1));
      const t = maxMcc === minMcc ? 0.5 : (e.private_mcc - minMcc) / (maxMcc - minMcc);
      const y = MARGIN.top + innerH - t * innerH;
      return { x, y, e };
    });

    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    return { points, pathD, minMcc, maxMcc };
  }, [experiments]);

  return (
    <div ref={ref} className={revealed ? "research-ladder revealed" : "research-ladder"}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby="research-ladder-title" className="research-ladder-svg">
        <title id="research-ladder-title">
          Private MCC ladder from {experiments[0]?.experiment_id} to {experiments[experiments.length - 1]?.experiment_id}
        </title>
        <path
          d={pathD}
          fill="none"
          stroke="var(--color-primary)"
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          className="research-ladder-path"
          style={{ strokeDasharray: DASH_LENGTH, strokeDashoffset: revealed ? 0 : DASH_LENGTH }}
        />
        {points.map(({ x, y, e }) => (
          <g key={e.experiment_id}>
            <circle
              cx={x}
              cy={y}
              r={5.5}
              className="research-ladder-point"
              fill={e.oof_status === "honest" ? "var(--color-fate-caught)" : "var(--color-surface)"}
              stroke="var(--color-fate-caught)"
              strokeWidth={2}
              style={{ opacity: revealed ? 1 : 0, transitionDelay: revealed ? "500ms" : "0ms" }}
            />
            <text x={x} y={HEIGHT - 6} textAnchor="middle" className="research-ladder-label">
              {e.experiment_id}
            </text>
            {ANNOTATED[e.experiment_id] && (
              <text
                x={x}
                y={y - 12}
                textAnchor={x > WIDTH - 100 ? "end" : "middle"}
                className="research-ladder-annotation"
                style={{ opacity: revealed ? 1 : 0, transitionDelay: revealed ? "700ms" : "0ms" }}
              >
                {ANNOTATED[e.experiment_id]}
              </text>
            )}
          </g>
        ))}
      </svg>
      <p className="visually-hidden">
        Private MCC by experiment, honest unless noted contaminated:{" "}
        {experiments
          .map((e) => `${e.experiment_id} ${e.private_mcc.toFixed(3)} (${e.oof_status})`)
          .join(", ")}
        . Range {minMcc.toFixed(3)} to {maxMcc.toFixed(3)}.
      </p>
    </div>
  );
}
