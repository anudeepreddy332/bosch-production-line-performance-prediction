import { useId, useState } from "react";

interface MetricReadoutProps {
  value: string;
  label: string;
  definition: string;
}

/**
 * A small stat with a plain-English definition available on hover or keyboard focus (never
 * hover-only) -- the definitions live as real text nodes, not tooltip-only markup, so they're
 * readable with CSS disabled and indexable by assistive tech.
 */
export default function MetricReadout({ value, label, definition }: MetricReadoutProps) {
  const [visible, setVisible] = useState(false);
  const id = useId();

  return (
    <div
      className="metric-readout"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      <span className="metric-readout-value">{value}</span>
      <button
        type="button"
        className="metric-readout-label"
        aria-describedby={id}
        aria-expanded={visible}
      >
        {label}
      </button>
      <span role="tooltip" id={id} className={visible ? "metric-readout-def visible" : "metric-readout-def"}>
        {definition}
      </span>
    </div>
  );
}
