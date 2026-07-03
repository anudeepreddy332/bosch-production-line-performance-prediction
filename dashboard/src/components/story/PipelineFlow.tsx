import { useRevealOnScroll } from "../../lib/useRevealOnScroll";

interface Stage {
  label: string;
  detail: string;
}

interface PipelineFlowProps {
  stages: Stage[];
}

/** The system, as one quiet diagram: each stage fades/slides in staggered, once, the first time
 * it scrolls into view. Reduced motion renders every stage settled immediately. */
export default function PipelineFlow({ stages }: PipelineFlowProps) {
  const { ref, revealed } = useRevealOnScroll<HTMLDivElement>();

  return (
    <div ref={ref} className="flow-strip" aria-label="System pipeline">
      {stages.map((stage, i) => (
        <span key={stage.label} className="flow-step-group">
          {i > 0 && (
            <span className="flow-arrow" aria-hidden="true">
              →
            </span>
          )}
          <span
            className={revealed ? "flow-step flow-step-revealed" : "flow-step"}
            style={{ transitionDelay: revealed ? `${i * 90}ms` : "0ms" }}
          >
            {stage.label}
            <small>{stage.detail}</small>
          </span>
        </span>
      ))}
    </div>
  );
}
