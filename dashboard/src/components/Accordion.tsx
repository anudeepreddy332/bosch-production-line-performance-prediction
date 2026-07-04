import type { ReactNode } from "react";

interface AccordionProps {
  summary: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
  /** Uncontrolled initial state. Ignored if `open`/`onToggle` are both provided. */
  defaultOpen?: boolean;
  /** Controlled open state -- pass alongside onToggle to drive this accordion externally. */
  open?: boolean;
  onToggle?: (open: boolean) => void;
  id?: string;
}

/** Thin wrapper around the native <details>/<summary> pair -- free keyboard support, no JS
 * required for basic operation, and a text-node fallback with no styling if CSS fails to load. */
export default function Accordion({
  summary,
  hint,
  children,
  defaultOpen = false,
  open,
  onToggle,
  id,
}: AccordionProps) {
  const isControlled = open !== undefined;

  return (
    <details
      className="accordion"
      id={id}
      {...(isControlled ? { open } : { open: defaultOpen })}
      onToggle={onToggle ? (e) => onToggle((e.target as HTMLDetailsElement).open) : undefined}
    >
      <summary>
        {summary}
        {hint && <span className="summary-hint">{hint}</span>}
      </summary>
      <div className="accordion-body">{children}</div>
    </details>
  );
}
