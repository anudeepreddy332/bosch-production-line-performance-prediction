import { useId, useState, type ReactNode } from "react";

export interface ProvenanceLink {
  label: string;
  href: string;
}

interface ProvenancePopoverProps {
  /** The value being annotated -- rendered exactly as passed, unchanged. */
  children: ReactNode;
  /** source -> KDR -> tag chain, in order. Every entry must come from data already in
   * governance.json / repo_links.json -- never invented here. */
  chain: ProvenanceLink[];
  /** "inline" wraps a bare value (a number, a short string) in a <span> -- valid HTML. "block"
   * wraps a card-shaped child (e.g. StatCard, which renders a <div>) in a <div> instead, so a
   * block-level child never ends up nested inside an inline element. */
  as?: "inline" | "block";
}

/**
 * Wraps a headline number. On hover or keyboard focus, shows the chain of evidence that backs
 * it (source file -> decision record -> git tag) so "verify this number" is a gesture, not a
 * paragraph of instructions. Dismissible via Escape or blur; works on hover AND focus so it's
 * not hover-only information.
 */
export default function ProvenancePopover({ children, chain, as = "inline" }: ProvenancePopoverProps) {
  const [visible, setVisible] = useState(false);
  const popoverId = useId();

  if (chain.length === 0) {
    return <>{children}</>;
  }

  const Wrapper = as === "block" ? "div" : "span";
  const Trigger = as === "block" ? "div" : "span";

  return (
    <Wrapper
      className={as === "block" ? "provenance-trigger provenance-trigger-block" : "provenance-trigger"}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
      onKeyDown={(e) => {
        if (e.key === "Escape") setVisible(false);
      }}
    >
      <Trigger tabIndex={0} aria-describedby={visible ? popoverId : undefined} className="provenance-value">
        {children}
      </Trigger>
      {visible && (
        <span role="tooltip" id={popoverId} className="provenance-popover">
          {chain.map((link, i) => (
            <span key={link.href} className="provenance-step">
              {i > 0 && <span className="provenance-arrow" aria-hidden="true">→</span>}
              <a href={link.href} target="_blank" rel="noreferrer" className="external-link">
                {link.label}
              </a>
            </span>
          ))}
        </span>
      )}
    </Wrapper>
  );
}
