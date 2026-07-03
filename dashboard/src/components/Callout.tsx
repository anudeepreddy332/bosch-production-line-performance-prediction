import type { ReactNode } from "react";

interface CalloutProps {
  children: ReactNode;
  /** Marks a callout whose content updates in response to user interaction (e.g. a slider) --
   * pairs with aria-live so assistive tech announces changes. */
  live?: boolean;
  tone?: "default" | "warn";
}

export default function Callout({ children, live = false, tone = "default" }: CalloutProps) {
  const className = ["callout", live ? "callout-live" : "", tone === "warn" ? "note-warn" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={className} aria-live={live ? "polite" : undefined}>
      {children}
    </div>
  );
}
