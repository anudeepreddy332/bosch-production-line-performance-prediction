import { useEffect, useState } from "react";
import { usePrefersReducedMotion } from "../../lib/usePrefersReducedMotion";

interface GapHeroProps {
  deployableRange: string; // e.g. "0.06-0.18", already formatted -- a range never counts up
  ceiling: number | null;
}

/**
 * The whole first screen. Two numbers start close together and spring apart on load -- the one
 * animated flourish this page spends, used exactly once. Under prefers-reduced-motion the gap
 * renders already-open (--motion-med resolves to 0ms globally, so the transition is instant
 * rather than removed piecemeal here).
 */
export default function GapHero({ deployableRange, ceiling }: GapHeroProps) {
  const reducedMotion = usePrefersReducedMotion();
  const [open, setOpen] = useState(reducedMotion);

  useEffect(() => {
    if (reducedMotion) return;
    const t = setTimeout(() => setOpen(true), 250);
    return () => clearTimeout(t);
  }, [reducedMotion]);

  return (
    <div className={open ? "gap-hero gap-hero-open" : "gap-hero"}>
      <div className="gap-hero-cell gap-hero-left">
        <span className="big-number deployable">{deployableRange}</span>
        <span className="hero-contrast-label">MCC that honestly ships — measured the way a live deployment experiences it</span>
      </div>
      <div className="gap-hero-divider">VS</div>
      <div className="gap-hero-cell gap-hero-right">
        <span className="big-number ceiling">{ceiling !== null ? ceiling.toFixed(2) : "…"}</span>
        <span className="hero-contrast-label">Leaderboard ceiling — signals a real factory stream can't have</span>
      </div>
    </div>
  );
}
