import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../lib/usePrefersReducedMotion";

interface CountUpNumberProps {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  durationMs?: number;
}

const formatValue = (n: number, decimals: number, prefix: string, suffix: string) =>
  `${prefix}${n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;

/**
 * Animates a number up once, the first time it scrolls into view. Renders the final value
 * immediately (no animation) under prefers-reduced-motion, or before intersection support is
 * confirmed, so there is never a "flash of zero" for assistive tech or SSR-adjacent tooling.
 *
 * `value` commonly starts at a placeholder (0) while a page's async data load is in flight, then
 * changes once to the real number. That change can land before OR after this component has
 * scrolled into view, so a `latestValue` ref (not a closure) is what the reveal-time animation
 * reads -- otherwise a value update that arrives while off-screen would be silently discarded by
 * a stale closure the moment the user actually scrolls to it.
 */
export default function CountUpNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  durationMs = 900,
}: CountUpNumberProps) {
  const reducedMotion = usePrefersReducedMotion();
  const ref = useRef<HTMLSpanElement>(null);
  const [display, setDisplay] = useState(value);
  const hasRevealed = useRef(false);
  const latestValue = useRef(value);
  const displayRef = useRef(value);
  const rafRef = useRef<number | null>(null);

  latestValue.current = value;

  const animateTo = (target: number) => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const from = displayRef.current;
    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      const next = from + (target - from) * eased;
      displayRef.current = next;
      setDisplay(next);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
      else {
        displayRef.current = target;
        setDisplay(target);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  // Reveal-on-scroll: fires once, reading whatever value is current at that moment (never a
  // value frozen at mount time).
  useEffect(() => {
    if (reducedMotion) {
      hasRevealed.current = true;
      setDisplay(latestValue.current);
      displayRef.current = latestValue.current;
      return;
    }
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      hasRevealed.current = true;
      setDisplay(latestValue.current);
      displayRef.current = latestValue.current;
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasRevealed.current) {
          hasRevealed.current = true;
          animateTo(latestValue.current);
          observer.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(node);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reducedMotion]);

  // If `value` changes after the reveal already fired, animate to the new value immediately --
  // this is what a placeholder-then-real-data update after the card is already on screen needs.
  useEffect(() => {
    if (!hasRevealed.current) return;
    if (reducedMotion) {
      setDisplay(value);
      displayRef.current = value;
    } else {
      animateTo(value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  useEffect(() => () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); }, []);

  return (
    <span ref={ref} className="stat-value">
      {formatValue(display, decimals, prefix, suffix)}
    </span>
  );
}
