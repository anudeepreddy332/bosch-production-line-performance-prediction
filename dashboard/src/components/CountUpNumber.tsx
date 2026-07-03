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
 * Animates a number up from 0 once, the first time it scrolls into view. Renders the final
 * value immediately (no animation) under prefers-reduced-motion, or before intersection support
 * is confirmed, so there is never a "flash of zero" for assistive tech or SSR-adjacent tooling.
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
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (reducedMotion) {
      setDisplay(value);
      return;
    }
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setDisplay(value);
      return;
    }

    setDisplay(0);
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const start = performance.now();
          const tick = (now: number) => {
            const progress = Math.min(1, (now - start) / durationMs);
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplay(value * eased);
            if (progress < 1) requestAnimationFrame(tick);
            else setDisplay(value);
          };
          requestAnimationFrame(tick);
          observer.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(node);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, reducedMotion, durationMs]);

  return (
    <span ref={ref} className="stat-value">
      {formatValue(display, decimals, prefix, suffix)}
    </span>
  );
}
