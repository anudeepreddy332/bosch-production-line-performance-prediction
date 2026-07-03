import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

/**
 * Fires once, the first time the returned ref scrolls into view -- driven entirely by
 * IntersectionObserver, never a per-frame scroll listener. Under prefers-reduced-motion, the
 * "revealed" flag is true immediately so components render their fully-settled static state
 * with no animation to disable piecemeal.
 */
export function useRevealOnScroll<T extends HTMLElement>(threshold = 0.35) {
  const ref = useRef<T>(null);
  const reducedMotion = usePrefersReducedMotion();
  const [revealed, setRevealed] = useState(reducedMotion);

  useEffect(() => {
    if (reducedMotion) {
      setRevealed(true);
      return;
    }
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setRevealed(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setRevealed(true);
          observer.disconnect();
        }
      },
      { threshold },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reducedMotion, threshold]);

  return { ref, revealed };
}
