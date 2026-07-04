import { useMemo } from "react";
import { useRevealOnScroll } from "../../lib/useRevealOnScroll";

const TOTAL_DOTS = 1000;
// 0.58% is the project's well-established, project-wide-cited failure rate (README,
// SYSTEM_OVERVIEW, case study all quote it) -- not a number invented for this visual.
const FAILURE_RATE = 0.0058;

/** A deterministic (not Math.random(), so it's stable across re-renders) pseudo-random spread
 * of `count` positions among `total` slots, via a fixed-seed linear congruential generator. */
function seededPositions(total: number, count: number, seed = 7): Set<number> {
  let state = seed;
  const rand = () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  };
  const positions = new Set<number>();
  while (positions.size < count) {
    positions.add(Math.floor(rand() * total));
  }
  return positions;
}

/**
 * One glance instead of a paragraph: 1,000 dots, ~6 red. The rarity that makes this problem hard
 * is felt, not explained. Dots pulse once on reveal so the eye finds the red ones, then settle.
 */
export default function RarityField() {
  const { ref, revealed } = useRevealOnScroll<HTMLDivElement>();
  const failCount = Math.round(TOTAL_DOTS * FAILURE_RATE);
  const failPositions = useMemo(() => seededPositions(TOTAL_DOTS, failCount), [failCount]);

  return (
    <div ref={ref} className={revealed ? "rarity-field revealed" : "rarity-field"}>
      <div className="rarity-grid" role="img" aria-hidden="true">
        {Array.from({ length: TOTAL_DOTS }, (_, i) => (
          <span key={i} className={failPositions.has(i) ? "rarity-dot rarity-dot-fail" : "rarity-dot"} />
        ))}
      </div>
      <p className="rarity-caption">
        {failCount} in {TOTAL_DOTS.toLocaleString()}. Miss them and they ship.
      </p>
      <p className="visually-hidden">
        {failCount} out of every {TOTAL_DOTS.toLocaleString()} parts fail ({(FAILURE_RATE * 100).toFixed(2)}%).
        The other {(TOTAL_DOTS - failCount).toLocaleString()} are healthy.
      </p>
    </div>
  );
}
