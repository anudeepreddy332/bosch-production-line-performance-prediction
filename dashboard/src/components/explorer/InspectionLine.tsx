import { useEffect, useMemo, useRef, useState } from "react";
import { MODEL_KEYS, type ModelKey, type ModelsMeta, type SweepPoint } from "../../lib/types";
import { usePrefersReducedMotion } from "../../lib/usePrefersReducedMotion";
import { useMediaQuery } from "../../lib/useMediaQuery";
import { datasetTotals, nearestSweepPoint } from "../../lib/inspectionMath";
import Dial from "./Dial";
import PartField from "./PartField";
import WaffleFallback from "./WaffleFallback";
import OutcomeLedger from "./OutcomeLedger";
import WhatMattersDial from "./WhatMattersDial";

const NUDGE_DISMISSED_KEY = "inspection-line-nudge-dismissed";

interface InspectionLineProps {
  modelsMeta: ModelsMeta | null;
  model: ModelKey;
  onModelChange: (model: ModelKey) => void;
  sweep: SweepPoint[] | null;
  threshold: number;
  onThresholdChange: (t: number) => void;
  costFn: number;
  costFp: number;
  onChangeCosts: (fn: number, fp: number) => void;
}

/** The centerpiece. Fully controlled by DecisionExplorer.tsx so the demoted PR/ROC charts below
 * the fold share the exact same model/threshold state -- one control surface, not two. */
export default function InspectionLine({
  modelsMeta,
  model,
  onModelChange,
  sweep,
  threshold,
  onThresholdChange,
  costFn,
  costFp,
  onChangeCosts,
}: InspectionLineProps) {
  const reducedMotion = usePrefersReducedMotion();
  const isMobile = useMediaQuery("(max-width: 599px)");
  const useWaffle = reducedMotion || isMobile;

  const [showNudge, setShowNudge] = useState(
    () => typeof window !== "undefined" && !sessionStorage.getItem(NUDGE_DISMISSED_KEY),
  );
  const [binPulse, setBinPulse] = useState(0);
  const [shipFlash, setShipFlash] = useState(0);

  const currentPoint = useMemo(() => (sweep ? nearestSweepPoint(sweep, threshold) : null), [sweep, threshold]);
  const totals = useMemo(() => (sweep ? datasetTotals(sweep) : null), [sweep]);
  const tunedThreshold = modelsMeta ? modelsMeta[model].best_threshold : threshold;

  const dismissNudge = () => {
    if (showNudge) {
      setShowNudge(false);
      sessionStorage.setItem(NUDGE_DISMISSED_KEY, "1");
    }
  };

  const binRef = useRef<HTMLDivElement>(null);
  const shipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (binPulse === 0) return;
    const el = binRef.current;
    if (!el) return;
    el.classList.add("bin-settle");
    const t = setTimeout(() => el.classList.remove("bin-settle"), 260);
    return () => clearTimeout(t);
  }, [binPulse]);

  useEffect(() => {
    if (shipFlash === 0) return;
    const el = shipRef.current;
    if (!el) return;
    el.classList.add("ship-flash");
    const t = setTimeout(() => el.classList.remove("ship-flash"), 500);
    return () => clearTimeout(t);
  }, [shipFlash]);

  return (
    <div className="inspection-line">
      <div className="controls-row">
        <div className="model-select" role="group" aria-label="Model selection">
          {MODEL_KEYS.map((key) => (
            <button
              key={key}
              className={key === model ? "active" : ""}
              onClick={() => {
                onModelChange(key);
                dismissNudge();
              }}
              type="button"
            >
              {modelsMeta ? modelsMeta[key].label : key}
            </button>
          ))}
        </div>
      </div>

      <p className="illustrative-label">Illustrative flow — the four counts and rates below are exact.</p>

      <div
        className={useWaffle ? "inspection-stage inspection-stage-compact" : "inspection-stage"}
        onPointerDown={dismissNudge}
      >
        {showNudge && !useWaffle && (
          <div className="inspection-nudge" aria-hidden="true">
            Drag the dial. Try to catch every red part.
          </div>
        )}

        {useWaffle ? (
          sweep && currentPoint ? <WaffleFallback sweep={sweep} point={currentPoint} /> : null
        ) : (
          <div className="inspection-lane">
            {sweep && currentPoint && totals && (
              <PartField
                sweep={sweep}
                point={currentPoint}
                pDefect={totals.totalDefects / totals.total}
                onMissedShip={() => setShipFlash((n) => n + 1)}
                onBinArrival={() => setBinPulse((n) => n + 1)}
              />
            )}
            <div ref={binRef} className="inspection-bin" aria-hidden="true">
              <span>Inspection bin</span>
            </div>
            <div ref={shipRef} className="inspection-ships" aria-hidden="true">
              <span>Ships</span>
            </div>
          </div>
        )}

        <div className={useWaffle ? "dial-column dial-column-horizontal" : "dial-column"}>
          <Dial
            value={threshold}
            onChange={(v) => {
              onThresholdChange(v);
              dismissNudge();
            }}
            tunedValue={tunedThreshold}
            onResetToTuned={() => onThresholdChange(tunedThreshold)}
          />
        </div>
      </div>

      {currentPoint && <OutcomeLedger point={currentPoint} />}

      {sweep && currentPoint && (
        <WhatMattersDial
          sweep={sweep}
          costFn={costFn}
          costFp={costFp}
          onChangeCosts={onChangeCosts}
          currentPoint={currentPoint}
          onJumpToThreshold={onThresholdChange}
        />
      )}
    </div>
  );
}
