import { useEffect, useMemo, useState } from "react";
import Plot from "../lib/plotly";
import { loadModelsMeta, loadSweep } from "../lib/data";
import { MODEL_KEYS, type ModelKey, type ModelsMeta, type SweepPoint } from "../lib/types";

function nearestPoint(sweep: SweepPoint[], threshold: number): SweepPoint {
  let best = sweep[0];
  let bestDist = Math.abs(best.threshold - threshold);
  for (const p of sweep) {
    const d = Math.abs(p.threshold - threshold);
    if (d < bestDist) {
      best = p;
      bestDist = d;
    }
  }
  return best;
}

export default function DecisionExplorer() {
  const [modelsMeta, setModelsMeta] = useState<ModelsMeta | null>(null);
  const [model, setModel] = useState<ModelKey>("dataset_h");
  const [sweep, setSweep] = useState<SweepPoint[] | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [costFn, setCostFn] = useState(100);
  const [costFp, setCostFp] = useState(5);

  useEffect(() => {
    loadModelsMeta().then((m) => {
      setModelsMeta(m);
      setThreshold(m.dataset_h.best_threshold);
    });
  }, []);

  useEffect(() => {
    setSweep(null);
    loadSweep(model).then(setSweep);
    if (modelsMeta) {
      setThreshold(modelsMeta[model].best_threshold);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const current = useMemo(() => (sweep ? nearestPoint(sweep, threshold) : null), [sweep, threshold]);
  const tunedThreshold = modelsMeta ? modelsMeta[model].best_threshold : null;
  const tuned = useMemo(
    () => (sweep && tunedThreshold !== null ? nearestPoint(sweep, tunedThreshold) : null),
    [sweep, tunedThreshold],
  );

  const minCostPoint = useMemo(() => {
    if (!sweep) return null;
    let best = sweep[0];
    let bestCost = Infinity;
    for (const p of sweep) {
      const cost = p.fn * costFn + p.fp * costFp;
      if (cost < bestCost) {
        bestCost = cost;
        best = p;
      }
    }
    return { point: best, cost: bestCost };
  }, [sweep, costFn, costFp]);

  const currentCost = current ? current.fn * costFn + current.fp * costFp : null;

  const totalFailures = current ? current.tp + current.fn : null;
  const caughtPer100 = current && totalFailures ? Math.round((current.tp / totalFailures) * 100) : null;
  const realPer100Flagged =
    current && current.tp + current.fp > 0 ? Math.round((current.tp / (current.tp + current.fp)) * 100) : null;

  // Where the current threshold sits relative to the tuned operating point (data-driven, from
  // the same sweep -- no new math, just a comparison of two precomputed grid points).
  const vsTuned =
    current && tuned && current.threshold !== tuned.threshold
      ? {
          direction: current.threshold > tuned.threshold ? ("above" as const) : ("below" as const),
          caughtDelta: current.tp - tuned.tp,
          falseAlarmDelta: current.fp - tuned.fp,
        }
      : null;

  const liveKey = current ? `${model}-${current.threshold}` : "loading";

  return (
    <section>
      <h1>Decision Explorer</h1>
      <p className="lede">
        Every part coming off the line gets a failure-risk score between 0 and 1. One number — the
        threshold — decides which parts get pulled for inspection. Move it and watch the real
        consequences, computed from the same honest out-of-fold predictions I trained on: 201
        pre-computed grid points, no interpolation, no client-side inference.
      </p>

      <div className="callout">
        <p>
          <strong>There is no free lunch here.</strong> Catch more failures and you inspect more
          healthy parts. Cut the false alarms and you let more failures through. The threshold
          doesn't remove that trade-off — it picks where on it you stand. That's why I treat this
          as a business decision with a cost model, not a modeling detail.
        </p>
      </div>

      <div className="controls-row">
        <div className="model-select" role="group" aria-label="Model selection">
          {MODEL_KEYS.map((key) => (
            <button
              key={key}
              className={key === model ? "active" : ""}
              onClick={() => setModel(key)}
              type="button"
            >
              {modelsMeta ? modelsMeta[key].label : key}
            </button>
          ))}
        </div>
      </div>
      <p className="chart-hint">
        Dataset H is the production candidate — it beats the stacked meta-model on honest OOF MCC,
        which is why the simpler model ships. Switching models resets the slider to that model's
        tuned threshold.
      </p>

      <div className="controls-row">
        <div className="slider-block">
          <label htmlFor="threshold-slider">
            Threshold: <strong>{threshold.toFixed(3)}</strong>
            {tunedThreshold !== null && threshold !== tunedThreshold && (
              <>
                {" "}
                <button type="button" className="btn btn-secondary" style={{ padding: "0.15rem 0.5rem", fontSize: "0.78rem" }} onClick={() => setThreshold(tunedThreshold)}>
                  reset to tuned ({tunedThreshold.toFixed(2)})
                </button>
              </>
            )}
          </label>
          <input
            id="threshold-slider"
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </div>
      </div>

      {current && (
        <>
          <h2 style={{ marginTop: "1.5rem" }}>
            What happens to the {(current.tp + current.fp + current.fn + current.tn).toLocaleString()} parts
          </h2>
          <div className="card-grid card-grid-tight" key={liveKey}>
            <div className="card stat stat-good stat-live">
              <span className="stat-value">{current.tp.toLocaleString()}</span>
              <span className="stat-label">Detected failures — flagged and really failing (TP)</span>
            </div>
            <div className="card stat stat-danger stat-live">
              <span className="stat-value">{current.fn.toLocaleString()}</span>
              <span className="stat-label">Missed failures — shipped as good, actually bad (FN)</span>
            </div>
            <div className="card stat stat-warn stat-live">
              <span className="stat-value">{current.fp.toLocaleString()}</span>
              <span className="stat-label">False alarms — healthy parts pulled for inspection (FP)</span>
            </div>
            <div className="card stat stat-live">
              <span className="stat-value">{current.tn.toLocaleString()}</span>
              <span className="stat-label">Healthy parts passed untouched (TN)</span>
            </div>
          </div>

          <div className="callout callout-live" aria-live="polite">
            <p>
              At threshold <strong>{threshold.toFixed(3)}</strong>, the system flags{" "}
              <strong>{current.flagged_pct.toFixed(2)}%</strong> of all parts.
              {realPer100Flagged !== null && (
                <>
                  {" "}
                  Of every 100 parts it flags, about <strong>{realPer100Flagged}</strong> are real
                  failures.
                </>
              )}
              {caughtPer100 !== null && (
                <>
                  {" "}
                  Of every 100 real failures, it catches about <strong>{caughtPer100}</strong> —
                  and <strong>{100 - caughtPer100}</strong> slip through.
                </>
              )}
            </p>
            {vsTuned && (
              <p>
                You're <strong>{vsTuned.direction}</strong> the tuned operating point: catching{" "}
                <strong>
                  {Math.abs(vsTuned.caughtDelta).toLocaleString()} {vsTuned.caughtDelta >= 0 ? "more" : "fewer"}
                </strong>{" "}
                failures, with{" "}
                <strong>
                  {Math.abs(vsTuned.falseAlarmDelta).toLocaleString()}{" "}
                  {vsTuned.falseAlarmDelta >= 0 ? "more" : "fewer"}
                </strong>{" "}
                false alarms.
              </p>
            )}
          </div>

          <div className="card-grid card-grid-tight" key={`${liveKey}-metrics`}>
            <div className="card stat stat-live">
              <span className="stat-value">{current.recall.toFixed(3)}</span>
              <span className="stat-label">Recall — share of real failures caught</span>
            </div>
            <div className="card stat stat-live">
              <span className="stat-value">{current.precision.toFixed(3)}</span>
              <span className="stat-label">Precision — share of flags that are real</span>
            </div>
            <div className="card stat stat-live">
              <span className="stat-value">{current.mcc.toFixed(3)}</span>
              <span className="stat-label">MCC — overall quality, imbalance-proof</span>
            </div>
            <div className="card stat stat-live">
              <span className="stat-value">{current.flagged_pct.toFixed(2)}%</span>
              <span className="stat-label">Inspection load — parts flagged</span>
            </div>
          </div>
        </>
      )}

      <div className="compare-grid">
        <div className="card compare-card">
          <h3>Raise the threshold ↑</h3>
          <ul className="consequences">
            <li className="good">Fewer parts pulled — lower inspection cost</li>
            <li className="good">Higher precision — flags you can trust</li>
            <li className="bad">Lower recall — more failures slip through</li>
            <li className="bad">Each missed failure ships to a customer</li>
          </ul>
        </div>
        <div className="card compare-card">
          <h3>Lower the threshold ↓</h3>
          <ul className="consequences">
            <li className="good">Higher recall — more failures caught</li>
            <li className="good">Fewer defective parts reach customers</li>
            <li className="bad">More false alarms — wasted inspections</li>
            <li className="bad">Inspection line saturates fast at 1.18M parts</li>
          </ul>
        </div>
      </div>

      <details className="accordion">
        <summary>
          What do these metrics actually mean?
          <span className="summary-hint">plain-English definitions</span>
        </summary>
        <div className="accordion-body">
          <dl className="metric-def">
            <dt>Threshold</dt>
            <dd>
              The score above which a part gets flagged. It's not learned — it's chosen, and the
              right choice depends on what a missed failure costs you versus a wasted inspection.
              Lower it when failures are expensive; raise it when inspections are.
            </dd>
            <dt>Recall — “of all the real failures, how many did I catch?”</dt>
            <dd>
              The safety metric. High recall matters when a missed failure is expensive or
              dangerous. You can always hit 100% recall by flagging everything — which is why
              recall alone is never enough.
            </dd>
            <dt>Precision — “of everything I flagged, how much was real?”</dt>
            <dd>
              The trust metric. Low precision means inspectors mostly see healthy parts, start
              ignoring the flags, and the system loses credibility. High precision matters when
              inspection capacity is scarce.
            </dd>
            <dt>False negative (missed failure)</dt>
            <dd>
              A defective part the system passed. In this cost model it's the expensive mistake —
              20× a false alarm — because it ships to a customer.
            </dd>
            <dt>False positive (false alarm)</dt>
            <dd>
              A healthy part the system flagged. Cheap individually, but at 1.18M parts even a 1%
              false-alarm rate means ~12,000 needless inspections.
            </dd>
            <dt>MCC — Matthews correlation coefficient</dt>
            <dd>
              One number summarizing all four outcomes: +1 is perfect, 0 is random guessing. Unlike
              accuracy, it can't be gamed by predicting “no failure” for everything — at a 0.58%
              failure rate, that shortcut scores 99.4% accuracy but an MCC of exactly 0. That's why
              this project reports MCC everywhere.
            </dd>
          </dl>
        </div>
      </details>

      <h2>The cost model</h2>
      <p className="section-intro">
        I priced the two mistakes instead of guessing a threshold: by default a missed failure
        costs 20× a false alarm (100 vs 5 — the same <code>CostConfig</code> as{" "}
        <code>src/evaluation/decision_system.py</code>). The units are relative; only the ratio
        matters. Change either number and see where the cost-minimizing threshold moves.
      </p>
      <div className="controls-row">
        <label>
          Cost per missed failure:{" "}
          <input
            type="number"
            min={0}
            value={costFn}
            onChange={(e) => setCostFn(Number(e.target.value))}
            style={{ width: "5rem" }}
          />
        </label>
        <label>
          Cost per false alarm:{" "}
          <input
            type="number"
            min={0}
            value={costFp}
            onChange={(e) => setCostFp(Number(e.target.value))}
            style={{ width: "5rem" }}
          />
        </label>
      </div>
      {current && currentCost !== null && minCostPoint && (
        <div className="callout callout-live">
          <p>
            Total cost at threshold {threshold.toFixed(3)}:{" "}
            <strong>{currentCost.toLocaleString()}</strong>. The cheapest threshold on this sweep is{" "}
            <strong>{minCostPoint.point.threshold.toFixed(3)}</strong> (cost{" "}
            {minCostPoint.cost.toLocaleString()})
            {currentCost > minCostPoint.cost && (
              <>
                {" "}
                — you're paying{" "}
                <strong>{(currentCost - minCostPoint.cost).toLocaleString()}</strong> above optimal
              </>
            )}
            .{" "}
            <button type="button" onClick={() => setThreshold(minCostPoint.point.threshold)}>
              jump to it
            </button>
          </p>
        </div>
      )}

      {sweep && (
        <>
          <h2>The full trade-off, in two charts</h2>
          <p className="chart-hint">
            Each point on these curves is one possible threshold. The dot is where your slider is
            now. Precision collapses as recall climbs — that cliff is the class imbalance doing its
            work.
          </p>
          <div className="chart-wrap">
            <Plot
              data={[
                {
                  x: sweep.map((p) => p.recall),
                  y: sweep.map((p) => p.precision),
                  type: "scatter",
                  mode: "lines",
                  name: "PR curve",
                  line: { color: "#1e3a5f" },
                },
                current
                  ? {
                      x: [current.recall],
                      y: [current.precision],
                      type: "scatter",
                      mode: "markers",
                      name: "current threshold",
                      marker: { color: "#2f9e8f", size: 10 },
                    }
                  : {},
              ]}
              layout={{
                title: { text: "Precision vs. recall" },
                xaxis: { title: { text: "Recall" }, range: [0, 1] },
                yaxis: { title: { text: "Precision" }, range: [0, 1] },
                margin: { t: 40, r: 20, l: 50, b: 40 },
                height: 360,
                autosize: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
          <p className="chart-hint">
            The ROC curve shows the same trade-off against the false-positive rate. The dashed
            diagonal is random guessing — the gap between it and the curve is all the signal the
            model has.
          </p>
          <div className="chart-wrap">
            <Plot
              data={[
                {
                  x: sweep.map((p) => p.fpr),
                  y: sweep.map((p) => p.recall),
                  type: "scatter",
                  mode: "lines",
                  name: "ROC curve",
                  line: { color: "#1e3a5f" },
                },
                {
                  x: [0, 1],
                  y: [0, 1],
                  type: "scatter",
                  mode: "lines",
                  name: "random",
                  line: { color: "#c3ccd6", dash: "dash" },
                },
                current
                  ? {
                      x: [current.fpr],
                      y: [current.recall],
                      type: "scatter",
                      mode: "markers",
                      name: "current threshold",
                      marker: { color: "#2f9e8f", size: 10 },
                    }
                  : {},
              ]}
              layout={{
                title: { text: "ROC (false positive rate vs. true positive rate)" },
                xaxis: { title: { text: "False positive rate" }, range: [0, 1] },
                yaxis: { title: { text: "True positive rate (recall)" }, range: [0, 1] },
                margin: { t: 40, r: 20, l: 50, b: 40 },
                height: 360,
                autosize: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        </>
      )}
    </section>
  );
}
