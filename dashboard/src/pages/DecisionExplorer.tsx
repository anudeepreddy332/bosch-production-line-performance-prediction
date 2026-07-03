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
  const [model, setModel] = useState<ModelKey>("meta_model");
  const [sweep, setSweep] = useState<SweepPoint[] | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [costFn, setCostFn] = useState(100);
  const [costFp, setCostFp] = useState(5);

  useEffect(() => {
    loadModelsMeta().then((m) => {
      setModelsMeta(m);
      setThreshold(m.meta_model.best_threshold);
    });
  }, []);

  useEffect(() => {
    setSweep(null);
    loadSweep(model).then(setSweep);
  }, [model]);

  const current = useMemo(() => (sweep ? nearestPoint(sweep, threshold) : null), [sweep, threshold]);

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

  return (
    <section>
      <h1>Decision Explorer</h1>
      <p className="lede">
        Move the threshold to see the recall/precision/MCC trade-off for each model, computed live
        from the same honest out-of-fold predictions used to train it — every point below is one of
        201 pre-computed, exact grid values (no interpolation, no client-side model inference).
      </p>

      <div className="controls-row">
        <div className="model-select">
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

      <div className="controls-row">
        <div className="slider-block">
          <label htmlFor="threshold-slider">
            Threshold: <strong>{threshold.toFixed(3)}</strong>
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
        <div className="card-grid">
          <div className="card stat">
            <span className="stat-value">{current.recall.toFixed(3)}</span>
            <span className="stat-label">Recall (failures caught)</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{current.precision.toFixed(3)}</span>
            <span className="stat-label">Precision</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{current.mcc.toFixed(3)}</span>
            <span className="stat-label">MCC</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{current.flagged_pct.toFixed(2)}%</span>
            <span className="stat-label">Rows flagged for inspection</span>
          </div>
        </div>
      )}

      {current && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>TN</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{current.tp.toLocaleString()}</td>
                <td>{current.fp.toLocaleString()}</td>
                <td>{current.fn.toLocaleString()}</td>
                <td>{current.tn.toLocaleString()}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <h2>Cost model</h2>
      <p>
        Default cost weights (a missed failure costs 20x a false alarm) match{" "}
        <code>src/evaluation/decision_system.py</code>'s <code>CostConfig</code>. Adjust them to see
        which threshold on this model's sweep minimizes total cost.
      </p>
      <div className="controls-row">
        <label>
          Cost per missed failure (FN):{" "}
          <input
            type="number"
            min={0}
            value={costFn}
            onChange={(e) => setCostFn(Number(e.target.value))}
            style={{ width: "5rem" }}
          />
        </label>
        <label>
          Cost per false alarm (FP):{" "}
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
        <p>
          Total cost at threshold {threshold.toFixed(3)}: <strong>{currentCost.toLocaleString()}</strong>.
          Minimum-cost threshold on this sweep:{" "}
          <strong>{minCostPoint.point.threshold.toFixed(3)}</strong> (cost{" "}
          {minCostPoint.cost.toLocaleString()}) —{" "}
          <button type="button" onClick={() => setThreshold(minCostPoint.point.threshold)}>
            jump to it
          </button>
        </p>
      )}

      {sweep && (
        <>
          <h2>Precision-recall and ROC</h2>
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
