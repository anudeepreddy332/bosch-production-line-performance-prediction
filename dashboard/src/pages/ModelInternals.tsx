import { useEffect, useMemo, useState } from "react";
import Plot from "../lib/plotly";
import { loadCalibration, loadImportances, loadModelsMeta } from "../lib/data";
import { MODEL_KEYS, type Calibration, type ImportanceRow, type ModelKey, type ModelsMeta } from "../lib/types";

const FAMILY_COLORS: Record<string, string> = {
  structural: "#1e3a5f",
  "rolling-window": "#2f9e8f",
  "path/target-rate": "#b45309",
  "transition/co-occurrence": "#7c3aed",
  "meta-stack": "#b3261e",
  other: "#8a94a3",
};

export default function ModelInternals() {
  const [modelsMeta, setModelsMeta] = useState<ModelsMeta | null>(null);
  const [importances, setImportances] = useState<Record<ModelKey, ImportanceRow[]> | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [model, setModel] = useState<ModelKey>("meta_model");

  useEffect(() => {
    loadModelsMeta().then(setModelsMeta);
    loadImportances().then(setImportances);
    loadCalibration().then(setCalibration);
  }, []);

  const meta = modelsMeta?.[model];
  const imp = importances?.[model];
  const cal = calibration?.[model];

  const families = useMemo(() => (imp ? Array.from(new Set(imp.map((r) => r.family))) : []), [imp]);

  return (
    <section>
      <h1>Model Internals</h1>
      <p className="lede">
        Four LightGBM models, stacked: three base models feed a meta-model. Importances, fold-level
        MCC spread, and calibration are computed from the same honest out-of-fold predictions shown
        in the Decision Explorer.
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

      {meta && (
        <div className="card-grid">
          <div className="card stat">
            <span className="stat-value">{meta.oof_mcc.toFixed(4)}</span>
            <span className="stat-label">OOF MCC (best threshold {meta.best_threshold.toFixed(2)})</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{meta.feature_count}</span>
            <span className="stat-label">Features</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{meta.rows.toLocaleString()}</span>
            <span className="stat-label">Training rows (5-fold OOF)</span>
          </div>
          <div className="card stat">
            <span className="stat-value" style={{ fontSize: "0.95rem" }} title={meta.data_fingerprint ?? ""}>
              {meta.data_fingerprint ?? "n/a"}
            </span>
            <span className="stat-label">Data fingerprint</span>
          </div>
        </div>
      )}

      {imp && (
        <>
          <h2>Feature importances</h2>
          <div className="chart-wrap">
            <Plot
              data={families.map((fam) => {
                const rows = imp.filter((r) => r.family === fam);
                return {
                  x: rows.map((r) => r.importance_pct),
                  y: rows.map((r) => r.feature),
                  type: "bar",
                  orientation: "h",
                  name: fam,
                  marker: { color: FAMILY_COLORS[fam] ?? "#8a94a3" },
                };
              })}
              layout={{
                barmode: "stack",
                yaxis: { automargin: true, categoryorder: "total ascending" },
                xaxis: { title: { text: "Share of total importance (%)" } },
                margin: { t: 20, r: 20, l: 160, b: 40 },
                height: Math.max(320, imp.length * 26),
                autosize: true,
                legend: { orientation: "h", y: -0.15 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        </>
      )}

      {meta && (
        <>
          <h2>Fold-level MCC spread</h2>
          <p>
            Per-fold thresholds and MCC vary because the failure rate is ~0.58% overall — each fold
            has few positive examples, so both quantities are naturally noisy at this scale.
          </p>
          <div className="chart-wrap">
            <Plot
              data={[
                {
                  x: meta.fold_metrics.map((f) => `Fold ${f.fold}`),
                  y: meta.fold_metrics.map((f) => f.mcc),
                  type: "bar",
                  marker: { color: "#1e3a5f" },
                  name: "Fold MCC",
                },
              ]}
              layout={{
                yaxis: { title: { text: "MCC" } },
                margin: { t: 20, r: 20, l: 50, b: 40 },
                height: 320,
                autosize: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        </>
      )}

      {cal && (
        <>
          <h2>Calibration</h2>
          <p>Mean predicted probability vs. observed failure rate, in 20 equal-width bins.</p>
          <div className="chart-wrap">
            <Plot
              data={[
                {
                  x: cal.map((c) => c.mean_predicted),
                  y: cal.map((c) => c.observed_rate),
                  type: "scatter",
                  mode: "markers+lines",
                  name: "Observed",
                  marker: { color: "#2f9e8f", size: 8 },
                },
                {
                  x: [0, 1],
                  y: [0, 1],
                  type: "scatter",
                  mode: "lines",
                  name: "Perfectly calibrated",
                  line: { color: "#c3ccd6", dash: "dash" },
                },
              ]}
              layout={{
                xaxis: { title: { text: "Mean predicted probability" }, range: [0, 1] },
                yaxis: { title: { text: "Observed failure rate" }, range: [0, 1] },
                margin: { t: 20, r: 20, l: 50, b: 40 },
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
