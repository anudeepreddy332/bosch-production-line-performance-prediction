import { useEffect, useMemo, useState } from "react";
import Plot from "../lib/plotly";
import { loadCalibration, loadImportances, loadModelsMeta } from "../lib/data";
import { MODEL_KEYS, type Calibration, type ImportanceRow, type ModelKey, type ModelsMeta } from "../lib/types";
import StackVerdict from "../components/internals/StackVerdict";
import MetricReadout from "../components/explorer/MetricReadout";
import Accordion from "../components/Accordion";

const FAMILY_COLORS: Record<string, string> = {
  structural: "#1e3a5f",
  "rolling-window": "#2f9e8f",
  "path/target-rate": "#b45309",
  "transition/co-occurrence": "#7c3aed",
  "meta-stack": "#b3261e",
  other: "#8a94a3",
};

const FAMILY_MEANING: { family: string; meaning: string }[] = [
  { family: "structural", meaning: "Physical flow of the part: when it entered, how long it took, sensor density." },
  { family: "rolling-window", meaning: "Short-term line congestion: how many parts passed in the last 1–24 hours." },
  { family: "path/target-rate", meaning: "Historical failure rates for the route a part took — computed fold-safe, never from its own fold." },
  { family: "transition/co-occurrence", meaning: "Risky station sequences: which station-to-station hops historically co-occur with failures." },
  { family: "meta-stack", meaning: "The base models' own predictions, fed to the stacked meta-model." },
];

export default function ModelInternals() {
  const [modelsMeta, setModelsMeta] = useState<ModelsMeta | null>(null);
  const [importances, setImportances] = useState<Record<ModelKey, ImportanceRow[]> | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [model, setModel] = useState<ModelKey>("dataset_h");

  useEffect(() => {
    loadModelsMeta().then(setModelsMeta);
    loadImportances().then(setImportances);
    loadCalibration().then(setCalibration);
  }, []);

  const meta = modelsMeta?.[model];
  const imp = importances?.[model];
  const cal = calibration?.[model];

  const families = useMemo(() => (imp ? Array.from(new Set(imp.map((r) => r.family))) : []), [imp]);
  const relevantFamilies = useMemo(
    () => FAMILY_MEANING.filter((f) => families.includes(f.family)),
    [families],
  );

  return (
    <section>
      <h1>Model Internals</h1>
      <p className="lede">Four LightGBM models, three base and one stacked — measured honestly.</p>

      {modelsMeta && <StackVerdict modelsMeta={modelsMeta} />}

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

      {meta && (
        <div className="card-grid card-grid-tight">
          <div className="card stat">
            <span className="stat-value">{meta.oof_mcc.toFixed(4)}</span>
            <span className="stat-label">Honest OOF MCC at tuned threshold {meta.best_threshold.toFixed(2)}</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{meta.feature_count}</span>
            <span className="stat-label">Features — deliberately few, all deployable</span>
          </div>
          <div className="card stat">
            <span className="stat-value">{meta.rows.toLocaleString()}</span>
            <span className="stat-label">Training rows, 5-fold chunk-aware CV</span>
          </div>
          <MetricReadout
            value={meta.data_fingerprint ? meta.data_fingerprint.slice(0, 10) + "…" : "n/a"}
            label="Data fingerprint"
            definition={`${meta.data_fingerprint ?? "n/a"} — a hash of rows + features + labels, so a rerun can prove it trained on identical data.`}
          />
        </div>
      )}

      {imp && (
        <>
          <h2>Feature importances</h2>
          <p className="chart-hint">No single feature dominates — nothing fragile here.</p>
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
          {relevantFamilies.length > 0 && (
            <Accordion summary="What the feature families mean" hint={`${relevantFamilies.length} families in this model`}>
              <dl className="metric-def">
                {relevantFamilies.map((f) => (
                  <div key={f.family}>
                    <dt>
                      <span
                        aria-hidden="true"
                        className="family-swatch"
                        style={{ background: FAMILY_COLORS[f.family] ?? "#8a94a3" }}
                      />
                      {f.family}
                    </dt>
                    <dd>{f.meaning}</dd>
                  </div>
                ))}
              </dl>
            </Accordion>
          )}
        </>
      )}

      {meta && (
        <Accordion
          summary="Fold-level stability"
          hint="5 folds — spread is expected, not a red flag"
        >
          <p className="chart-hint">
            At a ~0.58% failure rate each fold holds only ~1,400 positives, so per-fold MCC is
            naturally noisy. I report the spread rather than the best fold.
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
        </Accordion>
      )}

      {cal && (
        <>
          <h2>Calibration</h2>
          <p className="chart-hint">
            Dots near the dashed line mean a 0.8 score really means ~80% failure chance, not just a
            good ranking — the cost model prices mistakes in real units, so this matters.
          </p>
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
