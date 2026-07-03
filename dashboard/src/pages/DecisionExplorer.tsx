import { useEffect, useMemo, useState } from "react";
import Plot from "../lib/plotly";
import Accordion from "../components/Accordion";
import InspectionLine from "../components/explorer/InspectionLine";
import { loadModelsMeta, loadSweep } from "../lib/data";
import { nearestSweepPoint } from "../lib/inspectionMath";
import { type ModelKey, type ModelsMeta, type SweepPoint } from "../lib/types";

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
    if (modelsMeta) setThreshold(modelsMeta[model].best_threshold);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const current = useMemo(() => (sweep ? nearestSweepPoint(sweep, threshold) : null), [sweep, threshold]);

  return (
    <section>
      <h1>Decision Explorer</h1>
      <p className="lede">
        Every part gets a risk score. One dial decides which ones get pulled for inspection. Drag it.
      </p>

      <InspectionLine
        modelsMeta={modelsMeta}
        model={model}
        onModelChange={setModel}
        sweep={sweep}
        threshold={threshold}
        onThresholdChange={setThreshold}
        costFn={costFn}
        costFp={costFp}
        onChangeCosts={(fn, fp) => {
          setCostFn(fn);
          setCostFp(fp);
        }}
      />

      <Accordion summary="What do these words actually mean?" hint="plain-English definitions">
        <dl className="metric-def">
          <dt>Threshold</dt>
          <dd>
            The score above which a part gets flagged. It's not learned — it's chosen, and the right
            choice depends on what a missed failure costs against a wasted inspection.
          </dd>
          <dt>Recall — "of every real failure, how many did I catch?"</dt>
          <dd>
            The safety metric. Flag everything and recall hits 100% for free — which is why recall
            alone never proves anything.
          </dd>
          <dt>Precision — "of everything I flagged, how much was real?"</dt>
          <dd>
            The trust metric. Low precision means inspectors start ignoring the flags.
          </dd>
          <dt>Missed failure (false negative)</dt>
          <dd>A defective part the system passed. The expensive mistake by default — 20× a false alarm — because it ships.</dd>
          <dt>False alarm (false positive)</dt>
          <dd>A healthy part pulled anyway. Cheap alone, but 1% of 1.18M parts is still ~12,000 wasted inspections.</dd>
          <dt>MCC</dt>
          <dd>
            One number for all four outcomes. +1 is perfect, 0 is random. Predicting "no failure" for
            everything scores exactly 0 here, not the 99.4% accuracy it looks like at this failure rate.
          </dd>
        </dl>
      </Accordion>

      {sweep && (
        <details className="accordion">
          <summary>
            The full trade-off, in two charts
            <span className="summary-hint">precision/recall and ROC curves</span>
          </summary>
          <div className="accordion-body">
            <p className="chart-hint">
              Each point is one possible threshold. The dot is where the dial is now.
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
              The dashed diagonal is random guessing — the gap between it and the curve is all the
              signal the model has.
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
          </div>
        </details>
      )}
    </section>
  );
}
