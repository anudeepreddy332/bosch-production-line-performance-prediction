import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loadModelsMeta, loadRp2Summary } from "../lib/data";
import type { ModelsMeta, Rp2Summary } from "../lib/types";

export default function Story() {
  const [rp2, setRp2] = useState<Rp2Summary | null>(null);
  const [models, setModels] = useState<ModelsMeta | null>(null);

  useEffect(() => {
    loadRp2Summary().then(setRp2);
    loadModelsMeta().then(setModels);
  }, []);

  const s = rp2?.cross_origin_summary;

  return (
    <section>
      <h1>Detecting rare manufacturing failures — honestly measured</h1>
      <p className="lede">
        Bosch's production line dataset has a 0.58% failure rate. This system is built to be
        deployed, not to top a leaderboard: leakage-safe features, an explainable cost/threshold
        trade-off, and label-free production monitoring — measured under the same temporal drift a
        real deployment would face.
      </p>

      <div className="card-grid">
        <div className="card stat">
          <span className="stat-value">{s ? `${s.min_mcc.toFixed(2)}–${s.max_mcc.toFixed(2)}` : "…"}</span>
          <span className="stat-label">Honest deployable MCC, 5 rolling-origin windows</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{s ? s.mean_mcc.toFixed(3) : "…"}</span>
          <span className="stat-label">Mean MCC (std {s ? s.std_mcc.toFixed(3) : "…"})</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{s ? `${s.degradation_vs_incv_pct.toFixed(0)}%` : "…"}</span>
          <span className="stat-label">Degradation vs. in-CV (random-split, interpolation-optimistic)</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{models ? models.meta_model.rows.toLocaleString() : "…"}</span>
          <span className="stat-label">Rows, full-scale honest OOF run</span>
        </div>
      </div>

      <h2>Why the number looks low — and why that's the point</h2>
      <p>
        A rolling-origin (forward-chaining) evaluation, not a random split, is what a live
        deployment actually experiences: train on the past, score the future. Across 5 temporal
        windows the failure rate itself swings from 0.33% to 0.94%, and that non-stationarity — not
        model quality — drives most of the MCC variation. Ranking quality (AUC ≈ 0.55) barely moves
        across regimes; what moves is the operating threshold, which is why this system treats
        threshold recalibration, not model replacement, as the first response to drift.
      </p>

      {rp2 && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Window</th>
                <th>Test chunks</th>
                <th>Test failure rate</th>
                <th>Oracle threshold</th>
                <th>MCC at oracle</th>
                <th>MCC at fixed 0.91</th>
              </tr>
            </thead>
            <tbody>
              {rp2.fold_results.map((f) => (
                <tr key={f.fold_idx}>
                  <td>{f.fold_idx}</td>
                  <td>{f.test_chunks}</td>
                  <td>{(f.test_pos_rate * 100).toFixed(2)}%</td>
                  <td>{f.oot_best_threshold.toFixed(2)}</td>
                  <td>
                    <strong>{f.oot_mcc_best_threshold.toFixed(3)}</strong>
                  </td>
                  <td>{f.oot_mcc_fixed_threshold.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h2>The system</h2>
      <p>
        Three base LightGBM models — a lean structural baseline, one adding out-of-fold-safe
        target-rate features, one adding path-transition and station co-occurrence risk features —
        stack into a meta-model. A configurable cost model (default: a missed failure costs 100x a
        false alarm) turns raw scores into a threshold or inspection-budget decision, and an
        Evidently-based monitor watches for input/output drift without ever needing production
        labels. See the <Link to="/model-internals">Model Internals</Link> page for per-model
        importances and calibration, and the <Link to="/decision-explorer">Decision Explorer</Link>{" "}
        to try the cost/threshold trade-off yourself.
      </p>

      <h2>Scope note</h2>
      <p>
        A separate, fully quarantined research track explored how far a leaderboard-style score
        could be pushed using this same dataset (private MCC up to 0.419, via record-adjacency and
        raw-signal features that would leak future information in a real deployment). It never
        informs any number on this page or in the production decision system — see{" "}
        <Link to="/governance">Governance &amp; Reproducibility</Link> for the full, pre-registered
        research log.
      </p>

      <p className="note">
        Every number on this dashboard is exported directly from committed pipeline artifacts (
        <code>outputs/training_summary.json</code>, OOF prediction parquets, and{" "}
        <code>results/leaderboard.json</code>) by{" "}
        <code>scripts/ops/export_dashboard_data.py</code> — nothing here is hand-typed.
      </p>
    </section>
  );
}
