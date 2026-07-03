import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loadGovernance, loadModelsMeta, loadRp2Summary } from "../lib/data";
import type { Governance, ModelsMeta, Rp2Summary } from "../lib/types";

// Presentation-only short labels for the research ladder. All numbers come from
// governance.json (byte-checked against results/leaderboard.json by the export script);
// only these one-line descriptions are authored here.
const EXPERIMENT_LABELS: Record<string, string> = {
  K1: "Reproduced the frozen production model end-to-end — the honest baseline.",
  K2: "Added the record-adjacency “magic” features this leaderboard is famous for.",
  "K3-A": "Ablation: kept only the label-free position features. Matched K2 — the leak's label half added nothing.",
  "K3-B": "Ablation: kept only the neighbor-label lookups. Collapsed below the baseline — direction killed.",
  K4: "Tried timing-cohort geometry. Barely moved — I declared the record-order family saturated.",
  "K5-A": "Duplicate-identity keys, label-free. A real, modest signal.",
  "K5-B": "Identity-conditioned label lookup — the one label leak that does generalize.",
  P0: "Dropped in the full ~968-column raw sensor matrix. The single biggest jump in the program.",
  P1: "Station-temporal features + LightGBM capacity tuning. Program best — then I froze the track.",
};

export default function Story() {
  const [rp2, setRp2] = useState<Rp2Summary | null>(null);
  const [models, setModels] = useState<ModelsMeta | null>(null);
  const [gov, setGov] = useState<Governance | null>(null);

  useEffect(() => {
    loadRp2Summary().then(setRp2);
    loadModelsMeta().then(setModels);
    loadGovernance().then(setGov);
  }, []);

  const s = rp2?.cross_origin_summary;
  const experiments = gov?.leaderboard.experiments ?? null;
  const bestPrivate = experiments ? Math.max(...experiments.map((e) => e.private_mcc)) : null;
  const baselinePrivate = experiments?.find((e) => e.experiment_id === "K1")?.private_mcc ?? null;
  const minRate = rp2 ? Math.min(...rp2.fold_results.map((f) => f.test_pos_rate)) : null;
  const maxRate = rp2 ? Math.max(...rp2.fold_results.map((f) => f.test_pos_rate)) : null;

  return (
    <section>
      {/* ---------- Hero ---------- */}
      <div className="hero">
        <span className="eyebrow">Solo ML engineering project · 1,183,747 parts · 0.58% failure rate</span>
        <h1>I built the failure-detection system you could actually deploy — then measured what the leaderboard tricks are really worth.</h1>
        <p className="lede">
          Bosch's production-line dataset is famous for two things: extreme class imbalance and a
          leaderboard dominated by features that leak future information. I built the deployable
          version — leakage-safe features, a cost-based decision layer, label-free drift monitoring
          — and ran a separate, quarantined research track to decompose the leaderboard ceiling,
          mechanism by mechanism.
        </p>

        <div className="hero-contrast">
          <div className="hero-contrast-cell">
            <span className="big-number deployable">{s ? `${s.min_mcc.toFixed(2)}–${s.max_mcc.toFixed(2)}` : "…"}</span>
            <span className="hero-contrast-label">
              MCC that honestly ships — measured across 5 rolling time windows, the way a live
              deployment would experience it
            </span>
          </div>
          <div className="hero-contrast-divider">VS</div>
          <div className="hero-contrast-cell">
            <span className="big-number ceiling">{bestPrivate ? bestPrivate.toFixed(2) : "…"}</span>
            <span className="hero-contrast-label">
              Leaderboard ceiling — my quarantined research track's best private MCC, using signals
              a real factory stream can't have
            </span>
          </div>
        </div>
        <p className="hero-caption">
          The gap between these two numbers is the story most ML portfolios don't tell. I measured
          it, attributed it, and never let the right number contaminate the left one.
        </p>

        <div className="cta-row">
          <Link className="btn btn-primary" to="/decision-explorer">
            Try the Decision Explorer
          </Link>
          <a className="btn btn-secondary" href="#research-ladder">
            See the research ladder
          </a>
        </div>
      </div>

      {/* ---------- Why hard ---------- */}
      <h2>Why this problem is hard</h2>
      <div className="card-grid">
        <div className="card insight-card">
          <h3>1 failure in every ~172 parts</h3>
          <p>
            At a 0.58% failure rate, a model that predicts “everything is fine” is 99.4% accurate
            and completely useless. Accuracy is meaningless here — I report MCC, which punishes
            exactly that shortcut.
          </p>
        </div>
        <div className="card insight-card">
          <h3>The ground shifts under you</h3>
          <p>
            Across the five time windows I tested, the failure rate itself swings from{" "}
            {minRate !== null ? `${(minRate * 100).toFixed(2)}%` : "…"} to{" "}
            {maxRate !== null ? `${(maxRate * 100).toFixed(2)}%` : "…"}. A threshold tuned on the
            past quietly stops working on the future.
          </p>
        </div>
        <div className="card insight-card">
          <h3>The best scores can't ship</h3>
          <p>
            This competition's famous high scores lean on record-adjacency and duplicate-identity
            signals — information that only exists because the whole test set is visible at once. A
            factory scoring parts one at a time never has it.
          </p>
        </div>
      </div>

      {/* ---------- What I built ---------- */}
      <h2>What I built</h2>
      <p className="section-intro">
        An end-to-end decision system, not just a model: every stage below is committed, tested in
        CI, and released as v1.0.0.
      </p>
      <div className="flow-strip" aria-label="System pipeline">
        <span className="flow-step">
          Raw sensor data<small>1.18M rows × 3 sources</small>
        </span>
        <span className="flow-arrow" aria-hidden="true">→</span>
        <span className="flow-step">
          Leakage-safe features<small>chunk-aware, fold-scoped</small>
        </span>
        <span className="flow-arrow" aria-hidden="true">→</span>
        <span className="flow-step">
          4 LightGBM models<small>3 base + 1 stacked</small>
        </span>
        <span className="flow-arrow" aria-hidden="true">→</span>
        <span className="flow-step">
          Decision policy<small>cost model + budget</small>
        </span>
        <span className="flow-arrow" aria-hidden="true">→</span>
        <span className="flow-step">
          Drift monitoring<small>label-free, Evidently</small>
        </span>
      </div>
      <div className="card-grid">
        <div className="card insight-card">
          <h3>Cross-validation that can't leak</h3>
          <p>
            Rows are grouped into time chunks and no chunk ever spans train and validation. The
            guard raises on violation — and a CI test proves the guard itself works by injecting a
            leak and expecting the failure.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Decisions, not just scores</h3>
          <p>
            A configurable cost model (default: a missed failure costs 20× a false alarm) turns
            scores into an operating decision — inspect this part or pass it — with the trade-off
            explorable live on the <Link to="/decision-explorer">Decision Explorer</Link>.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Monitoring without labels</h3>
          <p>
            Production data has no ground truth, so the monitor never computes accuracy on it.
            Instead it watches the score distribution for drift — the honest signal you actually
            have on day one.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Governed like it matters</h3>
          <p>
            Every experiment was pre-registered before results existed, sealed with a git tag, and
            logged append-only. The <Link to="/governance">Governance</Link> page links every number
            on this site back to its evidence.
          </p>
        </div>
      </div>

      {/* ---------- Research ladder ---------- */}
      <h2 id="research-ladder">The research ladder: {baselinePrivate ? baselinePrivate.toFixed(2) : "…"} → {bestPrivate ? bestPrivate.toFixed(2) : "…"}</h2>
      <p className="section-intro">
        I ran nine pre-registered experiments to find out exactly where this leaderboard's
        performance comes from. Every hypothesis was written down before results existed. Two were
        rejected — and that negative result killed an entire follow-up direction before I spent a
        week on it. Green dots are label-free (“honest”) measurements; red-ringed dots leak label
        information and are only valid as leaderboard scores.
      </p>
      {experiments && (
        <ol className="timeline">
          {experiments.map((e, i) => (
            <li
              key={e.experiment_id}
              className={
                (e.oof_status === "contaminated" ? "contaminated" : "") +
                (i === experiments.length - 1 ? " milestone" : "")
              }
            >
              <span className="timeline-dot" aria-hidden="true" />
              <div className="timeline-head">
                <span className="timeline-id">{e.experiment_id}</span>
                <span className="timeline-mcc">private MCC {e.private_mcc.toFixed(3)}</span>
                <span className={e.oof_status === "honest" ? "badge badge-honest" : "badge badge-contaminated"}>
                  {e.oof_status}
                </span>
              </div>
              <p className="timeline-desc">{EXPERIMENT_LABELS[e.experiment_id] ?? e.mechanism}</p>
            </li>
          ))}
        </ol>
      )}
      <p className="note">
        This entire track is quarantined: its code lives behind an import firewall, and none of its
        numbers ever gate a production decision. Full pre-registration records on the{" "}
        <Link to="/governance">Governance</Link> page.
      </p>

      {/* ---------- Honest results ---------- */}
      <h2>What honestly ships</h2>
      <div className="card-grid">
        <div className="card stat stat-good">
          <span className="stat-value">{s ? `${s.min_mcc.toFixed(2)}–${s.max_mcc.toFixed(2)}` : "…"}</span>
          <span className="stat-label">MCC range across 5 rolling-origin time windows</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{s ? s.mean_mcc.toFixed(3) : "…"}</span>
          <span className="stat-label">Mean MCC (std {s ? s.std_mcc.toFixed(3) : "…"})</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{models ? models.meta_model.rows.toLocaleString() : "…"}</span>
          <span className="stat-label">Rows in the full-scale honest out-of-fold run</span>
        </div>
      </div>
      <p>
        These numbers look small next to the leaderboard — that's the point. I evaluated with a
        rolling-origin design: train on the past, score the future, five times. That's what a live
        deployment experiences, and it's{" "}
        {s ? `${Math.abs(s.degradation_vs_incv_pct).toFixed(0)}%` : "…"} harsher than the
        random-split number ({rp2 ? rp2.incv_mcc.toFixed(3) : "…"}) most projects report. Ranking
        quality barely moves across windows; what breaks is the fixed threshold. So the system
        treats threshold recalibration — not model replacement — as the first response to drift.
      </p>
      <details className="accordion">
        <summary>
          The five time windows, in detail
          <span className="summary-hint">rolling-origin evaluation table</span>
        </summary>
        <div className="accordion-body">
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
          <p className="chart-hint">
            “Oracle threshold” is the best threshold in hindsight for that window — an upper bound,
            not something a deployment can know in advance. The “fixed 0.91” column is what actually
            happens if the training-time threshold is never recalibrated: near-zero. That contrast
            is the strongest argument in this project for monitoring + recalibration.
          </p>
        </div>
      </details>

      {/* ---------- Decisions I'd defend ---------- */}
      <h2>Decisions I'd defend in a review</h2>
      <div className="card-grid">
        <div className="card insight-card">
          <h3>I gated on honest numbers only</h3>
          <p>
            Leaderboard scores never gated a production decision. A finding from the research track
            has to be re-derived leakage-free before it counts — a rule I wrote down before any
            experiment ran.
          </p>
        </div>
        <div className="card insight-card">
          <h3>I measured stacking — and shipped without it</h3>
          <p>
            The stacked meta-model scores{" "}
            {models ? models.meta_model.oof_mcc.toFixed(3) : "…"} OOF MCC versus{" "}
            {models ? models.dataset_h.oof_mcc.toFixed(3) : "…"} for its best base model. Stacking
            is regressive here, so the simpler model is the deployment candidate. Negative results
            get reported, not buried.
          </p>
        </div>
        <div className="card insight-card">
          <h3>I kept the tempting features out</h3>
          <p>
            Record-adjacency features would have tripled the offline score. They also can't exist
            in a one-part-at-a-time scoring stream — so the production feature contract bans them,
            and the research track proves what that ban costs.
          </p>
        </div>
      </div>

      {/* ---------- Go deeper ---------- */}
      <h2>Go deeper</h2>
      <div className="cta-grid">
        <Link className="card cta-card" to="/decision-explorer">
          <h3>Decision Explorer →</h3>
          <p>Move the threshold yourself and watch the catch-rate / false-alarm trade-off respond.</p>
        </Link>
        <Link className="card cta-card" to="/model-internals">
          <h3>Model Internals →</h3>
          <p>Feature importances, per-fold stability, and calibration for all four models.</p>
        </Link>
        <Link className="card cta-card" to="/governance">
          <h3>Governance →</h3>
          <p>The pre-registered experiment log, with every number linked to its evidence.</p>
        </Link>
        <a className="card cta-card" href="/docs/" target="_blank" rel="noreferrer">
          <h3>Documentation site →</h3>
          <p>Architecture, model & data cards, runbooks, and the full research summary.</p>
        </a>
      </div>

      <p className="note">
        Every number on this dashboard is exported directly from committed pipeline artifacts (
        <code>outputs/training_summary.json</code>, OOF prediction parquets, and{" "}
        <code>results/leaderboard.json</code>) by <code>scripts/ops/export_dashboard_data.py</code>{" "}
        — nothing here is hand-typed.
      </p>
    </section>
  );
}
