import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loadGovernance, loadModelsMeta, loadRp2Summary } from "../lib/data";
import type { Governance, ModelsMeta, Rp2Summary } from "../lib/types";
import GapHero from "../components/story/GapHero";
import RarityField from "../components/story/RarityField";
import PipelineFlow from "../components/story/PipelineFlow";
import ResearchLadder from "../components/story/ResearchLadder";
import StatCard from "../components/StatCard";
import Accordion from "../components/Accordion";

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

const PIPELINE_STAGES = [
  { label: "Raw sensor data", detail: "1.18M rows × 3 sources" },
  { label: "Leakage-safe features", detail: "chunk-aware, fold-scoped" },
  { label: "4 LightGBM models", detail: "3 base + 1 stacked" },
  { label: "Decision policy", detail: "cost model + budget" },
  { label: "Drift monitoring", detail: "label-free, Evidently" },
];

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

  return (
    <section>
      {/* ---------- Hero: the whole first screen ---------- */}
      <div className="hero">
        <span className="eyebrow">Solo ML engineering project · 1,183,747 parts</span>
        <h1>I built the failure-detection system you could actually deploy — then measured what the leaderboard tricks are really worth.</h1>

        <GapHero deployableRange={s ? `${s.min_mcc.toFixed(2)}–${s.max_mcc.toFixed(2)}` : "…"} ceiling={bestPrivate} />

        <p className="hero-caption">
          The gap between these two numbers is the story most ML portfolios don't tell.
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

      {/* ---------- Why hard: felt, not explained ---------- */}
      <h2>Why this problem is hard</h2>
      <RarityField />
      <ul className="rarity-labels">
        <li>A threshold tuned on the past quietly stops working on the future</li>
        <li>The leaderboard's best scores lean on signals a live stream can't have</li>
        <li>Accuracy is meaningless here — “always healthy” already scores 99.4%</li>
      </ul>

      {/* ---------- What I built ---------- */}
      <h2>What I built</h2>
      <p className="section-intro">
        An end-to-end decision system, not just a model — every stage below is committed, tested in
        CI, and released as v1.0.0.
      </p>
      <PipelineFlow stages={PIPELINE_STAGES} />
      <div className="card-grid">
        <div className="card insight-card">
          <h3>Cross-validation that can't leak</h3>
          <p>
            No time chunk ever sits in both train and test. A CI test proves the guard fires by
            injecting a leak and expecting the failure.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Decisions, not just scores</h3>
          <p>
            A cost model turns scores into an operating call — inspect this part or pass it —
            explorable live on the <Link to="/decision-explorer">Decision Explorer</Link>.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Monitoring without labels</h3>
          <p>
            Production data has no ground truth, so the monitor watches the score distribution for
            drift instead — the honest signal available on day one.
          </p>
        </div>
        <div className="card insight-card">
          <h3>Governed like it matters</h3>
          <p>
            Every experiment was pre-registered before results existed, sealed with a git tag. The{" "}
            <Link to="/governance">Governance</Link> page links every number back to its evidence.
          </p>
        </div>
      </div>

      {/* ---------- Research ladder ---------- */}
      <h2 id="research-ladder">The research ladder</h2>
      <p className="section-intro">
        Nine pre-registered experiments, hypothesis written down before results existed. Two were
        rejected — one killed a whole follow-up direction before I spent a week on it.
      </p>
      {experiments && <ResearchLadder experiments={experiments} />}
      {experiments && (
        <ol className="visually-hidden">
          {experiments.map((e) => (
            <li key={e.experiment_id}>{EXPERIMENT_LABELS[e.experiment_id] ?? e.mechanism}</li>
          ))}
        </ol>
      )}
      <p className="note">
        This entire track is quarantined behind an import firewall, and none of its numbers ever
        gate a production decision. Full pre-registration records on the{" "}
        <Link to="/governance">Governance</Link> page.
      </p>

      {/* ---------- Honest results ---------- */}
      <h2>What honestly ships</h2>
      <div className="card-grid">
        <StatCard value={s ? `${s.min_mcc.toFixed(2)}–${s.max_mcc.toFixed(2)}` : "…"} label="MCC range across 5 rolling-origin time windows" tone="good" />
        <StatCard value={s ? s.mean_mcc : 0} countUp decimals={3} label={`Mean MCC (std ${s ? s.std_mcc.toFixed(3) : "…"})`} />
        <StatCard value={models ? models.meta_model.rows : 0} countUp label="Rows in the full-scale honest out-of-fold run" />
      </div>
      <p>
        These numbers look small next to the leaderboard — that's the point. A rolling-origin
        evaluation (train on the past, score the future) is {s ? `${Math.abs(s.degradation_vs_incv_pct).toFixed(0)}%` : "…"} harsher than the random-split number most projects report.
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
            Ranking quality (AUC) barely moves across windows — what breaks is the fixed threshold.
            “Oracle threshold” is the best choice in hindsight, an upper bound no deployment can know
            in advance; “fixed 0.91” is what actually happens if the threshold is never recalibrated:
            near zero. That gap is why this system treats recalibration, not model replacement, as
            the first response to drift.
          </p>
        </div>
      </details>

      {/* ---------- Decisions I'd defend ---------- */}
      <h2>Decisions I'd defend in a review</h2>
      <ul className="teaser-row">
        <li>Gated production on honest numbers only</li>
        <li>Measured stacking — and shipped without it</li>
        <li>Kept the tempting leaky features out</li>
      </ul>
      <Accordion summary="Read the full reasoning" hint="3 decisions, one paragraph each">
        <h3>I gated on honest numbers only</h3>
        <p>
          Leaderboard scores never gated a production decision. A finding from the research track
          has to be re-derived leakage-free before it counts — a rule I wrote down before any
          experiment ran.
        </p>
        <h3>I measured stacking — and shipped without it</h3>
        <p>
          The stacked meta-model scores {models ? models.meta_model.oof_mcc.toFixed(3) : "…"} OOF MCC
          versus {models ? models.dataset_h.oof_mcc.toFixed(3) : "…"} for its best base model.
          Stacking is regressive here, so the simpler model ships. Negative results get reported, not
          buried.
        </p>
        <h3>I kept the tempting features out</h3>
        <p>
          Record-adjacency features would have tripled the offline score. They also can't exist in a
          one-part-at-a-time scoring stream — so the production feature contract bans them, and the
          research track proves what that ban costs.
        </p>
      </Accordion>

      {/* ---------- Go deeper ---------- */}
      <h2>Go deeper</h2>
      <div className="cta-grid">
        <Link className="card cta-card" to="/decision-explorer">
          <h3>Decision Explorer →</h3>
          <p>Drag the dial yourself and watch the catch-rate / false-alarm trade-off respond.</p>
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
    </section>
  );
}
