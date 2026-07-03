import { useEffect, useState } from "react";
import { loadGovernance, loadRepoLinks } from "../lib/data";
import { githubHeadingSlug } from "../lib/slug";
import type { Governance, KaggleExperiment, KdrHeading, RepoLinks } from "../lib/types";
import StatCard from "../components/StatCard";
import Callout from "../components/Callout";
import Accordion from "../components/Accordion";
import ProvenancePopover, { type ProvenanceLink } from "../components/ProvenancePopover";

function ExperimentRow({
  e,
  repoUrl,
  headingByKdr,
}: {
  e: KaggleExperiment;
  repoUrl: string;
  headingByKdr: Map<string, string>;
}) {
  const heading = headingByKdr.get(e.kdr);
  const href = heading
    ? `${repoUrl}/blob/main/${e.kdr_file}#${githubHeadingSlug(heading)}`
    : `${repoUrl}/blob/main/${e.kdr_file}`;
  return (
    <tr>
      <td>{e.experiment_id}</td>
      <td>
        <a className="external-link" href={href} target="_blank" rel="noreferrer">
          {e.kdr}
        </a>
      </td>
      <td>{e.date}</td>
      <td>
        <span className={e.oof_status === "honest" ? "badge badge-honest" : "badge badge-contaminated"}>
          {e.oof_status}
        </span>
      </td>
      <td>{e.oof_mcc.toFixed(5)}</td>
      <td>{e.public_mcc.toFixed(5)}</td>
      <td>{e.private_mcc.toFixed(5)}</td>
      <td>
        <a className="external-link" href={`${repoUrl}/tree/${e.git_tag}`} target="_blank" rel="noreferrer">
          {e.git_tag}
        </a>
      </td>
    </tr>
  );
}

export default function GovernancePage() {
  const [gov, setGov] = useState<Governance | null>(null);
  const [repoLinks, setRepoLinks] = useState<RepoLinks | null>(null);

  useEffect(() => {
    loadGovernance().then(setGov);
    loadRepoLinks().then(setRepoLinks);
  }, []);

  if (!gov) {
    return (
      <section>
        <h1>Governance &amp; Reproducibility</h1>
        <p>Loading…</p>
      </section>
    );
  }

  const { leaderboard, kdr_headings, repo_url } = gov;
  const headingByKdr = new Map<string, string>(kdr_headings.map((k: KdrHeading) => [k.kdr, k.heading]));
  const experiments = leaderboard.experiments;
  const honestCount = experiments.filter((e) => e.oof_status === "honest").length;
  const best = experiments.reduce((a, b) => (b.private_mcc > a.private_mcc ? b : a), experiments[0]);
  const k1 = experiments.find((e) => e.experiment_id === "K1");

  const sourceHref = `${repo_url}/blob/main/${leaderboard.source_of_truth}`;
  const bestHeading = headingByKdr.get(best.kdr);
  const bestChain: ProvenanceLink[] = [
    { label: "leaderboard.json", href: sourceHref },
    {
      label: best.kdr,
      href: bestHeading ? `${repo_url}/blob/main/${best.kdr_file}#${githubHeadingSlug(bestHeading)}` : `${repo_url}/blob/main/${best.kdr_file}`,
    },
    { label: best.git_tag, href: `${repo_url}/tree/${best.git_tag}` },
  ];
  const sourceOnlyChain: ProvenanceLink[] = [{ label: "leaderboard.json", href: sourceHref }];

  return (
    <section>
      <h1>Governance &amp; Reproducibility</h1>
      <p className="lede">Every number here traces to written-down, pre-registered evidence — hover any headline figure to see its chain.</p>

      <div className="card-grid card-grid-tight">
        <ProvenancePopover chain={sourceOnlyChain} as="block">
          <StatCard value={experiments.length} label="Sealed experiments — hypothesis fixed before results" />
        </ProvenancePopover>
        <ProvenancePopover chain={sourceOnlyChain} as="block">
          <StatCard
            value={kdr_headings.length}
            label={`Decision records (${kdr_headings[0]?.kdr} … ${kdr_headings[kdr_headings.length - 1]?.kdr})`}
          />
        </ProvenancePopover>
        <ProvenancePopover chain={sourceOnlyChain} as="block">
          <StatCard
            value={`${honestCount} / ${experiments.length - honestCount}`}
            label="Honest vs. label-contaminated measurements — always labeled, never mixed"
          />
        </ProvenancePopover>
        <ProvenancePopover chain={bestChain} as="block">
          <StatCard
            value={best.private_mcc.toFixed(5)}
            label={`Private MCC the program froze at${k1 ? ` — up from ${k1.private_mcc.toFixed(5)} at baseline` : ""}`}
          />
        </ProvenancePopover>
      </div>

      <Callout tone="warn">{leaderboard.program_status}</Callout>

      <Accordion
        summary="The full experiment ladder"
        hint={`${experiments.length} rows, every value traceable`}
        defaultOpen
      >
        <p className="chart-hint">
          “Honest” = label-free measurement, valid as a real quality estimate. “Contaminated” =
          leaks label information across CV folds; only its leaderboard score is meaningful. I
          never compare one to the other.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Experiment</th>
                <th>KDR</th>
                <th>Date</th>
                <th>OOF status</th>
                <th>OOF MCC</th>
                <th>Public LB</th>
                <th>Private LB</th>
                <th>Tag</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <ExperimentRow key={e.experiment_id} e={e} repoUrl={repo_url} headingByKdr={headingByKdr} />
              ))}
            </tbody>
          </table>
        </div>
      </Accordion>

      <Accordion
        summary={`Decision record log (${kdr_headings[0]?.kdr} – ${kdr_headings[kdr_headings.length - 1]?.kdr})`}
        hint="pre-registration → evidence → decision"
      >
        <p className="chart-hint">
          Each record fixes the hypothesis, success criteria, and contamination safeguards before
          any result existed. Anchors are computed client-side against the live heading text, so
          these links keep working even if the document grows.
        </p>
        <ul className="plain">
          {kdr_headings.map((k) => (
            <li key={k.kdr}>
              <a
                className="external-link"
                href={`${repo_url}/blob/main/${gov.kaggle_decisions_path}#${githubHeadingSlug(k.heading)}`}
                target="_blank"
                rel="noreferrer"
              >
                {k.heading}
              </a>
            </li>
          ))}
        </ul>
      </Accordion>

      <Accordion summary="Metric definitions & contamination rules" hint="how I keep honest and leaky numbers apart">
        <ul className="plain">
          {leaderboard.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      </Accordion>

      <Accordion summary="Reproducibility" hint="what you can regenerate from a clean clone">
        <p>
          The production models' honest OOF metrics (shown on Model Internals and the Decision
          Explorer) are reproducible end-to-end from the repo:{" "}
          <code>prepare_data.py</code> → <code>build_dataset_{"{"}baseline,g,h{"}"}.py</code> →{" "}
          <code>train_*.py</code> → <code>train_meta_model.py</code>. Each trained model records a
          data fingerprint so a rerun can prove it saw identical data. See{" "}
          <a
            className="external-link"
            href={`${repo_url}/blob/main/docs/reproducible_metrics_report.md`}
            target="_blank"
            rel="noreferrer"
          >
            docs/reproducible_metrics_report.md
          </a>{" "}
          for exactly what is and isn't reproducible from committed code — including one
          historical result set that explicitly is <em>not</em> (its training artifacts were
          deleted for repo size), which is why I excluded it from this dashboard entirely.
        </p>
        <p>
          The full documentation site — architecture, model &amp; data cards, research summary,
          runbooks — lives at{" "}
          <a className="external-link" href="/docs/" target="_blank" rel="noreferrer">
            /docs/
          </a>
          .
        </p>
      </Accordion>

      <Accordion summary="Repository & evidence tags" hint={repoLinks ? `${repoLinks.tags.length} sealed tags` : "…"}>
        <p>
          <a className="external-link" href={repo_url} target="_blank" rel="noreferrer">
            {repo_url.replace("https://github.com/", "")}
          </a>{" "}
          — every experiment's exact code state is sealed under an annotated tag:
        </p>
        {repoLinks && (
          <div className="card-grid card-grid-tight">
            {repoLinks.tags.map((tag) => (
              <a
                key={tag}
                className="badge badge-frozen"
                style={{ textAlign: "center" }}
                href={`${repoLinks.repo_url}/tree/${tag}`}
                target="_blank"
                rel="noreferrer"
              >
                {tag}
              </a>
            ))}
          </div>
        )}
      </Accordion>
    </section>
  );
}
