import { useEffect, useState } from "react";
import { loadGovernance, loadRepoLinks } from "../lib/data";
import { githubHeadingSlug } from "../lib/slug";
import type { Governance, KaggleExperiment, KdrHeading, RepoLinks } from "../lib/types";

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

  return (
    <section>
      <h1>Governance &amp; Reproducibility</h1>
      <p className="lede">
        This project runs two independent tracks under a written governance log: a production
        decision system (what this dashboard is mostly about) and a fully quarantined Kaggle
        leaderboard-optimization track, pre-registered and frozen. Neither informs the other.
      </p>

      <div className="note note-warn">{leaderboard.program_status}</div>

      <h2>Kaggle track (Track 2) — pre-registered decision record</h2>
      <p>
        Every row below is copied verbatim from <code>{leaderboard.source_of_truth}</code> — the
        single source of truth for every result number quoted anywhere in this repository — and the
        export script asserts byte-for-byte fidelity against it.
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
            {leaderboard.experiments.map((e) => (
              <ExperimentRow key={e.experiment_id} e={e} repoUrl={repo_url} headingByKdr={headingByKdr} />
            ))}
          </tbody>
        </table>
      </div>

      <h3>Notes</h3>
      <ul className="plain">
        {leaderboard.notes.map((n, i) => (
          <li key={i}>{n}</li>
        ))}
      </ul>

      <h2>Decision Record log (KDR-001 – KDR-009)</h2>
      <p>
        The full pre-registration → evidence → decision record for the Kaggle track, in order.
        Anchors are computed client-side against the live heading text, so every link below tracks
        the document even if it's edited later.
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

      <h2>Reproducibility</h2>
      <p>
        The production models' honest OOF metrics (shown on Model Internals / Decision Explorer) are
        reproducible end-to-end from this repo: <code>prepare_data.py</code> →{" "}
        <code>build_dataset_{"{"}baseline,g,h{"}"}.py</code> → <code>train_*.py</code> →{" "}
        <code>train_meta_model.py</code>. See{" "}
        <a
          className="external-link"
          href={`${repo_url}/blob/main/docs/reproducible_metrics_report.md`}
          target="_blank"
          rel="noreferrer"
        >
          docs/reproducible_metrics_report.md
        </a>{" "}
        for exactly what is and isn't reproducible from currently committed code, including one
        historical result set that is explicitly <em>not</em> reproducible (the underlying training
        artifacts were deleted for repo size) and is therefore excluded from this dashboard entirely.
      </p>

      <h2>Repository &amp; tags</h2>
      <p>
        <a className="external-link" href={repo_url} target="_blank" rel="noreferrer">
          {repo_url.replace("https://github.com/", "")}
        </a>
      </p>
      {repoLinks && (
        <div className="card-grid">
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
    </section>
  );
}
