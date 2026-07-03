import { Suspense, lazy } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

const Story = lazy(() => import("./pages/Story"));
const DecisionExplorer = lazy(() => import("./pages/DecisionExplorer"));
const ModelInternals = lazy(() => import("./pages/ModelInternals"));
const GovernancePage = lazy(() => import("./pages/Governance"));

const NAV_ITEMS = [
  { to: "/", label: "Story", end: true },
  { to: "/decision-explorer", label: "Decision Explorer" },
  { to: "/model-internals", label: "Model Internals" },
  { to: "/governance", label: "Governance" },
];

const REPO_URL = "https://github.com/anudeepreddy332/bosch-production-line-defect-analysis";

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <span className="app-title">Bosch Production Line Decision System</span>
          <nav className="app-nav">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? "nav-link nav-link-active" : "nav-link")}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="app-main">
        <Suspense fallback={<div className="page-loading">Loading…</div>}>
          <Routes>
            <Route path="/" element={<Story />} />
            <Route path="/decision-explorer" element={<DecisionExplorer />} />
            <Route path="/model-internals" element={<ModelInternals />} />
            <Route path="/governance" element={<GovernancePage />} />
          </Routes>
        </Suspense>
      </main>
      <footer className="app-footer">
        <span className="footer-links">
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a href="/docs/" target="_blank" rel="noreferrer">
            Documentation
          </a>
          <a href={`${REPO_URL}/blob/main/docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md`} target="_blank" rel="noreferrer">
            Case study
          </a>
          <a href={`${REPO_URL}/releases/tag/v1.0.0`} target="_blank" rel="noreferrer">
            v1.0.0 release
          </a>
          <a
            href={`${REPO_URL}/blob/main/scripts/ops/export_dashboard_data.py`}
            target="_blank"
            rel="noreferrer"
            title="Every number on this dashboard is exported directly from committed pipeline artifacts by this script — nothing here is hand-typed."
          >
            ⌁ traceable
          </a>
        </span>
        <span>
          Built by Anudeep Reddy. Client-side only — no data leaves your browser.
        </span>
      </footer>
    </div>
  );
}
