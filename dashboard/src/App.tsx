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
  { to: "/governance", label: "Governance & Reproducibility" },
];

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
        <span>
          Client-side only. No data leaves your browser. Source:{" "}
          <a href="https://github.com/anudeepreddy332/bosch-production-line-defect-analysis" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </span>
      </footer>
    </div>
  );
}
