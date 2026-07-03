"""Export a static JSON data bundle for the recruiter dashboard (PF4 of
docs/implementation/portfolio_master_plan.md).

This script reads only artifacts that are reproducible from the code currently
committed to this repository -- the four production models' honest OOF
predictions (``data/features/oof_predictions_{baseline,dataset_g,dataset_h,
final}.parquet``, produced by ``scripts/pipeline/train_*.py``) and their
``outputs/training_summary.json`` / ``outputs/feature_importance_*.csv``
companions -- plus the frozen, hand-authored ``results/leaderboard.json``
(Kaggle Track 2, quarantined) and ``outputs/e3_rolling_origin_results.json``
(the RP2 honest deployable-MCC headline already quoted in README.md and
docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md).

It deliberately does NOT read ``outputs/production_decision_summary.json`` or
its sibling threshold-sweep CSVs: per
docs/reproducible_metrics_report.md Section 2 ("World B"), those were computed
from ``data/features/oof_predictions_context_meta_v2_blend.parquet``, whose
generating artifacts were deliberately deleted from this repo and which no
committed script reproduces. Feeding non-reproducible numbers into a dashboard
that markets itself on reproducibility would violate standing rule 5 ("honesty
semantics survive all reframing").

Output is committed to ``dashboard/public/data/*.json`` (small, tens of KB
each) -- never the source parquets, per the PF4 deliverables list and frozen
technical decision #3 (CI has no access to gitignored parquets, so the bundle
must be committed, not CI-built).

Usage: ``python scripts/ops/export_dashboard_data.py`` (run from repo root or
anywhere -- paths are resolved from this file's location).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # noqa: E402 -- see scripts/pipeline/* for the same standalone-invocation pattern

from src.inference.decision_engine import metrics_from_labels  # noqa: E402

OUT_DIR = ROOT / "dashboard" / "public" / "data"
LEADERBOARD_PATH = ROOT / "results" / "leaderboard.json"
TRAINING_SUMMARY_PATH = ROOT / "outputs" / "training_summary.json"
E3_RESULTS_PATH = ROOT / "outputs" / "e3_rolling_origin_results.json"
KAGGLE_DECISIONS_PATH = ROOT / "docs" / "research" / "kaggle_decisions.md"

REPO_URL = "https://github.com/anudeepreddy332/bosch-production-line-defect-analysis"

N_THRESHOLD_POINTS = 201  # ~200 grid points, per PF4 deliverables list
N_CALIBRATION_BINS = 20

# model_key -> (oof parquet filename, feature-importance CSV filename, display label)
MODELS: dict[str, dict[str, str]] = {
    "baseline": {
        "oof_parquet": "oof_predictions_baseline.parquet",
        "importance_csv": "feature_importance_baseline.csv",
        "label": "Baseline",
    },
    "dataset_g": {
        "oof_parquet": "oof_predictions_dataset_g.parquet",
        "importance_csv": "feature_importance_dataset_g.csv",
        "label": "Dataset G (target-rate features)",
    },
    "dataset_h": {
        "oof_parquet": "oof_predictions_dataset_h.parquet",
        "importance_csv": "feature_importance_dataset_h.csv",
        "label": "Dataset H (transition + co-occurrence features)",
    },
    "meta_model": {
        "oof_parquet": "oof_predictions_final.parquet",
        "importance_csv": "feature_importance_meta_model.csv",
        "label": "Meta-Model (stacked)",
    },
}

# Feature-name -> family tag, for the Model Internals importances chart. Matched by
# exact name first, then by prefix -- covers every column that appears in any of
# the four feature_importance_*.csv files above.
_FAMILY_EXACT = {
    "start_time": "structural",
    "duration": "structural",
    "feature_mean": "structural",
    "chunk_id": "structural",
    "chunk_size": "structural",
    "records_last_1hr": "rolling-window",
    "records_last_24hr": "rolling-window",
    "density_ratio": "rolling-window",
    "path_count": "path/target-rate",
    "chunk_failure_rate": "path/target-rate",
    "path_failure_rate": "path/target-rate",
    "station_risk_mean": "path/target-rate",
    "baseline_pred": "meta-stack",
    "dataset_g_pred": "meta-stack",
    "dataset_h_pred": "meta-stack",
    "mean_prediction": "meta-stack",
    "std_prediction": "meta-stack",
    "max_prediction": "meta-stack",
    "agreement_count": "meta-stack",
}
_FAMILY_PREFIX = [
    ("transition_fail_rate", "transition/co-occurrence"),
    ("pair_cooccur", "transition/co-occurrence"),
    ("signature", "path/target-rate"),
]


def family_tag(feature_name: str) -> str:
    if feature_name in _FAMILY_EXACT:
        return _FAMILY_EXACT[feature_name]
    for prefix, tag in _FAMILY_PREFIX:
        if feature_name.startswith(prefix):
            return tag
    return "other"


def build_threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    thresholds = np.linspace(0.0, 1.0, N_THRESHOLD_POINTS)
    rows = []
    for t in thresholds:
        y_hat = (y_score >= t).astype(np.int8)
        m = metrics_from_labels(y_true, y_hat)
        mcc = matthews_corrcoef(y_true, y_hat) if 0 < (m["tp"] + m["fp"]) < len(y_true) else 0.0
        fpr = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else 0.0
        rows.append(
            {
                "threshold": round(float(t), 4),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
                "precision": round(m["precision"], 6),
                "recall": round(m["recall"], 6),
                "fpr": round(fpr, 6),
                "mcc": round(float(mcc), 6),
                "flagged_pct": round(m["flagged_pct"], 6),
            }
        )
    return rows


def build_calibration(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bin_idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, N_CALIBRATION_BINS - 1)
    rows = []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin_lo": round(float(edges[b]), 3),
                "bin_hi": round(float(edges[b + 1]), 3),
                "mean_predicted": round(float(y_score[mask].mean()), 6),
                "observed_rate": round(float(y_true[mask].mean()), 6),
                "count": n,
            }
        )
    return rows


def extract_kdr_headings(text: str) -> list[dict]:
    """Parse '## KDR-00N — <title>' headings; slugging happens client-side
    (github-slugger) so the anchor always matches whatever GitHub renders,
    even if headings are edited after this export runs."""
    pattern = re.compile(r"^## (KDR-\d+) — (.+)$", re.MULTILINE)
    return [{"kdr": m.group(1), "heading": f"{m.group(1)} — {m.group(2)}"} for m in pattern.finditer(text)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_summary = json.loads(TRAINING_SUMMARY_PATH.read_text())["models"]
    leaderboard = json.loads(LEADERBOARD_PATH.read_text())
    e3_results = json.loads(E3_RESULTS_PATH.read_text())

    models_meta: dict[str, dict] = {}
    importances: dict[str, list[dict]] = {}
    calibration: dict[str, list[dict]] = {}

    for key, spec in MODELS.items():
        summary = training_summary[key]
        oof_path = ROOT / "data" / "features" / spec["oof_parquet"]
        df = pd.read_parquet(oof_path, columns=["Response", "oof_pred"])
        y_true = df["Response"].to_numpy()
        y_score = df["oof_pred"].to_numpy()

        best_threshold = float(summary["best_threshold"])
        y_hat_best = (y_score >= best_threshold).astype(np.int8)
        recomputed_mcc = float(matthews_corrcoef(y_true, y_hat_best))
        stored_mcc = float(summary["oof_mcc"])
        assert abs(recomputed_mcc - stored_mcc) < 1e-6, (
            f"{key}: recomputed OOF MCC {recomputed_mcc} at stored best_threshold "
            f"{best_threshold} does not match outputs/training_summary.json's "
            f"stored oof_mcc {stored_mcc} -- data drift between the OOF parquet "
            "and the training summary. Rerun the training pipeline before exporting."
        )

        sweep = build_threshold_sweep(y_true, y_score)
        (OUT_DIR / f"sweep_{key}.json").write_text(json.dumps(sweep))

        calibration[key] = build_calibration(y_true, y_score)

        importance_path = ROOT / "outputs" / spec["importance_csv"]
        imp_df = pd.read_csv(importance_path).sort_values("importance", ascending=False)
        total = float(imp_df["importance"].sum())
        top = imp_df.head(25)
        importances[key] = [
            {
                "feature": row.feature,
                "importance": float(row.importance),
                "importance_pct": round(float(row.importance) / total * 100.0, 3) if total else 0.0,
                "family": family_tag(row.feature),
            }
            for row in top.itertuples(index=False)
        ]

        models_meta[key] = {
            "label": spec["label"],
            "rows": int(summary["rows"]),
            "feature_count": len(summary["features"]),
            "best_threshold": best_threshold,
            "oof_mcc": stored_mcc,
            "data_fingerprint": summary.get("data_fingerprint"),
            "fold_metrics": summary["fold_metrics"],
        }
        if key == "meta_model":
            models_meta[key]["base_thresholds"] = summary.get("base_thresholds")

    (OUT_DIR / "models.json").write_text(json.dumps(models_meta, indent=2))
    (OUT_DIR / "importances.json").write_text(json.dumps(importances, indent=2))
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))

    # Governance page: Kaggle Track 2 (frozen) results, copied verbatim from the
    # single source of truth. Assert byte-for-byte fidelity so the dashboard can
    # never silently drift from results/leaderboard.json.
    kdr_headings = extract_kdr_headings(KAGGLE_DECISIONS_PATH.read_text())
    governance = {
        "repo_url": REPO_URL,
        "leaderboard": leaderboard,
        "kdr_headings": kdr_headings,
        "kaggle_decisions_path": "docs/research/kaggle_decisions.md",
    }
    (OUT_DIR / "governance.json").write_text(json.dumps(governance, indent=2))
    reloaded = json.loads((OUT_DIR / "governance.json").read_text())
    assert reloaded["leaderboard"] == leaderboard, (
        "Governance export drifted from results/leaderboard.json -- the single source of truth."
    )

    # Story page hero stat: the RP2 honest rolling-origin headline, exactly the
    # numbers already quoted in README.md / CASE_STUDY (traceable, reproducible
    # via `python scripts/research/train_e3_rolling_origin.py`).
    rp2_summary = {
        "reproduce": e3_results["reproduce"],
        "cross_origin_summary": e3_results["cross_origin_summary"],
        "fold_results": [
            {
                "fold_idx": f["fold_idx"],
                "test_chunks": f["test_chunks"],
                "test_pos_rate": f["test_pos_rate"],
                "oot_mcc_best_threshold": f["oot_mcc_best_threshold"],
                "oot_best_threshold": f["oot_best_threshold"],
                "oot_mcc_fixed_threshold": f["oot_mcc_fixed_threshold"],
            }
            for f in e3_results["fold_results"]
        ],
        "incv_mcc": e3_results["reference_numbers"]["incv_mcc"],
        "incv_threshold": e3_results["reference_numbers"]["incv_threshold"],
    }
    (OUT_DIR / "rp2_summary.json").write_text(json.dumps(rp2_summary, indent=2))

    tags = [
        "K1-result", "K2-result", "K3-result", "K4-result", "K5-result",
        "P0-result", "P1-result", "track1-frozen", "track2-frozen", "track3-frozen",
    ]
    (OUT_DIR / "repo_links.json").write_text(
        json.dumps({"repo_url": REPO_URL, "tags": tags}, indent=2)
    )

    print(f"Wrote {len(list(OUT_DIR.glob('*.json')))} JSON files to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
