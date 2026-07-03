from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.decision_system import CostConfig, build_decision_table, summarize_operating_points

# local (default): read data/features + outputs/production directly from disk, no AWS
# credentials required. s3: read from the bucket named by AWS_BUCKET_NAME in .env -- see
# docs/runbooks/aws_s3.md. src.utils.s3_utils (which creates a boto3 client at import time) is
# only imported inside the two functions below that actually need it, so DATA_SOURCE=local never
# touches boto3/dotenv and never requires AWS credentials to be present.
DATA_SOURCE = os.getenv("DATA_SOURCE", "local").strip().lower()

MONITORING_JSON = ROOT / "outputs" / "monitoring" / "evidently_summary.json"
LOCAL_META_DATASET = ROOT / "data" / "features" / "meta_dataset.parquet"
LOCAL_OOF_PREDICTIONS_FINAL = ROOT / "data" / "features" / "oof_predictions_final.parquet"
LOCAL_PRODUCTION_GLOB = "outputs/production/*/cycle=*/batch=*/predictions.parquet"


def load_parquet_from_s3(key: str):
    from src.utils.s3_utils import BUCKET_NAME, s3  # lazy: only needed in DATA_SOURCE=s3 mode

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def _local_production_batch_paths() -> list[Path]:
    return sorted(ROOT.glob(LOCAL_PRODUCTION_GLOB))


@st.cache_data(ttl=60, show_spinner=False)
def load_monitoring_summary() -> dict | None:
    if not MONITORING_JSON.exists():
        return None
    with open(MONITORING_JSON) as f:
        data = json.load(f)
    mtime = os.path.getmtime(MONITORING_JSON)
    data["_file_mtime"] = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return data


# Track 3 (label-free production batch inference): cycle/batch-partitioned output written
# by scripts/pipeline/run_production_inference.py, never containing a Response column by construction.
PRODUCTION_PREFIX = "predictions/"
_PRODUCTION_KEY_RE = re.compile(r"^predictions/cycle=\d+/batch=\d+/predictions\.parquet$")


def list_production_batch_keys() -> list[str]:
    from src.utils.s3_utils import BUCKET_NAME, s3  # lazy: only needed in DATA_SOURCE=s3 mode

    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PRODUCTION_PREFIX):
        for obj in page.get("Contents", []):
            if _PRODUCTION_KEY_RE.match(obj["Key"]):
                keys.append(obj["Key"])
    return sorted(keys)


@st.cache_data(ttl=60, show_spinner=False)
def load_production_batches() -> pd.DataFrame:
    if DATA_SOURCE == "s3":
        keys = list_production_batch_keys()
        if not keys:
            return pd.DataFrame()
        frames = [load_parquet_from_s3(key) for key in keys]
    else:
        paths = _local_production_batch_paths()
        if not paths:
            return pd.DataFrame()
        frames = [pd.read_parquet(p) for p in paths]

    df = pd.concat(frames, ignore_index=True)

    if "Response" in df.columns:
        raise RuntimeError(
            "Track 3 production data unexpectedly contains a Response column -- this view "
            "is label-free by contract and refuses to render data that could be labeled."
        )

    return df.sort_values("run_seq", kind="mergesort").reset_index(drop=True)


COLOR_RECALL = "#2ca02c"
COLOR_PRECISION = "#1f77b4"
COLOR_COST = "#d62728"

st.set_page_config(page_title="Bosch Decision Dashboard", layout="wide")
st.title("Bosch Failure Decision System")
st.caption("Production-focused decision analytics for failure detection under inspection and cost constraints")
st.caption(
    "Note: every page below except \"Production Monitoring (Track 3)\" is an Offline "
    "Evaluation / Decision Analysis view over labeled OOF validation data, not live production scores."
)


@st.cache_data(show_spinner=False)
def load_scoring_data() -> pd.DataFrame:
    try:
        if DATA_SOURCE == "s3":
            meta = load_parquet_from_s3("data/features/meta_dataset.parquet")
            pred = load_parquet_from_s3("data/features/oof_predictions_final.parquet")
        else:
            if not (LOCAL_META_DATASET.exists() and LOCAL_OOF_PREDICTIONS_FINAL.exists()):
                st.error(
                    f"Local scoring data not found at {LOCAL_META_DATASET.relative_to(ROOT)} / "
                    f"{LOCAL_OOF_PREDICTIONS_FINAL.relative_to(ROOT)}. Run the training pipeline "
                    "first (see README Quickstart), or set DATA_SOURCE=s3 with AWS credentials in "
                    ".env to read from S3 instead."
                )
                st.stop()
            meta = pd.read_parquet(LOCAL_META_DATASET)
            pred = pd.read_parquet(LOCAL_OOF_PREDICTIONS_FINAL)
    except Exception as e:
        st.error(f"Data load failed (DATA_SOURCE={DATA_SOURCE!r}): {str(e)}")
        st.stop()

    df = meta[["Id", "Response"]].merge(
        pred[["Id", "oof_pred"]].rename(columns={"oof_pred": "pred"}),
        on="Id",
        how="left"
    )

    df["pred"] = df["pred"].fillna(0.0).astype(np.float32)
    df["Response"] = df["Response"].astype(np.int8)

    return df.sort_values("Id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def precompute_sorted_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    df = load_scoring_data()
    pred = df["pred"].to_numpy(dtype=np.float32, copy=False)
    y = df["Response"].to_numpy(dtype=np.int8, copy=False)

    order = np.argsort(-pred, kind="mergesort")
    pred_sorted = pred[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted, dtype=np.int64)
    fp_cum = np.cumsum(1 - y_sorted, dtype=np.int64)
    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)
    return pred_sorted, tp_cum, fp_cum, total_pos, total_neg


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, n: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if denom <= 0 else ((tp * tn) - (fp * fn)) / (denom ** 0.5)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
        "positives_pct": float((tp + fp) / n * 100.0),
    }


@st.cache_data(show_spinner=False)
def compute_threshold_sweep() -> pd.DataFrame:
    pred_sorted, tp_cum, fp_cum, total_pos, total_neg = precompute_sorted_arrays()
    n = len(pred_sorted)
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows: list[dict[str, float | int]] = []

    for thr in thresholds:
        k = int(np.searchsorted(-pred_sorted, -thr, side="right"))
        if k > 0:
            tp = int(tp_cum[k - 1])
            fp = int(fp_cum[k - 1])
        else:
            tp = 0
            fp = 0
        fn = total_pos - tp
        tn = total_neg - fp
        row = {"threshold": float(thr)}
        row.update(metrics_from_counts(tp, fp, fn, tn, n))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_budget_curve() -> pd.DataFrame:
    pred_sorted, tp_cum, fp_cum, total_pos, total_neg = precompute_sorted_arrays()
    n = len(pred_sorted)
    budgets = np.arange(1, 11, dtype=np.int32)
    rows: list[dict[str, float | int]] = []

    for budget_pct in budgets:
        k = int(np.ceil(n * (float(budget_pct) / 100.0)))
        k = min(max(k, 0), n)

        if k > 0:
            tp = int(tp_cum[k - 1])
            fp = int(fp_cum[k - 1])
            score_cutoff = float(pred_sorted[k - 1])
        else:
            tp = 0
            fp = 0
            score_cutoff = 1.0

        fn = total_pos - tp
        tn = total_neg - fp

        row = {
            "inspection_budget_pct": float(budget_pct),
            "selected_rows": int(k),
            "score_cutoff": score_cutoff,
        }
        row.update(metrics_from_counts(tp, fp, fn, tn, n))
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_fixed_precision_table() -> pd.DataFrame:
    sweep = compute_threshold_sweep()
    targets = np.round(np.arange(0.05, 0.55, 0.05), 2)
    rows = []
    for target in targets:
        eligible = sweep[sweep["precision"] >= float(target)]
        if eligible.empty:
            rows.append(
                {
                    "target_precision": float(target),
                    "available": False,
                    "threshold": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "mcc": np.nan,
                }
            )
            continue

        best = eligible.sort_values(["recall", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
        rows.append(
            {
                "target_precision": float(target),
                "available": True,
                "threshold": float(best["threshold"]),
                "recall": float(best["recall"]),
                "precision": float(best["precision"]),
                "mcc": float(best["mcc"]),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_live_decision_table(cost_fn: float, cost_fp: float) -> tuple[pd.DataFrame, dict[str, object]]:
    sweep = compute_threshold_sweep()
    budget = compute_budget_curve()
    n = len(load_scoring_data())

    tables = {
        "max_recall_threshold_sweep": sweep.copy(),
        "production_threshold_sweep": sweep.copy(),
        "inspection_budget_results": budget.copy(),
    }

    decision_df = build_decision_table(
        tables=tables,
        dataset_size=n,
        cost_cfg=CostConfig(cost_false_negative=float(cost_fn), cost_false_positive=float(cost_fp)),
    )
    summary = summarize_operating_points(decision_df)
    return decision_df, summary


@st.cache_data(show_spinner=False)
def compute_risk_group_table() -> pd.DataFrame:
    df = load_scoring_data().sort_values("pred", ascending=False, kind="mergesort").reset_index(drop=True)
    q_high = float(df["pred"].quantile(0.90))
    q_medium = float(df["pred"].quantile(0.70))

    df["risk_group"] = np.where(
        df["pred"] >= q_high,
        "HIGH",
        np.where(df["pred"] >= q_medium, "MEDIUM", "LOW"),
    )
    order_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    df["risk_rank"] = df["risk_group"].map(order_map).astype(np.int8)

    out = (
        df.groupby(["risk_rank", "risk_group"], as_index=False)
        .agg(
            number_of_parts=("Response", "size"),
            failure_rate_pct=("Response", lambda s: float(s.mean() * 100.0)),
            avg_risk_score=("pred", "mean"),
        )
        .sort_values("risk_rank")
        .drop(columns=["risk_rank"])
        .reset_index(drop=True)
    )
    return out


def confusion_label_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "True Positives (Correctly detected failures)",
                "False Positives (False alarms / unnecessary inspections)",
                "False Negatives (Missed failures / critical risk)",
                "True Negatives (Correctly passed parts)",
            ],
            "Number of Parts": [int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])],
        }
    )


def _budget_y_range(df: pd.DataFrame) -> list[float]:
    y = np.concatenate([
        df["recall"].to_numpy(dtype=np.float64),
        df["precision"].to_numpy(dtype=np.float64),
    ])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    pad = max(0.03, 0.1 * (y_max - y_min + 1e-9))
    return [max(0.0, y_min - pad), min(1.0, y_max + pad)]


nav = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Threshold Explorer",
        "Inspection Budget Simulator",
        "Recall at Fixed Precision",
        "Cost Simulator",
        "Model Insights",
        "Failure Analysis",
        "Production Monitoring (Track 3)",
    ],
)

if nav == "Overview":
    live_df = load_scoring_data()
    threshold_df = compute_threshold_sweep()
    budget_df = compute_budget_curve()
    fixed_precision_df = compute_fixed_precision_table()

    n = len(live_df)
    fail_rate = float(live_df["Response"].mean() * 100.0)
    best = threshold_df.sort_values("mcc", ascending=False).iloc[0]

    st.info("Live view of threshold behavior and business operating points using current prediction artifacts.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=threshold_df["threshold"], y=threshold_df["recall"], name="Recall", line={"color": COLOR_RECALL})
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_df["threshold"],
            y=threshold_df["precision"],
            name="Precision",
            line={"color": COLOR_PRECISION},
        )
    )
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Metric", xaxis_range=[0.01, 0.99])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n:,}")
    c2.metric("Failure Rate", f"{fail_rate:.3f}%")
    c3.metric("Best MCC (live sweep)", f"{best['mcc']:.4f}")
    c4.metric("Best Threshold", f"{best['threshold']:.2f}")

    _, live_summary = build_live_decision_table(cost_fn=100.0, cost_fp=5.0)
    min_cost = live_summary["minimum_cost_configuration"]
    b1, b2, b3 = st.columns(3)
    b1.metric("Min-Cost Threshold", f"{float(min_cost['threshold']):.2f}")
    b2.metric("Min-Cost Recall", f"{float(min_cost['recall']):.4f}")
    b3.metric("Min-Cost Precision", f"{float(min_cost['precision']):.4f}")

elif nav == "Threshold Explorer":
    threshold_df = compute_threshold_sweep()
    st.subheader("Threshold Explorer")
    st.info("Adjust the threshold and see how detection quality and missed-failure risk change.")

    thr = st.slider(
        "Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.23,
        step=0.01,
        help="Higher threshold triggers fewer inspections but can miss more true failures.",
    )
    row = threshold_df.loc[(threshold_df["threshold"] - thr).abs().idxmin()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["recall"], name="Recall", line={"color": COLOR_RECALL}))
    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["precision"], name="Precision", line={"color": COLOR_PRECISION}))
    fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df["mcc"], name="MCC", line={"color": "#7f7f7f"}))
    fig.add_vline(x=float(row["threshold"]), line_dash="dash", line_color="#444444")
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Metric", xaxis_range=[0.01, 0.99])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{row['recall']:.4f}")
    c2.metric("Precision", f"{row['precision']:.4f}")
    c3.metric("MCC", f"{row['mcc']:.4f}")
    c4.metric("Predicted Positives", f"{row['positives_pct']:.3f}%")

    st.dataframe(confusion_label_table(row), use_container_width=True, hide_index=True)

elif nav == "Inspection Budget Simulator":
    budget_df = compute_budget_curve()
    st.subheader("Inspection Budget Simulator")
    st.info(
        "Inspection budget is the percentage of highest-risk parts selected for manual inspection. "
        "Use this to match model policy to available inspection capacity."
    )

    budget = st.slider(
        "Inspection %",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Choose the share of highest-risk parts that will be inspected.",
    )
    row = budget_df.loc[(budget_df["inspection_budget_pct"] - budget).abs().idxmin()]

    y_range = _budget_y_range(budget_df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=budget_df["inspection_budget_pct"],
            y=budget_df["recall"],
            name="Recall",
            line={"color": COLOR_RECALL},
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=budget_df["inspection_budget_pct"],
            y=budget_df["precision"],
            name="Precision",
            line={"color": COLOR_PRECISION},
            mode="lines+markers",
        )
    )
    fig.update_layout(
        xaxis_title="Inspection Budget (%)",
        yaxis_title="Metric",
        yaxis_range=y_range,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        margin={"t": 50, "b": 40, "l": 40, "r": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", f"{row['recall']:.4f}")
    c2.metric("Precision", f"{row['precision']:.4f}")
    c3.metric("Flagged Rows", f"{int(row['selected_rows']):,}")

    st.dataframe(
        budget_df[["inspection_budget_pct", "selected_rows", "score_cutoff", "recall", "precision", "mcc"]],
        use_container_width=True,
    )

elif nav == "Recall at Fixed Precision":
    fixed_precision_df = compute_fixed_precision_table()
    st.subheader("Recall at Fixed Precision")
    st.info(
        "Precision shows how often alerts are correct. Recall shows how many failures are captured. "
        "Use this page when business rules require minimum alert quality."
    )

    ok = fixed_precision_df[fixed_precision_df["available"] == True]  # noqa: E712
    if not ok.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ok["target_precision"],
                y=ok["recall"],
                mode="lines+markers",
                line={"color": COLOR_RECALL},
                name="Recall",
            )
        )
        fig.update_layout(xaxis_title="Target Precision", yaxis_title="Best Achievable Recall")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fixed_precision_df, use_container_width=True)

elif nav == "Cost Simulator":
    live_df = load_scoring_data()
    threshold_df = compute_threshold_sweep()
    budget_df = compute_budget_curve()
    st.subheader("Cost Simulator")
    st.info("Set business costs for missed failures and false alarms, then find the lowest-cost operating threshold.")
    st.code("Total Cost = FN * cost_FN + FP * cost_FP")

    cost_fn = st.number_input(
        "Cost of False Negative",
        min_value=1,
        max_value=10000,
        value=100,
        step=1,
        help="Business impact of missing one true failure.",
    )
    cost_fp = st.number_input(
        "Cost of False Positive",
        min_value=1,
        max_value=1000,
        value=5,
        step=1,
        help="Business impact of one unnecessary inspection.",
    )

    decision_df, live_summary = build_live_decision_table(cost_fn=float(cost_fn), cost_fp=float(cost_fp))
    best = live_summary["minimum_cost_configuration"]

    cost_df = decision_df[(decision_df["mode"] == "threshold") & (decision_df["source"] == "production_threshold_sweep")].sort_values("threshold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_df["threshold"], y=cost_df["total_cost"], name="Total Cost", line={"color": COLOR_COST}))
    fig.add_trace(
        go.Scatter(
            x=[best["threshold"]],
            y=[best["total_cost"]],
            mode="markers",
            name="Optimal",
            marker={"size": 10, "color": "#000000", "symbol": "diamond"},
        )
    )
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Total Cost", xaxis_range=[0.01, 0.99])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Threshold", f"{best['threshold']:.2f}")
    c2.metric("Minimum Cost", f"{int(best['total_cost']):,}")
    c3.metric("Recall @ Optimum", f"{best['recall']:.4f}")

elif nav == "Model Insights":
    live_df = load_scoring_data()
    st.subheader("Model Insights")
    st.info("Prediction Confidence Distribution")
    st.caption("Most parts should have low risk scores; failures should appear in higher scores")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=live_df["pred"], nbinsx=100, marker_color=COLOR_PRECISION, name="Predictions"))
    fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Parts")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Risk groups are bucketed into LOW / MEDIUM / HIGH based on score quantiles.")
    risk_tbl = compute_risk_group_table().rename(
        columns={
            "risk_group": "Risk Group",
            "number_of_parts": "Number of Parts",
            "failure_rate_pct": "Failure Rate (%)",
            "avg_risk_score": "Avg Risk Score",
        }
    )
    st.dataframe(risk_tbl, use_container_width=True, hide_index=True)

elif nav == "Failure Analysis":
    live_df = load_scoring_data()
    st.subheader("Failure Analysis")
    st.info("This section analyzes MISSED FAILURES (false negatives)")
    st.warning("False negatives are critical because defective parts can pass without intervention.")

    thr = st.slider(
        "Failure Analysis Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.23,
        step=0.01,
        help="Threshold used to classify missed vs detected failures in this analysis.",
    )

    y_hat = (live_df["pred"].to_numpy() >= thr).astype(np.int8)
    y_true = live_df["Response"].to_numpy(dtype=np.int8, copy=False)
    fn_mask = (y_true == 1) & (y_hat == 0)
    tp_mask = (y_true == 1) & (y_hat == 1)

    fn_df = live_df.loc[fn_mask, ["Id", "pred"]]
    tp_df = live_df.loc[tp_mask, ["Id", "pred"]]

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_fn = go.Figure()
        fig_fn.add_trace(go.Histogram(x=fn_df["pred"], nbinsx=80, marker_color=COLOR_COST, name="False Negatives"))
        fig_fn.update_layout(title="False Negatives", xaxis_title="Risk Score", yaxis_title="Number of Parts")
        st.plotly_chart(fig_fn, use_container_width=True)

    with chart_right:
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Histogram(x=tp_df["pred"], nbinsx=80, marker_color=COLOR_RECALL, name="True Positives"))
        fig_tp.update_layout(title="True Positives", xaxis_title="Risk Score", yaxis_title="Number of Parts")
        st.plotly_chart(fig_tp, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Missed Failures (FN)", f"{len(fn_df):,}")
    c2.metric("Detected Failures (TP)", f"{len(tp_df):,}")
    c3.metric("FN Avg Risk Score", f"{fn_df['pred'].mean():.4f}" if len(fn_df) else "n/a")

    compare = pd.DataFrame(
        {
            "Segment": ["False Negatives", "True Positives"],
            "Number of Parts": [int(len(fn_df)), int(len(tp_df))],
            "Avg Risk Score": [
                float(fn_df["pred"].mean()) if len(fn_df) else 0.0,
                float(tp_df["pred"].mean()) if len(tp_df) else 0.0,
            ],
        }
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

elif nav == "Production Monitoring (Track 3)":
    st.subheader("Production Monitoring (Track 3)")
    if DATA_SOURCE == "s3":
        from src.utils.s3_utils import BUCKET_NAME  # lazy: only needed for this display string

        _production_source = f"s3://{BUCKET_NAME}/{PRODUCTION_PREFIX}cycle=*/batch=*/predictions.parquet"
    else:
        _production_source = LOCAL_PRODUCTION_GLOB
    st.info(
        "Label-free view of real, unlabeled Track 3 batch inference output "
        "(scripts/pipeline/run_production_inference.py), read directly from "
        f"{_production_source}. "
        "This page never shows MCC, precision, recall, accuracy, or a confusion matrix -- "
        "production batches are unlabeled by construction."
    )

    # --- Evidently Drift Monitoring ---
    st.markdown("### Evidently Drift Monitoring")
    mon = load_monitoring_summary()
    if mon is None:
        st.warning(
            "No monitoring output found. Run `scripts/pipeline/run_drift_monitoring.py` to generate "
            "`outputs/monitoring/evidently_summary.json`."
        )
    else:
        pred_drift = mon.get("summary", {}).get("prediction_drift", {})
        dataset_drift = mon.get("summary", {}).get("dataset_drift", {})

        pred_drift_detected: bool = bool(pred_drift.get("drift_detected", False))
        pred_drift_score: float | None = pred_drift.get("drift_score")
        drifted_cols: int = int(dataset_drift.get("drifted_columns_count", 0))
        drift_share: float | None = dataset_drift.get("drift_share")
        dataset_drift_detected: bool = drifted_cols > 0

        run_ts: str = mon.get("_file_mtime", "unknown")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Prediction Drift",
            "YES" if pred_drift_detected else "NO",
            delta=None,
        )
        m2.metric(
            "Prediction Drift Score",
            f"{pred_drift_score:.4f}" if pred_drift_score is not None else "n/a",
        )
        m3.metric(
            "Dataset Drift",
            "YES" if dataset_drift_detected else "NO",
        )
        m4.metric(
            "Drifted Columns",
            str(drifted_cols),
        )

        extra_cols = st.columns(2)
        if drift_share is not None:
            extra_cols[0].metric("Drift Share", f"{drift_share:.2%}")
        extra_cols[1].caption(f"Last monitoring run: {run_ts}")

        if pred_drift_detected or dataset_drift_detected:
            st.error(
                "Drift detected — review the Evidently report at "
                "`outputs/monitoring/evidently_report.html` for details."
            )
        else:
            st.success("No drift detected in the current monitoring window.")

    st.markdown("---")
    # --- End Evidently Drift Monitoring ---

    if st.button(f"🔄 Refresh ({DATA_SOURCE})"):
        load_production_batches.clear()
        st.rerun()

    prod_df = load_production_batches()

    if prod_df.empty:
        st.warning(
            f"No production batches found yet under {_production_source}. Run "
            "scripts/pipeline/run_production_inference.py to generate the first batch."
        )
    else:
        latest = prod_df.sort_values("run_seq").iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", f"{len(prod_df):,}")
        c2.metric("Latest Cycle", int(latest["cycle_id"]))
        c3.metric("Latest Batch", int(latest["batch_id"]))
        c4.metric("Latest run_seq", int(latest["run_seq"]))
        st.caption(f"Latest scored_at_utc: {latest['scored_at_utc']}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Flagged (decision=1)", f"{int(prod_df['decision'].sum()):,}")
        d2.metric("Auto-Reject", f"{int(prod_df['auto_reject'].sum()):,}")
        d3.metric("Manual Inspect", f"{int(prod_df['manual_inspect'].sum()):,}")

        st.subheader("Risk Score Distribution")
        fig_risk = go.Figure()
        fig_risk.add_trace(
            go.Histogram(x=prod_df["risk_score"], nbinsx=100, marker_color=COLOR_PRECISION, name="Risk Score")
        )
        fig_risk.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Parts")
        st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("Batch Growth")
        batch_growth = (
            prod_df.groupby(["run_seq", "cycle_id", "batch_id"], as_index=False)
            .agg(
                rows=("Id", "size"),
                flagged=("decision", "sum"),
                scored_at_utc=("scored_at_utc", "first"),
            )
            .sort_values("run_seq")
            .reset_index(drop=True)
        )
        batch_growth["cumulative_rows"] = batch_growth["rows"].cumsum()

        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Scatter(
                x=batch_growth["run_seq"],
                y=batch_growth["cumulative_rows"],
                mode="lines+markers",
                name="Cumulative Predictions",
                line={"color": COLOR_RECALL},
            )
        )
        fig_growth.update_layout(xaxis_title="run_seq", yaxis_title="Cumulative Predictions")
        st.plotly_chart(fig_growth, use_container_width=True)

        st.dataframe(batch_growth, use_container_width=True, hide_index=True)

        st.subheader("Top 100 Risky Parts")
        top_risky = (
            prod_df[["Id", "risk_score", "decision", "cycle_id", "batch_id", "scored_at_utc"]]
            .sort_values("risk_score", ascending=False)
            .head(100)
            .reset_index(drop=True)
        )
        st.dataframe(top_risky, use_container_width=True, hide_index=True)
