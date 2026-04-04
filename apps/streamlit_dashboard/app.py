from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.decision_system import CostConfig, build_decision_table, summarize_operating_points
from io import BytesIO

AWS_BUCKET = "bosch-ml-production-anudeep-193116635897-ap-south-2-an"
AWS_REGION = "ap-south-2"

s3 = boto3.client("s3", region_name=AWS_REGION)


def load_parquet_from_s3(key: str):
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


COLOR_RECALL = "#2ca02c"
COLOR_PRECISION = "#1f77b4"
COLOR_COST = "#d62728"

st.set_page_config(page_title="Bosch Decision Dashboard", layout="wide")
st.title("Bosch Failure Decision System")
st.caption("Production-focused decision analytics for failure detection under inspection and cost constraints")


@st.cache_data(show_spinner=False)
def load_scoring_data() -> pd.DataFrame:
    try:
        meta = load_parquet_from_s3("data/features/meta_dataset.parquet")
        pred = load_parquet_from_s3("data/features/oof_predictions_final.parquet")
    except Exception as e:
        st.error(f"S3 Load Failed: {str(e)}")
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
