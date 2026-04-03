from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FEAT = ROOT / "data/features"

st.set_page_config(page_title="Bosch Decision Dashboard", layout="wide")
st.title("Bosch Failure Decision System")


def load_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def load_json(name: str) -> dict:
    path = OUT / name
    if not path.exists():
        st.error(f"Missing file: {path}")
        st.stop()
    return json.loads(path.read_text())


def confusion_from_row(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Actual Fail": [int(row["tp"]), int(row["fn"])],
            "Actual Pass": [int(row["fp"]), int(row["tn"])],
        },
        index=["Pred Fail", "Pred Pass"],
    )


nav = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Threshold Explorer",
        "Inspection Budget Simulator",
        "Cost Simulator",
        "Model Insights",
        "Failure Analysis",
    ],
)

threshold_df = load_csv("max_recall_threshold_sweep.csv")
budget_df = load_csv("inspection_budget_results.csv")
prod_df = load_csv("production_threshold_sweep.csv")
summary = load_json("max_recall_system_summary.json")

if nav == "Overview":
    meta = pd.read_parquet(FEAT / "meta_dataset.parquet", columns=["Response"])
    n = len(meta)
    fail_rate = float(meta["Response"].mean() * 100.0)
    best = threshold_df.sort_values("mcc", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n:,}")
    c2.metric("Failure Rate", f"{fail_rate:.3f}%")
    c3.metric("Best MCC (sweep)", f"{best['mcc']:.4f}")
    c4.metric("Best Threshold", f"{best['threshold']:.2f}")

    st.subheader("Current Max-Recall Recommendation")
    rec = summary["final_recommendation"]
    st.json(rec)

    st.subheader("Recall vs Precision (Threshold Sweep)")
    fig = px.line(threshold_df, x="threshold", y=["recall", "precision"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

elif nav == "Threshold Explorer":
    st.subheader("Threshold Explorer")
    thr = st.slider("Threshold", min_value=0.05, max_value=0.50, value=0.20, step=0.01)
    row = threshold_df.loc[(threshold_df["threshold"] - thr).abs().idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{row['recall']:.4f}")
    c2.metric("Precision", f"{row['precision']:.4f}")
    c3.metric("MCC", f"{row['mcc']:.4f}")
    c4.metric("Predicted Positives", f"{row['positives_pct']:.3f}%")

    st.dataframe(confusion_from_row(row))

    fig = px.line(threshold_df, x="threshold", y=["recall", "precision", "mcc"], markers=False)
    fig.add_vline(x=float(row["threshold"]), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

elif nav == "Inspection Budget Simulator":
    st.subheader("Inspection Budget Simulator")
    budget = st.slider("Inspection %", min_value=1, max_value=10, value=5, step=1)
    row = budget_df.loc[(budget_df["inspection_budget_pct"] - budget).abs().idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", f"{row['recall']:.4f}")
    c2.metric("Precision", f"{row['precision']:.4f}")
    c3.metric("Flagged Rows", f"{int(row['selected_rows']):,}")

    fig1 = px.line(budget_df, x="inspection_budget_pct", y="recall", markers=True, title="Recall vs Budget")
    fig2 = px.line(budget_df, x="inspection_budget_pct", y="precision", markers=True, title="Precision vs Budget")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

elif nav == "Cost Simulator":
    st.subheader("Cost Simulator")
    cost_fn = st.number_input("Cost of False Negative", min_value=1, max_value=10000, value=100, step=1)
    cost_fp = st.number_input("Cost of False Positive", min_value=1, max_value=1000, value=5, step=1)

    cost_df = prod_df.copy()
    cost_df["total_cost"] = cost_df["fn"] * cost_fn + cost_df["fp"] * cost_fp
    best = cost_df.sort_values("total_cost", ascending=True).iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Threshold", f"{best['threshold']:.2f}")
    c2.metric("Minimum Cost", f"{int(best['total_cost']):,}")
    c3.metric("Recall @ Optimum", f"{best['recall']:.4f}")

    fig = px.line(cost_df, x="threshold", y="total_cost", title="Total Cost vs Threshold")
    fig.add_vline(x=float(best["threshold"]), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

elif nav == "Model Insights":
    st.subheader("Model Insights")
    fi_path = OUT / "meta_v3_dataset_h_feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(20)
        fig = px.bar(fi.iloc[::-1], x="importance_gain", y="feature", orientation="h", title="Top 20 Features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance file not found.")

    pred_files = [
        ("context_meta_v2_blend", FEAT / "oof_predictions_context_meta_v2_blend.parquet"),
        ("meta_v3_dataset_h_blend", FEAT / "oof_predictions_meta_v3_dataset_h_blend.parquet"),
    ]
    rows = []
    for name, path in pred_files:
        if path.exists():
            p = pd.read_parquet(path)["oof_pred"]
            rows.append(pd.DataFrame({"source": name, "prediction": p}))
    if rows:
        dist = pd.concat(rows, ignore_index=True)
        fig = px.histogram(dist, x="prediction", color="source", nbins=100, barmode="overlay", opacity=0.55)
        st.plotly_chart(fig, use_container_width=True)

elif nav == "Failure Analysis":
    st.subheader("Failure Analysis")
    fn_path = OUT / "false_negatives.parquet"
    tpfn_path = OUT / "fn_vs_tp_comparison.csv"

    if fn_path.exists():
        fn = pd.read_parquet(fn_path)
        c1, c2 = st.columns(2)
        c1.metric("False Negatives", f"{len(fn):,}")
        c2.metric("FN Mean Score", f"{fn['pred'].mean():.4f}")

        fig1 = px.histogram(fn, x="pred", nbins=80, title="FN Prediction Score Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        for col in ["path_failure_rate", "dataset_g_pred", "dataset_h_pred"]:
            if col in fn.columns:
                fig = px.histogram(fn, x=col, nbins=60, title=f"FN Distribution: {col}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run failure analysis first to generate false_negatives.parquet")

    if tpfn_path.exists():
        st.subheader("FN vs TP Comparison")
        st.dataframe(pd.read_csv(tpfn_path), use_container_width=True)
