from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset


ID_COLUMNS = {"id", "row_id", "index"}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c.lower() not in ID_COLUMNS]
    return df[keep].copy()


def _extract_summary(report_dict: dict) -> dict[str, object]:
    metrics = report_dict.get("metrics", [])

    dataset_drift = {"drifted_columns_count": None, "number_of_columns": None, "drift_share": None}
    drifted_columns = []
    pred_drift = None

    def _is_drift(metric_name: str, metric_value: float, metric_threshold: float) -> bool:
        name = metric_name.lower()
        if "p_value" in name:
            return metric_value < metric_threshold
        return metric_value > metric_threshold

    for metric in metrics:
        cfg = metric.get("config", {})
        mtype = str(cfg.get("type", ""))
        mname = str(metric.get("metric_name", ""))
        value = metric.get("value")
        threshold = cfg.get("threshold", 0.05)

        if mtype.endswith("DriftedColumnsCount"):
            count_value = value.get("count") if isinstance(value, dict) else value
            share_value = value.get("share") if isinstance(value, dict) else None
            dataset_drift = {
                "drifted_columns_count": int(count_value) if count_value is not None else None,
                "number_of_columns": len([m for m in metrics if str(m.get("metric_name", "")).startswith("ValueDrift(column=")]),
                "drift_share": float(share_value) if share_value is not None else None,
            }

        if mtype.endswith("ValueDrift"):
            column = cfg.get("column")
            if column == "pred":
                drift_detected = bool(
                    value is not None
                    and _is_drift(
                        metric_name=mname,
                        metric_value=float(value),
                        metric_threshold=float(threshold),
                    )
                )
                pred_drift = {
                    "column": "pred",
                    "drift_detected": drift_detected,
                    "drift_score": float(value) if value is not None else None,
                    "threshold": float(threshold),
                }
            elif column is not None and column.lower() not in ID_COLUMNS:
                is_drift = bool(
                    value is not None
                    and _is_drift(
                        metric_name=mname,
                        metric_value=float(value),
                        metric_threshold=float(threshold),
                    )
                )
                if is_drift:
                    drifted_columns.append(
                        {
                            "column": str(column),
                            "drift_score": float(value),
                            "threshold": float(threshold),
                            "metric_name": mname,
                        }
                    )

    if dataset_drift["drifted_columns_count"] is None:
        dataset_drift["drifted_columns_count"] = len(drifted_columns)
    if dataset_drift["number_of_columns"] in (None, 0):
        dataset_drift["number_of_columns"] = max(len(drifted_columns), 1)
    if dataset_drift["drift_share"] is None:
        dataset_drift["drift_share"] = float(dataset_drift["drifted_columns_count"] / dataset_drift["number_of_columns"])

    return {
        "dataset_drift": dataset_drift,
        "prediction_drift": pred_drift,
        "drifted_columns": sorted(drifted_columns, key=lambda x: x.get("drift_score", 0.0)),
    }


def generate_evidently_report(
    reference_path: Path,
    current_path: Path,
    output_json: Path,
    output_html: Path,
) -> dict[str, object]:
    reference = _clean_columns(pd.read_parquet(reference_path))
    current = _clean_columns(pd.read_parquet(current_path))

    report = Report(
        metrics=[
            DataDriftPreset(),
            DriftedColumnsCount(),
            ValueDrift(column="pred"),
        ]
    )
    snapshot = report.run(reference_data=reference, current_data=current)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_html))

    raw = snapshot.dict()
    summary = {
        "engine": "evidently",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "excluded_columns": sorted(ID_COLUMNS),
        "summary": _extract_summary(raw),
    }
    output_json.write_text(json.dumps(summary, indent=2))
    return summary
