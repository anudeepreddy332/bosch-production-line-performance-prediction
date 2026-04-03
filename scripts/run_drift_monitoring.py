from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_detection import generate_evidently_report


OUT = ROOT / "outputs"
FEAT = ROOT / "data/features"
MON = OUT / "monitoring"


def build_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_parquet(FEAT / "meta_dataset.parquet", columns=["Id", "Response", "dataset_g_pred", "dataset_h_pred"])
    p = pd.read_parquet(FEAT / "oof_predictions_context_meta_v2_blend.parquet")[["Id", "oof_pred"]]
    df = meta.merge(p.rename(columns={"oof_pred": "pred"}), on="Id", how="left", validate="one_to_one")
    # Stable random split keeps reference/current comparable and avoids ID-order artifacts.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.7)
    ref = df.iloc[:split].copy()
    cur = df.iloc[split:].copy()

    return ref, cur


if __name__ == "__main__":
    MON.mkdir(parents=True, exist_ok=True)
    ref, cur = build_reference_current()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path = tmp_dir / "reference.parquet"
        cur_path = tmp_dir / "current.parquet"
        ref.to_parquet(ref_path, index=False)
        cur.to_parquet(cur_path, index=False)
        report = generate_evidently_report(
            reference_path=ref_path,
            current_path=cur_path,
            output_json=MON / "evidently_summary.json",
            output_html=MON / "evidently_report.html",
        )
    print("Drift engine:", report.get("engine"))
    print("Saved:", MON / "evidently_summary.json")
