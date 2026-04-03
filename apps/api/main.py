from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.decision_engine import DecisionPolicy, apply_hybrid

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs/max_recall_system_summary.json"


def _load_default_policy() -> DecisionPolicy:
    if not SUMMARY_PATH.exists():
        return DecisionPolicy()
    payload = json.loads(SUMMARY_PATH.read_text())
    rec = payload.get("final_recommendation", {})
    return DecisionPolicy(
        threshold_high=float(rec.get("threshold_high", 0.60)),
        inspection_budget_pct=float(rec.get("inspection_budget_pct", 5.0)),
    )


DEFAULT_POLICY = _load_default_policy()
app = FastAPI(title="Bosch Failure Decision API", version="1.0.0")


class PredictRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    threshold_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    scores: list[float] = Field(..., min_length=1)
    threshold_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    inspection_budget_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, float | int | str]:
    thr = req.threshold_high if req.threshold_high is not None else DEFAULT_POLICY.threshold_high
    decision = int(req.score >= thr)
    return {
        "score": float(req.score),
        "threshold_high": float(thr),
        "decision": decision,
        "action": "auto_reject" if decision == 1 else "pass",
    }


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest) -> dict[str, object]:
    policy = DecisionPolicy(
        threshold_high=req.threshold_high if req.threshold_high is not None else DEFAULT_POLICY.threshold_high,
        inspection_budget_pct=(
            req.inspection_budget_pct
            if req.inspection_budget_pct is not None
            else DEFAULT_POLICY.inspection_budget_pct
        ),
    )
    pred = np.array(req.scores, dtype=np.float32)
    labels, auto, manual = apply_hybrid(pred, policy)

    return {
        "threshold_high": float(policy.threshold_high),
        "inspection_budget_pct": float(policy.inspection_budget_pct),
        "rows": int(len(pred)),
        "auto_reject_count": int(auto.sum()),
        "manual_inspection_count": int(manual.sum()),
        "decisions": labels.tolist(),
    }
