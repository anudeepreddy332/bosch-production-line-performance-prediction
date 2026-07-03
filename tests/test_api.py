"""FastAPI TestClient tests for apps/api/main.py. apps.api.main computes DEFAULT_POLICY once at
import time by calling the module-level _load_default_policy(), which reads the module-level
SUMMARY_PATH constant. These tests monkeypatch SUMMARY_PATH to a fixture JSON and then
monkeypatch DEFAULT_POLICY to the result of calling _load_default_policy() again -- _load_
default_policy() looks up SUMMARY_PATH from the module's global namespace at call time, so this
picks up the patched path. (importlib.reload is deliberately not used: reload re-executes the
whole module top-to-bottom, which would overwrite the patched SUMMARY_PATH right back to its
original value before DEFAULT_POLICY is recomputed.) No change to apps/api/main.py itself."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main


@pytest.fixture
def client_with_fixture_policy(tmp_path, monkeypatch):
    """A TestClient whose DEFAULT_POLICY comes from a fixture summary JSON, not any real
    outputs/max_recall_system_summary.json."""
    summary = tmp_path / "max_recall_system_summary.json"
    summary.write_text(
        json.dumps({"final_recommendation": {"threshold_high": 0.42, "inspection_budget_pct": 12.5}})
    )
    monkeypatch.setattr(api_main, "SUMMARY_PATH", summary)
    monkeypatch.setattr(api_main, "DEFAULT_POLICY", api_main._load_default_policy())
    return TestClient(api_main.app)


@pytest.fixture
def client_with_default_policy(tmp_path, monkeypatch):
    """A TestClient pointed at a path that doesn't exist, exercising the documented
    DecisionPolicy() fallback."""
    monkeypatch.setattr(api_main, "SUMMARY_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(api_main, "DEFAULT_POLICY", api_main._load_default_policy())
    return TestClient(api_main.app)


def test_health(client_with_default_policy):
    resp = client_with_default_policy.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_default_policy_fallback_values(client_with_default_policy):
    # score just above the documented default threshold_high (0.60)
    resp = client_with_default_policy.post("/predict", json={"score": 0.61})
    assert resp.status_code == 200
    body = resp.json()
    assert body["threshold_high"] == pytest.approx(0.60)
    assert body["decision"] == 1
    assert body["action"] == "auto_reject"


def test_predict_uses_fixture_policy_threshold(client_with_fixture_policy):
    below = client_with_fixture_policy.post("/predict", json={"score": 0.30}).json()
    above = client_with_fixture_policy.post("/predict", json={"score": 0.50}).json()
    assert below["threshold_high"] == pytest.approx(0.42)
    assert below["decision"] == 0
    assert below["action"] == "pass"
    assert above["decision"] == 1
    assert above["action"] == "auto_reject"


def test_predict_explicit_threshold_override(client_with_fixture_policy):
    resp = client_with_fixture_policy.post(
        "/predict", json={"score": 0.50, "threshold_high": 0.90}
    )
    body = resp.json()
    assert body["threshold_high"] == pytest.approx(0.90)
    assert body["decision"] == 0  # 0.50 < overridden 0.90


@pytest.mark.parametrize("bad_score", [-0.1, 1.1])
def test_predict_rejects_out_of_range_score(client_with_default_policy, bad_score):
    resp = client_with_default_policy.post("/predict", json={"score": bad_score})
    assert resp.status_code == 422


def test_batch_predict_matches_apply_hybrid_semantics(client_with_fixture_policy):
    # 10 scores; fixture policy threshold_high=0.42 auto-rejects >= 0.42; budget=12.5% ->
    # ceil(10 * 0.125) = 2 total slots.
    scores = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.90, 0.95]
    resp = client_with_fixture_policy.post("/batch_predict", json={"scores": scores})
    assert resp.status_code == 200
    body = resp.json()
    assert body["threshold_high"] == pytest.approx(0.42)
    assert body["inspection_budget_pct"] == pytest.approx(12.5)
    assert body["rows"] == 10
    assert body["auto_reject_count"] == 4  # scores >= 0.42: 0.50, 0.60, 0.90, 0.95
    assert body["manual_inspection_count"] == 0  # budget (2) already exceeded by auto-reject
    assert sum(body["decisions"]) == 4


def test_batch_predict_empty_scores_rejected(client_with_default_policy):
    resp = client_with_default_policy.post("/batch_predict", json={"scores": []})
    assert resp.status_code == 422
