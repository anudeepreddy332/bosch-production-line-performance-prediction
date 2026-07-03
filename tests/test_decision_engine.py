"""Unit tests for src/inference/decision_engine.py -- the pure decision-policy functions
shared by the FastAPI service and the batch simulator."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.inference.decision_engine import (
    DecisionPolicy,
    apply_hybrid,
    apply_threshold,
    apply_topk_budget,
    load_policy,
    metrics_from_labels,
)


def test_decision_policy_defaults():
    policy = DecisionPolicy()
    assert policy.threshold_high == 0.60
    assert policy.inspection_budget_pct == 5.0


def test_apply_threshold_basic():
    pred = np.array([0.1, 0.5, 0.59, 0.6, 0.61, 0.9], dtype=np.float32)
    out = apply_threshold(pred, threshold=0.6)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "budget_pct,expected_k",
    [(0.0, 0), (10.0, 1), (50.0, 5), (100.0, 10)],
)
def test_apply_topk_budget_selects_correct_count(budget_pct, expected_k):
    pred = np.arange(10, dtype=np.float32) / 10.0  # 0.0 .. 0.9, strictly increasing
    out = apply_topk_budget(pred, budget_pct)
    assert out.dtype == np.int8
    assert int(out.sum()) == expected_k
    if expected_k > 0:
        # top-k must be the highest-scoring rows (indices 9, 8, ... for strictly increasing pred)
        expected_idx = set(np.argsort(-pred, kind="mergesort")[:expected_k].tolist())
        assert set(np.where(out == 1)[0].tolist()) == expected_idx


def test_apply_topk_budget_rounds_up():
    # 3 rows, 34% budget -> ceil(3 * 0.34) = 2, not 1 (floor) or 0 (truncation)
    pred = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    out = apply_topk_budget(pred, budget_pct=34.0)
    assert int(out.sum()) == 2


def test_apply_hybrid_auto_reject_only_when_budget_already_exceeded():
    # 10 rows, threshold_high=0.6 catches indices where pred>=0.6 (3 rows), but the budget
    # (10% of 10 = 1 row, ceil'd) is already smaller than that -- no manual slots should open.
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.61, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
    policy = DecisionPolicy(threshold_high=0.6, inspection_budget_pct=10.0)
    decisions, auto_reject, manual = apply_hybrid(pred, policy)

    assert int(auto_reject.sum()) == 5  # pred >= 0.6: indices 5..9
    assert int(manual.sum()) == 0  # budget (ceil(10*0.10)=1) already exceeded by auto_reject
    np.testing.assert_array_equal(decisions, auto_reject)


def test_apply_hybrid_fills_remaining_budget_with_next_highest():
    # 10 rows, nothing clears threshold_high=0.99; budget=30% -> 3 manual slots, filled by
    # the 3 highest-scoring rows overall.
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
    policy = DecisionPolicy(threshold_high=0.99, inspection_budget_pct=30.0)
    decisions, auto_reject, manual = apply_hybrid(pred, policy)

    assert int(auto_reject.sum()) == 0
    assert int(manual.sum()) == 3
    assert set(np.where(manual == 1)[0].tolist()) == {7, 8, 9}
    np.testing.assert_array_equal(decisions, manual)


def test_apply_hybrid_partial_budget_remaining():
    # 10 rows; threshold_high=0.85 catches 2 rows (0.9, 0.95); budget=40% -> 4 total slots,
    # so 2 more manual slots should go to the next-highest-scoring non-auto-reject rows.
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
    policy = DecisionPolicy(threshold_high=0.85, inspection_budget_pct=40.0)
    decisions, auto_reject, manual = apply_hybrid(pred, policy)

    assert int(auto_reject.sum()) == 2  # indices 8, 9
    assert int(manual.sum()) == 2
    assert set(np.where(manual == 1)[0].tolist()) == {6, 7}  # next-highest: 0.7, 0.8
    assert int(decisions.sum()) == 4


def test_metrics_from_labels_confusion_counts():
    y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
    y_hat = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    m = metrics_from_labels(y_true, y_hat)

    assert m["tp"] == 2  # indices 0, 4
    assert m["fp"] == 1  # index 3
    assert m["fn"] == 1  # index 1
    assert m["tn"] == 2  # indices 2, 5
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["flagged_pct"] == pytest.approx(3 / 6 * 100.0)


def test_metrics_from_labels_handles_zero_denominators():
    y_true = np.array([0, 0, 0], dtype=np.int8)
    y_hat = np.array([0, 0, 0], dtype=np.int8)
    m = metrics_from_labels(y_true, y_hat)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_load_policy_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    policy = load_policy(missing)
    assert policy == DecisionPolicy(threshold_high=0.60, inspection_budget_pct=5.0)


def test_load_policy_reads_fixture_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"final_recommendation": {"threshold_high": 0.42, "inspection_budget_pct": 12.5}})
    )
    policy = load_policy(summary)
    assert policy.threshold_high == 0.42
    assert policy.inspection_budget_pct == 12.5


def test_load_policy_missing_recommendation_key_falls_back(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"something_else": {}}))
    policy = load_policy(summary)
    assert policy == DecisionPolicy(threshold_high=0.60, inspection_budget_pct=5.0)
