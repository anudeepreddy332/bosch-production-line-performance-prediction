"""Schema + value-locking tests for results/leaderboard.json -- the single source of truth
every downstream artifact (README, SYSTEM_OVERVIEW, case study, and eventually the dashboard)
traces its Kaggle numbers to (docs/implementation/portfolio_master_plan.md, standing rule 8).
This test locks the registry against silent future drift: any accidental edit to a value below
must fail CI, not just a manual read-through."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = ROOT / "results" / "leaderboard.json"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "generated",
    "source_of_truth",
    "program_status",
    "notes",
    "experiments",
}

REQUIRED_EXPERIMENT_KEYS = {
    "experiment_id",
    "kdr",
    "kdr_file",
    "date",
    "mechanism",
    "feature_count",
    "oof_mcc",
    "oof_status",
    "threshold",
    "public_mcc",
    "private_mcc",
    "data_fingerprint",
    "git_tag",
    "hypothesis",
    "hypothesis_result",
    "notes",
}

# Locked values: (oof_mcc, oof_status, threshold, public_mcc, private_mcc, git_tag, data_fingerprint).
# Sourced from docs/research/kaggle_decisions.md via results/leaderboard.json at KDR-009 freeze
# (2026-07-03) -- any diff here is either a real (re-registered, re-frozen) program change or a
# registry bug, and either way must not pass silently.
EXPECTED_ROWS = {
    "K1": (0.15337, "honest", 0.91, 0.14389, 0.16160, "K1-result", "a5bb652f2b20aca6"),
    "K2": (0.37530, "contaminated", 0.98, 0.31699, 0.32702, "K2-result", "3dc7fb742ce24ecf"),
    "K3-A": (0.31761, "honest", 0.98, 0.31791, 0.33161, "K3-result", "e02a1d4e1106fbaa"),
    "K3-B": (0.21171, "contaminated", 0.95, 0.10065, 0.10530, "K3-result", "1c98082a35397fd5"),
    "K4": (0.32192, "honest", 0.98, 0.31697, 0.33447, "K4-result", "00b4ed30ea58762d"),
    "K5-A": (0.32506, "honest", 0.98, 0.32330, 0.33711, "K5-result", "e9df7ffff186b6fa"),
    "K5-B": (0.57828, "contaminated", 0.95, 0.33571, 0.33989, "K5-result", "e6faaf3d92fdaa43"),
    "P0": (0.37892, "honest", 0.96, 0.39226, 0.40391, "P0-result", "f4a25438ad901355"),
    "P1": (0.39484, "honest", 0.89, 0.40447, 0.41917, "P1-result", "c602cd26810cf626"),
}


@pytest.fixture(scope="module")
def leaderboard() -> dict:
    return json.loads(LEADERBOARD_PATH.read_text())


def test_leaderboard_file_exists_and_parses():
    assert LEADERBOARD_PATH.exists()
    json.loads(LEADERBOARD_PATH.read_text())  # must not raise


def test_leaderboard_top_level_schema(leaderboard):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(leaderboard.keys())
    assert leaderboard["source_of_truth"] == "docs/research/kaggle_decisions.md"
    assert isinstance(leaderboard["experiments"], list)


def test_leaderboard_has_exactly_nine_experiments(leaderboard):
    assert len(leaderboard["experiments"]) == 9
    ids = [r["experiment_id"] for r in leaderboard["experiments"]]
    assert ids == list(EXPECTED_ROWS.keys())  # exact order, exact set


def test_leaderboard_every_row_has_required_keys(leaderboard):
    for row in leaderboard["experiments"]:
        missing = REQUIRED_EXPERIMENT_KEYS - row.keys()
        assert not missing, f"{row['experiment_id']} missing keys: {missing}"


@pytest.mark.parametrize("experiment_id", list(EXPECTED_ROWS.keys()))
def test_leaderboard_locked_values(leaderboard, experiment_id):
    row = next(r for r in leaderboard["experiments"] if r["experiment_id"] == experiment_id)
    oof_mcc, oof_status, threshold, public_mcc, private_mcc, git_tag, fingerprint = EXPECTED_ROWS[
        experiment_id
    ]
    assert row["oof_mcc"] == pytest.approx(oof_mcc, abs=1e-9)
    assert row["oof_status"] == oof_status
    assert row["threshold"] == pytest.approx(threshold, abs=1e-9)
    assert row["public_mcc"] == pytest.approx(public_mcc, abs=1e-9)
    assert row["private_mcc"] == pytest.approx(private_mcc, abs=1e-9)
    assert row["git_tag"] == git_tag
    assert row["data_fingerprint"] == fingerprint


def test_leaderboard_program_best_is_p1(leaderboard):
    best = max(leaderboard["experiments"], key=lambda r: r["private_mcc"])
    assert best["experiment_id"] == "P1"
    assert best["private_mcc"] == pytest.approx(0.41917, abs=1e-9)


def test_leaderboard_oof_status_only_two_values(leaderboard):
    statuses = {row["oof_status"] for row in leaderboard["experiments"]}
    assert statuses == {"honest", "contaminated"}


def test_leaderboard_contaminated_rows_are_the_expected_three(leaderboard):
    contaminated = {r["experiment_id"] for r in leaderboard["experiments"] if r["oof_status"] == "contaminated"}
    assert contaminated == {"K2", "K3-B", "K5-B"}
