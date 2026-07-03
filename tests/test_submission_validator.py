"""Tests for the Kaggle submission pipeline's validation/loading helpers:
scripts/generate_submission.py (load_test_features, check_against_sample_submission, main) and
src/inference/payload.py (load_validated_payload, predict_proba_ensemble). Uses the
tiny_model_payload fixture (tests/conftest.py) -- no committed model pickle anywhere here."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.generate_submission import check_against_sample_submission, load_test_features
from scripts.generate_submission import main as generate_submission_main
from src.inference.payload import load_validated_payload, predict_proba_ensemble


class _FakeBareEstimator:
    """A stand-in for a pre-Phase-2 bare LGBMClassifier (not a payload dict). Defined at
    module level, not nested in a test function, so joblib/pickle can locate it by
    qualified name when dumping/loading."""


# --- load_test_features -----------------------------------------------------------------


def test_load_test_features_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Test feature file not found"):
        load_test_features(tmp_path / "nope.parquet", feature_cols=["a"], id_col="Id")


def test_load_test_features_missing_id_column_raises(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing required id column"):
        load_test_features(path, feature_cols=["a"], id_col="Id")


def test_load_test_features_missing_feature_columns_raises(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"Id": [1, 2], "a": [1, 2]}).to_parquet(path)
    with pytest.raises(ValueError, match="missing 2 feature column"):
        load_test_features(path, feature_cols=["a", "b", "c"], id_col="Id")


def test_load_test_features_warns_on_response_column(tmp_path, capsys):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"Id": [1, 2], "a": [0.1, 0.2], "Response": [0, 1]}).to_parquet(path)
    df = load_test_features(path, feature_cols=["a"], id_col="Id")
    assert "Response" in df.columns  # not dropped, just flagged
    assert "ignored" in capsys.readouterr().out


def test_load_test_features_happy_path(tmp_path):
    path = tmp_path / "features.parquet"
    expected = pd.DataFrame({"Id": [1, 2, 3], "a": [0.1, 0.2, 0.3], "b": [1.0, 2.0, 3.0]})
    expected.to_parquet(path)
    df = load_test_features(path, feature_cols=["a", "b"], id_col="Id")
    pd.testing.assert_frame_equal(df, expected)


# --- check_against_sample_submission ----------------------------------------------------


def test_check_against_sample_submission_no_sample_file_is_a_noop(tmp_path, capsys):
    check_against_sample_submission(pd.Series([1, 2, 3]), tmp_path / "missing.parquet")
    assert "NOTE" in capsys.readouterr().out


def test_check_against_sample_submission_row_count_mismatch_warns(tmp_path, capsys):
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({"Id": [1, 2, 3, 4]}).to_parquet(sample_path)
    check_against_sample_submission(pd.Series([1, 2, 3]), sample_path)
    assert "row count" in capsys.readouterr().out


def test_check_against_sample_submission_id_mismatch_warns(tmp_path, capsys):
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({"Id": [1, 2, 3]}).to_parquet(sample_path)
    check_against_sample_submission(pd.Series([1, 2, 4]), sample_path)
    assert "Id mismatch" in capsys.readouterr().out


def test_check_against_sample_submission_happy_path_silent(tmp_path, capsys):
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({"Id": [1, 2, 3]}).to_parquet(sample_path)
    check_against_sample_submission(pd.Series([3, 1, 2]), sample_path)  # order doesn't matter
    out = capsys.readouterr().out
    assert "WARNING" not in out


# --- load_validated_payload ---------------------------------------------------------------


def test_load_validated_payload_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model payload not found"):
        load_validated_payload(tmp_path / "nope.pkl")


def test_load_validated_payload_bare_estimator_raises(tmp_path):
    path = tmp_path / "bare.pkl"
    joblib.dump(_FakeBareEstimator(), path)
    with pytest.raises(ValueError, match="not the Phase-2 payload dict"):
        load_validated_payload(path)


def test_load_validated_payload_invalid_dict_raises(tmp_path):
    path = tmp_path / "invalid.pkl"
    joblib.dump({"models": [], "feature_cols": ["a"], "threshold": 0.5}, path)  # missing keys
    with pytest.raises(ValueError, match="Invalid model payload"):
        load_validated_payload(path)


def test_load_validated_payload_valid_fixture_roundtrips(tiny_model_payload_path, tiny_model_payload):
    loaded = load_validated_payload(tiny_model_payload_path)
    assert loaded["model_name"] == tiny_model_payload["model_name"]
    assert loaded["feature_cols"] == tiny_model_payload["feature_cols"]
    assert loaded["threshold"] == tiny_model_payload["threshold"]


# --- predict_proba_ensemble ----------------------------------------------------------------


class _StubModel:
    """A minimal stand-in for a fitted LGBMClassifier: predict_proba returns a fixed,
    hand-chosen array so the fold-averaging arithmetic can be verified exactly."""

    def __init__(self, proba_class_1: list[float]):
        self._proba_class_1 = np.array(proba_class_1, dtype=np.float64)

    def predict_proba(self, X):
        p1 = self._proba_class_1
        return np.column_stack([1 - p1, p1])


def test_predict_proba_ensemble_averages_folds_and_inner_models():
    # 2 folds; fold 0 has 2 inner models (inner-mean), fold 1 has 1 inner model.
    payload = {
        "feature_cols": ["a"],
        "models": [
            [_StubModel([0.2, 0.8]), _StubModel([0.4, 0.6])],  # fold 0 inner-mean: [0.3, 0.7]
            [_StubModel([0.9, 0.1])],  # fold 1: [0.9, 0.1]
        ],
    }
    features = pd.DataFrame({"a": [10.0, 20.0]})
    out = predict_proba_ensemble(payload, features)
    # fold-mean of [0.3, 0.7] and [0.9, 0.1] -> [0.6, 0.4]
    np.testing.assert_allclose(out, [0.6, 0.4], rtol=1e-9)


def test_predict_proba_ensemble_uses_only_payload_feature_cols():
    payload = {"feature_cols": ["a"], "models": [[_StubModel([0.5, 0.5])]]}
    features = pd.DataFrame({"a": [1.0, 2.0], "unused_extra_col": [999.0, -999.0]})
    out = predict_proba_ensemble(payload, features)  # must not raise on the extra column
    np.testing.assert_allclose(out, [0.5, 0.5])


# --- end-to-end: main() against the fixture model ------------------------------------------


def test_generate_submission_main_end_to_end(tiny_model_payload_path, tiny_model_payload, tmp_path):
    feature_cols = tiny_model_payload["feature_cols"]
    n = 12
    features_path = tmp_path / "test_features.parquet"
    pd.DataFrame(
        {"Id": np.arange(1, n + 1, dtype=np.int64), **{c: np.linspace(-1, 1, n) for c in feature_cols}}
    ).to_parquet(features_path)

    output_path = tmp_path / "submission.csv"
    rc = generate_submission_main(
        [
            "--model-path",
            str(tiny_model_payload_path),
            "--test-features",
            str(features_path),
            "--output",
            str(output_path),
            "--sample-submission",
            str(tmp_path / "no_such_sample.parquet"),
        ]
    )
    assert rc == 0
    submission = pd.read_csv(output_path)
    assert list(submission.columns) == ["Id", "Response"]
    assert len(submission) == n
    assert set(submission["Response"].unique().tolist()) <= {0, 1}
