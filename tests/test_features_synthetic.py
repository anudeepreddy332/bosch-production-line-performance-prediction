"""Synthetic-fixture tests for src/features/core_pipeline.py::build_core_features -- the lean
core feature block shared by dataset_baseline/G/H. Values in the synthetic_core_df fixture
(tests/conftest.py) are hand-chosen so rolling-window counts and chunk assignment can be
verified by direct computation, not by eyeballing real Bosch data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.core_pipeline import CORE_FEATURE_COLUMNS, CorePipelineConfig, build_core_features


def test_build_core_features_output_columns(synthetic_core_df):
    out = build_core_features(synthetic_core_df)
    assert list(out.columns) == ["Id", "Response", *CORE_FEATURE_COLUMNS]
    assert len(out) == len(synthetic_core_df)


def test_build_core_features_omits_response_when_absent(synthetic_core_df):
    df = synthetic_core_df.drop(columns=["Response"])
    out = build_core_features(df)
    assert "Response" not in out.columns
    assert list(out.columns) == ["Id", *CORE_FEATURE_COLUMNS]


def test_build_core_features_missing_required_column_raises(synthetic_core_df):
    df = synthetic_core_df.drop(columns=["feature_mean"])
    with pytest.raises(ValueError, match="Missing required columns"):
        build_core_features(df)


def test_build_core_features_chunking_respects_chunk_size_rows(synthetic_core_df):
    # 10 rows, chunk_size_rows=3 -> 4 chunks of sizes [3, 3, 3, 1] in start_time-sorted order
    # (arange(10)//3 = [0,0,0,1,1,1,2,2,2,3]), so only the very last row has chunk_size 1.
    out = build_core_features(synthetic_core_df, config=CorePipelineConfig(chunk_size_rows=3))
    sizes = out.sort_values("start_time")["chunk_size"].to_numpy()
    np.testing.assert_array_equal(sizes, [3, 3, 3, 3, 3, 3, 3, 3, 3, 1])

    chunk_ids = out.sort_values("start_time")["chunk_id"].to_numpy()
    np.testing.assert_array_equal(chunk_ids, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3])


def test_build_core_features_rolling_counts_hand_verified(synthetic_core_df):
    # start_time (sorted): 0.0, 0.2, 0.5, 0.9, 1.5, 2.0, 2.1, 2.2, 30.0, 30.1
    # records_last_1hr counts rows within 1.0 of the current row's start_time, inclusive,
    # looking backward only (window is [current - window_size, current]).
    out = build_core_features(synthetic_core_df).sort_values("start_time").reset_index(drop=True)

    expected_last_1hr = [1, 2, 3, 4, 3, 2, 3, 4, 1, 2]
    np.testing.assert_array_equal(out["records_last_1hr"].to_numpy(), expected_last_1hr)

    # records_last_24hr (window=24.0): everything up to and including 2.2 falls within 24 of
    # itself given the earliest point is 0.0 (2.2 - 0.0 = 2.2 <= 24); the two rows at ~30 are
    # isolated from the first eight (30.0 - 2.2 = 27.8 > 24) but see each other.
    expected_last_24hr = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    np.testing.assert_array_equal(out["records_last_24hr"].to_numpy(), expected_last_24hr)

    density_ratio = out["records_last_1hr"].to_numpy(dtype=np.float32) / np.maximum(
        out["records_last_24hr"].to_numpy(dtype=np.float32), 1.0
    )
    np.testing.assert_allclose(out["density_ratio"].to_numpy(), density_ratio, rtol=1e-6)


def test_build_core_features_handles_nan_start_time():
    df = pd.DataFrame(
        {
            "Id": np.arange(1, 6, dtype=np.int64),
            "start_time": [0.0, np.nan, 2.0, np.nan, 4.0],
            "duration": [1.0] * 5,
            "feature_mean": [0.1] * 5,
        }
    )
    out = build_core_features(df)
    # NaN start_time rows must not crash chunking/rolling-count logic and must sort before
    # every finite value (min - 1.0), per _fill_start_time's documented behavior.
    assert out["start_time"].isna().sum() == 2
    assert len(out) == 5
    assert out["chunk_id"].notna().all()


def test_build_core_features_is_deterministic(synthetic_core_df):
    out_a = build_core_features(synthetic_core_df)
    out_b = build_core_features(synthetic_core_df)
    pd.testing.assert_frame_equal(out_a, out_b)
