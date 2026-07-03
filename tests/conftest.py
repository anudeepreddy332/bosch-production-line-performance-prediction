"""Shared fixtures for the PF3 test suite (docs/implementation/portfolio_master_plan.md).

No model pickle is committed anywhere under tests/ -- the `tiny_model_payload` fixture below
trains a real, tiny LightGBM model on an in-memory synthetic dataset for every test that needs
a valid payload, reusing the same self-test helpers scripts/ops/validate_model_payload.py already
exercises as a standalone script.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.ops.validate_model_payload import _run_self_test


@pytest.fixture(scope="session")
def tiny_model_payload(tmp_path_factory) -> dict:
    """A real, valid Phase-2 payload dict trained on a tiny synthetic dataset -- not a
    committed pickle. Session-scoped: training happens once (~2s) and is reused read-only
    by every test in the session.
    """
    tmp_dir = tmp_path_factory.mktemp("tiny_model_payload")
    payload, problems = _run_self_test(tmp_dir)
    assert not problems, f"fixture self-test produced problems: {problems}"
    return payload


@pytest.fixture
def tiny_model_payload_path(tiny_model_payload, tmp_path) -> Path:
    """The same fixture payload, joblib-dumped to a fresh temp file per test."""
    import joblib

    path = tmp_path / "tiny_model.pkl"
    joblib.dump(tiny_model_payload, path)
    return path


@pytest.fixture
def synthetic_core_df() -> pd.DataFrame:
    """A tiny, hand-constructed DataFrame for src.features.core_pipeline.build_core_features --
    values chosen so rolling-window counts and chunk assignment can be hand-verified in tests,
    not read off real Bosch data."""
    return pd.DataFrame(
        {
            "Id": np.arange(1, 11, dtype=np.int64),
            "Response": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "start_time": [0.0, 0.2, 0.5, 0.9, 1.5, 2.0, 2.1, 2.2, 30.0, 30.1],
            "duration": [1.0] * 10,
            "feature_mean": [0.1] * 10,
        }
    )


@pytest.fixture
def synthetic_chunked_df() -> pd.DataFrame:
    """A small, chunk-labeled, labeled dataset for CV-guard tests -- enough unique chunk_ids
    (20) to satisfy make_chunk_aware_splits' n_splits=5 default with room to spare."""
    rng = np.random.default_rng(0)
    n_chunks = 20
    rows_per_chunk = 10
    n = n_chunks * rows_per_chunk
    chunk_id = np.repeat(np.arange(n_chunks, dtype=np.int32), rows_per_chunk)
    return pd.DataFrame(
        {
            "Id": np.arange(1, n + 1, dtype=np.int64),
            "Response": (rng.random(n) < 0.1).astype(np.int8),
            "chunk_id": chunk_id,
            "feat_a": rng.normal(size=n).astype(np.float32),
        }
    )
