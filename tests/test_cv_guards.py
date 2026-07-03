"""Tests for src/training/cv.py -- the chunk-aware, leakage-safe CV harness. The
leak-injection test (test_validate_chunk_aware_splits_raises_on_chunk_split_across_folds) is
the PF3 checklist's binding guard test: it must raise, and disabling the guard it exercises
must make it fail (verified manually during PF3 validation, see the CP3 report)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.cv import (
    ChunkCVConfig,
    assign_fold_ids,
    make_chunk_aware_splits,
    validate_chunk_aware_splits,
    verify_persisted_fold_assignment,
)


def test_validate_chunk_aware_splits_passes_on_clean_partition():
    # 4 chunks, 2 folds, each fold's train/valid drawn from disjoint chunks -- no leakage.
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    splits = [
        (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),  # fold 0: valid = chunks {2, 3}
        (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),  # fold 1: valid = chunks {0, 1}
    ]
    validate_chunk_aware_splits(splits, groups=groups)  # must not raise


def test_validate_chunk_aware_splits_raises_on_chunk_split_across_folds():
    """The leak-injection test: chunk_id 1 appears in BOTH the train and validation index
    sets of fold 0 -- this is exactly the leakage make_chunk_aware_splits exists to prevent
    (a chunk's rows partially train, partially validate the same model). Must raise."""
    groups = np.array([0, 0, 1, 1, 1, 2, 2])
    # fold 0: train includes indices 0,1,2 (chunks 0,0,1); valid includes indices 3,4 (chunk 1)
    # -- chunk_id 1 is in both train_groups and valid_groups.
    leaked_splits = [
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([3, 4, 5, 6]), np.array([0, 1])),
    ]
    with pytest.raises(ValueError, match="Chunk leakage detected"):
        validate_chunk_aware_splits(leaked_splits, groups=groups)


def test_validate_chunk_aware_splits_raises_on_validation_chunk_reuse():
    """A different leakage mode: the same chunk_id is used as validation data in two
    different folds (no train/valid overlap in any single fold, but breaks the "each row
    validated exactly once" CV contract and can bias aggregate OOF metrics). fold 2's train
    set deliberately excludes chunk 0 entirely so this trips the reuse check, not the
    same-fold leakage check checked first."""
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    splits = [
        (np.array([2, 3, 4, 5, 6, 7, 8, 9]), np.array([0, 1])),  # fold 0: valid = chunk 0
        (np.array([0, 1, 4, 5, 6, 7, 8, 9]), np.array([2, 3])),  # fold 1: valid = chunk 1
        (np.array([4, 5, 6, 7, 8, 9]), np.array([0, 1])),  # fold 2: valid = chunk 0 again -- reused
    ]
    with pytest.raises(ValueError, match="Validation chunk reuse detected"):
        validate_chunk_aware_splits(splits, groups=groups)


def test_make_chunk_aware_splits_never_splits_a_chunk(synthetic_chunked_df):
    splits = make_chunk_aware_splits(synthetic_chunked_df, config=ChunkCVConfig(n_splits=5))
    groups = synthetic_chunked_df["chunk_id"].to_numpy()

    for train_idx, valid_idx in splits:
        train_chunks = set(groups[train_idx].tolist())
        valid_chunks = set(groups[valid_idx].tolist())
        assert not train_chunks.intersection(valid_chunks)

    # every row must appear in exactly one fold's validation set
    all_valid_idx = np.concatenate([valid_idx for _, valid_idx in splits])
    assert len(all_valid_idx) == len(synthetic_chunked_df)
    assert len(set(all_valid_idx.tolist())) == len(synthetic_chunked_df)


def test_make_chunk_aware_splits_missing_target_column_raises(synthetic_chunked_df):
    df = synthetic_chunked_df.drop(columns=["Response"])
    with pytest.raises(ValueError, match="Missing target column"):
        make_chunk_aware_splits(df)


def test_make_chunk_aware_splits_missing_group_column_raises(synthetic_chunked_df):
    df = synthetic_chunked_df.drop(columns=["chunk_id"])
    with pytest.raises(ValueError, match="Missing group column"):
        make_chunk_aware_splits(df)


def test_make_chunk_aware_splits_too_few_groups_raises():
    df = pd.DataFrame(
        {
            "Response": [0, 1, 0, 1],
            "chunk_id": [0, 0, 1, 1],  # only 2 unique chunks, default n_splits=5
        }
    )
    with pytest.raises(ValueError, match="Not enough unique groups"):
        make_chunk_aware_splits(df)


def test_assign_fold_ids_covers_every_row(synthetic_chunked_df):
    splits = make_chunk_aware_splits(synthetic_chunked_df, config=ChunkCVConfig(n_splits=5))
    fold_ids = assign_fold_ids(len(synthetic_chunked_df), splits)
    assert (fold_ids >= 0).all()
    assert set(fold_ids.tolist()) == {0, 1, 2, 3, 4}


def test_assign_fold_ids_raises_if_a_row_is_never_assigned():
    # Deliberately incomplete splits: only rows 0-1 ever appear in a validation set.
    incomplete_splits = [(np.array([2, 3, 4]), np.array([0, 1]))]
    with pytest.raises(ValueError, match="Fold assignment failed"):
        assign_fold_ids(n_rows=5, splits=incomplete_splits)


def test_verify_persisted_fold_assignment_passes_when_consistent(synthetic_chunked_df):
    splits = make_chunk_aware_splits(synthetic_chunked_df, config=ChunkCVConfig(n_splits=5))
    df = synthetic_chunked_df.copy()
    df["cv_fold"] = assign_fold_ids(len(df), splits)
    verify_persisted_fold_assignment(df, config=ChunkCVConfig(n_splits=5))  # must not raise


def test_verify_persisted_fold_assignment_raises_on_mismatch(synthetic_chunked_df):
    splits = make_chunk_aware_splits(synthetic_chunked_df, config=ChunkCVConfig(n_splits=5))
    df = synthetic_chunked_df.copy()
    df["cv_fold"] = assign_fold_ids(len(df), splits)
    # Corrupt one row's persisted fold id so it disagrees with what would be recomputed.
    df.loc[df.index[0], "cv_fold"] = (df.loc[df.index[0], "cv_fold"] + 1) % 5

    with pytest.raises(ValueError, match="does not match the chunk-aware fold assignment"):
        verify_persisted_fold_assignment(df, config=ChunkCVConfig(n_splits=5))
