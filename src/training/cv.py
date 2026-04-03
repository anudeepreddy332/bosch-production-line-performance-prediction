from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class ChunkCVConfig:
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True


def validate_chunk_aware_splits(
    splits: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> None:
    """Validate that no chunk_id is split across train/validation folds."""
    seen_val_groups: set[int] = set()

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        train_groups = set(groups[train_idx].tolist())
        valid_groups = set(groups[valid_idx].tolist())

        overlap = train_groups.intersection(valid_groups)
        if overlap:
            sample = sorted(list(overlap))[:5]
            raise ValueError(
                f"Chunk leakage detected in fold {fold_idx}: {len(overlap)} chunk_id values overlap. Sample={sample}"
            )

        reused = seen_val_groups.intersection(valid_groups)
        if reused:
            sample = sorted(list(reused))[:5]
            raise ValueError(
                f"Validation chunk reuse detected in fold {fold_idx}: {len(reused)} chunk_id values repeated across folds. Sample={sample}"
            )

        seen_val_groups.update(valid_groups)


def make_chunk_aware_splits(
    df: pd.DataFrame,
    target_col: str = "Response",
    group_col: str = "chunk_id",
    config: ChunkCVConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = config or ChunkCVConfig()

    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column '{group_col}'")

    groups = df[group_col].to_numpy()
    y = df[target_col].to_numpy()

    unique_groups = pd.unique(groups)
    if len(unique_groups) < cfg.n_splits:
        raise ValueError(
            f"Not enough unique groups for CV: unique_groups={len(unique_groups)}, n_splits={cfg.n_splits}"
        )

    try:
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
        )
        splits = list(splitter.split(df, y=y, groups=groups))
    except Exception:
        # Fallback for difficult class distributions.
        splitter = GroupKFold(n_splits=cfg.n_splits)
        splits = list(splitter.split(df, y=y, groups=groups))

    validate_chunk_aware_splits(splits, groups=groups)
    return splits


def assign_fold_ids(n_rows: int, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    fold_ids = np.full(n_rows, -1, dtype=np.int16)
    for fold_idx, (_, valid_idx) in enumerate(splits):
        fold_ids[valid_idx] = fold_idx

    if (fold_ids < 0).any():
        missing = int((fold_ids < 0).sum())
        raise ValueError(f"Fold assignment failed: {missing} rows were never assigned to a validation fold.")

    return fold_ids
