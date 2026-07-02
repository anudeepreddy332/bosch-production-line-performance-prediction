"""P0 (KDR-007): quarantined, feature-agnostic LightGBM trainer for wide (raw-numeric) experiments.

Reuses `src.training.cv` (chunk-aware splits, persisted-fold verification) and
`src.training.modeling` (MCC threshold search, data fingerprint, model payload assembly)
unchanged -- CV scheme, OOF semantics, threshold convention, determinism/provenance
contract, and payload shape are frozen for the life of P0-P4 (KDR-007 SS5a). The only
evolvable surface is the LightGBM hyperparameter dict passed to `train_wide_lgbm_oof`.

`LEGACY_LGB_PARAMS` reproduces `src.training.modeling.train_lightgbm_oof`'s hardcoded
hyperparameters exactly, so it serves as the standing regression anchor: `LEGACY_LGB_PARAMS`
on K5-A's 51-column feature stack must reproduce K5-A's honest OOF MCC (0.32506).

Permanently out of scope here (KDR-007 SS5a): alternative learners, feature selection,
feature engineering, stacking/meta-modeling, threshold-strategy exploration.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.logger import setup_logger
from src.training.cv import (
    ChunkCVConfig,
    assign_fold_ids,
    make_chunk_aware_splits,
    verify_persisted_fold_assignment,
)
from src.training.modeling import compute_data_fingerprint, search_best_mcc_threshold

logger = setup_logger(__name__)

LEGACY_LGB_PARAMS: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 50,
    "class_weight": "balanced",
    "n_jobs": -1,
    "verbosity": -1,
}
LEGACY_EARLY_STOPPING_ROUNDS = 100
LEGACY_RANDOM_STATE_BASE = 42

HIGH_CAPACITY_LGB_PARAMS: dict[str, object] = {
    **LEGACY_LGB_PARAMS,
    "n_estimators": 2500,
    "learning_rate": 0.02,
}
HIGH_CAPACITY_EARLY_STOPPING_ROUNDS = 150

K5A_HONEST_OOF_MCC = 0.32506


def train_wide_lgbm_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    output_oof_path: Path,
    output_importance_path: Path,
    lgb_params: dict[str, object] | None = None,
    early_stopping_rounds: int = LEGACY_EARLY_STOPPING_ROUNDS,
    random_state_base: int = LEGACY_RANDOM_STATE_BASE,
    target_col: str = "Response",
    group_col: str = "chunk_id",
    cv_config: ChunkCVConfig | None = None,
    persisted_fold_col: str = "cv_fold",
) -> tuple[dict[str, object], list[lgb.LGBMClassifier]]:
    """Feature-agnostic chunk-aware-CV LightGBM trainer -- same CV/OOF/threshold/
    fingerprint contract as `src.training.modeling.train_lightgbm_oof`; only the LightGBM
    hyperparameters are configurable. `random_state` is set per fold as
    `random_state_base + fold_idx`, matching `train_lightgbm_oof` exactly when
    `lgb_params=None` (i.e. `LEGACY_LGB_PARAMS`) -- this is what makes the regression
    anchor an exact reproduction rather than an approximation."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    params = dict(LEGACY_LGB_PARAMS if lgb_params is None else lgb_params)
    if "random_state" in params:
        raise ValueError("Pass random_state_base, not random_state, inside lgb_params")

    cv_cfg = cv_config or ChunkCVConfig()

    if persisted_fold_col in df.columns:
        verify_persisted_fold_assignment(
            df, persisted_col=persisted_fold_col, target_col=target_col, group_col=group_col, config=cv_cfg
        )
        logger.info("model=%s persisted-fold verification passed", model_name)
    else:
        logger.info("model=%s no persisted '%s' column found -- skipping persisted-fold cross-check", model_name, persisted_fold_col)

    splits = make_chunk_aware_splits(df, target_col=target_col, group_col=group_col, config=cv_cfg)
    fold_ids = assign_fold_ids(len(df), splits=splits)

    X = df[feature_cols].copy()
    y = df[target_col].astype(np.int8).to_numpy()

    oof_pred = np.zeros(len(df), dtype=np.float32)
    feature_importance = np.zeros(len(feature_cols), dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []
    fold_models: list[lgb.LGBMClassifier] = []
    fold_best_iterations: list[int] = []

    n_estimators = int(params.get("n_estimators", 0))

    logger.info(
        "Training wide model=%s with %d rows and %d features (params=%s, early_stopping_rounds=%d)",
        model_name, len(df), len(feature_cols), {k: v for k, v in params.items() if k != "n_jobs"}, early_stopping_rounds,
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]

        model = lgb.LGBMClassifier(random_state=random_state_base + fold_idx, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        fold_models.append(model)

        raw_best_iter = getattr(model, "best_iteration_", None)
        best_iter = int(raw_best_iter) if raw_best_iter else n_estimators
        fold_best_iterations.append(best_iter)

        pred_valid = model.predict_proba(X_valid)[:, 1].astype(np.float32)
        oof_pred[valid_idx] = pred_valid
        feature_importance += model.feature_importances_

        fold_thr, fold_mcc = search_best_mcc_threshold(y_valid, pred_valid)
        fold_metrics.append(
            {
                "fold": float(fold_idx),
                "rows": float(len(valid_idx)),
                "best_threshold": float(fold_thr),
                "mcc": float(fold_mcc),
                "best_iteration": float(best_iter),
            }
        )
        logger.info(
            "model=%s fold=%d valid_rows=%d best_thr=%.2f mcc=%.5f best_iteration=%d/%d",
            model_name, fold_idx, len(valid_idx), fold_thr, fold_mcc, best_iter, n_estimators,
        )

    best_thr, best_mcc = search_best_mcc_threshold(y, oof_pred)
    capacity_bound = any(it >= n_estimators for it in fold_best_iterations)

    oof_df = pd.DataFrame(
        {
            "Id": df["Id"].astype(np.int64),
            "Response": y.astype(np.int8),
            "oof_pred": oof_pred.astype(np.float32),
            "cv_fold": fold_ids.astype(np.int16),
        }
    )
    output_oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(output_oof_path, index=False)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": (feature_importance / max(len(splits), 1)).astype(np.float64),
        }
    ).sort_values("importance", ascending=False)
    output_importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_importance_path, index=False)

    logger.info(
        "model=%s done best_thr=%.2f oof_mcc=%.5f capacity_bound=%s oof_path=%s",
        model_name, best_thr, best_mcc, capacity_bound, output_oof_path,
    )

    return {
        "model_name": model_name,
        "rows": int(len(df)),
        "features": feature_cols,
        "oof_path": str(output_oof_path),
        "feature_importance_path": str(output_importance_path),
        "best_threshold": float(best_thr),
        "oof_mcc": float(best_mcc),
        "fold_metrics": fold_metrics,
        "fold_best_iterations": fold_best_iterations,
        "n_estimators": n_estimators,
        "capacity_bound": bool(capacity_bound),
        "lgb_params": params,
        "early_stopping_rounds": int(early_stopping_rounds),
        "random_state_base": int(random_state_base),
        "data_fingerprint": compute_data_fingerprint(df, feature_cols, target_col),
    }, fold_models
