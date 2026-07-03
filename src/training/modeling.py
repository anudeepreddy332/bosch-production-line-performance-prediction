from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.logger import setup_logger
from src.training.cv import ChunkCVConfig, assign_fold_ids, make_chunk_aware_splits

logger = setup_logger(__name__)


def search_best_mcc_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    thresholds = threshold_grid if threshold_grid is not None else np.round(np.arange(0.01, 1.0, 0.01), 2)
    best_thr = 0.5
    best_mcc = -1.0

    for thr in thresholds:
        y_hat = (y_prob >= float(thr)).astype(np.int8)
        mcc = float(matthews_corrcoef(y_true, y_hat))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)

    return best_thr, best_mcc


def compute_data_fingerprint(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> str:
    """Cheap, deterministic fingerprint of which rows/columns produced a
    model, for evidence-based provenance (not a full content hash)."""
    hasher = hashlib.sha256()
    hasher.update(repr(sorted(feature_cols)).encode())
    hasher.update(str(len(df)).encode())
    id_hash = pd.util.hash_pandas_object(df[["Id"]].sort_values("Id"), index=False).sum()
    hasher.update(str(int(id_hash)).encode())
    if target_col in df.columns:
        hasher.update(str(int(df[target_col].sum())).encode())
    return hasher.hexdigest()[:16]


def train_lightgbm_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    output_oof_path: Path,
    output_importance_path: Path,
    target_col: str = "Response",
    group_col: str = "chunk_id",
    cv_config: ChunkCVConfig | None = None,
) -> tuple[dict[str, object], list[lgb.LGBMClassifier]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    cv_cfg = cv_config or ChunkCVConfig()
    splits = make_chunk_aware_splits(df, target_col=target_col, group_col=group_col, config=cv_cfg)
    fold_ids = assign_fold_ids(len(df), splits=splits)

    X = df[feature_cols].copy()
    y = df[target_col].astype(np.int8).to_numpy()

    oof_pred = np.zeros(len(df), dtype=np.float32)
    feature_importance = np.zeros(len(feature_cols), dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []
    fold_models: list[lgb.LGBMClassifier] = []

    logger.info("Training model=%s with %d rows and %d features", model_name, len(df), len(feature_cols))

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=50,
            random_state=42 + fold_idx,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        fold_models.append(model)

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
            }
        )
        logger.info(
            "model=%s fold=%d valid_rows=%d best_thr=%.2f mcc=%.5f",
            model_name,
            fold_idx,
            len(valid_idx),
            fold_thr,
            fold_mcc,
        )

    best_thr, best_mcc = search_best_mcc_threshold(y, oof_pred)

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
        "model=%s done best_thr=%.2f oof_mcc=%.5f oof_path=%s",
        model_name,
        best_thr,
        best_mcc,
        output_oof_path,
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
        "data_fingerprint": compute_data_fingerprint(df, feature_cols, target_col),
    }, fold_models


def build_model_payload(
    result: dict[str, object],
    fold_models: list[lgb.LGBMClassifier],
) -> dict[str, object]:
    """Assemble the on-disk model artifact from train_lightgbm_oof's output.

    `models` is nested as one list per CV fold (`[[fold_0_model], [fold_1_model], ...]`)
    to match the {fold: [inner-ensemble models]} shape every consumer of this payload
    iterates over (`for fold_models in payload["models"]: for model in fold_models: ...`,
    see src/inference/payload.py::predict_proba_ensemble); each fold here holds exactly one
    model since this phase serves a CV-fold ensemble, not a per-fold multi-seed ensemble or
    a single full-data refit.
    """
    fold_metrics = list(result["fold_metrics"])
    return {
        "models": [[model] for model in fold_models],
        "feature_cols": list(result["features"]),
        "threshold": float(result["best_threshold"]),
        "model_name": result["model_name"],
        "oof_mcc": float(result["oof_mcc"]),
        "fold_metrics": fold_metrics,
        "fold_mccs": [float(m["mcc"]) for m in fold_metrics],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(result["rows"]),
        "data_fingerprint": result.get("data_fingerprint"),
    }
