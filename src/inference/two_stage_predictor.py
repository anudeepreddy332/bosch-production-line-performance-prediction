from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features import FeaturePipeline
from src.inference.predictor import BoschPredictor
from src.logger import setup_logger

logger = setup_logger(__name__)


class TwoStagePredictor:
    """
    Two-stage inference interface:
    - Stage 1: canonical production model on all rows
    - Stage 2: duplicate-only batch-focused model on rows with chunk_size > 1
    """

    def __init__(
        self,
        base_predictor: BoschPredictor,
        batch_model_payload: dict[str, Any],
        batch_model_path: str | Path,
    ) -> None:
        self.base_predictor = base_predictor
        self.batch_model_payload = batch_model_payload
        self.batch_model_path = Path(batch_model_path)

        self.batch_models = list(batch_model_payload["models"])
        self.batch_feature_cols = list(batch_model_payload["feature_cols"])
        self.batch_threshold = float(batch_model_payload["threshold"])
        self.batch_oof_mcc = float(batch_model_payload.get("oof_mcc", np.nan))
        self.neighbor_source_cols = list(batch_model_payload.get("neighbor_source_cols", []))

        self.model_version = self._compute_model_version()

    @classmethod
    def load(
        cls,
        base_model_path: str | Path,
        pipeline_path: Optional[str | Path] = None,
        batch_model_path: str | Path = "data/models/model_batch_focused_v1.pkl",
    ) -> "TwoStagePredictor":
        base_predictor = BoschPredictor.load(base_model_path, pipeline_path=pipeline_path)

        batch_model_path = Path(batch_model_path)
        with batch_model_path.open("rb") as handle:
            batch_model_payload = pickle.load(handle)

        return cls(
            base_predictor=base_predictor,
            batch_model_payload=batch_model_payload,
            batch_model_path=batch_model_path,
        )

    def predict(
        self,
        df_raw: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame] = None,
        allow_temporal_fallback: bool = True,
        filter_singletons: bool = False,
    ) -> pd.DataFrame:
        features, prep_info = self.base_predictor._prepare_features(
            df_raw=df_raw,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
        )

        base_proba = self.base_predictor._ensemble_predict_proba(features)
        final_proba = base_proba.copy()
        batch_proba = np.full(len(features), np.nan, dtype=np.float32)
        enriched_features = features.copy()
        enriched_features.insert(0, "Id", df_raw["Id"].values)

        duplicate_mask = features["chunk_size"].values > 1
        if np.any(duplicate_mask):
            batch_features = self._build_batch_features(
                enriched_features.loc[duplicate_mask].copy(),
                base_proba[duplicate_mask],
            )
            batch_pred = self._predict_batch_proba(batch_features)
            batch_proba[duplicate_mask] = batch_pred
            final_proba[duplicate_mask] = batch_pred

        if filter_singletons:
            final_proba[~duplicate_mask] = 0.0

        predicted_label = (final_proba >= self.batch_threshold).astype(np.int8)
        timestamp = datetime.now(timezone.utc).isoformat()
        return pd.DataFrame(
            {
                "Id": df_raw["Id"].values,
                "base_predicted_proba": base_proba.astype(np.float32),
                "batch_predicted_proba": batch_proba,
                "final_predicted_proba": final_proba.astype(np.float32),
                "predicted_label": predicted_label,
                "is_duplicate_row": duplicate_mask.astype(np.int8),
                "prediction_timestamp": timestamp,
                "model_version": self.model_version,
                "missing_features_count": prep_info["missing_features_count"],
            }
        )

    def _predict_batch_proba(self, batch_features: pd.DataFrame) -> np.ndarray:
        predictions = []
        for model in self.batch_models:
            predictions.append(model.predict_proba(batch_features[self.batch_feature_cols])[:, 1])
        return np.mean(predictions, axis=0)

    def _build_batch_features(
        self,
        duplicate_features: pd.DataFrame,
        base_pred: np.ndarray,
    ) -> pd.DataFrame:
        df = duplicate_features.copy()
        df = df.sort_values(["chunk_id", "chunk_rank_asc", "Id"]).reset_index(drop=True)
        df["base_oof_pred"] = base_pred.astype(np.float32)

        df["position_in_chunk"] = df["chunk_rank_asc"].astype(np.int16)
        df["position_from_end"] = df["chunk_rank_desc"].astype(np.int16)
        df["position_ratio"] = np.where(
            df["chunk_size"] > 1,
            (df["position_in_chunk"] - 1) / (df["chunk_size"] - 1),
            0.0,
        ).astype(np.float32)
        df["is_chunk_start"] = (df["position_in_chunk"] == 1).astype(np.int8)
        df["is_chunk_end"] = (df["position_from_end"] == 1).astype(np.int8)

        for col in self.neighbor_source_cols:
            prev_col = f"{col}_prev"
            next_col = f"{col}_next"
            df[prev_col] = df.groupby("chunk_id")[col].shift(1)
            df[next_col] = df.groupby("chunk_id")[col].shift(-1)

            df[f"{prev_col}_missing"] = df[prev_col].isna().astype(np.int8)
            df[f"{next_col}_missing"] = df[next_col].isna().astype(np.int8)

            df[prev_col] = df[prev_col].fillna(-1.0).astype(np.float32)
            df[next_col] = df[next_col].fillna(-1.0).astype(np.float32)
            df[f"{col}_delta_prev"] = (df[col] - df[prev_col]).astype(np.float32)
            df[f"{col}_delta_next"] = (df[next_col] - df[col]).astype(np.float32)

        df["response_lag_prev_oof"] = df["base_oof_pred_prev"]
        df["response_lag_next_oof"] = df["base_oof_pred_next"]
        df["response_lag_prev_oof_cummean"] = (
            df.groupby("chunk_id")["base_oof_pred"].transform(lambda s: s.shift(1).expanding().mean())
        )
        df["response_lag_prev_oof_cummean"] = df["response_lag_prev_oof_cummean"].fillna(0.0).astype(np.float32)

        df["chunk_base_oof_mean"] = df.groupby("chunk_id")["base_oof_pred"].transform("mean").astype(np.float32)
        df["chunk_base_oof_max"] = df.groupby("chunk_id")["base_oof_pred"].transform("max").astype(np.float32)
        df["chunk_base_oof_std"] = (
            df.groupby("chunk_id")["base_oof_pred"].transform("std").fillna(0.0).astype(np.float32)
        )
        df["chunk_duration_mean"] = df.groupby("chunk_id")["duration"].transform("mean").astype(np.float32)
        df["chunk_duration_range"] = (
            df.groupby("chunk_id")["duration"].transform("max")
            - df.groupby("chunk_id")["duration"].transform("min")
        ).astype(np.float32)
        df["base_oof_rank_in_chunk"] = (
            df.groupby("chunk_id")["base_oof_pred"].rank(method="first", ascending=False).astype(np.int16)
        )
        df["duration_rank_in_chunk"] = (
            df.groupby("chunk_id")["duration"].rank(method="first", ascending=False).astype(np.int16)
        )

        for col in self.batch_feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _compute_model_version(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.base_predictor.model_path.read_bytes())
        digest.update(self.batch_model_path.read_bytes())
        return digest.hexdigest()[:8]
