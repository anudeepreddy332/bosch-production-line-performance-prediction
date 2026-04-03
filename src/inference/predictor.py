from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features import FeaturePipeline
from src.logger import setup_logger

logger = setup_logger(__name__)

LEAKY_FEATURE_PREFIXES = ("mean_timediff_till_next_",)


class BoschPredictor:
    """
    Raw-data inference interface for the canonical Bosch production model.

    The predictor owns:
    - a fitted FeaturePipeline
    - the production LightGBM ensemble payload
    - schema validation and output formatting
    """

    def __init__(
        self,
        pipeline: FeaturePipeline,
        model_payload: dict[str, Any],
        model_path: str | Path,
        allow_leaky_features: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.model_payload = model_payload
        self.model_path = Path(model_path)
        self.allow_leaky_features = allow_leaky_features

        self.models = model_payload["models"]
        self.feature_cols = list(model_payload["feature_cols"])
        self.threshold = float(model_payload["threshold"])
        self.oof_mcc = float(model_payload.get("oof_mcc", np.nan))
        self.fold_mccs = list(model_payload.get("fold_mccs", []))
        self.model_version = self._compute_model_version(self.model_path)

        self._validate_loaded_model()

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        pipeline_path: Optional[str | Path] = None,
        allow_leaky_features: bool = False,
    ) -> "BoschPredictor":
        model_path = Path(model_path)
        with model_path.open("rb") as handle:
            model_payload = pickle.load(handle)

        if pipeline_path is None:
            payload_pipeline_path = model_payload.get("pipeline_path")
            if payload_pipeline_path is None:
                raise ValueError("pipeline_path was not provided and is missing from the model payload.")
            pipeline_path = payload_pipeline_path

        pipeline = FeaturePipeline.load(pipeline_path)
        return cls(
            pipeline=pipeline,
            model_payload=model_payload,
            model_path=model_path,
            allow_leaky_features=allow_leaky_features,
        )

    def predict_proba(
        self,
        df_raw: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame] = None,
        allow_temporal_fallback: bool = True,
    ) -> np.ndarray:
        features, _ = self._prepare_features(
            df_raw=df_raw,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
        )

        fold_predictions: list[np.ndarray] = []
        for fold_models in self.models:
            seed_predictions = []
            for model in fold_models:
                seed_predictions.append(model.predict_proba(features)[:, 1])
            fold_predictions.append(np.mean(seed_predictions, axis=0))

        return np.mean(fold_predictions, axis=0)

    def predict(
        self,
        df_raw: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame] = None,
        allow_temporal_fallback: bool = True,
    ) -> pd.DataFrame:
        features, prep_info = self._prepare_features(
            df_raw=df_raw,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
        )
        predicted_proba = self._ensemble_predict_proba(features)
        predicted_label = (predicted_proba >= self.threshold).astype(np.int8)

        timestamp = datetime.now(timezone.utc).isoformat()
        return pd.DataFrame(
            {
                "Id": df_raw["Id"].values,
                "predicted_proba": predicted_proba,
                "predicted_label": predicted_label,
                "prediction_timestamp": timestamp,
                "model_version": self.model_version,
                "n_features_used": len(self.feature_cols),
                "missing_features_count": prep_info["missing_features_count"],
            }
        )

    def _prepare_features(
        self,
        df_raw: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame],
        allow_temporal_fallback: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if "Id" not in df_raw.columns:
            raise ValueError("Raw inference input must include an Id column.")

        transformed = self.pipeline.transform(
            df_raw,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
        )
        missing_features = [col for col in self.feature_cols if col not in transformed.columns]
        prep_info = {"missing_features_count": len(missing_features)}
        features = self._impute_missing(transformed)
        self._validate_schema(features)
        return features[self.feature_cols], prep_info

    def _validate_schema(self, df_features: pd.DataFrame) -> None:
        missing = [col for col in self.feature_cols if col not in df_features.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing[:20]}")

        leaky_present = [
            col for col in self.feature_cols if col.startswith(LEAKY_FEATURE_PREFIXES)
        ]
        if leaky_present and not self.allow_leaky_features:
            raise ValueError(
                f"Model requires leaky features but allow_leaky_features=False: {leaky_present}"
            )

        extra = [col for col in df_features.columns if col not in self.feature_cols]
        if extra:
            logger.info("Ignoring %s extra pipeline features not used by the model.", len(extra))

        null_count = int(df_features[self.feature_cols].isna().sum().sum())
        if null_count > 0:
            raise ValueError(f"Feature frame still contains {null_count} null values after imputation.")

    def _impute_missing(self, df_features: pd.DataFrame) -> pd.DataFrame:
        aligned = df_features.copy()
        for col in self.feature_cols:
            if col not in aligned.columns:
                aligned[col] = self.pipeline.fill_values_.get(col, 0.0)
            fill_value = self.pipeline.fill_values_.get(col)
            if fill_value is not None:
                aligned[col] = aligned[col].fillna(fill_value)
        return aligned

    def _ensemble_predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        fold_predictions: list[np.ndarray] = []
        for fold_models in self.models:
            seed_predictions = []
            for model in fold_models:
                seed_predictions.append(model.predict_proba(features[self.feature_cols])[:, 1])
            fold_predictions.append(np.mean(seed_predictions, axis=0))
        return np.mean(fold_predictions, axis=0)

    def _validate_loaded_model(self) -> None:
        leaky_features = [col for col in self.feature_cols if col.startswith(LEAKY_FEATURE_PREFIXES)]
        if leaky_features and not self.allow_leaky_features:
            raise ValueError(
                f"Loaded model requires leaky features but allow_leaky_features=False: {leaky_features}"
            )

        pipeline_missing = [col for col in self.feature_cols if col not in self.pipeline.feature_columns_]
        if pipeline_missing:
            raise ValueError(
                "Loaded pipeline does not expose all required model features: "
                f"{pipeline_missing[:20]}"
            )

    def _compute_model_version(self, model_path: Path) -> str:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return digest[:8]
