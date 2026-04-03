from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

from src.logger import setup_logger

logger = setup_logger(__name__)

NUMERIC_FEATURE_PATTERN = re.compile(r"^L\d+_S\d+_F\d+$")
DATE_FEATURE_PATTERN = re.compile(r"^L\d+_S\d+_D\d+$")

LEGACY_TIME_WINDOWS = (25, 240, 1680)
TEMPORAL_HISTORY_KS = (1, 5, 10)
LEAKY_TEMPORAL_COLUMNS = [
    "mean_timediff_till_next_1",
    "mean_timediff_till_next_5",
    "mean_timediff_till_next_10",
]

DUPLICATE_PATTERN_COLUMNS = [
    "is_duplicate",
    "chunk_id",
    "chunk_size",
    "chunk_rank_asc",
    "chunk_rank_desc",
    "L0_S0_dup_count",
    "L0_S0_dup_rank",
    "L0_S12_dup_count",
    "L0_S12_dup_rank",
    "L3_S29_dup_count",
    "L3_S29_dup_rank",
    "L3_S30_dup_count",
    "L3_S30_dup_rank",
    "L3_S33_dup_count",
    "L3_S33_dup_rank",
    "total_station_dups",
    "L0_S0_concat_count",
    "L0_S12_concat_count",
    "L3_S29_concat_count",
    "L3_S30_concat_count",
    "L3_S33_concat_count",
    "L3_S35_concat_count",
    "total_concat_count",
]

PATH_STAT_COLUMNS = [
    "path_signature",
    "path_count",
    "path_failure_rate",
    "path_risk_tier",
]

DATE_FEATURE_COLUMNS = [
    "start_time",
    "end_time",
    "duration",
    "start_time_L0",
    "end_time_L0",
    "duration_L0",
    "n_stations_with_time_L0",
    "start_time_L1",
    "end_time_L1",
    "duration_L1",
    "n_stations_with_time_L1",
    "start_time_L2",
    "end_time_L2",
    "duration_L2",
    "n_stations_with_time_L2",
    "start_time_L3",
    "end_time_L3",
    "duration_L3",
    "n_stations_with_time_L3",
    "n_stations_with_time",
    "part_of_week",
    "day_of_week",
    "hour_of_day",
    "log_duration",
    "duration_z",
    "is_long_duration",
    "is_short_duration",
]

LEGACY_TIME_COLUMNS = [
    "records_last_25",
    "records_next_25",
    "records_last_240",
    "records_next_240",
    "records_last_1680",
    "records_next_1680",
    "id_mod_1680",
    "shift",
]

TEMPORAL_HISTORY_COLUMNS = [
    "mean_timediff_since_last_1",
    "mean_timediff_since_last_5",
    "mean_timediff_since_last_10",
    "records_same_6min",
]

ENGINEERED_BASE_COLUMNS = {
    "feature_mean",
    "feature_std",
    "feature_min",
    "feature_range",
    "S24_std",
    "S24_min",
    "S26_std",
    "sparsity_variance_interaction",
}


class FeaturePipeline:
    """
    Production-oriented Phase 5 feature pipeline.

    v1 scope is intentionally narrow:
    - Input is a single merged raw DataFrame
    - Output reproduces the current Phase 5 feature matrix contract
    - `till_next_*` temporal leakage features are excluded
    - Duplicate-pattern features are preserved as an isolated block
    """

    def __init__(
        self,
        min_path_count: int = 10,
        numeric_feature_list: Optional[list[str]] = None,
        categorical_encoded_feature_list: Optional[list[str]] = None,
    ) -> None:
        self.min_path_count = min_path_count
        self.numeric_feature_columns = numeric_feature_list or self._read_feature_list(
            Path("data/features/selected_features_top150.txt")
        )
        self.engineered_source_feature_columns = self._read_parquet_columns(
            Path("data/features/train_selected.parquet"), exclude={"Id", "Response"}
        )
        self.categorical_encoded_feature_columns = (
            categorical_encoded_feature_list
            or self._read_feature_list(Path("data/features/selected_categorical_top100.txt"))
        )
        self.categorical_raw_feature_columns = [
            col[: -len("_target_enc")] for col in self.categorical_encoded_feature_columns
        ]

        self.feature_columns_: list[str] = []
        self.feature_dtypes_: dict[str, str] = {}
        self.fill_values_: dict[str, Any] = {}
        self.feature_blocks_: dict[str, list[str]] = {}

        self.target_global_mean_: float = 0.0
        self.target_encoding_maps_: dict[str, dict[Any, float]] = {}
        self.path_global_mean_: float = 0.0
        self.path_failure_rate_map_: dict[str, float] = {}

        self.duration_mean_: float = 0.0
        self.duration_std_: float = 1.0
        self.long_duration_threshold_: float = 0.0
        self.short_duration_threshold_: float = 0.0

        self.station_keys_: list[str] = []
        self.date_station_keys_: list[str] = []
        self.date_entry_columns_: list[str] = []
        self.numeric_sensor_columns_: list[str] = []
        self.temporal_fallback_columns_: list[str] = [
            f"mean_timediff_since_last_{k}" for k in TEMPORAL_HISTORY_KS
        ]

        self.is_fitted_: bool = False
        self.version_: str = "feature_pipeline_v1"
        self.last_transform_info: dict[str, Any] = {}

    def fit(self, df_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "FeaturePipeline":
        y = self._resolve_target(df_train, y_train)
        self._initialize_structure(df_train)

        raw_features, feature_blocks, transform_info = self._build_features(
            df=df_train,
            y=y,
            temporal_context=None,
            allow_temporal_fallback=False,
            fit_mode=True,
        )

        self.feature_blocks_ = feature_blocks
        self.feature_columns_ = raw_features.columns.tolist()
        self.fill_values_ = self._derive_fill_values(raw_features)
        aligned = self._align_to_contract(raw_features)
        self.feature_dtypes_ = {col: str(aligned[col].dtype) for col in aligned.columns}
        self.is_fitted_ = True
        self.last_transform_info = transform_info
        return self

    def transform(
        self,
        df: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame] = None,
        allow_temporal_fallback: bool = True,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("FeaturePipeline must be fitted before calling transform().")

        raw_features, _, transform_info = self._build_features(
            df=df,
            y=None,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
            fit_mode=False,
        )
        aligned = self._align_to_contract(raw_features)
        self.last_transform_info = transform_info
        return aligned

    def fit_transform(
        self, df_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        y = self._resolve_target(df_train, y_train)
        self._initialize_structure(df_train)

        raw_features, feature_blocks, transform_info = self._build_features(
            df=df_train,
            y=y,
            temporal_context=None,
            allow_temporal_fallback=False,
            fit_mode=True,
        )

        self.feature_blocks_ = feature_blocks
        self.feature_columns_ = raw_features.columns.tolist()
        self.fill_values_ = self._derive_fill_values(raw_features)
        aligned = self._align_to_contract(raw_features)
        self.feature_dtypes_ = {col: str(aligned[col].dtype) for col in aligned.columns}
        self.is_fitted_ = True
        self.last_transform_info = transform_info
        return aligned

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version_,
            "min_path_count": self.min_path_count,
            "numeric_feature_columns": self.numeric_feature_columns,
            "categorical_encoded_feature_columns": self.categorical_encoded_feature_columns,
            "categorical_raw_feature_columns": self.categorical_raw_feature_columns,
            "feature_columns_": self.feature_columns_,
            "feature_dtypes_": self.feature_dtypes_,
            "fill_values_": self.fill_values_,
            "feature_blocks_": self.feature_blocks_,
            "target_global_mean_": self.target_global_mean_,
            "target_encoding_maps_": self.target_encoding_maps_,
            "path_global_mean_": self.path_global_mean_,
            "path_failure_rate_map_": self.path_failure_rate_map_,
            "duration_mean_": self.duration_mean_,
            "duration_std_": self.duration_std_,
            "long_duration_threshold_": self.long_duration_threshold_,
            "short_duration_threshold_": self.short_duration_threshold_,
            "station_keys_": self.station_keys_,
            "date_station_keys_": self.date_station_keys_,
            "date_entry_columns_": self.date_entry_columns_,
            "numeric_sensor_columns_": self.numeric_sensor_columns_,
            "temporal_fallback_columns_": self.temporal_fallback_columns_,
            "is_fitted_": self.is_fitted_,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        pipeline = cls(
            min_path_count=payload["min_path_count"],
            numeric_feature_list=payload["numeric_feature_columns"],
            categorical_encoded_feature_list=payload["categorical_encoded_feature_columns"],
        )

        pipeline.categorical_raw_feature_columns = payload["categorical_raw_feature_columns"]
        pipeline.feature_columns_ = payload["feature_columns_"]
        pipeline.feature_dtypes_ = payload["feature_dtypes_"]
        pipeline.fill_values_ = payload["fill_values_"]
        pipeline.feature_blocks_ = payload["feature_blocks_"]
        pipeline.target_global_mean_ = payload["target_global_mean_"]
        pipeline.target_encoding_maps_ = payload["target_encoding_maps_"]
        pipeline.path_global_mean_ = payload["path_global_mean_"]
        pipeline.path_failure_rate_map_ = payload["path_failure_rate_map_"]
        pipeline.duration_mean_ = payload["duration_mean_"]
        pipeline.duration_std_ = payload["duration_std_"]
        pipeline.long_duration_threshold_ = payload["long_duration_threshold_"]
        pipeline.short_duration_threshold_ = payload["short_duration_threshold_"]
        pipeline.station_keys_ = payload["station_keys_"]
        pipeline.date_station_keys_ = payload["date_station_keys_"]
        pipeline.date_entry_columns_ = payload["date_entry_columns_"]
        pipeline.numeric_sensor_columns_ = payload["numeric_sensor_columns_"]
        pipeline.temporal_fallback_columns_ = payload["temporal_fallback_columns_"]
        pipeline.is_fitted_ = payload["is_fitted_"]
        return pipeline

    def _build_features(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        temporal_context: Optional[pd.DataFrame],
        allow_temporal_fallback: bool,
        fit_mode: bool,
    ) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
        transform_info = {
            "temporal_context_used": False,
            "temporal_fallback_used": False,
            "warnings": [],
        }

        base_numeric = self._build_base_numeric_block(df)
        categorical = self._build_target_encoded_categorical_block(df, y, fit_mode)
        duplicate_pattern = self._build_duplicate_pattern_block(df)
        path_block = self._build_path_block(df, y, fit_mode)
        raw_date_block = self._build_raw_date_block(df, fit_mode)
        date_block = self._fill_block_with_defaults(raw_date_block)
        legacy_time = self._build_legacy_time_block(df)
        temporal_history, temporal_info = self._build_temporal_history_block(
            df=df,
            y=y,
            date_block=date_block,
            temporal_context=temporal_context,
            allow_temporal_fallback=allow_temporal_fallback,
            fit_mode=fit_mode,
        )
        transform_info.update(temporal_info)

        feature_blocks = {
            "base_numeric": base_numeric.columns.tolist(),
            "target_encoded_categorical": categorical.columns.tolist(),
            "duplicate_pattern": duplicate_pattern.columns.tolist(),
            "path": path_block.columns.tolist(),
            "date": date_block.columns.tolist(),
            "legacy_time": legacy_time.columns.tolist(),
            "temporal_history": temporal_history.columns.tolist(),
        }

        features = pd.concat(
            [
                base_numeric,
                categorical,
                duplicate_pattern,
                path_block,
                date_block,
                legacy_time,
                temporal_history,
            ],
            axis=1,
        )

        return features, feature_blocks, transform_info

    def _build_base_numeric_block(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_sensor_cols = self._numeric_sensor_columns(df)
        aggregate_source_cols = [
            col for col in self.engineered_source_feature_columns if col in df.columns
        ]
        sensor_df = (
            df[aggregate_source_cols] if aggregate_source_cols else pd.DataFrame(index=df.index)
        )

        derived_values: dict[str, pd.Series] = {}
        if not sensor_df.empty:
            derived_values["feature_mean"] = sensor_df.mean(axis=1)
            derived_values["feature_std"] = sensor_df.std(axis=1)
            derived_values["feature_min"] = sensor_df.min(axis=1)
            feature_max = sensor_df.max(axis=1)
            derived_values["feature_range"] = feature_max - derived_values["feature_min"]

            s24_cols = [col for col in aggregate_source_cols if "_S24_" in col]
            if s24_cols:
                s24_df = df[s24_cols]
                derived_values["S24_std"] = s24_df.std(axis=1)
                derived_values["S24_min"] = s24_df.min(axis=1)

            s26_cols = [col for col in aggregate_source_cols if "_S26_" in col]
            if s26_cols:
                derived_values["S26_std"] = df[s26_cols].std(axis=1)

            overall_sparsity = sensor_df.isnull().sum(axis=1) / len(aggregate_source_cols)
            derived_values["sparsity_variance_interaction"] = (
                overall_sparsity * derived_values["feature_std"]
            )

        block_values: dict[str, pd.Series | float] = {}
        for column in self.numeric_feature_columns:
            if column in df.columns:
                block_values[column] = df[column]
            elif column in derived_values:
                block_values[column] = derived_values[column]
            elif column in ENGINEERED_BASE_COLUMNS:
                block_values[column] = np.nan
            else:
                block_values[column] = np.nan

        return pd.DataFrame(block_values, index=df.index)

    def _build_target_encoded_categorical_block(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        fit_mode: bool,
    ) -> pd.DataFrame:
        block = pd.DataFrame(index=df.index)

        if fit_mode:
            if y is None:
                raise ValueError("Training target is required to fit categorical target encodings.")
            self.target_global_mean_ = float(pd.Series(y).mean())
            self.target_encoding_maps_ = {}
            for raw_col in self.categorical_raw_feature_columns:
                if raw_col not in df.columns:
                    self.target_encoding_maps_[raw_col] = {}
                    continue
                means = pd.DataFrame(
                    {"value": df[raw_col].astype(object), "target": y}, index=df.index
                ).groupby("value", observed=False)["target"].mean()
                self.target_encoding_maps_[raw_col] = means.to_dict()

        for raw_col, encoded_col in zip(
            self.categorical_raw_feature_columns, self.categorical_encoded_feature_columns
        ):
            if raw_col in df.columns:
                mapping = self.target_encoding_maps_.get(raw_col, {})
                encoded = df[raw_col].astype(object).map(mapping)
                block[encoded_col] = encoded.fillna(self.target_global_mean_).astype(np.float32)
            else:
                block[encoded_col] = np.float32(self.target_global_mean_)

        return block

    def _build_duplicate_pattern_block(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = self._numeric_sensor_columns(df)
        block = pd.DataFrame(index=df.index, columns=DUPLICATE_PATTERN_COLUMNS, dtype=float)
        if not sensor_cols:
            return self._finalize_duplicate_block(block.fillna(0))

        raw_start_time = self._compute_raw_start_time(df)
        sorted_index = (
            pd.DataFrame({"start_time": raw_start_time, "Id": df["Id"], "_idx": df.index})
            .sort_values(["start_time", "Id"], kind="stable", na_position="last")["_idx"]
            .to_numpy()
        )

        sorted_sensor_df = df.loc[sorted_index, sensor_cols]
        sorted_result = pd.DataFrame(index=sorted_index)

        row_hash = hash_pandas_object(sorted_sensor_df, index=False)
        sorted_result["is_duplicate"] = row_hash.eq(row_hash.shift(1)).astype(np.int8)
        sorted_result["chunk_id"] = (~sorted_result["is_duplicate"].astype(bool)).cumsum().astype(
            np.int32
        )

        chunk_sizes = sorted_result.groupby("chunk_id").size().astype(np.int32)
        sorted_result["chunk_size"] = sorted_result["chunk_id"].map(chunk_sizes)
        sorted_result["chunk_rank_asc"] = (
            sorted_result.groupby("chunk_id").cumcount() + 1
        ).astype(np.int32)
        sorted_result["chunk_rank_desc"] = (
            sorted_result["chunk_size"] - sorted_result["chunk_rank_asc"] + 1
        ).astype(np.int32)

        dup_stations = ["L0_S0", "L0_S12", "L3_S29", "L3_S30", "L3_S33"]
        for station in dup_stations:
            count_col = f"{station}_dup_count"
            rank_col = f"{station}_dup_rank"
            counts = np.zeros(len(sorted_result), dtype=np.int32)
            ranks = np.zeros(len(sorted_result), dtype=np.int32)

            station_cols = [col for col in sensor_cols if col.startswith(f"{station}_")]
            if station_cols:
                station_df = sorted_sensor_df[station_cols]
                has_data = station_df.notna().any(axis=1)
                if has_data.any():
                    hashed = pd.Series(
                        hash_pandas_object(station_df.loc[has_data], index=False).values,
                        index=station_df.index[has_data],
                    )
                    value_counts = hashed.value_counts()
                    counts_series = hashed.map(value_counts).astype(np.int32)
                    rank_series = (hashed.groupby(hashed, sort=False).cumcount() + 1).astype(
                        np.int32
                    )
                    target_positions = sorted_result.index.get_indexer(hashed.index)
                    counts[target_positions] = counts_series.to_numpy()
                    ranks[target_positions] = rank_series.to_numpy()

            sorted_result[count_col] = counts
            sorted_result[rank_col] = ranks

        dup_count_cols = [f"{station}_dup_count" for station in dup_stations]
        sorted_result["total_station_dups"] = (
            sorted_result[dup_count_cols].sum(axis=1).astype(np.int32)
        )

        concat_stations = ["L0_S0", "L0_S12", "L3_S29", "L3_S30", "L3_S33", "L3_S35"]
        concat_count_cols: list[str] = []
        for station in concat_stations:
            count_col = f"{station}_concat_count"
            concat_count_cols.append(count_col)
            counts = np.zeros(len(sorted_result), dtype=np.int32)
            station_cols = [col for col in sensor_cols if col.startswith(f"{station}_")]
            if station_cols:
                station_df = sorted_sensor_df[station_cols]
                has_data = station_df.notna().any(axis=1)
                if has_data.any():
                    valid_df = station_df.loc[has_data].round(6)
                    valid_df = valid_df.astype(object).where(valid_df.notna(), "__nan__")
                    hashed = pd.Series(
                        hash_pandas_object(valid_df, index=False).values,
                        index=valid_df.index,
                    )
                    n_valid = len(hashed)
                    n_unique = hashed.nunique()
                    if n_valid > 0 and (n_unique / n_valid) >= 0.01:
                        value_counts = hashed.map(hashed.value_counts()).astype(np.int32)
                        target_positions = sorted_result.index.get_indexer(hashed.index)
                        counts[target_positions] = value_counts.to_numpy()
            sorted_result[count_col] = counts

        sorted_result["total_concat_count"] = np.log1p(sorted_result[concat_count_cols]).sum(
            axis=1
        )

        block = sorted_result.reindex(df.index)
        return self._finalize_duplicate_block(block)

    def _build_path_block(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        fit_mode: bool,
    ) -> pd.DataFrame:
        sensor_cols = self._numeric_sensor_columns(df)
        block = pd.DataFrame(index=df.index)
        station_keys = self.station_keys_ or self._sorted_station_keys(sensor_cols)

        visited_columns: list[str] = []
        for station in station_keys:
            visited_col = f"visited_{station}"
            station_cols = [col for col in sensor_cols if col.startswith(f"{station}_")]
            if station_cols:
                block[visited_col] = df[station_cols].notna().any(axis=1).astype(np.int8)
            else:
                block[visited_col] = 0
            visited_columns.append(visited_col)

        for line in ["L0", "L1", "L2", "L3"]:
            line_cols = [col for col in visited_columns if col.startswith(f"visited_{line}_")]
            block[f"n_stations_{line}"] = (
                block[line_cols].sum(axis=1).astype(np.int8) if line_cols else 0
            )

        path_count_inputs = [f"n_stations_{line}" for line in ["L0", "L1", "L2", "L3"]]
        block["n_stations_total"] = block[path_count_inputs].sum(axis=1).astype(np.int16)

        def row_to_path(row: pd.Series) -> str:
            visited = [col[8:] for col in visited_columns if row[col] == 1]
            return "|".join(visited) if visited else "no_stations"

        block["path_signature"] = block[visited_columns].apply(row_to_path, axis=1)
        block["path_count"] = block["path_signature"].map(block["path_signature"].value_counts())

        if fit_mode:
            if y is None:
                raise ValueError("Training target is required to fit path failure rate statistics.")
            self.path_global_mean_ = float(pd.Series(y).mean())
            path_stats = pd.DataFrame({"path": block["path_signature"], "target": y})
            aggregated = (
                path_stats.groupby("path")["target"].agg(["mean", "count"]).rename(
                    columns={"mean": "rate", "count": "cnt"}
                )
            )
            aggregated["smoothed_rate"] = (
                aggregated["cnt"] * aggregated["rate"] + self.min_path_count * self.path_global_mean_
            ) / (aggregated["cnt"] + self.min_path_count)
            self.path_failure_rate_map_ = aggregated["smoothed_rate"].to_dict()

        block["path_failure_rate"] = block["path_signature"].map(self.path_failure_rate_map_)
        block["path_failure_rate"] = block["path_failure_rate"].fillna(self.path_global_mean_)
        block["path_risk_tier"] = pd.cut(
            block["path_count"],
            bins=[0, 10, 100, float("inf")],
            labels=[0, 1, 2],
        ).astype(np.int8)

        path_columns = visited_columns + [
            "n_stations_L0",
            "n_stations_L1",
            "n_stations_L2",
            "n_stations_L3",
            "n_stations_total",
        ] + PATH_STAT_COLUMNS
        return block[path_columns]

    def _build_raw_date_block(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        block = pd.DataFrame(index=df.index)
        date_cols = self._date_columns(df)
        station_keys = self.date_station_keys_ or self._sorted_station_keys(date_cols)
        entry_columns = [f"entry_{station}" for station in station_keys]

        station_times = pd.DataFrame(index=df.index)
        for station in station_keys:
            station_cols = [col for col in date_cols if col.startswith(f"{station}_")]
            entry_col = f"entry_{station}"
            if station_cols:
                station_times[entry_col] = df[station_cols].min(axis=1)
            else:
                station_times[entry_col] = np.nan

        if fit_mode:
            duration = station_times.min(axis=1)
            end_time = station_times.max(axis=1)
            raw_duration = end_time - duration
            self.duration_mean_ = float(raw_duration.mean())
            duration_std = float(raw_duration.std())
            self.duration_std_ = duration_std if duration_std > 0 else 1.0
            self.long_duration_threshold_ = float(raw_duration.quantile(0.90))
            self.short_duration_threshold_ = float(raw_duration.quantile(0.10))

        block["start_time"] = station_times.min(axis=1)
        block["end_time"] = station_times.max(axis=1)
        block["duration"] = block["end_time"] - block["start_time"]

        for line in ["L0", "L1", "L2", "L3"]:
            line_entry_cols = [col for col in entry_columns if col.startswith(f"entry_{line}_")]
            block[f"start_time_{line}"] = (
                station_times[line_entry_cols].min(axis=1) if line_entry_cols else np.nan
            )
            block[f"end_time_{line}"] = (
                station_times[line_entry_cols].max(axis=1) if line_entry_cols else np.nan
            )
            block[f"duration_{line}"] = block[f"end_time_{line}"] - block[f"start_time_{line}"]
            if line_entry_cols:
                block[f"n_stations_with_time_{line}"] = (
                    station_times[line_entry_cols].notna().sum(axis=1).astype(np.int16)
                )
            else:
                block[f"n_stations_with_time_{line}"] = 0

        block["n_stations_with_time"] = station_times[entry_columns].notna().sum(axis=1).astype(
            np.int16
        )
        block["part_of_week"] = block["start_time"] % 1680
        block["day_of_week"] = (block["start_time"] // 240).astype("float32") % 7
        block["hour_of_day"] = (block["start_time"] // 10).astype("float32") % 24
        block["log_duration"] = np.log1p(block["duration"])
        block["duration_z"] = (block["duration"] - self.duration_mean_) / self.duration_std_
        block["is_long_duration"] = (
            block["duration"] > self.long_duration_threshold_
        ).astype("float32")
        block["is_short_duration"] = (
            block["duration"] < self.short_duration_threshold_
        ).astype("float32")

        for entry_col in entry_columns:
            block[entry_col] = station_times[entry_col]

        ordered_columns = DATE_FEATURE_COLUMNS + entry_columns
        return block[ordered_columns]

    def _build_legacy_time_block(self, df: pd.DataFrame) -> pd.DataFrame:
        id_values = pd.Series(df["Id"], index=df.index)
        sort_order = id_values.sort_values(kind="stable").index.to_numpy()
        n_rows = len(df)
        block = pd.DataFrame(index=df.index)

        for window in LEGACY_TIME_WINDOWS:
            last_counts = np.minimum(np.arange(n_rows) + 1, window).astype(np.int32)
            next_counts = np.minimum(np.arange(n_rows, 0, -1), window).astype(np.int32)
            block[f"records_last_{window}"] = pd.Series(last_counts, index=sort_order).reindex(
                df.index
            )
            block[f"records_next_{window}"] = pd.Series(next_counts, index=sort_order).reindex(
                df.index
            )

        block["id_mod_1680"] = df["Id"] % 1680
        hour_proxy = (df["Id"] // 10) % 24
        block["shift"] = pd.cut(hour_proxy, bins=[0, 8, 16, 24], labels=[0, 1, 2]).astype(float)
        return block[LEGACY_TIME_COLUMNS]

    def _build_temporal_history_block(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series],
        date_block: pd.DataFrame,
        temporal_context: Optional[pd.DataFrame],
        allow_temporal_fallback: bool,
        fit_mode: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        info = {
            "temporal_context_used": False,
            "temporal_fallback_used": False,
            "warnings": [],
        }

        block = pd.DataFrame(index=df.index)
        end_times = date_block["end_time"].to_numpy()
        start_times = date_block["start_time"].to_numpy()

        if fit_mode:
            if y is None:
                raise ValueError("Training target is required to fit temporal history features.")
            block = self._compute_since_last_features(end_times, np.asarray(y))
        else:
            if temporal_context is not None:
                failure_times = self._extract_failure_times_from_context(temporal_context)
                if len(failure_times) > 0:
                    block = self._compute_since_last_features_from_failure_times(
                        end_times=end_times,
                        failure_times=failure_times,
                    )
                    info["temporal_context_used"] = True
                elif allow_temporal_fallback:
                    block = self._temporal_fallback_block(df.index)
                    info["temporal_fallback_used"] = True
                    info["warnings"].append(
                        "Temporal context was provided but did not contain any usable failure history."
                    )
                else:
                    raise ValueError(
                        "Temporal context did not contain usable failure history and fallback is disabled."
                    )
            elif allow_temporal_fallback:
                block = self._temporal_fallback_block(df.index)
                info["temporal_fallback_used"] = True
                info["warnings"].append(
                    "Temporal history context is missing; using fitted fallback values for since_last features."
                )
            else:
                raise ValueError(
                    "Temporal history context is required for since_last features when fallback is disabled."
                )

        block["records_same_6min"] = self._compute_same_window_count(start_times, window=0.01)
        return block[TEMPORAL_HISTORY_COLUMNS], info

    def _compute_since_last_features(
        self, end_times: np.ndarray, responses: np.ndarray
    ) -> pd.DataFrame:
        sort_idx = np.argsort(end_times, kind="stable")
        sorted_end_times = end_times[sort_idx]
        sorted_responses = responses[sort_idx]
        failure_times = sorted_end_times[sorted_responses == 1]
        sorted_block = self._compute_since_last_from_sorted_end_times(sorted_end_times, failure_times)
        unsort_idx = np.argsort(sort_idx, kind="stable")
        return sorted_block.iloc[unsort_idx].reset_index(drop=True)

    def _compute_since_last_features_from_failure_times(
        self, end_times: np.ndarray, failure_times: np.ndarray
    ) -> pd.DataFrame:
        sort_idx = np.argsort(end_times, kind="stable")
        sorted_end_times = end_times[sort_idx]
        sorted_failure_times = np.sort(np.asarray(failure_times), kind="stable")
        sorted_block = self._compute_since_last_from_sorted_end_times(
            sorted_end_times, sorted_failure_times
        )
        unsort_idx = np.argsort(sort_idx, kind="stable")
        return sorted_block.iloc[unsort_idx].reset_index(drop=True)

    def _compute_since_last_from_sorted_end_times(
        self, sorted_end_times: np.ndarray, failure_times: np.ndarray
    ) -> pd.DataFrame:
        failure_cumsum = np.concatenate([[0.0], np.cumsum(failure_times)])
        idx_before = np.searchsorted(failure_times, sorted_end_times, side="left")
        results: dict[str, np.ndarray] = {}

        for k in TEMPORAL_HISTORY_KS:
            values = np.full(len(sorted_end_times), np.nan)
            has_k = idx_before >= k
            valid_idx = idx_before[has_k]
            if len(valid_idx) > 0:
                mean_last_k = (failure_cumsum[valid_idx] - failure_cumsum[valid_idx - k]) / k
                values[has_k] = sorted_end_times[has_k] - mean_last_k
            results[f"mean_timediff_since_last_{k}"] = values

        return pd.DataFrame(results)

    def _compute_same_window_count(self, start_times: np.ndarray, window: float) -> np.ndarray:
        sort_idx = np.argsort(start_times, kind="stable")
        times_sorted = start_times[sort_idx]
        half_window = window / 2.0
        right_idx = np.searchsorted(times_sorted, times_sorted + half_window, side="right")
        left_idx = np.searchsorted(times_sorted, times_sorted - half_window, side="left")
        counts = (right_idx - left_idx - 1).astype(np.int32)

        result = np.empty(len(start_times), dtype=np.int32)
        result[sort_idx] = counts
        return result

    def _extract_failure_times_from_context(self, temporal_context: pd.DataFrame) -> np.ndarray:
        if "end_time" in temporal_context.columns:
            end_times = temporal_context["end_time"]
        else:
            end_times = self._compute_raw_end_time(temporal_context)

        if "Response" not in temporal_context.columns:
            raise ValueError(
                "temporal_context must include a Response column or a precomputed end_time + Response pair."
            )

        failure_times = end_times[temporal_context["Response"] == 1].dropna().to_numpy()
        return np.sort(failure_times, kind="stable")

    def _temporal_fallback_block(self, index: pd.Index) -> pd.DataFrame:
        block = pd.DataFrame(index=index)
        for column in self.temporal_fallback_columns_:
            block[column] = self.fill_values_.get(column, 0.0)
        return block

    def _fill_block_with_defaults(self, block: pd.DataFrame) -> pd.DataFrame:
        filled = block.copy()
        for column in filled.columns:
            fill_value = self.fill_values_.get(column)
            if fill_value is None:
                if pd.api.types.is_numeric_dtype(filled[column]):
                    fill_value = self._safe_numeric_fill(filled[column])
                else:
                    fill_value = "unknown_path" if column == "path_signature" else ""
            filled[column] = filled[column].fillna(fill_value)
        return filled

    def _derive_fill_values(self, features: pd.DataFrame) -> dict[str, Any]:
        fill_values: dict[str, Any] = {}
        for column in features.columns:
            if pd.api.types.is_numeric_dtype(features[column]):
                fill_values[column] = self._safe_numeric_fill(features[column])
            else:
                fill_values[column] = "unknown_path" if column == "path_signature" else ""
        return fill_values

    def _safe_numeric_fill(self, series: pd.Series) -> float:
        median = series.median()
        if pd.isna(median):
            return 0.0
        return float(median)

    def _align_to_contract(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise ValueError("Feature schema is not available. Fit the pipeline first.")

        aligned = features.copy()
        missing_columns = [col for col in self.feature_columns_ if col not in aligned.columns]
        extra_columns = [col for col in aligned.columns if col not in self.feature_columns_]

        for column in missing_columns:
            aligned[column] = self.fill_values_.get(column, 0.0)

        if extra_columns:
            aligned = aligned.drop(columns=extra_columns)

        aligned = aligned[self.feature_columns_]

        for column in self.feature_columns_:
            fill_value = self.fill_values_.get(column)
            if fill_value is not None:
                aligned[column] = aligned[column].fillna(fill_value)

            target_dtype = self.feature_dtypes_.get(column)
            if target_dtype:
                if target_dtype.startswith("int"):
                    aligned[column] = np.round(aligned[column]).astype(target_dtype)
                elif target_dtype == "object":
                    aligned[column] = aligned[column].astype(object)
                else:
                    aligned[column] = aligned[column].astype(target_dtype)

        return aligned

    def _resolve_target(
        self, df: pd.DataFrame, y_train: Optional[pd.Series | np.ndarray]
    ) -> pd.Series:
        if y_train is not None:
            return pd.Series(y_train, index=df.index, name="Response")
        if "Response" in df.columns:
            return df["Response"]
        raise ValueError("Training target is required. Pass y_train or include Response in df.")

    def _initialize_structure(self, df: pd.DataFrame) -> None:
        self.numeric_sensor_columns_ = self._numeric_sensor_columns(df)
        self.station_keys_ = self._sorted_station_keys(self.numeric_sensor_columns_)
        self.date_station_keys_ = self._sorted_station_keys(self._date_columns(df))
        self.date_entry_columns_ = [f"entry_{station}" for station in self.date_station_keys_]

    def _numeric_sensor_columns(self, df: pd.DataFrame) -> list[str]:
        if self.numeric_sensor_columns_:
            return [col for col in self.numeric_sensor_columns_ if col in df.columns]
        return [
            col
            for col in df.columns
            if NUMERIC_FEATURE_PATTERN.match(col) and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _date_columns(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if DATE_FEATURE_PATTERN.match(col)]

    def _compute_raw_start_time(self, df: pd.DataFrame) -> pd.Series:
        date_cols = self._date_columns(df)
        if not date_cols:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return df[date_cols].min(axis=1)

    def _compute_raw_end_time(self, df: pd.DataFrame) -> pd.Series:
        date_cols = self._date_columns(df)
        if not date_cols:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return df[date_cols].max(axis=1)

    def _sorted_station_keys(self, columns: list[str]) -> list[str]:
        station_keys = {self._station_key(col) for col in columns}
        return sorted(station_keys, key=self._station_sort_key)

    def _station_key(self, column_name: str) -> str:
        parts = column_name.split("_")
        return f"{parts[0]}_{parts[1]}"

    def _station_sort_key(self, station_key: str) -> tuple[int, int]:
        line, station = station_key.split("_")
        return int(line[1:]), int(station[1:])

    def _finalize_duplicate_block(self, block: pd.DataFrame) -> pd.DataFrame:
        finalized = block.copy()
        int_columns = [
            "is_duplicate",
            "chunk_id",
            "chunk_size",
            "chunk_rank_asc",
            "chunk_rank_desc",
            "L0_S0_dup_count",
            "L0_S0_dup_rank",
            "L0_S12_dup_count",
            "L0_S12_dup_rank",
            "L3_S29_dup_count",
            "L3_S29_dup_rank",
            "L3_S30_dup_count",
            "L3_S30_dup_rank",
            "L3_S33_dup_count",
            "L3_S33_dup_rank",
            "total_station_dups",
            "L0_S0_concat_count",
            "L0_S12_concat_count",
            "L3_S29_concat_count",
            "L3_S30_concat_count",
            "L3_S33_concat_count",
            "L3_S35_concat_count",
        ]
        for column in int_columns:
            finalized[column] = finalized[column].fillna(0).astype(np.int32)
        finalized["is_duplicate"] = finalized["is_duplicate"].astype(np.int8)
        finalized["total_concat_count"] = finalized["total_concat_count"].fillna(0.0).astype(
            np.float32
        )
        return finalized[DUPLICATE_PATTERN_COLUMNS]

    def _read_feature_list(self, path: Path) -> list[str]:
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise FileNotFoundError(f"Required feature list not found: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _read_parquet_columns(self, path: Path, exclude: Optional[set[str]] = None) -> list[str]:
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise FileNotFoundError(f"Required parquet artifact not found: {path}")
        columns = pq.ParquetFile(path).schema.names
        exclude = exclude or set()
        return [column for column in columns if column not in exclude]
