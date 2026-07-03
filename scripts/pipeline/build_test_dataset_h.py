"""Track 2/3 shared dependency: build dataset_h features for unlabeled test/incoming rows.

Both Track 2 (Kaggle submission, via scripts/generate_submission.py) and a future Track 3
production batch scorer need the same thing: turn unlabeled raw rows into the exact feature
contract models/dataset_h_model.pkl was trained on. This script is the one place that happens.

Reads data/processed/test_numeric.parquet + test_date.parquet (no Response column -- this is
unlabeled data and this script never reads or requires one), computes the same 8 baseline core
columns (start_time, duration, feature_mean, records_last_1hr, records_last_24hr, density_ratio,
chunk_id, chunk_size) and path_signature that scripts/build_dataset_baseline.py computes for
train data -- all self-contained, no train-derived statistics needed for these. Then applies the
train-derived lookup artifacts persisted by scripts/build_dataset_h.py
(data/features/dataset_h_lookup.json: global_mean, station_rate, trans_rate, path_count_train,
pair_count_train, all fit on labeled train data only) via
src.features.dataset_h_pipeline.apply_dataset_h_lookup to compute the remaining 8 train-derived
features. Output has Id plus every column in
src.features.dataset_h_pipeline.DATASET_H_FEATURE_COLS.

data/features/dataset_h_lookup.json is gitignored (regenerable via scripts/build_dataset_h.py,
not committed) and is NOT part of the models/dataset_h_model.pkl payload -- it is a second,
separate artifact this script depends on. Before doing any feature-building work, this script
cross-checks the lookup's embedded data_fingerprint against --model-path's (default
models/dataset_h_model.pkl) to fail fast, with a precise error, if they were fit from different
training runs rather than silently producing features the model wasn't trained to expect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from scripts.build_dataset_baseline import _build_date_core, _build_numeric_core
from src.features.core_pipeline import CorePipelineConfig, build_core_features
from src.features.dataset_h_pipeline import (
    DATASET_H_FEATURE_COLS,
    apply_dataset_h_lookup,
    load_dataset_h_lookup,
    validate_dataset_h_lookup_compatibility,
)
from src.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset_h features for unlabeled test/incoming rows (Track 2/3 shared contract)."
    )
    parser.add_argument("--numeric-path", type=Path, default=PROCESSED_DIR / "test_numeric.parquet")
    parser.add_argument("--date-path", type=Path, default=PROCESSED_DIR / "test_date.parquet")
    parser.add_argument("--lookup-path", type=Path, default=FEATURES_DIR / "dataset_h_lookup.json")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "dataset_h_model.pkl",
        help=(
            "Used only to cross-check the lookup's data_fingerprint before building features "
            "(fail fast if they were fit from different training runs). Pass --no-model-check "
            "to skip this if no model exists yet."
        ),
    )
    parser.add_argument(
        "--no-model-check",
        action="store_true",
        help="Skip the lookup<->model fingerprint cross-check (e.g. building features before any model is trained).",
    )
    parser.add_argument("--output", type=Path, default=FEATURES_DIR / "test_dataset_h.parquet")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--chunk-size-rows", type=int, default=10_000)
    args = parser.parse_args()

    for required in (args.numeric_path, args.date_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required input: {required}. Run scripts/prepare_data.py first."
            )

    lookup = load_dataset_h_lookup(args.lookup_path)

    if args.no_model_check:
        logger.warning("--no-model-check set: skipping lookup<->model fingerprint cross-check.")
    elif not args.model_path.exists():
        logger.warning(
            "%s does not exist; skipping lookup<->model fingerprint cross-check. The produced "
            "feature table's column set is still fixed by DATASET_H_FEATURE_COLS regardless.",
            args.model_path,
        )
    else:
        payload = joblib.load(args.model_path)
        problems = validate_dataset_h_lookup_compatibility(payload, lookup)
        if problems:
            raise ValueError(
                f"{args.lookup_path} is not compatible with {args.model_path}: {problems}. "
                f"Refusing to build a test feature table that would not match the model it's "
                f"meant to serve."
            )
        logger.info(
            "Lookup<->model fingerprint check passed (%s).", payload.get("data_fingerprint")
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_numeric = FEATURES_DIR / "_tmp_test_numeric_core.parquet"
    tmp_date = FEATURES_DIR / "_tmp_test_date_core.parquet"

    logger.info("Building numeric core features (unlabeled) from %s", args.numeric_path)
    _build_numeric_core(args.numeric_path, tmp_numeric, batch_size=args.batch_size)

    logger.info("Building date core + path signature (unlabeled) from %s", args.date_path)
    _build_date_core(args.date_path, tmp_date, batch_size=args.batch_size)

    numeric_df = pd.read_parquet(tmp_numeric)
    date_df = pd.read_parquet(tmp_date)

    if "Response" in numeric_df.columns:
        logger.warning(
            "%s unexpectedly has a Response column; dropping it -- this builder never reads "
            "labels for unlabeled test/incoming data.",
            args.numeric_path,
        )
        numeric_df = numeric_df.drop(columns=["Response"])

    merged = numeric_df.merge(date_df, on="Id", how="inner", validate="one_to_one")
    merged = merged.sort_values("Id", kind="mergesort").reset_index(drop=True)

    core = build_core_features(
        merged[["Id", "start_time", "duration", "feature_mean"]],
        config=CorePipelineConfig(chunk_size_rows=args.chunk_size_rows),
    )
    core_with_sig = core.merge(
        merged[["Id", "path_signature"]], on="Id", how="inner", validate="one_to_one"
    )

    featured = apply_dataset_h_lookup(core_with_sig, lookup)

    output_cols = ["Id", *DATASET_H_FEATURE_COLS]
    result = featured[output_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    for tmp_path in (tmp_numeric, tmp_date):
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info(
        "Saved unlabeled dataset_h test features: %s rows=%d cols=%d",
        args.output,
        len(result),
        len(output_cols),
    )


if __name__ == "__main__":
    main()
