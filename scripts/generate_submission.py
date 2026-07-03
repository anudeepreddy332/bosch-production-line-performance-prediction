"""Track 2: Kaggle Submission (see docs/ml_system_tracks.md).

Loads an approved Phase-2 model payload (the joblib dict produced by
build_model_payload: {"models", "feature_cols", "threshold", ...}), applies it
to an already feature-engineered, unlabeled test table, and writes
submission.csv with exactly Id,Response.

This script does NOT do feature engineering and does NOT compute supervised
metrics. Kaggle test labels are hidden, and this script never reads or scores
against a Response/label column even if one happens to be present in the
input file -- only Id and the payload's feature_cols are used. --test-features
must already contain Id plus every column in the model payload's feature_cols;
see docs/kaggle_submission.md for why no such engineered test table currently
exists in this repo for the real Bosch test set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.payload import load_validated_payload, predict_proba_ensemble

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_SUBMISSION = ROOT / "data" / "processed" / "sample_submission.parquet"
DEFAULT_OUTPUT = ROOT / "outputs" / "submission.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission.csv (Id,Response) from an approved model payload."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to a joblib-dumped Phase-2 model payload (e.g. models/meta_model.pkl).",
    )
    parser.add_argument(
        "--test-features",
        required=True,
        type=Path,
        help=(
            "Parquet file with an Id column plus every column in the model payload's "
            "feature_cols. Must be unlabeled test data; any Response column present is ignored."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write submission.csv.")
    parser.add_argument("--id-col", default="Id", help="Id column name in --test-features (default: Id).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the payload's stored decision threshold (default: use payload['threshold']).",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=DEFAULT_SAMPLE_SUBMISSION,
        help="Used only to warn on row-count/Id mismatch; never read for prediction.",
    )
    return parser.parse_args(argv)


def load_test_features(test_features_path: Path, feature_cols: list[str], id_col: str) -> pd.DataFrame:
    if not test_features_path.exists():
        raise FileNotFoundError(f"Test feature file not found: {test_features_path}")

    df = pd.read_parquet(test_features_path)

    if id_col not in df.columns:
        raise ValueError(f"{test_features_path} is missing required id column {id_col!r}.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{test_features_path} is missing {len(missing)} feature column(s) required by the "
            f"model payload: {missing}. This script does not perform feature engineering -- build "
            f"a matching engineered test table first (see docs/kaggle_submission.md)."
        )

    if "Response" in df.columns:
        print(
            f"WARNING: {test_features_path} contains a 'Response' column; it is ignored. "
            "Kaggle test data is unlabeled and this script never scores against labels."
        )

    return df


def check_against_sample_submission(ids: pd.Series, sample_submission_path: Path) -> None:
    if not sample_submission_path.exists():
        print(f"NOTE: no sample_submission found at {sample_submission_path}; skipping row-count/Id check.")
        return

    sample = pd.read_parquet(sample_submission_path)
    if len(ids) != len(sample):
        print(
            f"WARNING: row count {len(ids)} != sample_submission row count {len(sample)}. "
            "A valid Kaggle submission must match the official test set row count exactly."
        )

    our_ids = set(ids.tolist())
    sample_ids = set(sample["Id"].tolist())
    missing = sample_ids - our_ids
    extra = our_ids - sample_ids
    if missing or extra:
        print(f"WARNING: Id mismatch vs sample_submission -- missing {len(missing)}, extra {len(extra)}.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        payload = load_validated_payload(args.model_path)
        threshold = args.threshold if args.threshold is not None else payload["threshold"]

        features = load_test_features(args.test_features, payload["feature_cols"], args.id_col)
        proba = predict_proba_ensemble(payload, features)
        response = (proba >= float(threshold)).astype(np.int8)

        submission = pd.DataFrame(
            {
                "Id": features[args.id_col].to_numpy(dtype=np.int64),
                "Response": response,
            }
        )

        check_against_sample_submission(submission["Id"], args.sample_submission)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # These are the user-facing failure modes this script defines on purpose (missing
        # files, a bare pre-Phase-2 estimator instead of a payload dict, missing feature
        # columns, malformed payload structure). Print one line and exit 1 instead of a
        # traceback. Anything else is an unanticipated programming error and should still
        # surface with its full traceback.
        print(f"ERROR: {exc}")
        return 1

    print(f"model_name={payload.get('model_name')!r}")
    print(f"threshold_used={float(threshold)}")
    print(f"output_path={args.output}")
    print(f"row_count={len(submission)}")
    print(f"positive_prediction_count={int(response.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
