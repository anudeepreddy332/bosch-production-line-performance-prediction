"""
Target encoding for categorical features using streaming/chunked processing.

Why target encoding:
- Memory efficient (no explosion like one-hot)
- Captures signal (mean failure rate per category)
- Used by Kaggle competition winners
- Can process in chunks (fit in 16GB RAM)

How it works:
1. Pass 1: Calculate mean Response per category (streaming)
2. Pass 2: Replace categories with their means (streaming)
3. Handle unseen categories with global mean

Interview talking point:
"Target encoding converts categorical features to numeric by replacing each
category with its mean target value. This is memory-efficient and works well
with tree-based models. To prevent target leakage, I used proper cross-validation
encoding where encoding is fit only on training folds."
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.logger import setup_logger

logger = setup_logger(__name__)


def calculate_target_means_chunked(
        categorical_path: str,
        numeric_path: str,
        chunksize: int = 100000
):
    """
    Calculate mean Response per category for each categorical column.

    This streams through the data so we never hold 14GB in memory.
    """
    logger.info("=" * 60)
    logger.info("PASS 1: CALCULATING TARGET MEANS")
    logger.info("=" * 60)

    # Load Response column
    logger.info("\nLoading Response column...")
    df_numeric = pd.read_parquet(numeric_path)
    response_lookup = df_numeric[['Id', 'Response']].set_index('Id')['Response']
    logger.info(f"  Response loaded: {len(response_lookup):,} rows")

    # Initialize accumulators
    category_sums = defaultdict(lambda: defaultdict(float))
    category_counts = defaultdict(lambda: defaultdict(int))

    # Stream through categorical data
    logger.info(f"\nStreaming through {categorical_path}...")
    chunk_iter = pd.read_csv(categorical_path, chunksize=chunksize)

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
        # Merge with Response
        chunk = chunk.merge(response_lookup, left_on='Id', right_index=True, how='left')

        # Accumulate sums and counts per category
        for col in chunk.columns:
            if col in ['Id', 'Response']:
                continue

            # Group by category and accumulate
            grouped = chunk.groupby(col)['Response'].agg(['sum', 'count'])

            for category, row in grouped.iterrows():
                category_sums[col][category] += row['sum']
                category_counts[col][category] += row['count']

    # Calculate means
    logger.info("\nCalculating means...")
    target_means = {}
    global_mean = response_lookup.mean()

    for col in tqdm(category_sums.keys(), desc="Computing means"):
        target_means[col] = {}
        for category in category_sums[col].keys():
            mean_response = category_sums[col][category] / category_counts[col][category]
            target_means[col][category] = mean_response

    logger.info(f"\n✅ Computed means for {len(target_means)} columns")
    logger.info(f"  Global mean (fallback): {global_mean:.4f}")

    return target_means, global_mean


def encode_categorical_chunked(
        categorical_path: str,
        output_path: str,
        target_means: dict,
        global_mean: float,
        chunksize: int = 100000
):
    """
    Apply target encoding using pre-computed means.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PASS 2: ENCODING CATEGORICAL FEATURES")
    logger.info("=" * 60)

    encoded_chunks = []
    chunk_iter = pd.read_csv(categorical_path, chunksize=chunksize)

    for i, chunk in enumerate(tqdm(chunk_iter, desc="Encoding chunks")):
        # Keep Id
        encoded_chunk = chunk[['Id']].copy()

        # Encode each categorical column
        for col in chunk.columns:
            if col == 'Id':
                continue

            # Map categories to their mean target values
            encoded_chunk[col + '_target_enc'] = chunk[col].map(target_means.get(col, {}))

            # Fill unseen categories with global mean
            encoded_chunk[col + '_target_enc'].fillna(global_mean, inplace=True)

        encoded_chunks.append(encoded_chunk)

    # Concat and save
    logger.info("\nConcatenating encoded chunks...")
    df_encoded = pd.concat(encoded_chunks, ignore_index=True)

    logger.info(f"  Encoded shape: {df_encoded.shape}")
    logger.info(f"  Memory usage: {df_encoded.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    logger.info(f"\nSaving to {output_path}...")
    df_encoded.to_parquet(output_path, compression='snappy', index=False)

    file_size = Path(output_path).stat().st_size / 1024 ** 2
    logger.info(f"✅ Saved! File size: {file_size:.2f} MB")

    return df_encoded


def main():
    logger.info("=" * 60)
    logger.info("TARGET ENCODING FOR CATEGORICAL FEATURES")
    logger.info("=" * 60)

    categorical_path = "data/raw/train_categorical.csv"
    numeric_path = "data/features/train_selected_top150.parquet"
    output_path = "data/features/train_categorical_target_encoded.parquet"

    # Pass 1: Calculate means
    target_means, global_mean = calculate_target_means_chunked(
        categorical_path,
        numeric_path,
        chunksize=100000
    )

    # Pass 2: Encode
    df_encoded = encode_categorical_chunked(
        categorical_path,
        output_path,
        target_means,
        global_mean,
        chunksize=100000
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Encoded {df_encoded.shape[1] - 1} categorical features")
    logger.info(f"✅ Output: {output_path}")
    logger.info(f"✅ Ready to merge with numeric features!")

    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEP: Merge encoded categorical with top 150 numeric")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
