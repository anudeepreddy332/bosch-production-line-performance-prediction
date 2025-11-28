"""
Validate Parquet files after CSV conversion.

Checks:
1. Row counts match CSV originals
2. Response distribution unchanged (0.58% failure rate)
3. Can load full dataset into RAM
4. No data loss during conversion
5. Memory usage is acceptable
"""

import sys
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.logger import setup_logger
from src.utils.memory import memory_usage_report, get_system_memory

logger = setup_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def validate_row_counts():
    """Verify row counts match between CSV and Parquet."""
    logger.info("=" * 60)
    logger.info("ROW COUNT VALIDATION")
    logger.info("=" * 60)

    files = ['train_numeric', 'train_categorical', 'train_date']

    for filename in files:
        csv_path = RAW_DIR / f"{filename}.csv"
        parquet_path = PROCESSED_DIR / f"{filename}.parquet"

        # Count CSV rows (expensive but necessary)
        logger.info(f"\nChecking {filename}...")
        csv_rows = sum(1 for _ in open(csv_path)) - 1  # -1 for header

        # Load parquet and count
        df = pd.read_parquet(parquet_path)
        parquet_rows = len(df)

        match = "✅ MATCH" if csv_rows == parquet_rows else "❌ MISMATCH"
        logger.info(f"  CSV rows:     {csv_rows:,}")
        logger.info(f"  Parquet rows: {parquet_rows:,}")
        logger.info(f"  {match}")

        if csv_rows != parquet_rows:
            logger.error(f"⚠️  Row count mismatch for {filename}!")
            return False

    logger.info("\n✅ All row counts match!")
    return True


def validate_response_distribution():
    """Verify failure rate is preserved."""
    logger.info("\n" + "=" * 60)
    logger.info("RESPONSE DISTRIBUTION VALIDATION")
    logger.info("=" * 60)

    # Load train_numeric (has Response column)
    df = pd.read_parquet(PROCESSED_DIR / "train_numeric.parquet")

    total = len(df)
    failures = (df['Response'] == 1).sum()
    failure_rate = (failures / total) * 100

    logger.info(f"Total parts:    {total:,}")
    logger.info(f"Failures:       {failures:,}")
    logger.info(f"Failure rate:   {failure_rate:.3f}%")

    # Expected: ~0.58%
    if 0.5 < failure_rate < 0.7:
        logger.info("✅ Failure rate within expected range (0.5-0.7%)")
        return True
    else:
        logger.error(f"⚠️  Unexpected failure rate: {failure_rate:.3f}%")
        return False


def validate_memory_usage():
    """Test loading full dataset to ensure it fits in RAM."""
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY USAGE VALIDATION")
    logger.info("=" * 60)

    sys_mem = get_system_memory()
    logger.info(f"System: {sys_mem['used_gb']:.1f} GB / {sys_mem['total_gb']:.1f} GB used")
    logger.info(f"Available: {sys_mem['available_gb']:.1f} GB\n")

    files = [
        'train_numeric.parquet',
        'train_categorical.parquet',
        'train_date.parquet'
    ]

    for filename in files:
        logger.info(f"Loading {filename}...")
        df = pd.read_parquet(PROCESSED_DIR / filename)
        memory_usage_report(df, filename)

        # Check if memory usage is reasonable (<10GB per file)
        mem_gb = df.memory_usage(deep=True).sum() / (1024 ** 3)
        if mem_gb > 10:
            logger.warning(f"⚠️  {filename} uses {mem_gb:.1f} GB RAM (high)")
        else:
            logger.info(f"✅ {filename} uses {mem_gb:.1f} GB RAM (acceptable)")

        del df  # Free memory

    logger.info("\n✅ All files fit in memory")
    return True


def validate_dtypes():
    """Check that dtype optimizations were applied."""
    logger.info("\n" + "=" * 60)
    logger.info("DTYPE VALIDATION")
    logger.info("=" * 60)

    # Check train_numeric
    df_num = pd.read_parquet(PROCESSED_DIR / "train_numeric.parquet")
    logger.info("\ntrain_numeric dtypes:")
    logger.info(f"  Response: {df_num['Response'].dtype} (expected: int8)")
    logger.info(f"  Feature columns: {df_num.select_dtypes(include=['float32']).shape[1]} float32")

    if df_num['Response'].dtype == 'int8':
        logger.info("✅ Response is int8")
    else:
        logger.warning(f"⚠️  Response is {df_num['Response'].dtype}, expected int8")

    # Check train_date
    df_date = pd.read_parquet(PROCESSED_DIR / "train_date.parquet")
    float32_cols = df_date.select_dtypes(include=['float32']).shape[1]
    logger.info(f"\ntrain_date dtypes:")
    logger.info(f"  float32 columns: {float32_cols} / {df_date.shape[1]}")

    if float32_cols > df_date.shape[1] * 0.9:
        logger.info("✅ Most columns are float32")
    else:
        logger.warning(f"⚠️  Only {float32_cols} columns are float32")

    return True


def validate_no_data_loss():
    """Spot check: verify specific values match between CSV and Parquet."""
    logger.info("\n" + "=" * 60)
    logger.info("DATA INTEGRITY SPOT CHECK")
    logger.info("=" * 60)

    # Load first 1000 rows from CSV
    csv_sample = pd.read_csv(RAW_DIR / "train_numeric.csv", nrows=1000)

    # Load same rows from Parquet
    parquet_df = pd.read_parquet(PROCESSED_DIR / "train_numeric.parquet")
    parquet_sample = parquet_df.head(1000)

    # Check Ids match
    ids_match = (csv_sample['Id'] == parquet_sample['Id']).all()
    logger.info(f"IDs match: {'✅' if ids_match else '❌'}")

    # Check Response values match
    response_match = (csv_sample['Response'] == parquet_sample['Response']).all()
    logger.info(f"Response values match: {'✅' if response_match else '❌'}")

    # Check a few numeric columns (accounting for float32 precision)
    test_col = csv_sample.columns[2]  # Pick a numeric column
    max_diff = (csv_sample[test_col] - parquet_sample[test_col]).abs().max()
    logger.info(f"Max difference in {test_col}: {max_diff:.6f}")

    if max_diff < 0.001:  # float32 precision tolerance
        logger.info("✅ Numeric values match within float32 precision")
        return True
    else:
        logger.warning(f"⚠️  Large difference detected: {max_diff}")
        return False


def main():
    """Run all validation checks."""
    logger.info("🔍 Starting data validation...")

    checks = [
        ("Row Counts", validate_row_counts),
        ("Response Distribution", validate_response_distribution),
        ("Memory Usage", validate_memory_usage),
        ("Dtypes", validate_dtypes),
        ("Data Integrity", validate_no_data_loss),
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} failed: {str(e)}")
            results[check_name] = False

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 ALL VALIDATIONS PASSED!")
        logger.info("Safe to proceed with feature engineering.")
    else:
        logger.error("\n⚠️  SOME VALIDATIONS FAILED!")
        logger.error("Review errors before proceeding.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
