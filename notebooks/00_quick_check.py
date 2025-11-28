"""
Quick data validation to confirm download success.
"""
import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")

# Check file existence
files = [
    "train_numeric.csv",
    "train_categorical.csv",
    "train_date.csv",
    "test_numeric.csv",
    "test_categorical.csv",
    "test_date.csv"
]

print("=" * 60)
print("FILE VALIDATION")
print("=" * 60)

for file in files:
    path = RAW_DIR / file
    if path.exists():
        size_mb = path.stat().st_size / (1024**2)
        print(f"✓ {file}: {size_mb:.2f} MB")
    else:
        print(f"✗ {file}: MISSING")

print("\n" + "=" * 60)
print("DATA SHAPE CHECK (First 10,000 rows)")
print("=" * 60)

# Load samples
train_num = pd.read_csv(RAW_DIR / "train_numeric.csv", nrows=10000)
train_cat = pd.read_csv(RAW_DIR / "train_categorical.csv", nrows=10000)
train_date = pd.read_csv(RAW_DIR / "train_date.csv", nrows=10000)

print(f"train_numeric:     {train_num.shape}")
print(f"train_categorical: {train_cat.shape}")
print(f"train_date:        {train_date.shape}")

print("\n" + "=" * 60)
print("RESPONSE DISTRIBUTION (First 10,000 rows)")
print("=" * 60)

response_counts = train_num['Response'].value_counts()
failure_rate = (response_counts.get(1, 0) / len(train_num)) * 100

print(f"Total rows:    {len(train_num)}")
print(f"Failures (1):  {response_counts.get(1, 0)}")
print(f"Normal (0):    {response_counts.get(0, 0)}")
print(f"Failure rate:  {failure_rate:.3f}%")

print("\n" + "=" * 60)
print("MEMORY USAGE (First 10,000 rows)")
print("=" * 60)

print(f"train_numeric: {train_num.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"train_categorical: {train_cat.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"train_date: {train_date.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("SAMPLE FEATURE NAMES")
print("=" * 60)

print("\nNumeric features (first 10):")
print(train_num.columns[:10].tolist())

print("\nCategorical features (first 5):")
print(train_cat.columns[:5].tolist())

print("\nDate features (first 5):")
print(train_date.columns[:5].tolist())

print("\n" + "=" * 60)
print("NULL PERCENTAGE (First 10,000 rows)")
print("=" * 60)

null_pct = (train_num.isnull().sum() / len(train_num)) * 100
print(f"Features with >99% nulls: {(null_pct > 99).sum()}")
print(f"Features with >90% nulls: {(null_pct > 90).sum()}")
print(f"Features with <10% nulls: {(null_pct < 10).sum()}")
