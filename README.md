# Bosch Production Line Performance

> Predictive maintenance ML pipeline for manufacturing failure detection.  
> Built on the [Kaggle Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance) dataset.

## Problem Statement

Bosch's production lines generate sensor readings across hundreds of stations for every part manufactured. The goal is to predict which parts will fail quality checks, with a failure rate of only **0.58%**, making this an extreme class imbalance problem.

**Primary metric:** Matthews Correlation Coefficient (MCC)  
**Target:** MCC ≥ 0.52

## Dataset

| Property | Value |
|---|---|
| Training rows | 1,183,747 |
| Total features | 4,268 (numeric, categorical, date) |
| Failure rate | ~0.58% (~171:1 imbalance) |
| Raw data size | ~12 GB CSV → 1.77 GB Parquet |

## Pipeline

```
Raw CSVs → Parquet conversion → Feature selection → Target encoding
→ Leak feature engineering → Time features → Model training → Threshold optimization
```

| Step | Script | Description |
|---|---|---|
| 00 | `00_quick_check.py` | Data sanity checks |
| 01 | `01_data_validation.py` | Schema and null validation |
| 02 | `02_eda.py` | Exploratory data analysis |
| 03 | `03_train_baseline.py` | LightGBM baseline |
| 04 | `04_feature_selection_numeric.py` | Select top 150 numeric features |
| 05 | `05_retrain_numeric_top150.py` | Retrain on selected numeric |
| 06 | `06_select_top_categorical.py` | Select top 100 categorical features |
| 07 | `07_merge_numeric_categorical_retrain.py` | Merge and retrain |
| 08 | `08_hyperparameter_tuning.py` | Optuna hyperparameter search |
| 09 | `09_merge_leak_features.py` | Merge engineered leak features |
| 10 | `10_retrain_with_leaks.py` | Retrain with leak features |
| 11 | `11_merge_time_features.py` | Merge time-based features |
| 12 | `12_retrain_with_time.py` | Retrain with full feature set |

## Feature Engineering

**Numeric features (150):** Selected from 970 by null rate, variance, and importance filtering.

**Leak features (23):** Based on manufacturing reality — consecutive parts with identical sensor readings tend to fail together.
- *Consecutive duplicate detection:* Chunk ID, chunk size, rank within chunk
- *Station-level duplication:* Per-station duplicate counts at key stations (L3_S29, L3_S30)
- *String concat patterns:* Unique pattern counting across priority stations

**Time features:** Rolling failure rates, failure distances, temporal cycle features.

## Results

| Stage | Features | MCC |
|---|---|---|
| Baseline LightGBM | 250 | 0.19 |
| + Hyperparameter tuning | 250 | 0.25 |
| + Leak features (fixed) | 273 | In progress |
| + Chunk-aware CV | TBD | Target: ≥ 0.52 |

## Setup

```bash
# Clone and set up environment
git clone https://github.com/anudeepreddy332/bosch-production-line-performance-prediction.git
cd bosch-production-line-performance-prediction

conda env create -f environment.yml
conda activate bosch

# Download data from Kaggle and place in data/raw/
# Then run pipeline sequentially from notebooks/00 → 12
```

## Hardware

Built and trained on MacBook Air M3 (16GB RAM, no GPU).  
Full pipeline run time: ~30 minutes. Training per fold: ~40 seconds.

## Tech Stack

- **Python 3.11**, pandas, numpy
- **LightGBM** — primary model
- **Optuna** — hyperparameter tuning
- **Streamlit** — results dashboard (`app/streamlit_app.py`)
- **Parquet + snappy** — compressed feature storage