# Bosch Production Line Performance

> Predictive maintenance ML pipeline for manufacturing failure detection.  
> Built on the [Kaggle Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance) dataset.

## Problem Statement

Bosch's production lines generate sensor readings across hundreds of stations for every part manufactured. The goal is to predict which parts will fail quality checks, with a failure rate of only **0.58%** — an extreme class imbalance problem (~171:1 ratio).

**Primary metric:** Matthews Correlation Coefficient (MCC)  
**Target:** MCC ≥ 0.52

---

## Dataset

| Property | Value |
|---|---|
| Training rows | 1,183,747 |
| Total raw features | 4,268 (numeric, categorical, date) |
| Failure rate | ~0.58% (~6,879 failures) |
| Raw data size | ~12 GB CSV → 1.77 GB Parquet |

---

## Current Results

| Phase | Script | Features | OOF MCC | Notes |
|---|---|---|---|---|
| Baseline LightGBM | 03 | 250 | 0.190 | No feature engineering |
| + Hyperparameter tuning | 08 | 250 | 0.250 | Optuna search |
| + Leak + chunk features | 10 | 273 | 0.247 | TimeSeriesSplit (wrong CV) |
| + Chunk-aware CV fix | 13 | 273 | **0.294** | Round-robin chunk fold assignment |
| + Path features | 16 | 331 | **0.332** | Station visit patterns, path failure rates |
| Date/time features | TBD | ~353 | Target: 0.40+ | Next phase |
| Full feature set | TBD | ~400+ | **Target: ≥ 0.52** | |

---

## Roadmap to MCC ≥ 0.52

The gap between current MCC (0.332) and target (0.52) comes from three specific signal sources not yet fully utilized:

### What's driving MCC right now
- **Chunk/leak features** — consecutive parts with identical sensors tend to fail together. 7% of failures caught this way.
- **Path features** — which combination of stations a part visited. Paths with 41% failure rates vs 0.58% global mean. Works for all 1.18M parts.

### What will close the gap

**Phase 4 — Date/time features (expected +0.06–0.10 MCC)**  
The dataset includes timestamps for every sensor reading. Time-ordering features — how many failures occurred in the surrounding time window, inter-part gaps, production cycle position — are the single strongest signal in this dataset. Top Kaggle competitors (MCC 0.48+) relied heavily on these.

**Phase 5 — Interaction features (expected +0.03–0.05 MCC)**  
Pairwise products between top sensor features and path features. Parts on high-risk paths that also have anomalous sensor readings are far more likely to fail than either signal alone.

**Phase 6 — Model stacking (expected +0.03–0.05 MCC)**  
LightGBM OOF predictions as meta-features fed into a second-level model. Historically adds 0.02–0.05 MCC on this specific dataset.

**Phase 7 — Threshold calibration + submission**  
The OOF threshold grid search already finds the MCC-optimal threshold. Final submission uses the threshold that maximizes MCC across all folds.

---

## Pipeline

```
Raw CSVs
  → Parquet conversion
  → Feature selection (numeric + categorical)
  → Leak/chunk feature engineering
  → Path feature engineering
  → Date/time feature engineering        ← in progress
  → Interaction features                 ← planned
  → Chunk-aware CV training
  → Threshold optimization
  → Submission
```

| Phase | Script | Description | MCC Impact |
|---|---|---|---|
| 0 | `00_quick_check.py` | Data sanity checks | — |
| 0 | `01_data_validation.py` | Schema and null validation | — |
| 0 | `02_eda.py` | Exploratory data analysis | — |
| 1 | `03_train_baseline.py` | LightGBM baseline (250 features) | 0.190 |
| 1 | `04_feature_selection_numeric.py` | Select top 150 numeric features by importance | — |
| 1 | `05_retrain_numeric_top150.py` | Retrain on top 150 numeric | — |
| 1 | `06_select_top_categorical.py` | Select top 100 categorical features | — |
| 1 | `07_merge_numeric_categorical_retrain.py` | Merge numeric + categorical, retrain | — |
| 1 | `08_hyperparameter_tuning.py` | Optuna hyperparameter search | 0.250 |
| 2 | `09_merge_leak_features.py` | Add chunk/leak features (chunk_id, chunk_size, ranks) | — |
| 2 | `10_retrain_with_leaks.py` | Retrain with leak features | 0.247 |
| 2 | `11_merge_time_features.py` | Merge time features | — |
| 2 | `12_retrain_with_time.py` | Retrain — TimeSeriesSplit (later identified as wrong CV) | — |
| 2 | `13_chunk_aware_cv.py` | **Fix CV:** round-robin chunk fold assignment | **0.294** |
| 3 | `14_create_path_features.py` | Station presence flags + path signatures + path failure rates | — |
| 3 | `15_merge_path_features.py` | Merge 58 path features into main matrix (334 cols) | — |
| 3 | `16_retrain_phase3.py` | Retrain with path features, 1000 trees, lr=0.02 | **0.332** |
| 4 | `17_create_date_features.py` | Time-ordering features from date parquet | planned |
| 4 | `18_retrain_phase4.py` | Retrain with date + path + chunk features | planned |

---

## Feature Engineering

### Chunk / Leak Features (23 features)
Parts manufactured consecutively often have identical sensor readings — a manufacturing reality where machine calibration drift affects batches. Features capture:
- `chunk_id`, `chunk_size` — group ID and size for consecutive identical-sensor parts
- `chunk_rank_asc/desc` — position within the chunk
- Per-station duplicate counts at high-signal stations (L3_S29, L3_S30, L3_S33)
- String concat pattern counts across priority stations

### Path Features (58 features) ← Phase 3
Each part travels through a subset of ~50 manufacturing stations. Station visit presence is determined by nullness in the raw data — if all columns for a station are null, the part never visited it.
- `visited_L{n}_S{m}` — 50 binary flags (did this part visit this station?)
- `n_stations_L{0-3}` — count of stations visited per production line
- `n_stations_total` — total breadth of manufacturing path
- `path_count` — how many parts share this exact same path
- `path_failure_rate` — smoothed % of failures among parts on same path (0.01%–41.7%)
- `path_risk_tier` — binned: 0=rare path, 1=common, 2=very common

**Key insight:** 7,911 unique paths found. Parts on certain paths fail at 41.7% vs 0.58% global mean — a 72× signal that works for all 1.18M parts including singletons.

### Date / Time Features (planned — Phase 4)
Timestamps for every sensor reading across all stations. Features will include:
- Part production start/end time, total duration
- Inter-part time gaps (time since previous part, time to next)
- Rolling failure rates in surrounding time windows
- Production cycle position (time-of-shift, time-of-day patterns)

---

## Setup

```bash
git clone https://github.com/anudeepreddy332/bosch-production-line-performance-prediction.git
cd bosch-production-line-performance-prediction

conda env create -f environment.yml
conda activate bosch

# Download data from Kaggle → place CSVs in data/raw/
# Run pipeline sequentially: notebooks/00 → 16
```

---

## Hardware

MacBook Air M3, 16GB RAM, Apple Silicon (no GPU).  
Full pipeline: ~2 hours (path signature computation is the bottleneck at ~90s).  
Training per fold: ~4 minutes (1000 trees, 331 features, 294K rows).

---

## Tech Stack

- **Python 3.11** — pandas, numpy, scikit-learn
- **LightGBM** — primary gradient boosting model
- **Optuna** — hyperparameter tuning (script 08)
- **Parquet + Snappy** — compressed feature storage (~15–177 MB per feature set)
- **tqdm** — progress bars for long feature engineering steps
- **Streamlit** — results dashboard (`app/streamlit_app.py`)