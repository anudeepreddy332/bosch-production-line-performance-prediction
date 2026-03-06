"""
Retrain with time features.

Expected MCC: 0.38-0.42 (vs 0.3247 with leaks only)
"""
import os
print("Current working directory:", os.getcwd())
import sys
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, confusion_matrix
import joblib

from src.logger import setup_logger

logger = setup_logger(__name__)

def optimize_threshold(y_true, y_pred_proba, min_recall=0.30):
    """
    Optimize threshold prioritizing recall.
    First try to achieve min_recall.
    Among those thresholds, choose best MCC.
    If none achieve min_recall, fallback to best MCC overall.
    """
    best_mcc = -1
    best_thresh = 0.5
    best_metrics = {}

    best_mcc_any = -1
    best_thresh_any = 0.5
    best_metrics_any = {}

    for thresh in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_pred_proba >= thresh).astype(int)

        mcc = matthews_corrcoef(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Track best MCC overall (fallback)
        if mcc > best_mcc_any:
            best_mcc_any = mcc
            best_thresh_any = thresh
            best_metrics_any = {
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm
            }

        # Only consider thresholds meeting recall constraint
        if recall >= min_recall:
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'confusion_matrix': cm
                }
    if best_mcc == -1:
        print("⚠️ No threshold met minimum recall. Falling back to best MCC overall.")
        return best_thresh_any, best_mcc_any, best_metrics_any

    return best_thresh, best_mcc, best_metrics


def main():
    logger.info("=" * 60)
    logger.info("RETRAIN WITH TIME FEATURES - TIMESERIES SPLIT")
    logger.info("=" * 60)
    logger.info("\nExpected MCC: 0.38-0.42 (vs 0.3247 with leaks)")

    # Load data
    logger.info("\nLoading data...")
    data_path = project_root / "data" / "features" / "train_all_features_complete.parquet"
    df = pd.read_parquet(data_path)
    logger.info(f"  Shape: {df.shape}")
    print("\nFIRST 20 ROWS:")
    print(df[['Id', 'chunk_id']].head(20))

    print("\nLAST 20 ROWS:")
    print(df[['Id', 'chunk_id']].tail(20))

    print("\nChunk distribution:")
    print(df['chunk_id'].value_counts().sort_index())

    df = df.sort_values('Id').reset_index(drop=True)

    # Real time features (Duration)
    # Load date features
    df_date = pd.read_parquet(project_root/"data"/"processed"/"train_date.parquet")

    date_cols = df_date.columns.drop("Id")

    df_date["start_time"] = df_date[date_cols].min(axis=1)
    df_date["end_time"] = df_date[date_cols].max(axis=1)
    df_date["duration"] = df_date["end_time"] - df_date["start_time"]

    # Keep only needed columns
    df_date = df_date[["Id", "start_time", "end_time", "duration"]]

    # Merge into main dataframe
    df = df.merge(df_date, on="Id", how="left")

    # Engineered duration features
    df["log_duration"] = np.log1p(df["duration"])
    df["is_long_duration"] = (df["duration"] > 35).astype(int)
    df["duration_z"] = (df["duration"] - df["duration"].mean()) / df["duration"].std()

    # Interaction with chunk structure
    df["duration_x_chunk"] = df["duration"] * df["chunk_size"]


    # Drop future leakage
    leak_cols = [c for c in df.columns if 'distance_to_next_' in c]
    rolling_cols = [c for c in df.columns if 'rolling_mean_target_' in c]
    print(f"Dropping future leakage columns: {leak_cols}")
    print("Dropping global rolling columns:", rolling_cols)

    df = df.drop(columns=leak_cols + rolling_cols, errors='ignore')

    # Load tuned params
    model_path = project_root / "data" / "models" / "optuna_study_results.pkl"
    tuning_results = joblib.load(model_path)
    params = tuning_results['best_params']


    params['objective'] = 'binary'
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['verbose'] = -1

    # Relative chunk features
    df['chunk_position_ratio'] = df['chunk_rank_asc'] / df['chunk_size']
    df['chunk_reverse_ratio'] = df['chunk_rank_desc'] / df['chunk_size']
    df['is_last_in_chunk'] = (df['chunk_rank_desc'] == 1).astype(int)
    df['is_first_in_chunk'] = (df['chunk_rank_asc'] == 1).astype(int)

    # Add duplicate density signal
    df['dup_density'] = df['total_station_dups'] / (df['chunk_size'] + 1)
    df['concat_density'] = df['total_concat_count'] / (df['chunk_size'] + 1)

    # Interaction features
    df['rank_x_size'] = df['chunk_rank_asc'] * df['chunk_size']
    df['reverse_x_size'] = df['chunk_rank_desc'] * df['chunk_size']


    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n========== Fold {fold + 1} ==========")

        # Full fold data
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # Controlled downsampling
        print("\n--- Applying Controlled Downsampling ---")

        # Separate groups
        df_fail = df_train[df_train['Response'] == 1]
        df_dup = df_train[(df_train['Response'] == 0) & (df_train['chunk_size'] > 1)]
        df_easy = df_train[(df_train['Response'] == 0) & (df_train['chunk_size'] == 1)]

        # Downsample easy negatives to 30%
        df_easy_sampled = df_easy.sample(frac=0.30, random_state=42)

        df_train_balanced = pd.concat([df_fail, df_dup, df_easy_sampled])

        df_train_balanced = df_train_balanced.sort_values('Id')

        print("Train size (balanced):", len(df_train_balanced))
        print("Val size (full):", len(df_val))

        # Compute correct imbalance after downsampling
        pos = (df_train_balanced['Response'] == 1).sum()
        neg = (df_train_balanced['Response'] == 0).sum()

        params_fold = params.copy()
        params_fold['scale_pos_weight'] = neg/pos

        X_train = df_train_balanced.drop(['Id', 'Response'], axis=1)
        y_train = df_train_balanced['Response']

        X_val = df_val.drop(['Id', 'Response'], axis=1)
        y_val = df_val['Response']

        #Recompute rolling mean on TRAIN only
        # for window in [5,10,20,100,1000,5000]:
        #
        #     temp_y = pd.concat([y_train, y_val])
        #     rolling_full = (
        #         temp_y.rolling(window=window, min_periods=1).mean().shift(1).fillna(y_train.mean())
        #     )
        #
        #     rolling_train = rolling_full.iloc[:len(y_train)].values
        #     rolling_val = rolling_full.iloc[len(y_train):].values
        #
        #     X_train[f'rolling_mean_target_{window}'] = rolling_train
        #     X_val[f'rolling_mean_target_{window}'] = rolling_val

        # Model 1
        model1 = lgb.LGBMClassifier(**params_fold)
        model1.fit(X_train, y_train)
        pred1 = model1.predict_proba(X_val)[:, 1]

        # Model 2
        params2 = params_fold.copy()
        params2["random_state"] = 99

        model2 = lgb.LGBMClassifier(**params2)
        model2.fit(X_train, y_train)
        pred2 = model2.predict_proba(X_val)[:, 1]

        # Average
        oof_preds[val_idx] = (pred1 + pred2) / 2

    best_thresh, best_mcc, best_metrics = optimize_threshold(df['Response'], oof_preds, min_recall=0.30)
    print("OOF MCC:", best_mcc)

    logger.info(f"\nOOF MCC: {best_mcc:.4f}")
    logger.info(f"Best threshold: {best_thresh:.3f}")
    logger.info(f"Precision: {best_metrics['precision']:.4f}")
    logger.info(f"Recall: {best_metrics['recall']:.4f}")

    cm = best_metrics['confusion_matrix']
    logger.info(f"Confusion Matrix:")
    logger.info(f"TN={cm[0, 0]}, FP={cm[0, 1]}")
    logger.info(f"FN={cm[1, 0]}, TP={cm[1, 1]}")


if __name__ == "__main__":
    main()