"""
Retrain baseline with top 150 features (skip categorical for now).

Goal: Prove that feature selection improves MCC while reducing training time.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

from src.config import Config
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)
config = Config("config/config.yaml")


def optimize_threshold(y_true, y_pred_proba):
    """Find best threshold by maximizing MCC."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_mcc = -1
    best_thresh = 0.5
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
            best_metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }

    return best_thresh, best_mcc, best_metrics


def main():
    logger.info("=" * 60)
    logger.info("RETRAIN WITH TOP 150 FEATURES")
    logger.info("=" * 60)

    # 1. Load top 150 features
    logger.info("\n1. Loading top 150 features...")
    df = pd.read_parquet("data/features/train_selected_top150.parquet")
    logger.info(f"  Shape: {df.shape}")

    memory_usage_report(df, "Top 150 Features")

    # 2. Train/val split
    logger.info("\n" + "=" * 60)
    logger.info("TRAIN/VAL SPLIT")
    logger.info("=" * 60)

    X = df.drop(['Id', 'Response'], axis=1)
    y = df['Response']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nTrain: {len(X_train):,} ({y_train.sum():,} failures)")
    logger.info(f"Val: {len(X_val):,} ({y_val.sum():,} failures)")

    # 3. Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL")
    logger.info("=" * 60)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    logger.info(f"\nClass imbalance: {scale_pos_weight:.2f}:1")

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 64,
        'subsample': 0.8,
        'subsample_freq': 5,
        'colsample_bytree': 0.8,
        'min_child_samples': 50,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMClassifier(**params)

    import time
    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.log_evaluation(period=50)]
    )

    training_time = time.time() - start_time

    logger.info(f"\n✅ Training complete in {training_time:.1f} seconds")
    logger.info(f"  Trees built: {model.n_estimators}")

    # 4. Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    best_thresh, best_mcc, best_metrics = optimize_threshold(y_val, y_pred_proba)

    logger.info(f"\n✅ Best threshold: {best_thresh:.3f}")
    logger.info(f"  MCC: {best_mcc:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1: {best_metrics['f1']:.4f}")

    cm = best_metrics['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
    logger.info(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    # 5. Compare with baseline
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("=" * 60)

    baseline_mcc = 0.1648
    baseline_time = 44  # seconds
    baseline_features = 465

    improvement_mcc = ((best_mcc - baseline_mcc) / baseline_mcc) * 100
    speedup = baseline_time / training_time

    logger.info(f"\nBaseline (465 features):")
    logger.info(f"  MCC: {baseline_mcc:.4f}")
    logger.info(f"  Training time: {baseline_time}s")
    logger.info(f"\nTop 150 features:")
    logger.info(f"  MCC: {best_mcc:.4f}")
    logger.info(f"  Training time: {training_time:.1f}s")
    logger.info(f"\nResults:")
    logger.info(f"  MCC change: {improvement_mcc:+.1f}%")
    logger.info(f"  Speedup: {speedup:.1f}x faster")
    logger.info(
        f"  Features: {baseline_features} → 150 ({((baseline_features - 150) / baseline_features) * 100:.1f}% reduction)")

    # 6. Save
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    model_path = "data/models/baseline_top150.pkl"
    joblib.dump({
        'model': model,
        'best_threshold': best_thresh,
        'feature_names': list(X.columns),
        'mcc': best_mcc,
        'training_time': training_time
    }, model_path)

    logger.info(f"✅ Saved to: {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
