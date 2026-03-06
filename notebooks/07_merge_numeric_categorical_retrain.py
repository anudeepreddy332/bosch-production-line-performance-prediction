"""
Merge top 150 numeric + top 100 categorical, then retrain.

Total features: 250 (manageable!)
Expected MCC boost: 0.1535 → 0.18-0.20
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
import time

from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


def optimize_threshold(y_true, y_pred_proba):
    """Find best threshold by maximizing MCC."""
    best_mcc = -1
    best_thresh = 0.5
    best_metrics = {}

    for thresh in np.arange(0.05, 0.95, 0.01):
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
    logger.info("MERGE TOP 150 NUMERIC + TOP 100 CATEGORICAL")
    logger.info("=" * 60)

    # 1. Load numeric (top 150)
    logger.info("\n1. Loading top 150 numeric features...")
    df_numeric = pd.read_parquet("data/features/train_selected_top150.parquet")
    logger.info(f"  Shape: {df_numeric.shape}")
    memory_usage_report(df_numeric, "Top 150 Numeric")

    # 2. Load categorical (top 100)
    logger.info("\n2. Loading top 100 categorical features...")
    df_cat = pd.read_parquet("data/features/train_categorical_top100.parquet")
    logger.info(f"  Shape: {df_cat.shape}")
    memory_usage_report(df_cat, "Top 100 Categorical")

    # 3. Merge
    logger.info("\n" + "=" * 60)
    logger.info("MERGING FEATURES")
    logger.info("=" * 60)

    df_merged = df_numeric.merge(df_cat, on='Id', how='left')
    logger.info(f"\nMerged shape: {df_merged.shape}")
    logger.info(f"  Numeric features: 150")
    logger.info(f"  Categorical features: 100")
    logger.info(f"  Total features: 250")

    memory_usage_report(df_merged, "Merged Dataset")

    # 4. Train/val split
    logger.info("\n" + "=" * 60)
    logger.info("TRAIN/VAL SPLIT")
    logger.info("=" * 60)

    X = df_merged.drop(['Id', 'Response'], axis=1)
    y = df_merged['Response']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nTrain: {len(X_train):,} ({y_train.sum():,} failures, {y_train.sum() / len(y_train) * 100:.3f}%)")
    logger.info(f"Val: {len(X_val):,} ({y_val.sum():,} failures, {y_val.sum() / len(y_val) * 100:.3f}%)")

    # 5. Train model
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

    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.log_evaluation(period=50)]
    )

    training_time = time.time() - start_time

    logger.info(f"\n✅ Training complete in {training_time:.1f} seconds")
    logger.info(f"  Trees built: {model.n_estimators}")

    # 6. Evaluate
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
    logger.info(f"  TN={cm[0, 0]:,}, FP={cm[0, 1]:,}")
    logger.info(f"  FN={cm[1, 0]:,}, TP={cm[1, 1]:,}")

    # 7. Compare with baselines
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH BASELINES")
    logger.info("=" * 60)

    baseline_465 = {'mcc': 0.1648, 'features': 465, 'time': 44}
    baseline_150 = {'mcc': 0.1535, 'features': 150, 'time': 31.2}

    logger.info(f"\n1. Baseline (465 numeric features):")
    logger.info(f"   MCC: {baseline_465['mcc']:.4f}")
    logger.info(f"   Training time: {baseline_465['time']:.1f}s")

    logger.info(f"\n2. Top 150 numeric only:")
    logger.info(f"   MCC: {baseline_150['mcc']:.4f}")
    logger.info(f"   Training time: {baseline_150['time']:.1f}s")
    logger.info(f"   Change: {((baseline_150['mcc'] - baseline_465['mcc']) / baseline_465['mcc']) * 100:+.1f}%")

    logger.info(f"\n3. Top 150 numeric + Top 100 categorical:")
    logger.info(f"   MCC: {best_mcc:.4f}")
    logger.info(f"   Training time: {training_time:.1f}s")
    logger.info(f"   Change vs 465 numeric: {((best_mcc - baseline_465['mcc']) / baseline_465['mcc']) * 100:+.1f}%")
    logger.info(f"   Change vs 150 numeric: {((best_mcc - baseline_150['mcc']) / baseline_150['mcc']) * 100:+.1f}%")

    # 8. Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FEATURES BY IMPORTANCE")
    logger.info("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 20 features:")
    for i, row in feature_importance.head(20).iterrows():
        feat_type = "CAT" if row['feature'].endswith('_target_enc') else "NUM"
        logger.info(f"  {i + 1:2d}. [{feat_type}] {row['feature']:50s} {row['importance']:8.1f}")

    # Count categorical in top 20
    top_20_cat = feature_importance.head(20)['feature'].str.endswith('_target_enc').sum()
    logger.info(f"\nCategorical features in top 20: {top_20_cat}/20")

    # 9. Save
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    model_path = "data/models/model_numeric_categorical.pkl"
    joblib.dump({
        'model': model,
        'best_threshold': best_thresh,
        'feature_names': list(X.columns),
        'mcc': best_mcc,
        'training_time': training_time,
        'metrics': best_metrics
    }, model_path)

    logger.info(f"✅ Model saved to: {model_path}")
    logger.info(f"  Features: 250 (150 numeric + 100 categorical)")
    logger.info(f"  MCC: {best_mcc:.4f}")
    logger.info(f"  Threshold: {best_thresh:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 60)

    if best_mcc > baseline_150['mcc']:
        improvement = ((best_mcc - baseline_150['mcc']) / baseline_150['mcc']) * 100
        logger.info(f"\n🎉 MCC improved by {improvement:.1f}% with categorical features!")
        logger.info(f"   Next step: Hyperparameter tuning to push MCC even higher")
    else:
        logger.info(f"\n⚠️  Categorical features didn't improve MCC significantly")
        logger.info(f"   Consider: Feature engineering or stick with numeric-only")


if __name__ == "__main__":
    main()
