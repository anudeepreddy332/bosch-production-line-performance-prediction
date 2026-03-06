"""
Retrain LightGBM with leak features.

Expected MCC improvement:
- Day 7 (without leaks): 0.1881
- Day 8 (with leaks): 0.30-0.35 (+60-80%!)

"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, confusion_matrix
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
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }

    return best_thresh, best_mcc, best_metrics


def main():
    logger.info("=" * 60)
    logger.info("RETRAIN WITH LEAK FEATURES - DAY 8")
    logger.info("=" * 60)
    logger.info("\nExpected MCC: 0.30-0.35 (vs 0.1881 without leaks)")
    logger.info("This should be a MASSIVE improvement! 🚀")

    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    df = pd.read_parquet("data/features/train_all_features_with_leaks.parquet")
    logger.info(f"\nShape: {df.shape}")
    memory_usage_report(df, "Training Data")

    # Train/val split
    X = df.drop(['Id', 'Response'], axis=1)
    y = df['Response']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nTrain: {len(X_train):,} ({y_train.sum():,} failures, {y_train.sum() / len(y_train) * 100:.3f}%)")
    logger.info(f"Val: {len(X_val):,} ({y_val.sum():,} failures, {y_val.sum() / len(y_val) * 100:.3f}%)")

    # Load best params from Day 7 tuning
    logger.info("\n" + "=" * 60)
    logger.info("LOADING BEST HYPERPARAMETERS")
    logger.info("=" * 60)

    try:
        tuning_results = joblib.load("data/models/optuna_study_results.pkl")
        params = tuning_results['best_params']
        logger.info("\n✅ Loaded optimized params from Day 7")
    except:
        logger.info("\n⚠️  Could not load tuned params, using defaults")
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 64,
            'max_depth': -1,
        }

    # Add fixed params
    params['scale_pos_weight'] = 171.09
    params['objective'] = 'binary'
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['verbose'] = -1

    logger.info("\nParameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL")
    logger.info("=" * 60)

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

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    best_thresh, best_mcc, best_metrics = optimize_threshold(y_val, y_pred_proba)

    logger.info(f"\n✅ Best threshold: {best_thresh:.3f}")
    logger.info(f"  MCC: {best_mcc:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")

    cm = best_metrics['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0, 0]:,}, FP={cm[0, 1]:,}")
    logger.info(f"  FN={cm[1, 0]:,}, TP={cm[1, 1]:,}")

    # Compare with Day 7
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH DAY 7")
    logger.info("=" * 60)

    day7_mcc = 0.1881
    improvement = ((best_mcc - day7_mcc) / day7_mcc) * 100

    logger.info(f"\nDay 7 (250 features, no leaks): {day7_mcc:.4f}")
    logger.info(f"Day 8 (with leak features): {best_mcc:.4f}")
    logger.info(f"Improvement: {improvement:+.1f}% 🚀")

    if best_mcc >= 0.30:
        logger.info("\n🎉🎉🎉 MCC >= 0.30! ON TRACK TO TOP 10%! 🎉🎉🎉")
    elif best_mcc >= 0.25:
        logger.info("\n✅ Good progress! MCC >= 0.25")
    else:
        logger.info("\n⚠️  MCC below expectations. Check leak features.")

    # Feature importance - check if leak features are important
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FEATURES BY IMPORTANCE")
    logger.info("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 20 features:")
    leak_count = 0
    for i, row in feature_importance.head(20).iterrows():
        is_leak = 'chunk' in row['feature'] or 'dup' in row['feature'] or 'concat' in row['feature']
        feat_type = "LEAK" if is_leak else "NUM/CAT"
        if is_leak:
            leak_count += 1
        logger.info(f"  {i + 1:2d}. [{feat_type}] {row['feature']:50s} {row['importance']:8.1f}")

    logger.info(f"\nLeak features in top 20: {leak_count}/20")

    if leak_count >= 5:
        logger.info("✅ Leak features are highly important! Good sign!")
    else:
        logger.info("⚠️  Leak features not dominating. May need investigation.")

    # Save
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    model_path = "data/models/model_day8_with_leaks.pkl"
    joblib.dump({
        'model': model,
        'best_threshold': best_thresh,
        'feature_names': list(X.columns),
        'mcc': best_mcc,
        'training_time': training_time,
        'metrics': best_metrics,
        'improvement_vs_day7': improvement
    }, model_path)

    logger.info(f"\n✅ Model saved to: {model_path}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  MCC: {best_mcc:.4f}")
    logger.info(f"  Threshold: {best_thresh:.3f}")

    # Roadmap
    logger.info("\n" + "=" * 60)
    logger.info("ROADMAP TO 0.51+")
    logger.info("=" * 60)
    logger.info(f"\n✅ Day 7: MCC 0.1881")
    logger.info(f"✅ Day 8: MCC {best_mcc:.4f} (+{improvement:.1f}%)")
    logger.info(f"🎯 Day 9: + Time features → MCC 0.38-0.43")
    logger.info(f"🎯 Day 10: + Station combos → MCC 0.40-0.45")
    logger.info(f"🎯 Day 11: + Advanced CV/tuning → MCC 0.42-0.47")
    logger.info(f"🎯 Day 12-13: + Ensemble → MCC 0.48-0.52 (TOP 10%!)")


if __name__ == "__main__":
    main()
