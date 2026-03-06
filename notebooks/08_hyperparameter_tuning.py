"""
Hyperparameter tuning with Optuna for LightGBM.

Goal: Find optimal params to maximize MCC on validation set.
Dataset: 150 numeric + 100 categorical = 250 features
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import optuna
import joblib

from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)
config = Config("config/config.yaml")


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function to maximize MCC.
    """
    # Hyperparameters to tune
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': 171.09,  # Fixed (class imbalance)
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Predict and optimize threshold
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Find best threshold
    best_mcc = -1
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_pred_proba >= thresh).astype(int)
        mcc = matthews_corrcoef(y_val, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc

    return best_mcc


def main():
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING WITH OPTUNA")
    logger.info("=" * 60)

    # Load data (150 numeric + 100 categorical)
    logger.info("\nLoading merged dataset (150 numeric + 100 categorical)...")

    df_numeric = pd.read_parquet("data/features/train_selected_top150.parquet")
    df_cat = pd.read_parquet("data/features/train_categorical_top100.parquet")
    df = df_numeric.merge(df_cat, on='Id', how='left')

    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Features: 250 (150 numeric + 100 categorical)")

    X = df.drop(['Id', 'Response'], axis=1)
    y = df['Response']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nTrain: {len(X_train):,}, Val: {len(X_val):,}")

    # Create Optuna study
    logger.info("\n" + "=" * 60)
    logger.info("STARTING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("\nThis will run 50 trials (~30-40 minutes)")
    logger.info("Each trial trains a model and evaluates MCC")
    logger.info("Grab a coffee and relax! ☕\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True
    )

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nBest MCC: {study.best_value:.4f}")
    logger.info(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # Compare with baseline (untuned 250-feature model)
    baseline_mcc = 0.1649  # ← Updated to 250-feature model
    improvement = ((study.best_value - baseline_mcc) / baseline_mcc) * 100

    logger.info(f"\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info(f"\nUntuned (250 features, default params): {baseline_mcc:.4f}")
    logger.info(f"Tuned (250 features, optimized params): {study.best_value:.4f}")
    logger.info(f"Improvement: {improvement:+.1f}%")

    # Show improvement from original baseline
    original_baseline = 0.1648
    total_improvement = ((study.best_value - original_baseline) / original_baseline) * 100
    logger.info(f"\nVs original baseline (465 features): {original_baseline:.4f}")
    logger.info(f"Total improvement: {total_improvement:+.1f}%")

    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    results = {
        'best_params': study.best_params,
        'best_mcc': study.best_value,
        'study': study,
        'baseline_mcc': baseline_mcc,
        'improvement': improvement
    }

    joblib.dump(results, "data/models/optuna_study_results.pkl")
    logger.info("✅ Saved to: data/models/optuna_study_results.pkl")

    # Save trials as CSV for analysis
    trials_df = study.trials_dataframe()
    trials_df.to_csv("data/models/optuna_trials.csv", index=False)
    logger.info("✅ Saved trials to: data/models/optuna_trials.csv")

    logger.info("\n" + "=" * 60)
    logger.info("✅ NEXT STEP: Train final model with best params")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
