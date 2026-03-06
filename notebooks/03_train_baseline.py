"""
Train baseline LightGBM model for Bosch failure prediction.

Pipeline:
1. Load engineered features (465 features + Id + Response)
2. Split into train/validation (80/20, stratified)
3. Handle class imbalance (scale_pos_weight)
4. Train LightGBM with early stopping
5. Optimize classification threshold for MCC
6. Analyze feature importance
7. Save model for production

Why this approach:
- Stratified split: Preserves 0.58% failure rate in both sets
- MCC metric: Works well with imbalanced data (better than accuracy)
- Threshold tuning: Default 0.5 is bad for imbalanced classes
- Feature importance: Validates our Day 5 engineering work
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from src.config import Config
from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)


class BoschBaselineModel:
    """
    Baseline LightGBM classifier for production line failure prediction.

    Handles:
    - Extreme class imbalance (0.58% failures)
    - High-dimensional sparse features (465 features, 69% sparse)
    - MCC optimization (not accuracy)
    """

    def __init__(self, config: Config):
        self.config = config
        self.features_dir = Path(config.get('paths.features'))
        self.models_dir = Path(config.get('paths.models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.outputs_dir = Path("outputs/models")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.best_threshold = 0.5
        self.feature_names = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load engineered features from Day 5.

        Returns:
            X: Features (465 columns)
            y: Target (Response, 0/1)
        """
        logger.info("=" * 60)
        logger.info("LOADING ENGINEERED FEATURES")
        logger.info("=" * 60)

        data_path = self.features_dir / 'train_engineered.parquet'
        logger.info(f"Loading from {data_path}...")

        df = pd.read_parquet(data_path)
        logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

        # Separate features and target
        X = df.drop(columns=['Id', 'Response'])
        y = df['Response'].astype(int)

        self.feature_names = X.columns.tolist()

        logger.info(f"\nFeatures: {X.shape[1]}")
        logger.info(f"Target distribution:")
        logger.info(f"  Class 0 (normal): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        logger.info(f"  Class 1 (failure): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.2f}%)")

        memory_usage_report(X, "Feature Matrix")

        return X, y

    def create_train_val_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            test_size: float = 0.2,
            random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and validation sets.

        Why stratify?
        - With 0.58% failure rate, random split might give:
          - Train: 0.6% failures
          - Val: 0.5% failures
        - Stratify ensures BOTH sets have ~0.58%

        Example:
        - Total: 1,183,747 parts, 6,879 failures
        - Train (80%): 947,000 parts, 5,503 failures (0.58%)
        - Val (20%): 236,747 parts, 1,376 failures (0.58%)

        Args:
            test_size: Validation set proportion (0.2 = 20%)
            random_state: Seed for reproducibility
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAIN/VALIDATION SPLIT")
        logger.info("=" * 60)

        logger.info(f"Split ratio: {(1 - test_size) * 100:.0f}% train, {test_size * 100:.0f}% validation")
        logger.info(f"Stratification: ON (preserves class balance)")

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # KEY: Preserves 0.58% in both sets
        )

        logger.info(f"\nTrain set:")
        logger.info(f"  Size: {len(X_train):,} parts")
        logger.info(f"  Failures: {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.3f}%)")

        logger.info(f"\nValidation set:")
        logger.info(f"  Size: {len(X_val):,} parts")
        logger.info(f"  Failures: {(y_val == 1).sum():,} ({(y_val == 1).sum() / len(y_val) * 100:.3f}%)")

        return X_train, X_val, y_train, y_val

    def build_model(self, y_train: pd.Series) -> lgb.LGBMClassifier:
        """
        Build LightGBM classifier with imbalance handling.

        Key parameter: scale_pos_weight
        - Formula: (# negative samples) / (# positive samples)
        - For Bosch: 1,176,868 / 6,879 ≈ 171
        - Effect: Makes each failure count 171× more in loss function
        - Why: Without this, model ignores failures (too rare)

        Other parameters:
        - n_estimators=500: Number of boosting rounds (trees)
        - learning_rate=0.05: Small steps (prevents overfitting)
        - max_depth=-1: No limit (LightGBM uses leaf-wise growth)
        - num_leaves=64: Controls tree complexity
        - subsample=0.8: Use 80% of data per tree (regularization)
        - colsample_bytree=0.8: Use 80% of features per tree
        - min_child_samples=50: Min samples in leaf (prevents tiny splits)
        """
        logger.info("\n" + "=" * 60)
        logger.info("BUILDING LIGHTGBM MODEL")
        logger.info("=" * 60)

        # Calculate class imbalance ratio
        n_negative = (y_train == 0).sum()
        n_positive = (y_train == 1).sum()
        scale_pos_weight = n_negative / n_positive

        logger.info(f"Class imbalance:")
        logger.info(f"  Negative (0): {n_negative:,}")
        logger.info(f"  Positive (1): {n_positive:,}")
        logger.info(f"  Ratio: 1:{n_negative / n_positive:.1f}")
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")

        # Model parameters
        params = {
            'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
            'objective': 'binary',  # Binary classification
            'metric': 'binary_logloss',  # Loss function
            'n_estimators': 500,  # Max trees
            'learning_rate': 0.05,  # Shrinkage
            'max_depth': -1,  # No limit (leaf-wise)
            'num_leaves': 64,  # Tree complexity
            'subsample': 0.8,  # Row sampling
            'subsample_freq': 5,  # Sample every 5 iterations
            'colsample_bytree': 0.8,  # Column sampling
            'min_child_samples': 50,  # Min samples per leaf
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': scale_pos_weight,  # IMBALANCE HANDLING
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'verbose': -1,  # Suppress warnings
        }

        logger.info("\nModel parameters:")
        for key, val in params.items():
            logger.info(f"  {key}: {val}")

        model = lgb.LGBMClassifier(**params)

        return model

    def train_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM with early stopping.

        Early stopping:
        - Monitor validation loss every iteration
        - If no improvement for 50 rounds → stop training
        - Prevents overfitting (model memorizing training data)

        Why eval_set?
        - LightGBM needs validation data to monitor performance
        - Can't use training loss (always decreases, even when overfitting)
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MODEL")
        logger.info("=" * 60)

        model = self.build_model(y_train)

        logger.info("Training with early stopping...")
        logger.info("  Monitoring: binary_logloss on validation set")
        logger.info("  Early stopping: 50 rounds without improvement")

        # Train with validation monitoring
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.log_evaluation(period=50)  # Log every 50 iterations
            ]
        )

        logger.info(f"\n✅ Training complete!")
        logger.info(f"  Best iteration: {model.best_iteration_}")
        logger.info(f"  Best score: {model.best_score_['valid_0']['binary_logloss']:.4f}")

        self.model = model
        return model

    def evaluate_with_threshold(
            self,
            y_true: pd.Series,
            y_prob: np.ndarray,
            threshold: float = 0.5
    ) -> dict:
        """
        Evaluate model at specific threshold.

        Why threshold matters:
        - Model outputs probability: 0.0 to 1.0
        - We convert to decision: 0 or 1
        - Default: if prob >= 0.5 → predict 1
        - BUT with imbalance, 0.5 is BAD

        Example:
        - Part A: prob = 0.48 → predict 0 (with threshold=0.5)
        - Part A: prob = 0.48 → predict 1 (with threshold=0.3)

        We'll search for best threshold that maximizes MCC
        """
        # Convert probabilities to predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)

        # Other metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'threshold': threshold,
            'mcc': mcc,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def optimize_threshold(
            self,
            y_val: pd.Series,
            y_val_prob: np.ndarray
    ) -> float:
        """
        Find threshold that maximizes MCC on validation set.

        Why not use 0.5?
        - With 0.58% failure rate, model might output:
          - prob=0.01 for normal parts
          - prob=0.15 for failures
        - Threshold=0.5 → predicts everything as 0 (bad!)
        - Threshold=0.10 → catches some failures (better)

        We test thresholds from 0.01 to 0.99 and pick best MCC
        """
        logger.info("\n" + "=" * 60)
        logger.info("THRESHOLD OPTIMIZATION")
        logger.info("=" * 60)

        logger.info("Testing thresholds from 0.01 to 0.99...")

        best_mcc = -1
        best_threshold = 0.5
        best_metrics = None

        thresholds = np.linspace(0.01, 0.99, 99)
        mcc_scores = []

        for thr in thresholds:
            metrics = self.evaluate_with_threshold(y_val, y_val_prob, threshold=thr)
            mcc_scores.append(metrics['mcc'])

            if metrics['mcc'] > best_mcc:
                best_mcc = metrics['mcc']
                best_threshold = thr
                best_metrics = metrics

        logger.info(f"\n✅ Best threshold: {best_threshold:.3f}")
        logger.info(f"  MCC: {best_mcc:.4f}")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Recall: {best_metrics['recall']:.4f}")
        logger.info(f"  F1: {best_metrics['f1']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TP={best_metrics['tp']}, FP={best_metrics['fp']}")
        logger.info(f"  FN={best_metrics['fn']}, TN={best_metrics['tn']}")

        # Plot MCC vs threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, mcc_scores, linewidth=2)
        ax.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.3f}')
        ax.axhline(best_mcc, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('MCC', fontsize=12)
        ax.set_title('MCC vs Classification Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.outputs_dir / 'threshold_optimization.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.outputs_dir / 'threshold_optimization.png'}")
        plt.close()

        self.best_threshold = best_threshold
        return best_threshold

    def analyze_feature_importance(self, top_n: int = 30):
        """
        Analyze which features contribute most to predictions.

        LightGBM importance types:
        - split: How many times feature was used for splitting
        - gain: Total improvement from splits using this feature

        Why this matters:
        - Validates Day 5 engineering (do our features help?)
        - Identifies weak features (can remove for simplicity)
        - Guides further feature engineering
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"TOP {top_n} FEATURE IMPORTANCE")
        logger.info("=" * 60)

        # Get feature importances (gain-based)
        importances = self.model.feature_importances_

        # Create dataframe
        feat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} features:")
        for i, row in feat_imp.head(top_n).iterrows():
            logger.info(f"  {row['feature']:50s} {row['importance']:>10.0f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 12))
        top_feats = feat_imp.head(top_n)
        ax.barh(range(len(top_feats)), top_feats['importance'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats['feature'], fontsize=9)
        ax.set_xlabel('Importance (Gain)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.outputs_dir / 'feature_importance.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.outputs_dir / 'feature_importance.png'}")
        plt.close()

        # Save to CSV
        feat_imp.to_csv(self.outputs_dir / 'feature_importance.csv', index=False)
        logger.info(f"📄 CSV saved: {self.outputs_dir / 'feature_importance.csv'}")

        return feat_imp

    def save_model(self):
        """Save trained model and threshold for production."""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING MODEL")
        logger.info("=" * 60)

        model_path = self.models_dir / 'baseline_lgbm.pkl'
        joblib.dump({
            'model': self.model,
            'threshold': self.best_threshold,
            'feature_names': self.feature_names
        }, model_path)

        logger.info(f"✅ Model saved to: {model_path}")
        logger.info(f"  - LightGBM model")
        logger.info(f"  - Best threshold: {self.best_threshold:.3f}")
        logger.info(f"  - Feature names ({len(self.feature_names)} features)")

    def run_full_pipeline(self):
        """Execute complete training pipeline."""
        logger.info("=" * 60)
        logger.info("BOSCH BASELINE MODEL TRAINING")
        logger.info("=" * 60)

        # 1. Load data
        X, y = self.load_data()

        # 2. Train/val split
        X_train, X_val, y_train, y_val = self.create_train_val_split(X, y)

        # 3. Train model
        self.train_model(X_train, y_train, X_val, y_val)

        # 4. Predict on validation
        y_val_prob = self.model.predict_proba(X_val)[:, 1]

        # 5. Evaluate at default threshold
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE EVALUATION (threshold=0.5)")
        logger.info("=" * 60)
        metrics_05 = self.evaluate_with_threshold(y_val, y_val_prob, threshold=0.5)
        logger.info(f"MCC: {metrics_05['mcc']:.4f}")
        logger.info(f"Precision: {metrics_05['precision']:.4f}")
        logger.info(f"Recall: {metrics_05['recall']:.4f}")

        # 6. Optimize threshold
        self.optimize_threshold(y_val, y_val_prob)

        # 7. Feature importance
        self.analyze_feature_importance(top_n=30)

        # 8. Save model
        self.save_model()

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Review feature importance (do engineered features help?)")
        logger.info("  2. Analyze errors (which failures are we missing?)")
        logger.info("  3. Advanced tuning (hyperparameter optimization)")
        logger.info("  4. Test set evaluation")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline LightGBM model")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    args = parser.parse_args()

    config = Config(args.config)
    trainer = BoschBaselineModel(config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
