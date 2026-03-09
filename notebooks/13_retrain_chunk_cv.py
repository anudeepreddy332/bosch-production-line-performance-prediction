"""
13_retrain_chunk_cv.py

Phase 2: Chunk-Aware Cross-Validation

Problem with TimeSeriesSplit (earlier approach):
- Sliced rows sequentially -> chunks bisected across folds
- Leak features (chunk_id, chunk_size) became noise in validation
- MCC dropped significantly

Fix - Chunk-Preserving Fold Assignment:
- Sort unique chunks by chunk_id (already in temporal order from start_time sort)
- Assign chunks to folds via round-robin -> each fold covers full timeline
- Chunks never split across folds -> leak features stay valid in validation

Expected MCC: 0.37-0.42
"""
import sys
import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

# Config
N_FOLDS = 4
SEEDS = [42, 99]
DOWNSAMPLE_RATIO = 0.30 # keep 30% of singleton non-failures

# These cols exist in parquet but must not be model features
NOT_FEATURE_COLS = ['Id', 'Response', 'next_response_in_chunk', 'fold']

# LightGBM base params - scale_pos_weight set dynamically per fold
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'scale_pos_weight': 15,
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 50,
    'feature_fraction': 0.70,
    'bagging_fraction': 0.70,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
}

def assign_chunk_folds(df: pd.DataFrame, n_folds: int) -> pd.Series:
    """
    Assign fold numbers to every row, keeping each chunk entirely within one fold.
    chunk_id is already in temporal order - created as a cumsum after sorting by start_time in
    create_leak_features.py.
    Round-robin assignment means fold 0 gets chunks 0,4,8,...
    fold 1 gets chunks 1,5,9,..etc. - time coverage is spread evenly across all folds
    without ever splitting a chunk.
    """
    unique_chunks = np.sort(df['chunk_id'].unique())

    chunk_to_fold = {
        chunk_id: idx % n_folds
        for idx, chunk_id in enumerate(unique_chunks)
    }

    return df['chunk_id'].map(chunk_to_fold)

def downsample_training_set(df_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Controlled downsampling on the training fold only - never on validation.

    Keep: all failures (Response == 1) -> rare, every one matters
    Keep: all chunk duplicates (Response == 0) -> carry leak signal
    Sample: easy neaatives (chunk_size == 1, Response == 0) at DOWNSAMPLE_RATIO

    Easy negatives are singleton parts with no duplicates - the model
    already classifies them easily. Keeping all 1.1M of them adds noise
    and slows training without improving recall.
    """
    failures = df_train[df_train['Response'] == 1]
    dup_negatives = df_train[(df_train['chunk_size'] > 1) & (df_train['Response'] == 0)]
    easy_neg = df_train[(df_train['chunk_size'] == 1) & (df_train['Response'] == 0)]

    easy_neg_sampled = easy_neg.sample(frac=DOWNSAMPLE_RATIO, random_state=seed)

    result = pd.concat([failures, dup_negatives, easy_neg_sampled], ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"    Failures kept:        {len(failures):>8,}")
    logger.info(f"    Dup negatives kept:   {len(dup_negatives):>8,}")
    logger.info(f"    Easy neg sampled:     {len(easy_neg_sampled):>8,}  "
                f"({DOWNSAMPLE_RATIO*100:.0f}% of {len(easy_neg):,})")
    logger.info(f"    Total training rows:  {len(result):>8,}")

    return result

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Grid search ovr [0.01, 0.50] to find threshold maximizing MCC.
    Returns (best_threshold, best_mcc).
    """
    best_mcc, best_thresh = -1.0, 0.5

    for thresh in np.arange(0.01, 1.00, 0.01):
        mcc = matthews_corrcoef(y_true, (y_proba >= thresh).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thresh = mcc, thresh
    return best_thresh, best_mcc

def log_confusion(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp+fp) if (tp+fp) > 0 else 0
    recall = tp / (tp+fn) if (tp+fn) > 0 else 0
    logger.info(f"  {label}Confusion Matrix:")
    logger.info(f"    TP={tp:>7,}  FP={fp:>7,}")
    logger.info(f"    FN={fn:>7,}  TN={tn:>7,}")
    logger.info(f"    Precision={precision:.4f}  Recall={recall:.4f}")
    return tn, fp, fn, tp

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2: CHUNK-AWARE CV — LightGBM")
    logger.info("=" * 60)
    logger.info(f"  Folds:  {N_FOLDS}")
    logger.info(f"  Seeds:  {SEEDS}")
    logger.info(f"  Neg downsample: {DOWNSAMPLE_RATIO*100:.0f}% of easy negatives")

    # ── Load ─────────────────────────────────────────────────────────
    logger.info("\nLoading merged feature set...")
    df = pd.read_parquet(
        project_root / "data/features/train_all_features_with_leaks.parquet"
    )
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Failures: {df['Response'].sum():,} ({df['Response'].mean():.4%})")
    memory_usage_report(df, "Feature Matrix")

    # ── Fold assignment ───────────────────────────────────────────────
    logger.info("\nAssigning chunk-aware folds...")
    df['fold'] = assign_chunk_folds(df, N_FOLDS)

    # Validate - no chunk should span more than one fold
    chunks_per_fold = df.groupby('chunk_id')['fold'].nunique()
    split_chunks = (chunks_per_fold > 1).sum()

    for f in range(N_FOLDS):
        fold_df = df[df['fold'] == f]
        logger.info(f"  Fold {f}: {len(fold_df):>9,} rows | "
                    f"{fold_df['chunk_id'].nunique():>7,} chunks | "
                    f"failure rate={fold_df['Response'].mean():.4%}")

    if split_chunks > 0:
        logger.error(f"  ❌ {split_chunks} chunks split across folds — aborting")
        return
    else:
        logger.info(f"  ✅ All chunks intact — no chunk spans multiple folds")

    # Feature columns
    feature_cols = [c for c in df.columns if c not in NOT_FEATURE_COLS]
    logger.info(f"\nFeatures used for training: {len(feature_cols)}")

    X_all = df[feature_cols]
    y_all = df['Response'].values

    # Cross-validation loop
    oof_preds = np.zeros(len(df))
    fold_mccs = []
    all_models = []

    for fold in range(N_FOLDS):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"FOLD {fold + 1} / {N_FOLDS}")
        logger.info(f"{'─' * 60}")

        train_mask = (df['fold'] != fold).values
        val_mask = (df['fold'] == fold).values

        # Downsample training fold
        logger.info("  Downsampling training fold...")
        train_df_sampled = downsample_training_set(
            df.loc[train_mask, feature_cols + ['Response']].copy().reset_index(drop=True),
            seed=SEEDS[0]
        )
        X_train = train_df_sampled[feature_cols]
        y_train = train_df_sampled['Response'].values

        X_val = X_all.loc[val_mask]
        y_val = y_all[val_mask]


        # Multi-seed training
        fold_proba = np.zeros(val_mask.sum())
        fold_proba_list = []
        fold_models = []

        for seed in SEEDS:
            logger.info(f"  → Training seed={seed}...")
            params = {**LGBM_PARAMS,
            'random_state': seed
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set = [(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=150, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            )
            logger.info(f"     Best iteration: {model.best_iteration_}")
            # Only include model in ensemble if it trained meaninfully
            if model.best_iteration_ >= 50:
                proba = model.predict_proba(X_val)[:, 1]
                fold_proba_list.append(proba)
                fold_models.append(model)
            else:
                logger.warning(f"      ⚠️ Skipping seed={seed} — stopped too early ({model.best_iteration_} trees)")

        # Average only valid models
        if len(fold_proba_list) == 0:
            logger.error("  ❌ All seeds stopped early — check params")
            return
        fold_proba = np.mean(fold_proba_list, axis=0)
        oof_preds[val_mask] = fold_proba
        all_models.append(fold_models)

        # Fold metrics
        thresh, fold_mcc = optimize_threshold(y_val, fold_proba)
        fold_mccs.append(fold_mcc)
        logger.info(f"\n  Fold {fold + 1} MCC: {fold_mcc:.4f}  (threshold={thresh:.2f})")
        log_confusion(y_val, (fold_proba >= thresh).astype(int))

    # ── OOF summary ───────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("OOF SUMMARY")
    logger.info(f"{'=' * 60}")

    oof_thresh, oof_mcc = optimize_threshold(y_all, oof_preds)

    logger.info(f"\n  Per-fold MCCs: {[f'{m:.4f}' for m in fold_mccs]}")
    logger.info(f"  Mean ± std:    {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}")
    logger.info(f"\n  OOF MCC (all folds):  {oof_mcc:.4f}")
    logger.info(f"  OOF threshold:        {oof_thresh:.2f}")
    log_confusion(y_all, (oof_preds >= oof_thresh).astype(int), "OOF ")

    # Save
    model_path = project_root / "data/models/model_phase2_chunk_cv.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models':       all_models,
            'feature_cols': feature_cols,
            'threshold':    oof_thresh,
            'oof_mcc':      oof_mcc,
            'fold_mccs':    fold_mccs,
        }, f)
    logger.info(f"\n✅ Model saved → {model_path}")

    oof_df = pd.DataFrame({
        'Id':       df['Id'].values,
        'oof_pred': oof_preds,
        'Response': y_all,
    })
    oof_path = project_root / "data/features/oof_predictions_phase2.parquet"
    oof_df.to_parquet(oof_path, index=False)
    logger.info(f"✅ OOF predictions saved → {oof_path}")

    # Final verdict
    logger.info(f"\n{'=' * 60}")
    if oof_mcc >= 0.37:
        logger.info(f"✅ Phase 2 target met: MCC {oof_mcc:.4f} ≥ 0.37")
        logger.info(f"🎯 NEXT: Phase 3 — path/station features (target 0.43–0.48)")
    else:
        logger.info(f"⚠️  MCC {oof_mcc:.4f} — below 0.37, check fold distributions")
    logger.info(f"{'=' * 60}")

if __name__ == "__main__":
    main()