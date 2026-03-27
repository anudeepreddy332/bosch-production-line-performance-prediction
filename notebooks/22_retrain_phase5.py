"""
Phase 5: Retrain LightGBM with Temporal + Date + Path + Chunk Features

Changes from Phase 4 (script 19):
  - Input: train_phase5_features.parquet (428 cols vs 421)
  - New features: 7 temporal (MeanTimeDiff since/till last K failures, records_same_6min)
  - Deeper trees: num_leaves=511 (was 63), max_depth=10, min_child_samples=20
  - Same chunk-aware CV, downsampling, seeds, spw=5

Goal: MCC jump from 0.3376 toward 0.39+ via temporal failure clustering signal
      + deeper tree interactions.

Expected lift sources:
  - till_next_1/5/10: strongest expected features (leaky, same tier as path_failure_rate)
  - since_last_1/5/10: clean backward signal, failure state of line
  - records_same_6min: batch density, quality control signal
  - num_leaves=511: captures 14-way feature interactions vs 6-way before
"""
import sys
import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import setup_logger
from src.utils.memory import memory_usage_report

logger = setup_logger(__name__)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# ── Config ────────────────────────────────────────────────────────
N_FOLDS        = 4
SEEDS          = [42, 99]
DOWNSAMPLE_RATIO = 0.30

NOT_FEATURE_COLS = [
    'Id', 'Response', 'next_response_in_chunk',
    'fold', 'path_signature'
]

LGBM_PARAMS = {
    'objective':          'binary',
    'verbosity':          -1,
    'boosting_type':      'gbdt',
    'scale_pos_weight':   5,
    'n_estimators':       1500,
    'learning_rate':      0.02,
    'num_leaves':         63,      # was 63 — captures deeper interactions
    'max_depth':          -1,       # explicit cap — prevents memory explosion
    'min_child_samples':  50,       # was 50 — allows finer splits on rare failures
    'feature_fraction':   0.70,
    'bagging_fraction':   0.70,
    'bagging_freq':       5,
    'reg_alpha':          0.1,
    'reg_lambda':         1.0,
    'n_jobs':             -1,
}

PHASE4_MCC = 0.3376   # baseline to beat

# ── Helpers (identical to script 19) ─────────────────────────────
def assign_chunk_fold(df: pd.DataFrame, n_folds: int) -> pd.Series:
    unique_chunks = np.sort(df['chunk_id'].unique())
    chunk_to_fold = {cid: idx % n_folds for idx, cid in enumerate(unique_chunks)}
    return df['chunk_id'].map(chunk_to_fold)

def downsample_training_set(df_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    failures  = df_train[df_train['Response'] == 1]
    dup_neg   = df_train[(df_train['chunk_size'] > 1) & (df_train['Response'] == 0)]
    easy_neg  = df_train[(df_train['chunk_size'] == 1) & (df_train['Response'] == 0)]
    easy_sampled = easy_neg.sample(frac=DOWNSAMPLE_RATIO, random_state=seed)

    result = pd.concat([failures, dup_neg, easy_sampled], ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"    Failures kept:        {len(failures):>8,}")
    logger.info(f"    Dup negatives kept:   {len(dup_neg):>8,}")
    logger.info(f"    Easy neg sampled:     {len(easy_sampled):>8,}  "
                f"({DOWNSAMPLE_RATIO*100:.0f}% of {len(easy_neg):,})")
    logger.info(f"    Total training rows:  {len(result):>8,}")
    return result

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray):
    best_mcc, best_thresh = -1.0, 0.5
    for thresh in np.arange(0.01, 1.00, 0.01):
        mcc = matthews_corrcoef(y_true, (y_proba >= thresh).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thresh = mcc, thresh
    return best_thresh, best_mcc

def log_confusion(y_true, y_pred, label=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    logger.info(f"  {label}Confusion Matrix:")
    logger.info(f"    TP={tp:>7,}  FP={fp:>7,}")
    logger.info(f"    FN={fn:>7,}  TN={tn:>7,}")
    logger.info(f"    Precision={precision:.4f}  Recall={recall:.4f}")
    return tn, fp, fn, tp


# ── Main ──────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("PHASE 5: CHUNK-AWARE CV — LightGBM + Temporal Features")
    logger.info("=" * 60)
    logger.info(f"  Folds:          {N_FOLDS}")
    logger.info(f"  Seeds:          {SEEDS}")
    logger.info(f"  Neg downsample: {DOWNSAMPLE_RATIO*100:.0f}% of easy negatives")
    logger.info(f"  scale_pos_weight: {LGBM_PARAMS['scale_pos_weight']}")
    logger.info(f"  num_leaves:       {LGBM_PARAMS['num_leaves']}  (was 63)")
    logger.info(f"  max_depth:        {LGBM_PARAMS['max_depth']}")
    logger.info(f"  Baseline to beat: Phase 4 MCC {PHASE4_MCC}")

    # 1. Load
    logger.info("\n1) Loading Phase 5 feature matrix...")
    df = pd.read_parquet(project_root / "data/features/train_phase5_features.parquet")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Failures: {df['Response'].sum():,} ({df['Response'].mean():.4%})")
    memory_usage_report(df, "Phase 5 Feature Matrix")

    # 2. Downcast float64 → float32
    logger.info("\n2) Downcasting float64 to float32 where safe...")
    float64_cols = df.select_dtypes(include=['float64']).columns
    logger.info(f"  float64 cols: {len(float64_cols)}")
    for col in float64_cols:
        df[col] = df[col].astype('float32')
    memory_usage_report(df, "Phase 5 After Downcast")

    # 3. Chunk-aware folds
    logger.info("\n3) Assigning chunk-aware folds...")
    df['fold'] = assign_chunk_fold(df, N_FOLDS)
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
    logger.info("  ✅ All chunks intact")

    # 4. Feature columns
    logger.info("\n4) Selecting feature columns...")
    feature_cols = [c for c in df.columns if c not in NOT_FEATURE_COLS]
    logger.info(f"  Total features: {len(feature_cols)}")

    # Log new temporal features explicitly
    temporal_cols = [c for c in feature_cols if 'timediff' in c or 'same_6min' in c]
    logger.info(f"  → Temporal features:  {len(temporal_cols)}")
    for c in temporal_cols:
        logger.info(f"       {c}")

    X_all = df[feature_cols]
    y_all = df['Response'].values

    # 5. CV loop
    oof_preds  = np.zeros(len(df))
    fold_mccs  = []
    all_models = []

    for fold in range(N_FOLDS):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"FOLD {fold + 1} / {N_FOLDS}")
        logger.info(f"{'─' * 60}")

        train_mask = (df['fold'] != fold).values
        val_mask   = (df['fold'] == fold).values

        logger.info("  Downsampling training fold...")
        train_df_sampled = downsample_training_set(
            df.loc[train_mask, feature_cols + ['Response']].copy().reset_index(drop=True),
            seed=SEEDS[0]
        )
        X_train = train_df_sampled[feature_cols]
        y_train = train_df_sampled['Response'].values
        X_val   = X_all.loc[val_mask]
        y_val   = y_all[val_mask]

        fold_proba_list = []
        fold_models     = []

        for seed in SEEDS:
            logger.info(f"  → Training seed={seed}...")
            params = {**LGBM_PARAMS, 'random_state': seed}
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=200)],
            )
            logger.info(f"     Trees trained: {model.n_estimators}")
            proba = model.predict_proba(X_val)[:, 1]
            fold_proba_list.append(proba)
            fold_models.append(model)

        fold_proba = np.mean(fold_proba_list, axis=0)
        oof_preds[val_mask] = fold_proba
        all_models.append(fold_models)

        thresh, fold_mcc = optimize_threshold(y_val, fold_proba)
        fold_mccs.append(fold_mcc)
        logger.info(f"\n  Fold {fold + 1} MCC: {fold_mcc:.4f}  (threshold={thresh:.2f})")
        log_confusion(y_val, (fold_proba >= thresh).astype(int))

    # 6. OOF summary
    logger.info(f"\n{'=' * 60}")
    logger.info("OOF SUMMARY")
    logger.info(f"{'=' * 60}")
    oof_thresh, oof_mcc = optimize_threshold(y_all, oof_preds)
    logger.info(f"\n  Per-fold MCCs: {[f'{m:.4f}' for m in fold_mccs]}")
    logger.info(f"  Mean ± std:    {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}")
    logger.info(f"\n  OOF MCC (all folds):  {oof_mcc:.4f}")
    logger.info(f"  OOF threshold:        {oof_thresh:.2f}")
    log_confusion(y_all, (oof_preds >= oof_thresh).astype(int), 'OOF')

    # 7. Save model
    model_path = project_root / "data/models/model_phase5_temporal.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models':       all_models,
            'feature_cols': feature_cols,
            'threshold':    oof_thresh,
            'oof_mcc':      oof_mcc,
            'fold_mccs':    fold_mccs,
        }, f)
    logger.info(f"\n✅ Model saved → {model_path}")

    # 8. Save OOF predictions
    oof_df = pd.DataFrame({
        'Id':       df['Id'].values,
        'oof_pred': oof_preds,
        'Response': y_all
    })
    oof_path = project_root / "data/features/oof_predictions_phase5.parquet"
    oof_df.to_parquet(oof_path, index=False)
    logger.info(f"✅ OOF predictions saved → {oof_path}")

    # 9. Verdict
    delta = oof_mcc - PHASE4_MCC
    logger.info(f"\n{'=' * 60}")
    if oof_mcc >= 0.45:
        logger.info(f"🚀 Excellent: MCC {oof_mcc:.4f} — temporal features delivering big lift")
        logger.info(f"🎯 NEXT: script 23 — feature combinations (top-20 × top-20)")
    elif oof_mcc >= 0.39:
        logger.info(f"✅ On track: MCC {oof_mcc:.4f} (+{delta:.4f} vs Phase 4)")
        logger.info(f"🎯 NEXT: script 23 — feature combinations (top-20 × top-20)")
    elif oof_mcc >= PHASE4_MCC:
        logger.info(f"⚠️  Modest gain: MCC {oof_mcc:.4f} (+{delta:.4f} vs Phase 4)")
        logger.info(f"🎯 Check feature importance — are till_next_* in top 20?")
    else:
        logger.info(f"❌ Regression: MCC {oof_mcc:.4f} ({delta:.4f} vs Phase 4)")
        logger.info(f"🎯 Check: are temporal cols included in feature_cols?")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n📊 PHASE COMPARISON:")
    logger.info(f"   Phase 2 (leak only):         MCC ~0.3247")
    logger.info(f"   Phase 3 (+ path feats):      MCC ~0.3323")
    logger.info(f"   Phase 4 (+ date feats):      MCC  0.3376")
    logger.info(f"   Phase 5 (+ temporal feats):  MCC  {oof_mcc:.4f}  ← today")


if __name__ == '__main__':
    main()