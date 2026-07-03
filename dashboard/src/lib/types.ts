export type ModelKey = "baseline" | "dataset_g" | "dataset_h" | "meta_model";

export const MODEL_KEYS: ModelKey[] = ["baseline", "dataset_g", "dataset_h", "meta_model"];

export interface FoldMetric {
  fold: number;
  rows: number;
  best_threshold: number;
  mcc: number;
}

export interface ModelMeta {
  label: string;
  rows: number;
  feature_count: number;
  best_threshold: number;
  oof_mcc: number;
  data_fingerprint: string | null;
  fold_metrics: FoldMetric[];
  base_thresholds?: Record<string, number>;
}

export type ModelsMeta = Record<ModelKey, ModelMeta>;

export interface SweepPoint {
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  fpr: number;
  mcc: number;
  flagged_pct: number;
}

export interface ImportanceRow {
  feature: string;
  importance: number;
  importance_pct: number;
  family: string;
}

export type Importances = Record<ModelKey, ImportanceRow[]>;

export interface CalibrationRow {
  bin_lo: number;
  bin_hi: number;
  mean_predicted: number;
  observed_rate: number;
  count: number;
}

export type Calibration = Record<ModelKey, CalibrationRow[]>;

export interface KaggleExperiment {
  experiment_id: string;
  kdr: string;
  kdr_file: string;
  date: string;
  mechanism: string;
  feature_count: number;
  oof_mcc: number;
  oof_status: "honest" | "contaminated";
  threshold: number;
  public_mcc: number;
  private_mcc: number;
  data_fingerprint: string;
  git_tag: string;
  hypothesis: string | null;
  hypothesis_result: string | null;
  notes: string;
}

export interface Leaderboard {
  schema_version: string;
  generated: string;
  source_of_truth: string;
  program_status: string;
  notes: string[];
  experiments: KaggleExperiment[];
}

export interface KdrHeading {
  kdr: string;
  heading: string;
}

export interface Governance {
  repo_url: string;
  leaderboard: Leaderboard;
  kdr_headings: KdrHeading[];
  kaggle_decisions_path: string;
}

export interface Rp2FoldResult {
  fold_idx: number;
  test_chunks: string;
  test_pos_rate: number;
  oot_mcc_best_threshold: number;
  oot_best_threshold: number;
  oot_mcc_fixed_threshold: number;
}

export interface Rp2Summary {
  reproduce: string;
  cross_origin_summary: {
    n_folds: number;
    mean_mcc: number;
    std_mcc: number;
    ci_95_lower: number;
    ci_95_upper: number;
    min_mcc: number;
    max_mcc: number;
    degradation_vs_incv_absolute: number;
    degradation_vs_incv_pct: number;
    corr_test_posrate_vs_mcc: number;
  };
  fold_results: Rp2FoldResult[];
  incv_mcc: number;
  incv_threshold: number;
}

export interface RepoLinks {
  repo_url: string;
  tags: string[];
}
