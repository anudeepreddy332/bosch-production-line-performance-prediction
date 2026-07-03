# Model Card

Two models, same underlying algorithm (LightGBM binary classifier), built for opposite goals. The
gap between them is the centerpiece of this repository, not an incidental detail — read the
[Deployability distinction](#deployability-distinction-the-centerpiece) section first if you read
nothing else on this page.

## At a glance

| | Production model (`dataset_h`) | Kaggle model (`P1`) |
|---|---|---|
| Track | 1 — Offline Training + Evaluation | 2 — Kaggle Research (frozen) |
| Deployable? | **Yes** — this is the approved production candidate | **No** — leaderboard-legal only, several input features cannot exist in a real one-at-a-time scoring stream |
| Feature count | 16 | 1,190 |
| Training rows | 1,183,747 | 1,183,747 (same underlying data) |
| OOF MCC | 0.15337 (honest) | 0.39484 (honest OOF; not comparable across tracks — see below) |
| Kaggle private MCC | 0.16160 (K1 reproduction) | **0.41917** (program best) |
| Threshold | 0.91 | 0.89 |
| Data fingerprint | `a5bb652f2b20aca6` | `c602cd26810cf626` |
| Governing log | `docs/research/decisions.md` | `docs/research/kaggle_decisions.md`, `KDR-008` |

## Deployability distinction (the centerpiece)

**`dataset_h` is deployed because every one of its 16 features can be computed for a single part
at the moment it needs to be scored:** start time, duration, rolling failure-rate statistics
computed only from prior training folds, path-transition and station-pair co-occurrence risk. None
of it depends on knowing anything about *other* rows in the same batch, their order, or their
labels.

**P1 cannot be deployed as-is**, for reasons that are structural, not merely "not yet productionized":

- It includes the full ~968-column raw numeric sensor matrix at high model capacity (P0's
  contribution) plus per-station timing and station/line aggregate features (P1's own addition) —
  wide, but every column is still a property of the single row being scored, so this *part* of P1
  is legitimately deployable in principle.
- The broader K1–K5 leakage-family lineage this model stack descends from is not: earlier winning
  Kaggle experiments (K2, K5-B) relied on record-adjacency and duplicate-identity effects that only
  exist because the *entire* test set is visible at once — information a real factory stream,
  scoring one part as it completes, structurally cannot have. P1 itself does not use those specific
  leaky columns, but it inherits the same research lineage and Track-2-only governance status; it
  has never been re-derived and re-validated inside the production track's stronger protocol
  (leakage-free, pre-registered, chunk-aware honest OOF), which is the only path from a Kaggle
  finding to a production decision (see `KDR-009`, "Kaggle → Production is never direct").
- **OOF MCC is not comparable across the two rows above even though both say "honest."** `dataset_h`'s
  0.15337 is measured under the production protocol's leakage-safe chunk-aware CV, gating an actual
  deployment decision. P1's 0.39484 is measured under the same *harness family* but answers a
  different question (private leaderboard optimization) on a training set that is Kaggle-legal but
  production-forbidden in composition. Put the two numbers side by side to see the gap; don't treat
  them as the same yardstick.

This is the deliberate design of the whole repository: two tracks answer "how good could a model
be at this task" (Kaggle) and "how good is the model I would actually run" (production)
separately, so that neither number can be mistaken for the other. See [Results](results.md#why-two-numbers-not-one).

## Production model: `dataset_h`

- **Algorithm:** LightGBM (`LGBMClassifier`), `objective="binary"`.
- **Hyperparameters:** `n_estimators=700`, `learning_rate=0.03`, `num_leaves=63`, `max_depth=-1`,
  `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`,
  `min_child_samples=50`, `class_weight="balanced"`, early stopping at 100 rounds on validation
  `binary_logloss`.
- **Features (16):** start_time, duration, feature_mean, rolling counts (1hr/24hr), density_ratio,
  chunk_id, chunk_size, plus dataset_g's OOF-safe target-rate features (chunk/signature/path/
  rolling failure rates) and dataset_h's path-transition/station-pair co-occurrence risk features.
- **Validation:** chunk-aware `StratifiedGroupKFold` (5 folds), `chunk_id` never split across
  train/validation (`validate_chunk_aware_splits` raises if it is).
- **Intended use:** score incoming, unlabeled manufacturing-line batches (Track 3); flag
  high-risk parts for inspection under a cost-weighted threshold/budget policy
  (`src/inference/decision_engine.py`).
- **Known limitation:** static threshold; a rolling-origin evaluation shows deployed MCC ranges
  0.06–0.18 across 5 out-of-time folds, not a single number — see [Track 1](track1.md#results-honest-with-caveats).
- **Persistence:** `models/dataset_h_model.pkl` (raw `LGBMClassifier`, joblib-dumped), committed
  to git despite `models/*.pkl` being gitignored (intentionally force-added).

## Kaggle model: `P1`

- **Algorithm:** LightGBM (`LGBMClassifier`), `objective="binary"`, `HIGH_CAPACITY` parameter set.
- **Hyperparameters:** `n_estimators=2500`, `learning_rate=0.02`, `num_leaves=255` (tuned up from
  63), `max_depth=-1`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`,
  `min_child_samples=20` (tuned down from 50), `class_weight="balanced"`, early stopping at 150
  rounds.
- **Features (1,190):** K5-A's 51-column honest label-free stack (record-position, timing-cohort,
  and duplicate/identity-key features) + the full ~968-column raw numeric sensor matrix (P0) +
  Family D (57 per-station date-offset/weekly-position/transit columns) + Family S/L (108
  station/line numeric mean/std aggregates) — all label-free by construction.
- **Validation:** same chunk-aware CV family as production, reusing persisted `chunk_id`/`cv_fold`
  assignment; a mandatory regression anchor (`LEGACY_LGB_PARAMS` on K5-A's exact 51 columns) had to
  reproduce OOF 0.32506 ± 1e-4 before this result was trusted.
- **Intended use:** Kaggle leaderboard submission only. **Not used, and not eligible to be used,
  for any production decision** — see [Track 2](track2.md#known-limitations-non-goals).
- **Governance:** pre-registered `KDR-008`, frozen alongside the rest of Track 2 at `KDR-009`
  (tag `track2-frozen`). No further tuning authorized without a new pre-registered KDR.
- **Persistence:** quarantined under `src/kaggle/` / `scripts/kaggle/`; not committed to the same
  `models/` directory as production artifacts.
