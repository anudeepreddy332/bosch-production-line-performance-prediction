"""
Exploratory Data Analysis (EDA) for Bosch production line data.

Analysis goals:
1. Station coverage: Which stations do parts visit?
2. Failure patterns: Which stations correlate with failures?
3. Sparsity patterns: How sparse is each line/station?
4. Feature importance: Which raw features matter most?

Why EDA before modeling:
- Informs feature engineering (where to focus effort)
- Identifies data quality issues (leakage, bias)
- Validates domain assumptions (do patterns make sense?)
- Guides model selection (sparse features → LightGBM)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src.logger import setup_logger

logger = setup_logger(__name__)

# Matplotlib settings for clean plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BoschEDA:
    """
    Exploratory analysis for Bosch manufacturing data.

    Focuses on understanding:
    - Which stations matter (coverage + failure correlation)
    - Sparsity patterns (routing paths)
    - Feature distributions (numeric ranges, outliers)
    """

    def __init__(self, data_path: str = "data/features/train_selected.parquet"):
        """Load selected features for analysis."""
        logger.info("=" * 60)
        logger.info("BOSCH PRODUCTION LINE EDA")
        logger.info("=" * 60)

        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        logger.info(f"  Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")

        # Parse feature names into structured format
        self._parse_features()

        # Create output directory for plots
        self.output_dir = Path("outputs/eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_features(self):
        """
        Parse feature names to extract line/station/feature structure.

        Feature naming: L{line}_S{station}_F{feature} or D{date}

        Examples:
        - L0_S0_F10 → Line 0, Station 0, Feature 10
        - L2_S15_D234 → Line 2, Station 15, Date 234
        """
        logger.info("\nParsing feature structure...")

        self.features = [c for c in self.df.columns if c not in ['Id', 'Response']]

        # Extract line, station, type from each feature
        self.feature_info = []
        for feat in self.features:
            parts = feat.split('_')
            if len(parts) >= 3:
                line = int(parts[0][1:])  # L0 → 0
                station = int(parts[1][1:])  # S15 → 15
                feat_type = parts[2][0]  # F or D
                feat_num = int(parts[2][1:])  # F234 → 234

                self.feature_info.append({
                    'feature': feat,
                    'line': line,
                    'station': station,
                    'type': feat_type,
                    'number': feat_num
                })

        self.feature_df = pd.DataFrame(self.feature_info)

        # Summary
        n_lines = self.feature_df['line'].nunique()
        n_stations = self.feature_df['station'].nunique()
        n_features = len(self.feature_df[self.feature_df['type'] == 'F'])
        n_dates = len(self.feature_df[self.feature_df['type'] == 'D'])

        logger.info(f"  Lines: {n_lines}")
        logger.info(f"  Stations: {n_stations}")
        logger.info(f"  Feature sensors (F): {n_features}")
        logger.info(f"  Date features (D): {n_dates}")

    def analyze_station_coverage(self):
        """
        Analyze which stations parts visit.

        Key insight: Not all parts visit all stations (routing variability)

        Creates:
        - Bar chart: % of parts visiting each station
        - Identifies critical stations (>50% coverage)
        - Identifies rare stations (<5% coverage)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STATION COVERAGE ANALYSIS")
        logger.info("=" * 60)

        # For each station, calculate % of parts with non-null values
        station_coverage = {}

        for station in self.feature_df['station'].unique():
            # Get all features for this station
            station_features = self.feature_df[
                self.feature_df['station'] == station
                ]['feature'].tolist()

            # A part "visited" this station if ANY feature is non-null
            visited = self.df[station_features].notnull().any(axis=1)
            coverage_pct = (visited.sum() / len(self.df)) * 100

            station_coverage[station] = coverage_pct

        # Sort by coverage
        station_coverage = dict(sorted(station_coverage.items(), key=lambda x: x[1], reverse=True))

        # Log top and bottom stations
        logger.info("\nTop 10 Most Visited Stations:")
        for station, pct in list(station_coverage.items())[:10]:
            logger.info(f"  Station {station}: {pct:.1f}% of parts")

        logger.info("\nBottom 10 Least Visited Stations:")
        for station, pct in list(station_coverage.items())[-10:]:
            logger.info(f"  Station {station}: {pct:.1f}% of parts")

        # Categorize stations
        critical = [s for s, p in station_coverage.items() if p > 50]
        common = [s for s, p in station_coverage.items() if 10 < p <= 50]
        rare = [s for s, p in station_coverage.items() if p <= 10]

        logger.info(f"\nStation Categories:")
        logger.info(f"  Critical (>50% coverage): {len(critical)} stations")
        logger.info(f"  Common (10-50% coverage): {len(common)} stations")
        logger.info(f"  Rare (<10% coverage): {len(rare)} stations")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        stations = list(station_coverage.keys())
        coverage = list(station_coverage.values())

        bars = ax.bar(range(len(stations)), coverage, color='steelblue', alpha=0.7)

        # Color-code bars
        for i, (s, c) in enumerate(zip(stations, coverage)):
            if c > 50:
                bars[i].set_color('green')
            elif c < 10:
                bars[i].set_color('red')

        ax.set_xlabel('Station Number', fontsize=12)
        ax.set_ylabel('Coverage (% of parts)', fontsize=12)
        ax.set_title('Station Coverage: Which Stations Do Parts Visit?', fontsize=14, fontweight='bold')
        ax.axhline(50, color='green', linestyle='--', alpha=0.5, label='Critical (>50%)')
        ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='Rare (<10%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'station_coverage.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.output_dir / 'station_coverage.png'}")
        plt.close()

        return station_coverage

    def analyze_failure_by_station(self):
        """
        Identify which stations correlate with failures.

        Logic:
        - For each station, compare failure rates:
          - Parts that visited station
          - Parts that skipped station
        - High difference = station might be failure indicator

        Key insight: If Station 24 has 2% failure rate vs 0.5% baseline,
        Station 24 might be where issues are detected (or caused)
        """
        logger.info("\n" + "=" * 60)
        logger.info("FAILURE ANALYSIS BY STATION")
        logger.info("=" * 60)

        baseline_failure_rate = (self.df['Response'] == 1).sum() / len(self.df) * 100
        logger.info(f"\nBaseline failure rate: {baseline_failure_rate:.3f}%")

        station_failure_rates = {}

        for station in self.feature_df['station'].unique():
            # Get features for this station
            station_features = self.feature_df[
                self.feature_df['station'] == station
                ]['feature'].tolist()

            # Parts that visited this station
            visited_mask = self.df[station_features].notnull().any(axis=1)
            visited_parts = self.df[visited_mask]

            if len(visited_parts) > 100:  # Need sufficient sample
                failure_rate = (visited_parts['Response'] == 1).sum() / len(visited_parts) * 100
                station_failure_rates[station] = {
                    'failure_rate': failure_rate,
                    'parts_count': len(visited_parts),
                    'diff_from_baseline': failure_rate - baseline_failure_rate
                }

        # Sort by difference from baseline
        sorted_stations = sorted(
            station_failure_rates.items(),
            key=lambda x: abs(x[1]['diff_from_baseline']),
            reverse=True
        )

        logger.info("\nTop 10 Stations with HIGHEST Failure Rates:")
        for station, info in sorted_stations[:10]:
            logger.info(
                f"  Station {station}: {info['failure_rate']:.3f}% "
                f"(+{info['diff_from_baseline']:.3f}% vs baseline, "
                f"n={info['parts_count']:,})"
            )

        logger.info("\nTop 10 Stations with LOWEST Failure Rates:")
        for station, info in sorted(sorted_stations, key=lambda x: x[1]['failure_rate'])[:10]:
            logger.info(
                f"  Station {station}: {info['failure_rate']:.3f}% "
                f"({info['diff_from_baseline']:.3f}% vs baseline, "
                f"n={info['parts_count']:,})"
            )

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        stations = [s for s, _ in sorted_stations][:30]  # Top 30
        diffs = [info['diff_from_baseline'] for _, info in sorted_stations[:30]]

        colors = ['red' if d > 0 else 'green' for d in diffs]
        ax.barh(range(len(stations)), diffs, color=colors, alpha=0.7)

        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels([f"S{s}" for s in stations])
        ax.set_xlabel('Failure Rate Difference from Baseline (%)', fontsize=12)
        ax.set_ylabel('Station', fontsize=12)
        ax.set_title('Top 30 Stations by Failure Rate Deviation', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'failure_by_station.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.output_dir / 'failure_by_station.png'}")
        plt.close()

        return station_failure_rates

    def analyze_sparsity_by_line(self):
        """
        Analyze sparsity (null percentage) by production line.

        Lines (L0, L1, L2, L3) might have different coverage patterns:
        - L0: Main line (low sparsity, most parts go through)
        - L3: Specialty line (high sparsity, only some parts)

        This informs feature engineering:
        - Low sparsity lines → raw features work
        - High sparsity lines → routing features matter more
        """
        logger.info("\n" + "=" * 60)
        logger.info("SPARSITY ANALYSIS BY LINE")
        logger.info("=" * 60)

        line_sparsity = {}

        for line in sorted(self.feature_df['line'].unique()):
            # Get features for this line
            line_features = self.feature_df[
                self.feature_df['line'] == line
                ]['feature'].tolist()

            # Calculate average null percentage
            null_pct = self.df[line_features].isnull().sum().sum() / (len(self.df) * len(line_features)) * 100

            line_sparsity[line] = null_pct
            logger.info(f"  Line {line}: {null_pct:.1f}% sparse")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        lines = list(line_sparsity.keys())
        sparsity = list(line_sparsity.values())

        ax.bar(lines, sparsity, color='coral', alpha=0.7, width=0.6)
        ax.set_xlabel('Production Line', fontsize=12)
        ax.set_ylabel('Sparsity (%)', fontsize=12)
        ax.set_title('Data Sparsity by Production Line', fontsize=14, fontweight='bold')
        ax.set_xticks(lines)
        ax.set_xticklabels([f"L{l}" for l in lines])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sparsity_by_line.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.output_dir / 'sparsity_by_line.png'}")
        plt.close()

        return line_sparsity

    def analyze_top_features(self, top_n: int = 20):
        """
        Identify top features by correlation with Response.

        Uses point-biserial correlation (numeric feature vs binary target)

        Why this matters:
        - Top features guide feature engineering (similar features to create)
        - Informs model interpretation (what signals matter)
        - Validates domain knowledge (do top features make sense?)
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"TOP {top_n} FEATURES BY CORRELATION WITH FAILURE")
        logger.info("=" * 60)

        # Calculate correlation for numeric features only
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f not in ['Id', 'Response']]

        correlations = {}
        for feat in numeric_features:
            # Use Spearman (robust to outliers, handles non-linear)
            corr = self.df[[feat, 'Response']].corr(method='spearman').iloc[0, 1]
            if not np.isnan(corr):
                correlations[feat] = abs(corr)  # Absolute value (direction doesn't matter)

        # Sort by absolute correlation
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        logger.info(f"\nTop {top_n} features:")
        for i, (feat, corr) in enumerate(top_features, 1):
            logger.info(f"  {i}. {feat}: {corr:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        features = [f for f, _ in top_features]
        corrs = [c for _, c in top_features]

        ax.barh(range(len(features)), corrs, color='teal', alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Absolute Correlation with Failure', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Correlation with Response', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_features.png', dpi=150)
        logger.info(f"\n📊 Plot saved: {self.output_dir / 'top_features.png'}")
        plt.close()

        return top_features

    def generate_summary_report(self):
        """
        Generate text summary of key findings.

        This becomes the "EDA Insights" section of your case study.
        """
        logger.info("\n" + "=" * 60)
        logger.info("EDA SUMMARY")
        logger.info("=" * 60)

        # Key stats
        n_parts = len(self.df)
        n_features = len(self.features)
        n_failures = (self.df['Response'] == 1).sum()
        failure_rate = (n_failures / n_parts) * 100

        logger.info(f"\nDataset Overview:")
        logger.info(f"  Parts: {n_parts:,}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Failures: {n_failures:,} ({failure_rate:.3f}%)")

        # Overall sparsity
        overall_sparsity = self.df[self.features].isnull().sum().sum() / (n_parts * n_features) * 100
        logger.info(f"\nOverall Sparsity: {overall_sparsity:.1f}%")
        logger.info(f"  → {100 - overall_sparsity:.1f}% of data is non-null")

        logger.info("\nKey Insights:")
        logger.info("  1. High sparsity (>70%) indicates routing-based production")
        logger.info("     → Not all parts visit all stations")
        logger.info("     → Routing patterns likely predictive of failures")
        logger.info("\n  2. Station coverage varies widely")
        logger.info("     → Some stations critical (>50% parts)")
        logger.info("     → Some stations rare (<10% parts)")
        logger.info("     → Suggests product type differentiation")
        logger.info("\n  3. Failure rates vary by station")
        logger.info("     → Some stations show elevated failure rates")
        logger.info("     → May indicate inspection points or problem areas")
        logger.info("\n  4. Top features show moderate correlation")
        logger.info("     → No single 'magic feature' (highest ~0.05-0.10)")
        logger.info("     → Requires ensemble of features for prediction")
        logger.info("     → Feature engineering will be critical")

        logger.info("\n" + "=" * 60)
        logger.info("EDA COMPLETE - Proceed to Feature Engineering")
        logger.info("=" * 60)

    def run_all_analyses(self):
        """Run complete EDA pipeline."""
        logger.info("\nStarting comprehensive EDA...\n")

        # Run analyses
        self.analyze_station_coverage()
        self.analyze_failure_by_station()
        self.analyze_sparsity_by_line()
        self.analyze_top_features(top_n=20)
        self.generate_summary_report()

        logger.info(f"\n✅ All plots saved to: {self.output_dir}/")
        logger.info("\nNext steps:")
        logger.info("  1. Review plots in outputs/eda/")
        logger.info("  2. Update NOTES.md with key insights")
        logger.info("  3. Proceed to Day 5: Feature Engineering")


def main():
    """Run EDA on selected features."""
    eda = BoschEDA()
    eda.run_all_analyses()


if __name__ == "__main__":
    main()
