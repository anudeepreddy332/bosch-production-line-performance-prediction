"""
Download Bosch dataset from Kaggle with validation.
"""
import os
import subprocess
from pathlib import Path
from typing import List
from src.config import Config
from src.logger import setup_logger

logger = setup_logger(__name__)


class KaggleDownloader:
    """Handle Kaggle dataset download with error handling."""

    COMPETITION = "bosch-production-line-performance"
    REQUIRED_FILES = [
        "train_numeric.csv",
        "train_categorical.csv",
        "train_date.csv",
        "test_numeric.csv",
        "test_categorical.csv",
        "test_date.csv",
        "sample_submission.csv"
    ]

    def __init__(self, config: Config):
        """
        Initialize downloader.

        Args:
            config: Project configuration
        """
        self.config = config
        self.raw_data_dir = Path(config.get('paths.raw_data'))
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def check_kaggle_auth(self) -> bool:
        """
        Check if Kaggle API is authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

        if not kaggle_json.exists():
            logger.error("✗ Kaggle API credentials not found")
            logger.info("→ Download kaggle.json from https://www.kaggle.com/settings/account")
            logger.info(f"→ Place it in: {kaggle_json.parent}")
            return False

        # Check permissions (must be 600)
        if kaggle_json.stat().st_mode & 0o777 != 0o600:
            logger.warning("⚠ Setting kaggle.json permissions to 600")
            kaggle_json.chmod(0o600)

        logger.info("✓ Kaggle API authenticated")
        return True

    def download(self, force: bool = False) -> bool:
        """
        Download dataset from Kaggle.

        Args:
            force: Force re-download even if files exist

        Returns:
            True if successful, False otherwise
        """
        # Check authentication
        if not self.check_kaggle_auth():
            return False

        # Check if already downloaded
        if not force and self._verify_files():
            logger.info("✓ All files already downloaded")
            return True

        logger.info(f"⬇ Downloading {self.COMPETITION} dataset...")

        try:
            # Download using Kaggle CLI
            cmd = [
                "kaggle", "competitions", "download",
                "-c", self.COMPETITION,
                "-p", str(self.raw_data_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"✗ Download failed: {result.stderr}")
                return False

            logger.info("✓ Download complete")

            # Unzip files
            zip_file = self.raw_data_dir / f"{self.COMPETITION}.zip"
            if zip_file.exists():
                logger.info("📦 Extracting files...")
                subprocess.run(
                    ["unzip", "-o", str(zip_file), "-d", str(self.raw_data_dir)],
                    capture_output=True
                )
                zip_file.unlink()  # Remove zip after extraction
                logger.info("✓ Extraction complete")

            # Verify all files present
            if self._verify_files():
                logger.info("✓ All files verified")
                return True
            else:
                logger.error("✗ Some files missing after download")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Download timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"✗ Download failed: {str(e)}")
            return False

    def _verify_files(self) -> bool:
        """
        Verify all required files exist.

        Returns:
            True if all files present, False otherwise
        """
        missing_files = []
        for filename in self.REQUIRED_FILES:
            if not (self.raw_data_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            logger.warning(f"⚠ Missing files: {missing_files}")
            return False

        return True

    def get_file_sizes(self) -> dict:
        """
        Get sizes of all downloaded files.

        Returns:
            Dictionary of {filename: size_mb}
        """
        sizes = {}
        for filename in self.REQUIRED_FILES:
            file_path = self.raw_data_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                sizes[filename] = f"{size_mb:.2f} MB"

        return sizes


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Bosch dataset from Kaggle")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Download
    downloader = KaggleDownloader(config)
    success = downloader.download(force=args.force)

    if success:
        logger.info("=" * 50)
        logger.info("FILE SIZES:")
        for filename, size in downloader.get_file_sizes().items():
            logger.info(f"  {filename}: {size}")
        logger.info("=" * 50)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
