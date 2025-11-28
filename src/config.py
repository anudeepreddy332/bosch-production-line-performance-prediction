"""
Configuration loader with validation.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from src.logger import setup_logger

logger = setup_logger(__name__)


class Config:
    """Load and validate project configuration."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"✗ Invalid YAML in config file: {e}")
            raise

        self._validate()

    def _validate(self):
        """Validate required configuration keys."""
        required_keys = ['project', 'paths', 'data_processing', 'features', 'model']
        missing = [key for key in required_keys if key not in self.config]

        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        logger.info("✓ Configuration validated")

    def get(self, key_path: str, default=None):
        """
        Get config value using dot notation (e.g., 'paths.raw_data').

        Args:
            key_path: Dot-separated path to config value
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default

        return value

    def __getitem__(self, key):
        """Allow dict-like access."""
        return self.config[key]
