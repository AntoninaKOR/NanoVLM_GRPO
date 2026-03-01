"""
Configuration loader for NanoVLM.

Loads settings from configs.yaml and provides easy access to all configuration parameters.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfigNamespace:
    """Namespace object for accessing config values as attributes."""
    _data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return ConfigNamespace(value)
            return value
        raise AttributeError(f"Config has no attribute '{key}'")
    
    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to configs.yaml. If None, looks for it in nanovlm/ directory.
        """
        if config_path is None:
            # Try to find configs.yaml relative to this file
            base_dir = Path(__file__).parent
            config_path = base_dir / "configs.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = config_path
        self.config = self._load_yaml()
        logger.info(f"✓ Loaded configuration from {config_path}")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'model.name', 'training.batch_size')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> ConfigNamespace:
        """
        Get entire configuration section as namespace object.
        
        Args:
            section: Section name (e.g., 'model', 'training', 'dataset')
        
        Returns:
            ConfigNamespace with all values in section
        """
        data = self.config.get(section, {})
        return ConfigNamespace(data)
    
    def get_all(self) -> ConfigNamespace:
        """Get entire configuration as namespace object."""
        return ConfigNamespace(self.config)
    
    def print_config(self, section: Optional[str] = None):
        """Print configuration in a readable format."""
        if section:
            data = self.config.get(section, {})
            print(f"\n{'='*60}")
            print(f"Configuration: {section.upper()}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("FULL CONFIGURATION")
            print(f"{'='*60}")
            data = self.config
        
        self._print_dict(data, indent=0)
        print(f"{'='*60}\n")
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """Recursively print dictionary in readable format."""
        indent_str = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{indent_str}{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{indent_str}{key}: {value}")
            else:
                print(f"{indent_str}{key}: {value}")


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Load configuration (creates singleton if not exists)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def get_config() -> ConfigNamespace:
    """Get entire configuration as namespace."""
    loader = load_config()
    return loader.get_all()


def get_config_section(section: str) -> ConfigNamespace:
    """Get specific configuration section."""
    loader = load_config()
    return loader.get_section(section)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = load_config()
    config.print_config()
    
    # Access specific values
    print("Model name:", config.get("model.name"))
    print("Batch size:", config.get("training.batch_size"))
    print("Max image size:", config.get("image_processor.max_img_size"))
    
    # Get entire sections
    model_cfg = config.get_section("model")
    print(f"\nLoRA enabled: {model_cfg.lora.enabled}")
    print(f"LoRA rank: {model_cfg.lora.r}")
