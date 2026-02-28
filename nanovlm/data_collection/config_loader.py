"""Config loader for YAML configuration files."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Optional[str] = None, preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. Defaults to data_collection module's config.yaml
        preset: Name of preset config to use (small, medium, large, dev)
    
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if preset:
        if "presets" in config and preset in config["presets"]:
            preset_config = config["presets"][preset]
            # Deep merge preset into main config
            config = _deep_merge(config, preset_config)
        else:
            raise ValueError(f"Preset '{preset}' not found in config")
    
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config dict to arguments for collect_data function."""
    return {
        "env_id": config["env"]["id"],
        "seed": config["env"]["seed"],
        "num_episodes": config["collection"]["num_episodes"],
        "max_steps": config["collection"]["max_steps_per_episode"],
        "output_dir": Path(config["output"]["dir"]),
        "mode": config["collection"]["mode"],
    }


if __name__ == "__main__":
    import json
    
    # Test loading main config
    config = load_config()
    print("Main config:")
    print(json.dumps(config, indent=2, default=str))
    
    # Test loading with preset
    print("\n\nSmall preset:")
    config_small = load_config(preset="small")
    print(json.dumps(config_small, indent=2, default=str))
