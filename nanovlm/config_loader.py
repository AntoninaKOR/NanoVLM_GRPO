import yaml
from pathlib import Path


def load_config(config_path=None):
    """Load YAML config and return as nested dict."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
