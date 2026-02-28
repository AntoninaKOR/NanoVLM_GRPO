"""Data collection configuration for MiniGrid."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataCollectionConfig:
    """Configuration for collecting data from MiniGrid environments."""
    
    # Environment
    env_id: str = "MiniGrid-Empty-8x8-v0"
    seed: int = 0
    
    # Data collection
    num_episodes: int = 10
    max_steps_per_episode: Optional[int] = None
    
    # Output
    output_dir: str = "data/minigrid_sft"
    save_frames: bool = True
    
    # Training mode
    mode: str = "action"  # "action" or "text_action"
    
    # Multiprocessing (future)
    num_workers: int = 1
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "output_dir": self.output_dir,
            "save_frames": self.save_frames,
            "mode": self.mode,
            "num_workers": self.num_workers,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "DataCollectionConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Preset configurations
CONFIGS = {
    "small": DataCollectionConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        num_episodes=10,
        max_steps_per_episode=50,
        output_dir="data/minigrid_small",
    ),
    "medium": DataCollectionConfig(
        env_id="MiniGrid-Empty-16x16-v0",
        num_episodes=50,
        max_steps_per_episode=200,
        output_dir="data/minigrid_medium",
    ),
    "large": DataCollectionConfig(
        env_id="MiniGrid-Empty-16x16-v0",
        num_episodes=200,
        max_steps_per_episode=None,
        output_dir="data/minigrid_large",
    ),
    "dev": DataCollectionConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        num_episodes=1,
        max_steps_per_episode=10,
        output_dir="data/minigrid_dev",
    ),
}


if __name__ == "__main__":
    import json
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(json.dumps(config.to_dict(), indent=2))
