from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper
from PIL import Image


@dataclass
class Transition:
    """Single environment transition."""
    observation: np.ndarray  # Image frame
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict


@dataclass
class Episode:
    """Complete episode trajectory."""
    transitions: List[Transition]
    return_: float = field(default=0.0)
    success: bool = field(default=False)
    length: int = field(default=0)

    def __post_init__(self):
        self.return_ = sum(t.reward for t in self.transitions)
        self.length = len(self.transitions)
        self.success = self.return_ > 0.0


class MiniGridRLEnv:
    """Wrapper for MiniGrid environment for RL training."""
    
    def __init__(
        self,
        env_id: str = "MiniGrid-Empty-8x8-v0",
        render_mode: Optional[str] = None,
        seed: int = 0,
        max_steps: int = 100,
    ):
        """Initialize environment.
        
        Args:
            env_id: Gymnasium environment ID
            render_mode: Render mode (None, 'rgb_array', 'human')
            seed: Random seed
            max_steps: Max steps per episode
        """
        self.env_id = env_id
        self.max_steps = max_steps
        self.seed = seed
        
        # Create base environment
        self.env = gym.make(env_id, render_mode=render_mode)
        
        # Wrap with RGB partial observation
        self.env = RGBImgPartialObsWrapper(self.env, tile_size=8)
        
        # Action space
        self.num_actions = 7
        self.action_names = {
            0: "turn_left",
            1: "turn_right",
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done",
        }
        
        self.obs = None
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return observation."""
        if seed is not None:
            self.seed = seed
        
        obs, _ = self.env.reset(seed=self.seed)
        self.obs = obs["image"]  # Extract RGB image from partial obs
        self.step_count = 0
        return self.obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment.
        
        Args:
            action: Action ID (0-6)
            
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if action >= self.num_actions:
            raise ValueError(f"Invalid action {action}, must be < {self.num_actions}")
        
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        next_obs = next_obs["image"]
        
        self.step_count += 1
        self.obs = next_obs
        
        # Check if episode should terminate due to max steps
        if self.step_count >= self.max_steps:
            truncated = True
        
        return next_obs, float(reward), terminated, truncated, info
    
    def render(self) -> np.ndarray:
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def run_episode(
        self,
        policy_fn: callable,
        seed: Optional[int] = None,
        return_frames: bool = False,
    ) -> Episode:
        """Run a single episode with given policy.
        
        Args:
            policy_fn: Function that takes observation and returns action ID
            seed: Episode seed
            return_frames: Whether to return rendered frames
            
        Returns:
            Episode object with trajectory
        """
        obs = self.reset(seed=seed)
        transitions = []
        
        while True:
            # Get action from policy
            action = policy_fn(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.step(action)
            
            # Create transition
            transition = Transition(
                observation=obs.copy(),
                action=action,
                reward=reward,
                next_observation=next_obs.copy(),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            transitions.append(transition)
            
            # Check termination
            if terminated or truncated:
                break
            
            obs = next_obs
        
        return Episode(transitions=transitions)
    
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self.obs.copy() if self.obs is not None else None


class EpisodeCollector:
    """Collect multiple episodes from environment."""
    
    def __init__(self, env: MiniGridRLEnv):
        self.env = env
    
    def collect_episodes(
        self,
        policy_fn: callable,
        num_episodes: int,
        base_seed: int = 0,
    ) -> List[Episode]:
        """Collect multiple episodes.
        
        Args:
            policy_fn: Policy function
            num_episodes: Number of episodes to collect
            base_seed: Base seed (incremented for each episode)
            
        Returns:
            List of episodes
        """
        episodes = []
        for i in range(num_episodes):
            episode = self.env.run_episode(
                policy_fn=policy_fn,
                seed=base_seed + i,
            )
            episodes.append(episode)
        
        return episodes
    
    def get_stats(self, episodes: List[Episode]) -> Dict[str, float]:
        """Compute statistics from episodes.
        
        Returns:
            Dict with keys: success_rate, avg_return, avg_length, min_length, max_length
        """
        if not episodes:
            return {
                "success_rate": 0.0,
                "avg_return": 0.0,
                "avg_length": 0.0,
                "min_length": 0,
                "max_length": 0,
            }
        
        successes = sum(1 for ep in episodes if ep.success)
        returns = [ep.return_ for ep in episodes]
        lengths = [ep.length for ep in episodes]
        
        return {
            "success_rate": successes / len(episodes),
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "avg_length": float(np.mean(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "num_episodes": len(episodes),
        }
