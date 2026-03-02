import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .model import NanoVLMActionPredictor
from .processors import get_tokenizer
from .device_utils import get_device, setup_device, print_device_stats
from .env import MiniGridRLEnv, Episode, EpisodeCollector
from .grpo_algorithm import GRPOTrainer, GRPOConfig

logger = logging.getLogger(__name__)


class DirectActionGRPOTrainer:
    """GRPO trainer for direct action prediction mode."""
    
    def __init__(
        self,
        model: NanoVLMActionPredictor,
        reference_model: NanoVLMActionPredictor,
        env: MiniGridRLEnv,
        grpo_config: GRPOConfig,
        device: torch.device,
        tokenizer,
    ):
        """Initialize direct action GRPO trainer."""
        self.model = model
        self.reference_model = reference_model
        self.env = env
        self.grpo_config = grpo_config
        self.device = device
        self.tokenizer = tokenizer
        
        # Optimizer for policy
        # Filter to only trainable parameters (LoRA adapters)
        trainable_params = [param for param in self.model.model.parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Optimizer - Total params: {total_params:,} | Trainable: {trainable_param_count:,} ({100*trainable_param_count/total_params:.2f}%)")
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=grpo_config.learning_rate,
            weight_decay=0.01,
        )
        
        # Episode collector
        self.collector = EpisodeCollector(env)
        
        # Metrics tracking
        self.metrics_history = []
        
        # Freeze reference policy
        self.reference_model.model.eval()
        for param in self.reference_model.model.parameters():
            param.requires_grad = False
    
    def policy_fn(self, observation: np.ndarray) -> int:
        """Policy function using model's built-in predict_action."""
        return self.model.predict_action(observation)
    
    def collect_trajectories(
        self,
        num_episodes: int,
        base_seed: int = 0,
    ) -> Tuple[List[Episode], Dict]:
        """Collect trajectories with current policy."""
        self.model.model.eval()
        episodes = self.collector.collect_episodes(
            policy_fn=self.policy_fn,
            num_episodes=num_episodes,
            base_seed=base_seed,
        )
        
        stats = self.collector.get_stats(episodes)
        return episodes, stats
    
    def training_step(
        self,
        episodes: List[Episode],
    ) -> Dict:
        """Single training step with GRPO loss."""
        self.model.model.train()
        
        # Extract trajectory data
        all_observations = []
        all_actions = []
        all_returns = []
        episode_returns = []
        
        for episode in episodes:
            obs, actions, returns = self._extract_episode_data(episode)
            all_observations.extend(obs)
            all_actions.extend(actions)
            all_returns.extend(returns)
            episode_returns.append(episode.return_)
        
        advantages_list = self._compute_advantages(all_returns)
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        
        current_logits, reference_logits = self._get_policy_logits(all_observations)
        
        grpo_trainer = GRPOTrainer(self.model.model, self.grpo_config, self.tokenizer, self.device)
        loss, metrics = grpo_trainer.compute_grpo_loss(
            current_logits=current_logits,
            reference_logits=reference_logits,
            actions=actions_tensor,
            advantages=advantages,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.grpo_config.max_grad_norm)
        self.optimizer.step()
        
        metrics["episode_return"] = float(np.mean(episode_returns))
        metrics["episode_std_return"] = float(np.std(episode_returns))
        
        return metrics
    
    def _extract_episode_data(self, episode: Episode) -> Tuple[List, List, List]:
        """Extract observations, actions, and returns-to-go from episode."""
        obs = [t.observation for t in episode.transitions]
        actions = [t.action for t in episode.transitions]
        rewards = [t.reward for t in episode.transitions]
        
        # Compute returns-to-go
        returns_togo = []
        cumulative = 0.0
        for r in reversed(rewards):
            cumulative = r + cumulative
            returns_togo.insert(0, cumulative)
        
        return obs, actions, returns_togo
    
    def _compute_advantages(self, returns: List[float]) -> List[float]:
        """Compute normalized advantages."""
        if not returns:
            return []
        
        if self.grpo_config.reward_baseline == "mean":
            baseline = sum(returns) / len(returns)
        elif self.grpo_config.reward_baseline == "min":
            baseline = min(returns)
        else:
            baseline = 0.0
        
        advantages = [r - baseline for r in returns]
        
        # Normalize
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]
        
        return advantages
    
    def _get_policy_logits(self, observations: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action logits from current and reference policies."""
        from PIL import Image
        
        # Convert observations to PIL Images
        images = []
        for obs in observations:
            if isinstance(obs, np.ndarray):
                if obs.dtype != np.uint8:
                    obs = (obs * 255).astype(np.uint8)
                images.append(Image.fromarray(obs))
            else:
                images.append(obs)
        
        prompt = "What action should the agent take to reach the goal?"
        prompts = [prompt] * len(images)
        
        # Current model needs gradients for loss.backward()
        current_output = self.model(images=images, prompts=prompts)
        current_logits = current_output["logits"]
        
        with torch.no_grad():
            reference_output = self.reference_model(images=images, prompts=prompts)
            reference_logits = reference_output["logits"]
        
        # Extract action logits
        current_action_logits = self._extract_action_logits(current_logits)
        reference_action_logits = self._extract_action_logits(reference_logits)
        
        return current_action_logits, reference_action_logits
    
    def _extract_action_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract logits for action tokens from last position."""
        batch_size = logits.shape[0]
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        
        action_logits = []
        for action_name in sorted(self.model.action_token_ids.keys()):
            token_id = self.model.action_token_ids[action_name]
            action_logits.append(last_logits[:, token_id])
        
        return torch.stack(action_logits, dim=1)  # [batch, num_actions]
    
    def save_checkpoint(self, checkpoint_dir: Path, step: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / f"model_step_{step}.pt"
        if hasattr(self.model.model, 'save_pretrained'):
            self.model.model.save_pretrained(str(model_path))
        else:
            torch.save(self.model.model.state_dict(), str(model_path))
        
        logger.info(f"Saved checkpoint to {model_path}")
