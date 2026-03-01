from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    
    # Sampling
    num_candidate_actions: int = 4  # Number of candidate actions to sample
    temperature: float = 0.7  # Sampling temperature
    
    # Loss weights
    kl_coeff: float = 0.1  # KL divergence coefficient (beta in GRPO paper)
    entropy_coeff: float = 0.01  # Entropy regularization
    
    # Optimization
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    
    # Training
    num_episodes_per_update: int = 4
    num_updates: int = 100
    max_grad_norm: float = 1.0
    
    # Reward normalization
    reward_baseline: str = "mean"  # "mean", "min", or "none"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50


class GRPOTrainer:
    """GRPO trainer for language models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: GRPOConfig,
        tokenizer: Any,
        device: torch.device = None,
    ):
        """Initialize GRPO trainer.
        
        Args:
            model: Language model to train (NanoVLMActionPredictor)
            config: GRPO configuration
            tokenizer: Tokenizer for encoding text
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        
        # Optimizer (only for model parameters, not for reference policy)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        
        self.global_step = 0
        self.episode_count = 0
    
    def forward_pass(
        self,
        images: List,
        prompts: List[str],
    ) -> torch.Tensor:
        """Get action logits from model.
        
        Args:
            images: List of PIL images
            prompts: List of text prompts
            
        Returns:
            Logits tensor of shape [batch_size, vocab_size]
        """
        with torch.no_grad():
            output = self.model(images=images, prompts=prompts)
            logits = output["logits"]  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def extract_action_logits(
        self,
        logits: torch.Tensor,
        action_token_ids: Dict[str, int],
        mask_prompt_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract logits for action tokens only.
        
        Args:
            logits: Full model logits [batch_size, seq_len, vocab_size]
            action_token_ids: Dict mapping action name to token ID
            mask_prompt_length: Length of prompt tokens to mask
            
        Returns:
            Action logits [batch_size, num_actions]
        """
        batch_size = logits.shape[0]
        action_logits = []
        
        # Get the last position logits (where action is predicted)
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Extract logits for each action token
        for action_name in sorted(action_token_ids.keys()):
            token_id = action_token_ids[action_name]
            action_logits.append(last_logits[:, token_id])
        
        # Stack to [batch_size, num_actions]
        action_logits = torch.stack(action_logits, dim=1)
        
        return action_logits
    
    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policies.
        
        Args:
            current_logits: Action logits from current policy [batch_size, num_actions]
            reference_logits: Action logits from reference policy [batch_size, num_actions]
            
        Returns:
            KL divergence [batch_size]
        """
        current_probs = torch.softmax(current_logits, dim=-1)
        reference_probs = torch.softmax(reference_logits, dim=-1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        current_log_probs = torch.log(current_probs + eps)
        
        kl = torch.sum(reference_probs * (torch.log(reference_probs + eps) - current_log_probs), dim=-1)
        return kl
    
    def compute_likelihood(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log likelihood of actions under logits.
        
        Args:
            logits: Action logits [batch_size, num_actions]
            actions: Action indices [batch_size]
            
        Returns:
            Log likelihood [batch_size]
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return selected_log_probs
    
    def compute_advantages(
        self,
        rewards: List[float],
        baseline: str = "mean",
    ) -> List[float]:
        """Compute advantages with optional baseline subtraction.
        
        Args:
            rewards: List of trajectory returns
            baseline: Baseline type ("mean", "min", "none")
            
        Returns:
            List of advantages
        """
        if not rewards:
            return []
        
        if baseline == "mean":
            baseline_value = sum(rewards) / len(rewards)
        elif baseline == "min":
            baseline_value = min(rewards)
        else:  # "none"
            baseline_value = 0.0
        
        advantages = [r - baseline_value for r in rewards]
        
        # Normalize advantages
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]
        
        return advantages
    
    def compute_grpo_loss(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss.
        
        GRPO loss = -E[max(0, advantage) * log(π_θ(a|s)) - β * D_KL(π_ref || π_θ)]
        
        Args:
            current_logits: Action logits from current policy [batch_size, num_actions]
            reference_logits: Action logits from reference policy [batch_size, num_actions]
            actions: Selected actions [batch_size]
            advantages: Computed advantages [batch_size]
            
        Returns:
            (loss, metrics_dict)
        """
        # Compute log likelihoods
        log_probs = self.compute_likelihood(current_logits, actions)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(current_logits, reference_logits)
        
        # GRPO objective: maximize advantage * log_prob - β * KL
        # We minimize the negative
        policy_loss = -torch.mean(torch.clamp(advantages, min=0.0) * log_probs)
        kl_loss = self.config.kl_coeff * torch.mean(kl_div)
        
        # Add entropy regularization
        current_probs = torch.softmax(current_logits, dim=-1)
        entropy = -torch.sum(current_probs * torch.log_softmax(current_logits, dim=-1), dim=-1)
        entropy_loss = -self.config.entropy_coeff * torch.mean(entropy)
        
        total_loss = policy_loss + kl_loss + entropy_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_advantage": torch.mean(advantages).item(),
            "mean_kl": torch.mean(kl_div).item(),
            "mean_entropy": torch.mean(entropy).item(),
        }
        
        return total_loss, metrics
    
    def training_step(
        self,
        trajectories: List[Dict],
        rewards: List[float],
        actions: List[int],
    ) -> Dict[str, float]:
        """Single training step with GRPO loss.
        
        Args:
            trajectories: List of trajectories (each is list of transitions)
            rewards: List of trajectory returns
            actions: List of selected actions per state in trajectory
            
        Returns:
            Dictionary of loss metrics
        """
        self.optimizer.zero_grad()
        
        # Compute advantages
        advantages_list = self.compute_advantages(
            rewards,
            baseline=self.config.reward_baseline,
        )
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)
        
        # Convert actions to tensor
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        # Compute loss (assuming you have current_logits and reference_logits)
        # This would be implemented in the specific training script
        
        logger.debug(f"Step {self.global_step}: advantages={advantages_list}, actions={actions}")
        
        self.global_step += 1
        
        return {
            "mean_advantage": float(torch.mean(advantages).item()),
            "num_trajectories": len(trajectories),
        }


class ReferencePolicy:
    """Wrapper for reference policy (usually SFT trained model)."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """Initialize reference policy.
        
        Args:
            model: Model to use as reference (will be frozen)
            device: Device to run on
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze reference policy
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_action_logits(self, images: List, prompts: List[str]) -> torch.Tensor:
        """Get action logits from reference policy."""
        with torch.no_grad():
            output = self.model(images=images, prompts=prompts)
            return output["logits"]
