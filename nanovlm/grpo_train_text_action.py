import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

from .model import NanoVLMActionPredictor
from .processors import get_tokenizer
from .device_utils import get_device, setup_device, print_device_stats
from .env import MiniGridRLEnv, Episode
from .grpo_algorithm import GRPOConfig
from .grpo_train_action import DirectActionGRPOTrainer, load_sft_checkpoint

logger = logging.getLogger(__name__)


class TextActionGRPOTrainer(DirectActionGRPOTrainer):
    """GRPO trainer for text+action prediction mode."""
    
    def __init__(
        self,
        model: NanoVLMActionPredictor,
        reference_model: NanoVLMActionPredictor,
        env: MiniGridRLEnv,
        grpo_config: GRPOConfig,
        device: torch.device,
        tokenizer,
        text_mode: str = "plan",
    ):
        """Initialize text+action GRPO trainer."""
        super().__init__(
            model=model,
            reference_model=reference_model,
            env=env,
            grpo_config=grpo_config,
            device=device,
            tokenizer=tokenizer,
        )
        
        self.text_mode = text_mode
        self.model.mode = "text_action"
        self.reference_model.mode = "text_action"
    
    def get_text_prompt(self) -> str:
        """Get prompt for text generation based on text_mode."""
        if self.text_mode == "plan":
            return "Briefly describe in 1-2 sentences what the agent should do next. Then predict the action."
        elif self.text_mode == "state":
            return "Describe what you see in the current state. Then predict the action."
        elif self.text_mode == "reasoning":
            return "Reason about the best strategy to reach the goal. Then predict the action."
        else:
            return "What action should the agent take? Describe your reasoning first."
    
    def _extract_episode_data(self, episode: Episode):
        """Extract data with text generation from base method."""
        obs, actions, returns_togo = super()._extract_episode_data(episode)
        
        # Generate text descriptions
        texts = [self._generate_state_text(o) for o in obs]
        
        return obs, actions, returns_togo
    
    def _generate_state_text(self, observation: np.ndarray) -> str:
        """Generate text description of state (simple deterministic version)."""
        descriptions = [
            "Agent needs to navigate towards the goal. Moving forward would be good.",
            "Agent is positioned away from the goal. Should turn and move forward.",
            "Goal is visible. Agent should move forward to reach it.",
            "Agent needs to find path. Should turn right and proceed.",
            "Getting closer to goal. Continue moving forward.",
        ]
        
        obs_hash = hash(observation.tobytes()) % len(descriptions)
        return descriptions[obs_hash]
    
    def _get_policy_logits(self, observations: List[np.ndarray]):
        """Get logits with text descriptions."""
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
        
        # Create prompts with text descriptions
        generated_texts = [self._generate_state_text(obs) for obs in observations]
        prompts = [
            f"{self.get_text_prompt()}\nAgent's thought: {text}"
            for text in generated_texts
        ]
        
        with torch.no_grad():
            current_output = self.model(images=images, prompts=prompts)
            current_logits = current_output["logits"]
            
            reference_output = self.reference_model(images=images, prompts=prompts)
            reference_logits = reference_output["logits"]
        
        # Extract action logits
        current_action_logits = self._extract_action_logits(current_logits)
        reference_action_logits = self._extract_action_logits(reference_logits)
        
        return current_action_logits, reference_action_logits
    
    def training_step(self, episodes: List[Episode]) -> Dict:
        """Training step with text descriptions."""
        metrics = super().training_step(episodes)
        metrics["text_mode"] = self.text_mode
        return metrics



