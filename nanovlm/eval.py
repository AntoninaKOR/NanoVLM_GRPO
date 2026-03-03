import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from .model import NanoVLMActionPredictor
from .processors import get_tokenizer
from .env import MiniGridRLEnv, EpisodeCollector

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models on MiniGrid."""
    
    def __init__(
        self,
        model: NanoVLMActionPredictor,
        env: MiniGridRLEnv,
        device: torch.device,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            env: MiniGrid environment
            device: Device to run on
        """
        self.model = model
        self.env = env
        self.device = device
        self.collector = EpisodeCollector(env)
    
    def policy_fn(self, observation: np.ndarray) -> int:
        """Policy function for evaluation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Action ID
        """
        patches = self.model.preprocess_image(observation)
        action = self.model.predict_action(patches)
        return action
    
    def evaluate(
        self,
        num_episodes: int,
        base_seed: int = 0,
    ) -> Dict:
        """Evaluate model on multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate on
            base_seed: Base random seed
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.model.eval()
        
        episodes = self.collector.collect_episodes(
            policy_fn=self.policy_fn,
            num_episodes=num_episodes,
            base_seed=base_seed,
        )
        
        stats = self.collector.get_stats(episodes)
        
        # Additional metrics
        returns = [ep.return_ for ep in episodes]
        lengths = [ep.length for ep in episodes]
        
        stats["min_return"] = float(np.min(returns))
        stats["max_return"] = float(np.max(returns))
        stats["p25_return"] = float(np.percentile(returns, 25))
        stats["p50_return"] = float(np.percentile(returns, 50))
        stats["p75_return"] = float(np.percentile(returns, 75))
        
        stats["min_length"] = int(np.min(lengths))
        stats["max_length"] = int(np.max(lengths))
        
        return stats, episodes
    
    def save_episode_gifs(
        self,
        episodes: List,
        output_dir: Path,
        num_gifs: int = 3,
    ):
        """Save episode visualizations as GIFs.
        
        Args:
            episodes: List of episodes to visualize
            output_dir: Directory to save GIFs
            num_gifs: Number of episodes to render as GIFs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select episodes to render (best, worst, median)
        returns = [ep.return_ for ep in episodes]
        if len(episodes) == 0:
            return
        
        indices_to_render = []
        
        # Best episode
        best_idx = np.argmax(returns)
        indices_to_render.append(best_idx)
        
        # Worst episode
        worst_idx = np.argmin(returns)
        if worst_idx != best_idx:
            indices_to_render.append(worst_idx)
        
        # Median episode
        median_idx = np.argsort(returns)[len(returns) // 2]
        if median_idx not in indices_to_render:
            indices_to_render.append(median_idx)
        
        indices_to_render = indices_to_render[:num_gifs]
        
        for idx in indices_to_render:
            episode = episodes[idx]
            return_val = episode.return_
            length = episode.length
            
            # Collect frames during episode rollout
            frames = []
            obs = self.env.reset(seed=idx)
            
            for transition in episode.transitions:
                frame = self.env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            
            # Save as GIF if we have frames
            if frames:
                gif_path = output_dir / f"episode_{idx:03d}_return_{return_val:.2f}_len_{length}.gif"
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,  # 200ms per frame
                    loop=0,
                )
                logger.info(f"Saved episode GIF: {gif_path}")


class ComparisonPlotter:
    """Plot comparisons between models."""
    
    @staticmethod
    def plot_metrics_comparison(
        results: Dict[str, Dict],
        output_path: Optional[Path] = None,
        figsize: Tuple = (15, 10),
    ):
        """Plot comparison of evaluation metrics across models.
        
        Args:
            results: Dict mapping model names to evaluation results
            output_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Model Comparison Metrics", fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        metrics_to_plot = [
            ("success_rate", "Success Rate"),
            ("avg_return", "Average Return"),
            ("std_return", "Return Std Dev"),
            ("avg_length", "Average Episode Length"),
            ("p50_return", "Median Return"),
            ("min_return", "Min Return"),
        ]
        
        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            values = [results[model].get(metric_key, 0) for model in models]
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(metric_name)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_training_curves(
        metrics_history: List[Dict],
        output_path: Optional[Path] = None,
        figsize: Tuple = (15, 10),
    ):
        """Plot training curves from GRPO training.
        
        Args:
            metrics_history: List of metrics dicts from training
            output_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("GRPO Training Curves", fontsize=16, fontweight='bold')
        
        updates = list(range(len(metrics_history)))
        
        # Extract different metric types
        collect_metrics = {key: [] for key in metrics_history[0]["collect_stats"].keys()}
        train_metrics = {key: [] for key in metrics_history[0]["train_metrics"].keys()}
        
        for step_metrics in metrics_history:
            for key in collect_metrics:
                collect_metrics[key].append(step_metrics["collect_stats"].get(key, 0))
            for key in train_metrics:
                train_metrics[key].append(step_metrics["train_metrics"].get(key, 0))
        
        # Plot collection metrics
        ax = axes[0, 0]
        ax.plot(updates, collect_metrics["success_rate"], marker='o', linewidth=2, markersize=4)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Update")
        ax.set_title("Success Rate Over Updates")
        ax.grid(alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(updates, collect_metrics["avg_return"], marker='o', label="Mean", linewidth=2, markersize=4)
        ax.fill_between(
            updates,
            np.array(collect_metrics["avg_return"]) - np.array(collect_metrics["std_return"]),
            np.array(collect_metrics["avg_return"]) + np.array(collect_metrics["std_return"]),
            alpha=0.3,
            label="±1 Std"
        )
        ax.set_ylabel("Return")
        ax.set_xlabel("Update")
        ax.set_title("Episode Returns Over Updates")
        ax.grid(alpha=0.3)
        ax.legend()
        
        ax = axes[0, 2]
        ax.plot(updates, collect_metrics["avg_length"], marker='o', linewidth=2, markersize=4)
        ax.set_ylabel("Length")
        ax.set_xlabel("Update")
        ax.set_title("Episode Length Over Updates")
        ax.grid(alpha=0.3)
        
        # Plot training losses
        ax = axes[1, 0]
        ax.plot(updates, train_metrics["policy_loss"], marker='s', label="Policy Loss", linewidth=2)
        ax.plot(updates, train_metrics.get("kl_loss", [0]*len(updates)), marker='^', label="KL Loss", linewidth=2)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Update")
        ax.set_title("Training Losses")
        ax.grid(alpha=0.3)
        ax.legend()
        
        ax = axes[1, 1]
        if "mean_advantage" in train_metrics:
            ax.plot(updates, train_metrics["mean_advantage"], marker='o', linewidth=2, markersize=4)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel("Mean Advantage")
            ax.set_xlabel("Update")
            ax.set_title("Mean Advantage Over Updates")
            ax.grid(alpha=0.3)
        
        ax = axes[1, 2]
        if "mean_kl" in train_metrics:
            ax.plot(updates, train_metrics["mean_kl"], marker='o', linewidth=2, markersize=4, color='green')
            ax.set_ylabel("Mean KL Divergence")
            ax.set_xlabel("Update")
            ax.set_title("KL Divergence Over Updates")
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training curves to {output_path}")
        
        plt.close()



