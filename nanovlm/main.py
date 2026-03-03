import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import gymnasium as gym
import imageio
import minigrid  # noqa: F401 — registers MiniGrid envs
from minigrid.wrappers import RGBImgPartialObsWrapper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .model import NanoVLMActionPredictor, ACTIONS
from .data_utils import get_dataset_and_collator
from .data_collection.dijkstra import Dijkstra
from .data_collection.env_utils import action_to_next
from .processors import get_tokenizer
from .device_utils import get_device, setup_device
from .config_loader import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def train_epoch(
    model: NanoVLMActionPredictor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch."""
    model.model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [TRAIN]")
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images = batch.get("images", None)
        
        # Forward pass 
        outputs = model.forward_with_vision(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss


def validate(
    model: NanoVLMActionPredictor,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model on validation set."""
    model.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="[VALIDATE]")
        for batch in progress_bar:
            if batch is None:
                continue
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch.get("images", None)
            
            outputs = model.forward_with_vision(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )
            
            loss = outputs["loss"]
            total_loss += loss.item()
            
            # Accuracy with causal-LM shift: logits[i] predicts labels[i+1]
            logits = outputs["logits"]
            shift_preds = torch.argmax(logits[:, :-1, :], dim=-1)  # [B, T-1]
            shift_labels = labels[:, 1:]                            # [B, T-1]
            
            mask = shift_labels != -100
            if mask.any():
                correct += (shift_preds[mask] == shift_labels[mask]).sum().item()
                total += mask.sum().item()
            
            progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / max(1, len(val_loader))
    accuracy = correct / max(1, total)
    
    return avg_loss, accuracy


def save_evaluation_gif(
    frames: List[np.ndarray],
    output_path: Path,
    duration: float = 0.5,
) -> None:
    """Save a list of frames as an animated GIF."""
    if frames:
        frames_uint8 = [np.uint8(f) if f.dtype != np.uint8 else f for f in frames]
        imageio.mimwrite(output_path, frames_uint8, duration=duration, loop=0)
        logger.info(f"GIF saved to {output_path}")


def create_grid_gif(
    episode_frames_list: List[List[np.ndarray]],
    output_path: Path,
    duration: float = 0.5,
    grid_size: Tuple[int, int] = (2, 2),
) -> None:
    """Create a GIF with multiple episodes displayed in a grid layout (e.g., 2x2)."""
    if not episode_frames_list or not episode_frames_list[0]:
        return
    
    # Pad all episodes to the same length (length of longest episode)
    max_length = max(len(frames) for frames in episode_frames_list)
    padded_episodes = []
    for frames in episode_frames_list:
        if len(frames) < max_length:
            # Repeat last frame to pad
            padding = [frames[-1]] * (max_length - len(frames))
            padded_episodes.append(frames + padding)
        else:
            padded_episodes.append(frames)
    
    # Create grid frames
    grid_frames = []
    for step in range(max_length):
        # Get frame from each episode at this step
        grid_row_frames = []
        for row in range(grid_size[0]):
            grid_col_frames = []
            for col in range(grid_size[1]):
                ep_idx = row * grid_size[1] + col
                if ep_idx < len(padded_episodes):
                    frame = padded_episodes[ep_idx][step]
                    grid_col_frames.append(frame)
            
            if grid_col_frames:
                # Stack frames horizontally
                row_frame = np.hstack(grid_col_frames)
                grid_row_frames.append(row_frame)
        
        if grid_row_frames:
            # Stack rows vertically
            combined_frame = np.vstack(grid_row_frames)
            grid_frames.append(np.uint8(combined_frame))
    
    # Save as GIF
    if grid_frames:
        imageio.mimwrite(output_path, grid_frames, duration=duration, loop=0)
        logger.info(f"Grid GIF saved to {output_path}")


def evaluate_in_env(
    model: NanoVLMActionPredictor,
    env_name: str,
    num_episodes: int = 5,
    max_steps: int = 20,
    seed: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run the model in MiniGrid and compute success rate / average return. Optionally save GIFs."""
    model.model.eval()
    base_env = gym.make(env_name, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(base_env, tile_size=8)
    planner = Dijkstra(base_env)

    successes = 0
    total_returns: List[float] = []
    episode_lengths: List[int] = []
    optimal_matches = 0
    total_steps = 0
    episode_gifs: List[Tuple[int, List[np.ndarray]]] = []  # (ep_idx, frames)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        frames = []

        # Capture initial frame
        if output_dir is not None:
            frames.append(base_env.render())

        for _ in range(max_steps):
            path = planner.shortest_path()
            optimal_action = None
            if path and len(path) > 1:
                optimal_action = action_to_next(base_env, path[1])

            frame = obs["image"]  # Partial observation (POMDP)
            patches = model.preprocess_image(frame)
            action_id = model.predict_action(patches)

            if optimal_action is not None:
                total_steps += 1
                if action_id == optimal_action:
                    optimal_matches += 1

            obs, reward, terminated, truncated, _ = env.step(action_id)
            ep_return += reward
            steps += 1

            # Capture frame after action
            if output_dir is not None:
                frames.append(base_env.render())

            if terminated or truncated:
                break

        if output_dir is not None:
            episode_gifs.append((ep, frames))

        if ep_return > 0:
            successes += 1
        total_returns.append(ep_return)
        episode_lengths.append(steps)

    env.close()

    # Save combined GIF for last 4 episodes in 2x2 grid
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Get last 4 episodes
        last_episodes = episode_gifs[-4:] if len(episode_gifs) > 0 else []
        if last_episodes:
            # Extract frames only (remove episode indices)
            frames_list = [frames for _, frames in last_episodes]
            gif_path = output_dir / "last_4_episodes_grid.gif"
            create_grid_gif(frames_list, gif_path, duration=0.5, grid_size=(2, 2))

    return {
        "success_rate": successes / max(1, num_episodes),
        "mean_return": float(np.mean(total_returns)),
        "std_return": float(np.std(total_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "optimal_action_rate": optimal_matches / max(1, total_steps),
    }


class MetricsSaver:
    """Real-time metrics tracker and saver."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_log = []
        self.metrics_file = output_dir / "metrics_realtime.json"
    
    def append(self, metric_dict: Dict) -> None:
        """Add new metrics and save immediately."""
        self.metrics_log.append(metric_dict)
        self._save()
    
    def _save(self) -> None:
        """Save metrics to JSON file."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def get_all(self) -> List[Dict]:
        """Get all metrics."""
        return self.metrics_log


def plot_learning_curves(
    metrics_log: List[Dict],
    output_dir: Path,
    filename: str = "learning_curves.png",
) -> None:
    """Save learning-curve plots from the collected per-epoch metrics."""
    epochs = [m["epoch"] for m in metrics_log]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Learning Curves", fontsize=14)

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, [m["train_loss"] for m in metrics_log], "o-", label="train")
    ax.plot(epochs, [m["val_loss"] for m in metrics_log], "s-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Val accuracy ---
    ax = axes[0, 1]
    ax.plot(epochs, [m["val_accuracy"] for m in metrics_log], "o-", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Token Accuracy")
    ax.grid(True, alpha=0.3)

    # --- Success rate ---
    has_env = "success_rate" in metrics_log[0]
    ax = axes[1, 0]
    if has_env:
        ax.plot(epochs, [m["success_rate"] for m in metrics_log], "o-", color="tab:orange")
        ax.set_ylabel("Success Rate")
        ax.set_title("MiniGrid Success Rate")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # --- Mean return ---
    ax = axes[1, 1]
    if has_env:
        returns = [m["mean_return"] for m in metrics_log]
        stds = [m["std_return"] for m in metrics_log]
        ax.plot(epochs, returns, "o-", color="tab:red")
        ax.fill_between(
            epochs,
            [r - s for r, s in zip(returns, stds)],
            [r + s for r, s in zip(returns, stds)],
            alpha=0.2, color="tab:red",
        )
        ax.set_ylabel("Return")
        ax.set_title("MiniGrid Episode Return (mean ± std)")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # --- Mean episode length ---
    ax = axes[2, 0]
    if has_env:
        ax.plot(epochs, [m["mean_episode_length"] for m in metrics_log], "o-", color="tab:purple")
        ax.set_ylabel("Steps")
        ax.set_title("Mean Episode Length")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # --- Optimal action rate ---
    ax = axes[2, 1]
    if has_env:
        ax.plot(epochs, [m["optimal_action_rate"] for m in metrics_log], "o-", color="tab:cyan")
        ax.set_ylabel("Rate")
        ax.set_title("Optimal Action Rate")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    logger.info(f"Learning curves saved to {output_file}")


def plot_grpo_curves(
    metrics_log: List[Dict],
    output_dir: Path,
    filename: str = "grpo_curves.png",
) -> None:
    """Save GRPO training curves."""
    updates = [m["update"] for m in metrics_log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GRPO Training Curves", fontsize=14)

    # --- Total / Policy / KL loss ---
    ax = axes[0, 0]
    if "total_loss" in metrics_log[0]:
        ax.plot(updates, [m["total_loss"] for m in metrics_log], "o-", label="total")
    if "policy_loss" in metrics_log[0]:
        ax.plot(updates, [m["policy_loss"] for m in metrics_log], "s-", label="policy")
    if "kl_loss" in metrics_log[0]:
        ax.plot(updates, [m["kl_loss"] for m in metrics_log], "^-", label="kl")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Success rate ---
    ax = axes[0, 1]
    if "success_rate" in metrics_log[0]:
        ax.plot(updates, [m["success_rate"] for m in metrics_log], "o-", color="tab:orange")
        ax.set_ylabel("Success Rate")
        ax.set_title("Episode Success Rate")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # --- Episode return ---
    ax = axes[1, 0]
    if "episode_return" in metrics_log[0]:
        returns = [m["episode_return"] for m in metrics_log]
        ax.plot(updates, returns, "o-", color="tab:red")
        if "episode_std_return" in metrics_log[0]:
            stds = [m["episode_std_return"] for m in metrics_log]
            ax.fill_between(
                updates,
                [r - s for r, s in zip(returns, stds)],
                [r + s for r, s in zip(returns, stds)],
                alpha=0.2, color="tab:red",
            )
        ax.set_ylabel("Return")
        ax.set_title("Episode Return")
    elif "avg_return" in metrics_log[0]:
        ax.plot(updates, [m["avg_return"] for m in metrics_log], "o-", color="tab:red")
        ax.set_ylabel("Return")
        ax.set_title("Avg Episode Return")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Update")
    ax.grid(True, alpha=0.3)

    # --- Entropy / KL ---
    ax = axes[1, 1]
    if "mean_entropy" in metrics_log[0]:
        ax.plot(updates, [m["mean_entropy"] for m in metrics_log], "o-", color="tab:green", label="entropy")
    if "mean_kl" in metrics_log[0]:
        ax.plot(updates, [m["mean_kl"] for m in metrics_log], "s-", color="tab:purple", label="kl")
    ax.set_xlabel("Update")
    ax.set_ylabel("Value")
    ax.set_title("Entropy & KL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    logger.info(f"GRPO curves saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NanoVLM for MiniGrid action prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    # CLI overrides for common switches
    parser.add_argument("--mode", choices=["action", "text_action"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Episodes for MiniGrid env evaluation per epoch (0 to skip)")
    parser.add_argument("--eval-max-steps", type=int, default=None,
                        help="Max steps per episode during env evaluation")
    parser.add_argument("--prompt-type", choices=["simple", "with_description"], default=None,
                        help="Prompt format: 'simple' (baseline) or 'with_description' (includes state description)")
    parser.add_argument("--method", choices=["sft", "grpo"], default="sft",
                        help="Training method: 'sft' (supervised fine-tuning) or 'grpo' (guided regularized policy optimization)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for GRPO initialization (required for GRPO mode)")
    # GRPO-specific arguments
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Number of episodes to collect per GRPO update (GRPO only)")
    parser.add_argument("--num-updates", type=int, default=None,
                        help="Number of GRPO training updates/iterations (GRPO only)")
    parser.add_argument("--kl-beta", type=float, default=None,
                        help="KL divergence weight in GRPO loss (GRPO only)")
    parser.add_argument("--entropy-weight", type=float, default=None,
                        help="Entropy regularization weight in GRPO loss (GRPO only)")
    parser.add_argument("--num-trajectory-batch", type=int, default=None,
                        help="Number of trajectories per batch during GRPO training (GRPO only)")
    args = parser.parse_args()

    # Load config with CLI overrides
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    img_cfg = cfg["image_processor"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    prompt_cfg = cfg.get("prompts", {})
    prompt_type = prompt_cfg.get("prompt_type", "simple")

    # Apply CLI overrides
    if args.mode is not None:
        model_cfg["mode"] = args.mode
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.dataset is not None:
        data_cfg["path"] = args.dataset
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.prompt_type is not None:
        prompt_type = args.prompt_type
    env_cfg = cfg["env"]
    grpo_cfg = cfg.get("grpo", {})
    eval_cfg = cfg.get("eval", {})
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else eval_cfg.get("num_episodes", 5)
    eval_max_steps = args.eval_max_steps if args.eval_max_steps is not None else eval_cfg.get("max_steps", 20)

    # Setup
    torch.manual_seed(train_cfg["seed"])
    device = get_device(train_cfg["device"])
    setup_device(device)

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Initialize tokenizer once
    model_name = model_cfg["name"]
    special_tokens = {
        "image_token": "<image>",
        "global_image_token": "<global_image>",
        **{f"r{i}c{j}": f"<r{i}c{j}>" for i in range(1, 9) for j in range(1, 9)},
        **{action: f"<{action}>" for action in ACTIONS.values()},
    }
    tokenizer = get_tokenizer(
        name=model_name,
        extra_special_tokens=special_tokens,
        trust_remote_code=True,
    )
    logger.info(f"Tokenizer initialized: {model_name}")

    # Load model
    lora_cfg = model_cfg["lora"]
    model = NanoVLMActionPredictor(
        model_name=model_name,
        tokenizer=tokenizer,
        mode=model_cfg["mode"],
        use_lora=lora_cfg["enabled"],
        lora_r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        max_img_size=img_cfg["max_img_size"],
        splitted_image_size=img_cfg["splitted_image_size"],
        mp_image_token_length=img_cfg["mp_image_token_length"],
        dtype=model_cfg["dtype"],
        vit_model_type=model_cfg.get("vit_model_type", "google/siglip2-base-patch16-512"),
    )
    model.to(device) 

    # Load dataset
    dataset, collator = get_dataset_and_collator(
        dataset_path=data_cfg["path"],
        tokenizer=tokenizer,
        image_processor_config=img_cfg,
        mp_image_token_length=model.mp_image_token_length,
        mode=model_cfg["mode"],
        max_length=data_cfg["max_length"],
        prompt_type=prompt_type,
        collator_type="action_prediction",
    )

    # Train/val split
    val_size = int(len(dataset) * data_cfg["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=train_cfg["num_workers"],
    )

    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    trainable_params += list(model.modality_projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    warmup_steps = train_cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps)),
    )

    # Branch on training method
    if args.method == "sft":
        _train_sft(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            output_dir,
            train_cfg,
            env_cfg,
            eval_episodes,
            eval_max_steps,
        )
    elif args.method == "grpo":
        _train_grpo(
            model,
            device,
            output_dir,
            args.checkpoint,
            train_cfg,
            grpo_cfg,
            env_cfg,
            model_cfg,
            tokenizer,
            num_episodes=args.num_episodes,
            num_updates=args.num_updates,
            kl_beta=args.kl_beta,
            entropy_weight=args.entropy_weight,
            num_trajectory_batch=args.num_trajectory_batch,
        )

def _train_sft(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    output_dir,
    train_cfg,
    env_cfg,
    eval_episodes,
    eval_max_steps,
):
    """SFT (Supervised Fine-Tuning) training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Starting SFT Training")
    logger.info("="*60)
    
    best_val_loss = float("inf")
    metrics_saver = MetricsSaver(output_dir)

    for epoch in range(1, train_cfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{train_cfg['epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, train_cfg["epochs"])
        val_loss, val_accuracy = validate(model, val_loader, device)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "timestamp": datetime.now().isoformat(),
        }

        # Evaluate in MiniGrid environment
        if eval_episodes > 0:
            # Create GIF output directory for this epoch
            gif_dir = output_dir / f"gifs_epoch_{epoch}"
            env_metrics = evaluate_in_env(
                model,
                env_name=env_cfg["name"],
                num_episodes=eval_episodes,
                max_steps=eval_max_steps,
                seed=train_cfg["seed"],
                output_dir=gif_dir,
            )
            metrics.update(env_metrics)
            logger.info(
                f"Env eval: success_rate={env_metrics['success_rate']:.2f}, "
                f"mean_return={env_metrics['mean_return']:.4f}, "
                f"mean_ep_len={env_metrics['mean_episode_length']:.1f}, "
                f"optimal_action_rate={env_metrics['optimal_action_rate']:.2f}"
            )

        # Save metrics in real-time
        metrics_saver.append(metrics)
        logger.info(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save checkpoint
        if epoch % train_cfg["save_interval"] == 0:
            ckpt_dir = output_dir / f"checkpoint_epoch_{epoch}"
            ckpt_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(ckpt_dir)
            model.tokenizer.save_pretrained(ckpt_dir)
            
            # Save metrics and plot at checkpoint
            checkpoint_metrics_file = ckpt_dir / "metrics.json"
            with open(checkpoint_metrics_file, "w") as f:
                json.dump(metrics_saver.get_all(), f, indent=2)
            plot_learning_curves(metrics_saver.get_all(), ckpt_dir, filename="checkpoint_curves.png")
            logger.info(f"Checkpoint saved to {ckpt_dir}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best_model"
            best_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            
            # Save metrics and plot at best checkpoint
            best_metrics_file = best_dir / "metrics.json"
            with open(best_metrics_file, "w") as f:
                json.dump(metrics_saver.get_all(), f, indent=2)
            plot_learning_curves(metrics_saver.get_all(), best_dir, filename="best_curves.png")
            logger.info(f"New best model (val_loss={val_loss:.4f})")

        scheduler.step()

    # Save final metrics and plot
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_saver.get_all(), f, indent=2)
    plot_learning_curves(metrics_saver.get_all(), output_dir, filename="learning_curves.png")

    logger.info(f"SFT Training complete. Results saved to {output_dir}")
    return output_dir / "best_model"


def _train_grpo(
    model,
    device,
    output_dir,
    checkpoint_path,
    train_cfg,
    grpo_cfg,
    env_cfg,
    model_cfg,
    tokenizer,
    num_episodes=None,
    num_updates=None,
    kl_beta=None,
    entropy_weight=None,
    num_trajectory_batch=None,
):
    """GRPO (Guided Regularized Policy Optimization) training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Starting GRPO Training")
    logger.info("="*60)
    
    try:
        from .grpo_train_action import DirectActionGRPOTrainer
        from .grpo_algorithm import GRPOConfig
        from .env import MiniGridRLEnv
    except ImportError as e:
        logger.error(f"Failed to import GRPO modules: {e}")
        logger.error("Make sure grpo_algorithm.py, grpo_train_action.py, and env.py are available")
        raise
    
    if checkpoint_path is None:
        raise ValueError("--checkpoint required for GRPO mode. Use SFT checkpoint as initialization.")
    
    logger.info(f"Loading SFT checkpoint from {checkpoint_path}")
    
    # Setup environment
    env = MiniGridRLEnv(env_id=env_cfg.get("name", "MiniGrid-Empty-8x8-v0"))
    
    # Use CLI arguments if provided, else fallback to config
    effective_kl_beta = kl_beta if kl_beta is not None else grpo_cfg.get("kl_beta", 0.1)
    effective_entropy_weight = entropy_weight if entropy_weight is not None else grpo_cfg.get("entropy_weight", 0.01)
    effective_num_updates = num_updates if num_updates is not None else grpo_cfg.get("num_updates", 50)
    effective_num_episodes = num_episodes if num_episodes is not None else grpo_cfg.get("num_episodes", 100)
    effective_num_trajectory_batch = num_trajectory_batch if num_trajectory_batch is not None else grpo_cfg.get("num_trajectory_batch", 32)
    
    logger.info(f"GRPO Config:")
    logger.info(f"  num_updates: {effective_num_updates}")
    logger.info(f"  num_episodes: {effective_num_episodes}")
    logger.info(f"  kl_beta: {effective_kl_beta}")
    logger.info(f"  entropy_weight: {effective_entropy_weight}")
    logger.info(f"  num_trajectory_batch: {effective_num_trajectory_batch}")
    
    grpo_config = GRPOConfig(
        learning_rate=train_cfg.get("lr", 1e-4),
        kl_coeff=effective_kl_beta,
        entropy_coeff=effective_entropy_weight,
    )
    
    # Create a frozen copy of the model as the reference policy
    import copy
    reference_model = copy.deepcopy(model)

    # Create trainer
    trainer = DirectActionGRPOTrainer(
        model=model,
        reference_model=reference_model,
        env=env,
        grpo_config=grpo_config,
        device=device,
        tokenizer=tokenizer,
    )
    
    metrics_saver = MetricsSaver(output_dir)
    
    for update_idx in range(1, effective_num_updates + 1):
        logger.info(f"\nUpdate {update_idx}/{effective_num_updates}")
        
        # Collect trajectories
        logger.info(f"Collecting {effective_num_episodes} episodes...")
        episodes, collection_stats = trainer.collect_trajectories(
            num_episodes=effective_num_episodes,
            base_seed=train_cfg["seed"] + update_idx * 1000,
        )
        
        logger.info(
            f"Episode Stats - Success Rate: {collection_stats['success_rate']:.2%}, "
            f"Avg Return: {collection_stats['avg_return']:.2f}"
        )
        
        # Training step
        logger.info("Updating policy with GRPO loss...")
        train_metrics = trainer.training_step(episodes)
        
        # Collect metrics
        metrics = {
            "update": update_idx,
            **collection_stats,
            **train_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        metrics_saver.append(metrics)
        
        logger.info(
            f"Loss: {train_metrics.get('loss', 0):.4f}, "
            f"Policy Loss: {train_metrics.get('policy_loss', 0):.4f}"
        )
        
        # Save checkpoint
        if update_idx % train_cfg.get("save_interval", 5) == 0:
            ckpt_dir = output_dir / f"checkpoint_update_{update_idx}"
            ckpt_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            
            # Save metrics and plot
            checkpoint_metrics_file = ckpt_dir / "metrics.json"
            with open(checkpoint_metrics_file, "w") as f:
                json.dump(metrics_saver.get_all(), f, indent=2)
            plot_grpo_curves(metrics_saver.get_all(), ckpt_dir, filename="checkpoint_curves.png")
            logger.info(f"Checkpoint saved to {ckpt_dir}")
    
    # Save final metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_saver.get_all(), f, indent=2)
    plot_grpo_curves(metrics_saver.get_all(), output_dir, filename="grpo_curves.png")
    
    logger.info(f"GRPO Training complete. Results saved to {output_dir}")
    env.close()
    return output_dir





if __name__ == "__main__":
    main()