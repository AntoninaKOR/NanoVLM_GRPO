import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import NanoVLMActionPredictor, ACTIONS
from data_utils import get_dataset_and_collator
from processors import get_tokenizer
from device_utils import get_device, setup_device
from config_loader import load_config

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
        
        # Forward pass
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        loss = outputs.loss
        
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
            
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Simple accuracy: check if predicted token matches label (simplified)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Only count non-ignored labels
            mask = labels != -100
            if mask.any():
                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
            
            progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / max(1, len(val_loader))
    accuracy = correct / max(1, total)
    
    return avg_loss, accuracy


def evaluate_in_env(
    model: NanoVLMActionPredictor,
    env_name: str,
    num_episodes: int = 5,
    max_steps: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the model in MiniGrid and compute success rate / average return."""
    model.model.eval()
    env = gym.make(env_name, render_mode="rgb_array")

    successes = 0
    total_returns: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0

        for _ in range(max_steps):
            frame = env.render()
            action_id = model.predict_action(frame)
            obs, reward, terminated, truncated, _ = env.step(action_id)
            ep_return += reward
            steps += 1
            if terminated or truncated:
                break

        if ep_return > 0:
            successes += 1
        total_returns.append(ep_return)
        episode_lengths.append(steps)

    env.close()

    return {
        "success_rate": successes / max(1, num_episodes),
        "mean_return": float(np.mean(total_returns)),
        "std_return": float(np.std(total_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


def plot_learning_curves(
    metrics_log: List[Dict],
    output_dir: Path,
) -> None:
    """Save learning-curve plots from the collected per-epoch metrics."""
    epochs = [m["epoch"] for m in metrics_log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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

    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)
    logger.info(f"Learning curves saved to {output_dir / 'learning_curves.png'}")


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
    args = parser.parse_args()

    # Load config with CLI overrides
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    img_cfg = cfg["image_processor"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

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
    env_cfg = cfg["env"]
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
        json.dump(cfg, f, indent=2)

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
    )
    model.model.to(device)

    # Load dataset
    dataset, collator = get_dataset_and_collator(
        dataset_path=data_cfg["path"],
        tokenizer=tokenizer,
        image_processor_config=img_cfg,
        mp_image_token_length=img_cfg["mp_image_token_length"],
        mode=model_cfg["mode"],
        max_length=data_cfg["max_length"],
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

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    warmup_steps = train_cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps)),
    )

    # Training loop
    best_val_loss = float("inf")
    metrics_log = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{train_cfg['epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, train_cfg["epochs"])
        val_loss, val_accuracy = validate(model, val_loader, device)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        # Evaluate in MiniGrid environment
        if eval_episodes > 0:
            env_metrics = evaluate_in_env(
                model,
                env_name=env_cfg["name"],
                num_episodes=eval_episodes,
                max_steps=eval_max_steps,
                seed=train_cfg["seed"],
            )
            metrics.update(env_metrics)
            logger.info(
                f"Env eval: success_rate={env_metrics['success_rate']:.2f}, "
                f"mean_return={env_metrics['mean_return']:.4f}, "
                f"mean_ep_len={env_metrics['mean_episode_length']:.1f}"
            )

        metrics_log.append(metrics)
        logger.info(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save checkpoint
        if epoch % train_cfg["save_interval"] == 0:
            ckpt_dir = output_dir / f"checkpoint_epoch_{epoch}"
            ckpt_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(ckpt_dir)
            model.tokenizer.save_pretrained(ckpt_dir)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best_model"
            best_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            logger.info(f"New best model (val_loss={val_loss:.4f})")

        scheduler.step()

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Plot learning curves
    plot_learning_curves(metrics_log, output_dir)

    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()