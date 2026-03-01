#!/usr/bin/env python3
"""
SFT Fine-tuning script for NanoVLM on MiniGrid action prediction.

Usage:
    python nanovlm/main.py \
        --dataset nanovlm/data/minigrid_small/dataset.jsonl \
        --mode action \
        --epochs 3 \
        --batch-size 4 \
        --lr 1e-4 \
        --output-dir nanovlm/checkpoints/sft_baseline
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .model import NanoVLMActionPredictor
from .data_utils import get_dataset_and_collator
from .processors import get_tokenizer
from .device_utils import get_device, setup_device, print_device_stats
from .env import MiniGridRLEnv, EpisodeCollector
from .grpo_algorithm import GRPOTrainer, GRPOConfig
from .grpo_train_action import DirectActionGRPOTrainer
from .grpo_train_text_action import TextActionGRPOTrainer
from .eval import ModelEvaluator, ComparisonPlotter, load_checkpoint

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
    num_episodes: int = 50,
    seed: int = 42,
) -> Dict:
    """Evaluate model in MiniGrid environment.
    
    Args:
        model: Model to evaluate
        num_episodes: Number of episodes to run
        seed: Random seed
        
    Returns:
        Dict with success_rate, avg_return, avg_length, etc.
    """
    env = MiniGridRLEnv(env_id="MiniGrid-Empty-8x8-v0", seed=seed)
    evaluator = ModelEvaluator(model, env, model.model.device if hasattr(model.model, 'device') else torch.device('cpu'))
    
    stats, episodes = evaluator.evaluate(
        num_episodes=num_episodes,
        base_seed=seed,
    )
    
    env.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NanoVLM for MiniGrid action prediction")
    
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="nanovlm/data/minigrid_small/dataset.jsonl",
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation",
    )
    
    # Training method
    parser.add_argument(
        "--method",
        choices=["sft", "grpo", "eval"],
        default="sft",
        help="Training method: 'sft' (supervised fine-tuning), 'grpo' (guided regularized policy optimization), or 'eval' (evaluate models)",
    )
    
    # Model args
    parser.add_argument(
        "--mode",
        choices=["action", "text_action"],
        default="action",
        help="Training mode",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for fine-tuning",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (scaling factor)",
    )
    
    # Training args
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    
    # Output args
    parser.add_argument(
        "--output-dir",
        type=str,
        default="nanovlm/checkpoints/sft_baseline",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for training. 'auto' will use CUDA if available, else MPS if available, else CPU",
    )
    
    # GRPO-specific args
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanovlm/checkpoints/sft_baseline",
        help="Path to SFT checkpoint for GRPO initialization (required for GRPO training)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect per training iteration (GRPO only)",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=50,
        help="Number of training updates/iterations (GRPO only)",
    )
    parser.add_argument(
        "--num-trajectory-batch",
        type=int,
        default=32,
        help="Number of trajectories per batch (GRPO only)",
    )
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.1,
        help="KL divergence weight in GRPO loss (GRPO only)",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.01,
        help="Entropy regularization weight in GRPO loss (GRPO only)",
    )
    parser.add_argument(
        "--text-mode",
        choices=["plan", "state", "reasoning"],
        default="plan",
        help="Text generation mode for text_action mode (GRPO only)",
    )
    
    # Evaluation-specific args
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Model checkpoints to evaluate (eval mode only)",
    )
    parser.add_argument(
        "--checkpoint-names",
        type=str,
        nargs="+",
        help="Names for checkpoints in evaluation results",
    )
    parser.add_argument(
        "--training-curves",
        type=str,
        help="Path to training metrics JSON for plotting curves",
    )
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    # Device selection with MPS support
    device = get_device(args.device)
    setup_device(device)
    print_device_stats(device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {output_dir / 'config.json'}")
    
    # Initialize tokenizer once (dependency injection pattern)
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    logger.info(f"Initializing tokenizer from {model_name}")
    
    # Define special tokens needed for both dataset and model
    special_tokens = {
        "image_token": "<image>",
        "global_image_token": "<global_image>",
        **{f"r{i}c{j}": f"<r{i}c{j}>" for i in range(1, 9) for j in range(1, 9)},
        # Action tokens
        "forward": "<forward>",
        "turn_left": "<turn_left>",
        "turn_right": "<turn_right>",
        "pickup": "<pickup>",
        "drop": "<drop>",
        "toggle": "<toggle>",
        "done": "<done>",
    }
    
    tokenizer = get_tokenizer(
        name=model_name,
        extra_special_tokens=special_tokens,
        trust_remote_code=True
    )
    logger.info(f"✓ Tokenizer initialized with {len(special_tokens)} special tokens")
    
    # Branch on training method
    if args.method == "sft":
        _train_sft(args, device, tokenizer, special_tokens, output_dir)
    elif args.method == "grpo":
        _train_grpo(args, device, tokenizer, special_tokens, output_dir)
    elif args.method == "eval":
        _eval_models(args, device, tokenizer)


def _train_sft(args, device, tokenizer, special_tokens, output_dir):
    """SFT (Supervised Fine-Tuning) training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Starting SFT Training")
    logger.info("="*60)
    
    # Load model
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    logger.info("Loading NanoVLM model...")
    model = NanoVLMActionPredictor(
        model_name=model_name,
        tokenizer=tokenizer,
        mode=args.mode,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_img_size=224,
        splitted_image_size=112,
    )
    model.model.to(device)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    dataset, collator = get_dataset_and_collator(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        image_processor_config={
            "max_img_size": 224,
            "splitted_image_size": 112,
        },
        mp_image_token_length=2,
        mode=args.mode,
        max_length=512,
        collator_type="action_prediction",
    )
    
    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_loss = float("inf")
    metrics_log = []
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
        )
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, device)
        
        # Optionally evaluate in environment every 2 epochs
        env_stats = None
        if epoch % 2 == 0:
            logger.info("Evaluating in environment...")
            env_stats = evaluate_in_env(model, num_episodes=20,  seed=args.seed + epoch)
            logger.info(f"  Env Success Rate: {env_stats['success_rate']:.2%}, Avg Return: {env_stats['avg_return']:.3f}")
        
        # Log
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        if env_stats:
            metrics.update({f"env_{k}": v for k, v in env_stats.items()})
        metrics_log.append(metrics)
        
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
        
        if env_stats:
            logger.info(
                f"          Env Success: {env_stats['success_rate']:.2%}, "
                f"Avg Return: {env_stats['avg_return']:.3f}, "
                f"Avg Length: {env_stats['avg_length']:.1f}"
            )
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            model.model.save_pretrained(checkpoint_dir)
            model.tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best_model"
            best_dir.mkdir(exist_ok=True)
            model.model.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            logger.info(f"Saved best model (val_loss={val_loss:.4f}) to {best_dir}")
        
        # Update scheduler
        scheduler.step()
    
    # Save final metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"\nSFT Training complete. Metrics saved to {output_dir / 'metrics.json'}")
    logger.info(f"Best checkpoint: {output_dir / 'best_model'}")


def _train_grpo(args, device, tokenizer, special_tokens, output_dir):
    """GRPO (Guided Regularized Policy Optimization) training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Starting GRPO Training")
    logger.info("="*60)
    
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    
    # Load policy model from checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading policy model from {checkpoint_path}...")
    policy_model = NanoVLMActionPredictor(
        model_name=model_name,
        tokenizer=tokenizer,
        mode=args.mode,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_img_size=224,
        splitted_image_size=112,
    )
    policy_model.model = policy_model.model.from_pretrained(
        checkpoint_path,
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    # Load reference model (SFT baseline)
    logger.info(f"Loading reference model from {checkpoint_path}...")
    reference_model = NanoVLMActionPredictor(
        model_name=model_name,
        tokenizer=tokenizer,
        mode=args.mode,
        use_lora=False,
        max_img_size=224,
        splitted_image_size=112,
    )
    reference_model.model = reference_model.model.from_pretrained(
        checkpoint_path,
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    # Setup environment
    logger.info("Setting up MiniGrid environment...")
    env = MiniGridRLEnv(env_id="MiniGrid-Empty-8x8-v0")
    

    
    # Setup GRPO config
    grpo_config = GRPOConfig(
        learning_rate=args.lr,
        kl_beta=args.kl_beta,
        entropy_weight=args.entropy_weight,
        num_trajectory_batch=args.num_trajectory_batch,
    )
    
    # Create trainer based on mode
    if args.mode == "action":
        logger.info("Using DirectActionGRPOTrainer")
        trainer = DirectActionGRPOTrainer(
            model=policy_model,
            reference_model=reference_model,
            env=env,
            grpo_config=grpo_config,
            device=device,
            tokenizer=tokenizer,
        )
    elif args.mode == "text_action":
        logger.info(f"Using TextActionGRPOTrainer with text_mode={args.text_mode}")
        trainer = TextActionGRPOTrainer(
            model=policy_model,
            reference_model=reference_model,
            env=env,
            grpo_config=grpo_config,
            device=device,
            tokenizer=tokenizer,
            text_mode=args.text_mode,
        )
    
    # Training loop
    metrics_log = []
    
    for update_idx in range(1, args.num_updates + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Update {update_idx}/{args.num_updates}")
        logger.info(f"{'='*60}")
        
        # Collect trajectories
        logger.info(f"Collecting {args.num_episodes} episodes...")
        episodes, collection_stats = trainer.collect_trajectories(
            num_episodes=args.num_episodes,
            base_seed=args.seed + update_idx * 1000,
        )
        
        logger.info(
            f"Episode Stats - "
            f"Success Rate: {collection_stats['success_rate']:.2%}, "
            f"Avg Return: {collection_stats['avg_return']:.2f}, "
            f"Avg Length: {collection_stats['avg_episode_length']:.1f}"
        )
        
        # Training step
        logger.info("Updating policy with GRPO loss...")
        train_metrics = trainer.training_step(episodes)
        
        # Optionally run supervised validation to track if model forgets SFT knowledge
        metrics = {
            "update": update_idx,
            **collection_stats,
            **train_metrics,
        }
        
        if update_idx % 5 == 0:
            # Environment evaluation
            logger.info("  Evaluating policy in environment...")
            env_stats = evaluate_in_env(policy_model, num_episodes=20, seed=args.seed + update_idx)
            logger.info(
                f"  Env Success: {env_stats['success_rate']:.2%}, "
                f"Avg Return: {env_stats['avg_return']:.3f}, "
                f"Avg Length: {env_stats['avg_length']:.1f}"
            )
            metrics.update({f"env_{k}": v for k, v in env_stats.items()})
        
        metrics_log.append(metrics)
        
        logger.info(
            f"Update {update_idx} - "
            f"Loss: {train_metrics.get('loss', 0):.4f}, "
            f"Policy Loss: {train_metrics.get('policy_loss', 0):.4f}, "
            f"KL Div: {train_metrics.get('kl_divergence', 0):.4f}"
        )
        
        # Save checkpoint
        if update_idx % args.save_interval == 0:
            checkpoint_dir = output_dir / f"checkpoint_update_{update_idx}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            policy_model.model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final metrics
    with open(output_dir / "metrics_all.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"\nGRPO Training complete. Metrics saved to {output_dir / 'metrics_all.json'}")
    logger.info(f"Final checkpoint: {output_dir / f'checkpoint_update_{args.num_updates}'}")
    
    env.close()


def _eval_models(args, device, tokenizer):
    """Evaluate and compare multiple trained models."""
    logger.info("\n" + "="*60)
    logger.info("Starting Model Evaluation")
    logger.info("="*60)
    
    if not args.checkpoints:
        raise ValueError("--checkpoints required for eval mode")
    
    # Setup evaluation environment with rendering
    env = MiniGridRLEnv(env_id="MiniGrid-Empty-8x8-v0", max_steps=100, seed=args.seed, render_mode="rgb_array")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set checkpoint names
    checkpoint_names = args.checkpoint_names or [
        Path(cp).stem for cp in args.checkpoints
    ]
    
    # Evaluate each checkpoint
    logger.info(f"Evaluating {len(args.checkpoints)} model(s)...")
    results = {}
    
    for checkpoint, name in zip(args.checkpoints, checkpoint_names):
        logger.info(f"\nEvaluating {name} from {checkpoint}")
        
        # Load model
        model = load_checkpoint(checkpoint, device, tokenizer)
        
        # Evaluate
        evaluator = ModelEvaluator(model, env, device)
        stats, episodes = evaluator.evaluate(
            num_episodes=args.num_episodes,
            base_seed=args.seed,
        )
        
        results[name] = stats
        
        logger.info(f"  Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"  Avg Return: {stats['avg_return']:.3f} ± {stats['std_return']:.3f}")
        logger.info(f"  Avg Length: {stats['avg_length']:.1f}")
        logger.info(f"  Median Return: {stats['p50_return']:.3f}")
        
        # Save episode GIFs
        logger.info(f"Saving episode visualizations for {name}...")
        gifs_dir = output_dir / f"{name.replace(' ', '_')}_episodes"
        evaluator.save_episode_gifs(episodes, gifs_dir, num_gifs=3)
        logger.info(f"Saved GIFs to {gifs_dir}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved evaluation results to {results_path}")
    
    # Plot training curves if provided
    if args.training_curves:
        with open(args.training_curves, 'r') as f:
            metrics_history = json.load(f)
        
        ComparisonPlotter.plot_training_curves(
            metrics_history,
            output_path=output_dir / "training_curves.png",
        )
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Success':<12} {'Avg Return':<15} {'Median Return':<15} {'Avg Length':<12}")
    print("-"*80)
    
    for name in checkpoint_names:
        stats = results[name]
        print(f"{name:<20} {stats['success_rate']:>10.2%} "
              f"{stats['avg_return']:>13.3f} "
              f"{stats['p50_return']:>13.3f} "
              f"{stats['avg_length']:>10.1f}")
    
    print("="*80)
    
    env.close()
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()