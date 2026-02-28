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

from model import NanoVLMActionPredictor
from data_utils import get_dataset_and_collator
from processors import get_tokenizer

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
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    
    # Load model
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
    
    # Load dataset using existing utility (pass tokenizer instead of model_name)
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Scheduler (simple linear warmup)
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
        
        # Log
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        metrics_log.append(metrics)
        
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
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
    logger.info(f"\nTraining complete. Metrics saved to {output_dir / 'metrics.json'}")
    logger.info(f"Best checkpoint: {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()