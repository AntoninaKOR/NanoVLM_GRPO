from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace default ignore index for labels
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForActionPrediction:
    """Collator for batching action prediction examples.
    
    Handles:
    - Padding input_ids and labels to max length in batch
    - Creating attention masks
    - Stacking image patches (assumes same patch resolution)
    - Proper handling of variable batch sizes
    
    Attributes:
        pad_token_id: ID used for padding tokens
        model_max_length: Maximum sequence length (used for truncation)
        padding_side: 'left' or 'right' (typically 'right' for generation)
    """
    
    pad_token_id: int
    model_max_length: int = 2048
    padding_side: str = "right"
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Batch and pad a list of examples.
        
        Args:
            instances: List of dicts with keys 'input_ids', 'labels', 'attention_mask', 'images'
            
        Returns:
            Dict with batched tensors:
            - input_ids: (batch_size, seq_len) padded
            - labels: (batch_size, seq_len) padded with IGNORE_INDEX
            - attention_mask: (batch_size, seq_len)
            - images: List of image patch stacks or None
        """
        
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        attention_masks = [inst.get("attention_mask", torch.ones_like(iid)) for iid, inst in zip(input_ids, instances)]
        images_list = [inst.get("images", None) for inst in instances]
        
        # Pad sequences
        if self.padding_side == "right":
            input_ids = pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id
            )
            labels = pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            attention_masks = pad_sequence(
                attention_masks,
                batch_first=True,
                padding_value=0
            )
        else:
            raise NotImplementedError("Left padding not yet implemented")
        
        # Truncate if necessary
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        attention_masks = attention_masks[:, :self.model_max_length]
        
        # Stack images if present
        images = None
        if any(img is not None for img in images_list):
            # Flatten list of lists into single list of patches
            all_patches = []
            for img_patches in images_list:
                if img_patches is not None:
                    if isinstance(img_patches, list):
                        all_patches.extend(img_patches)
                    else:
                        all_patches.append(img_patches)
            
            if all_patches:
                try:
                    images = torch.cat(all_patches, dim=0)
                except RuntimeError as e:
                    # Handle case where patches have different shapes
                    import logging
                    logging.warning(f"Could not concatenate image patches: {e}")
                    images = None
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
            "images": images,
        }


@dataclass
class PaddedCollatorForLanguageModeling:
    """Collator for standard language modeling (similar to openvla).
    
    Extends PaddedCollatorForActionPrediction to handle general LM training with
    optional image inputs and multimodal indices tracking.
    
    Attributes:
        pad_token_id: ID used for padding tokens
        model_max_length: Maximum sequence length
        default_image_resolution: Tuple (C, H, W) for dummy image tensors
        padding_side: 'left' or 'right'
        pixel_values_dtype: Data type for image tensors (default float32)
    """
    
    pad_token_id: int
    model_max_length: int = 2048
    default_image_resolution: Tuple = (3, 448, 448)
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Initialize dummy pixel values for unimodal examples."""
        self.dummy_pixel_values = torch.zeros(
            self.default_image_resolution,
            dtype=self.pixel_values_dtype
        )
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Batch and pad a list of examples.
        
        Handles mixed unimodal (text-only) and multimodal (image+text) examples.
        
        Args:
            instances: List of dicts with keys 'input_ids', 'labels', 'pixel_values'
            
        Returns:
            Dict with batched tensors including 'multimodal_indices' tracking
        """
        # Extract components
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        pixel_values = [inst.get("pixel_values", None) for inst in instances]
        
        # Pad sequences
        if self.padding_side == "right":
            input_ids = pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id
            )
            labels = pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
        else:
            raise NotImplementedError("Left padding not yet implemented")
        
        # Truncate if necessary
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        
        # Compute attention mask
        attention_mask = input_ids.ne(self.pad_token_id)
        
        # Track which examples have images (for selective image processing)
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None],
            dtype=torch.long
        )
        
        # Handle pixel values - stack or create dummy
        if len(multimodal_indices) == 0:
            # All unimodal examples
            pixel_values = torch.stack(
                [self.dummy_pixel_values for _ in range(len(input_ids))]
            )
        elif isinstance(pixel_values[multimodal_indices[0]], torch.Tensor):
            # All examples have single tensor pixel values
            pixel_values = torch.stack([
                pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                for idx in range(len(input_ids))
            ])
        elif isinstance(pixel_values[multimodal_indices[0]], dict):
            # Pixel values are dicts (e.g., for models with multiple image processors)
            pv_example = pixel_values[multimodal_indices[0]]
            pixel_values = {
                k: torch.stack([
                    pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ])
                for k in pv_example.keys()
            }
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[multimodal_indices[0]])}")
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "multimodal_indices": multimodal_indices,
        }
