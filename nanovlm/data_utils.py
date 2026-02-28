from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import logging
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from processors import get_image_processor
from dataset import MiniGridSFTDataset
from collators import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def get_dataset_and_collator(
    dataset_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    image_processor_config: Dict[str, int],
    mp_image_token_length: int = 256,
    mode: str = "action",
    max_length: int = 512,
    collator_type: str = "action_prediction",
    collator_kwargs: Optional[Dict] = None,
) -> Tuple[Dataset, Union[PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling]]:
    """
    Args:
        dataset_path: Path to JSONL file with training examples
        tokenizer: Pre-initialized tokenizer instance
        image_processor_config: Dict with keys:
            - 'max_img_size': Maximum image dimension (recommended: 224 for simple envs, 448 for complex)
            - 'splitted_image_size': Patch size (recommended: 112 for simple envs, 224 for complex)
            - Optional 'resize_to_max_side_len': Bool flag
        mp_image_token_length: Number of image tokens per patch
        mode: Training mode ('action' or 'text_action')
        max_length: Maximum sequence length
        collator_type: 'action_prediction' or 'language_modeling'
        collator_kwargs: Additional kwargs for collator (e.g., model_max_length, pad_token_id)
        
    Returns:
        (dataset, collator) tuple
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info("Creating image processor...")
    image_processor = get_image_processor(
        max_img_size=image_processor_config["max_img_size"],
        splitted_image_size=image_processor_config["splitted_image_size"],
        resize_to_max_side_len=image_processor_config.get("resize_to_max_side_len", False),
    )

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = MiniGridSFTDataset(
        jsonl_path=dataset_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=mp_image_token_length,
        mode=mode,
        max_length=max_length,
    )
 
    logger.info(f"Creating {collator_type} collator")
    default_collator_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        "model_max_length": max_length,
    }
    if collator_kwargs:
        default_collator_kwargs.update(collator_kwargs)
    if collator_type == "action_prediction":
        collator = PaddedCollatorForActionPrediction(**default_collator_kwargs)
    elif collator_type == "language_modeling":
        lm_kwargs = {
            **default_collator_kwargs,
            "default_image_resolution": (3, 448, 448),  # RGB images
        }
        if collator_kwargs:
            lm_kwargs.update(collator_kwargs)
        collator = PaddedCollatorForLanguageModeling(**lm_kwargs)
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")
    
    logger.info(f"✓ Dataset loaded with {len(dataset)} examples")
    logger.info(f"✓ Dataset mode: {mode}")
    logger.info(f"✓ Pad token ID: {tokenizer.pad_token_id}")
    
    return dataset, collator


def create_dataloader(
    dataset_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    image_processor_config: Dict[str, int],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    mp_image_token_length: int = 256,
    mode: str = "action",
    max_length: int = 512,
    **dataloader_kwargs
):
    """
    Args:
        dataset_path: Path to JSONL dataset
        tokenizer: Pre-initialized tokenizer instance
        image_processor_config: Image processor configuration
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        mp_image_token_length: Image token length
        mode: Training mode
        max_length: Maximum sequence length
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        torch.utils.data.DataLoader with appropriate collator
    """
    
    dataset, collator = get_dataset_and_collator(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        image_processor_config=image_processor_config,
        mp_image_token_length=mp_image_token_length,
        mode=mode,
        max_length=max_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collator,
        **dataloader_kwargs
    )
