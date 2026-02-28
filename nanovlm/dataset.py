import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from processors import get_image_string

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _get_chat_template_prefix_len(tokenizer) -> int:
    """Calculate the length of tokenizer's chat template prefix.
    
    This helps us properly mask labels by finding where the assistant's response starts
    in the tokenized sequence.
    
    Args:
        tokenizer: The tokenizer to inspect
        
    Returns:
        int: Number of tokens in the chat template prefix before response content
    """
    dummy_text = "xzyvd"  # Use a dummy 5-letter string to find insertion point
    templated = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": dummy_text}],
        tokenize=False,
        add_special_tokens=False
    )
    dummy_pos = templated.find(dummy_text)
    if dummy_pos == -1:
        logger.warning("Could not find dummy text in templated output - using approximate prefix length")
        return 10  
    
    prefix_text = templated[:dummy_pos]
    prefix_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    return prefix_len


class BaseMiniGridDataset(Dataset):

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer,
        image_processor,
        mp_image_token_length: int,
        mode: str = "action",
        max_length: int = 512,
    ):
        """Initialize dataset.
        
        Args:
            jsonl_path: Path to JSONL file with examples
            tokenizer: Tokenizer for text
            image_processor: Image processor that returns (patches, split_ratio)
            mp_image_token_length: Number of image tokens per patch
            mode: Training mode ('action' or 'text_action')
            max_length: Maximum sequence length
        """
        assert mode in ["action", "text_action"], f"Mode must be 'action' or 'text_action', got {mode}" 
        assert mp_image_token_length > 0, "mp_image_token_length must be positive"
        assert max_length > 0, "max_length must be positive"
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.mode = mode
        self.max_length = max_length
        
        # Calculate prefix length for proper label masking
        self.prefix_len = _get_chat_template_prefix_len(tokenizer)
        
        with open(self.jsonl_path, 'r') as fh:
            self.examples = [json.loads(line) for line in fh]
        logger.info(f"Loaded {len(self.examples)} examples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def _get_prompt(self) -> str:
        if self.mode == "action":
            return "What action should the agent take to reach the goal? Reply with exactly one of: turn_left, turn_right, forward."
        elif self.mode == "text_action":
            return "Describe what you observe and what action the agent should take. End your reply with the action name."
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process a single example into model inputs.
        
        Returns dictionary with:
        - input_ids: Tokenized input sequence
        - attention_mask: Attention mask (1 for real tokens, 0 for padding)
        - labels: Labels for loss computation (IGNORE_INDEX for user prompts, token IDs for responses)
        - images: List of processed image patches
        """
    
        image_path = item["image"]
        if not image_path:
            logger.warning("Example missing 'image' key")
            return None
        
        image = Image.open(image_path).convert("RGB")
        processed_patches, split_ratio = self.image_processor(image)
        
        # Remove global patch if tokenizer doesn't support it but processor generated it
        # if not hasattr(self.tokenizer, "global_image_token"):
        #     n_patches = split_ratio[0] * split_ratio[1]
        #     if n_patches > 0 and len(processed_patches) == n_patches + 1:
        #         processed_patches = processed_patches[1:]
        
        image_str = get_image_string(
            self.tokenizer,
            [split_ratio],
            self.mp_image_token_length
        )
        target = item["target"]
        assert target, "Example missing 'target' key or value is empty"
        prompt = self._get_prompt()
        messages = [
            {"role": "user", "content": image_str + prompt},
            {"role": "assistant", "content": target}
        ]
        
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            add_special_tokens=False,
            return_dict=True
        )
        
        input_ids = tokenized["input_ids"]

        prompt_only = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False
        )
        
        input_ids = input_ids[:self.max_length]
        
        # mask everything except assistant response
        labels = input_ids.copy()
        prompt_len = min(len(prompt_only), len(input_ids))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len 
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": [processed_patches],
        }
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return self._process_item(item)


class MiniGridSFTDataset(BaseMiniGridDataset):
    """Supervised Fine-Tuning dataset for MiniGrid action prediction. """
    pass  
