import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .processors import get_image_string

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
        prompt_type: str = "simple",
    ):
        """Initialize dataset.
        
        Args:
            jsonl_path: Path to JSONL file with examples
            tokenizer: Tokenizer for text
            image_processor: Image processor that returns (patches, split_ratio)
            mp_image_token_length: Number of image tokens per patch
            mode: Training mode ('action' or 'text_action')
            max_length: Maximum sequence length
            prompt_type: Prompt format ('simple' or 'with_description')
        """
        assert mode in ["action", "text_action"], f"Mode must be 'action' or 'text_action', got {mode}" 
        assert mp_image_token_length > 0, "mp_image_token_length must be positive"
        assert max_length > 0, "max_length must be positive"
        assert prompt_type in ["simple", "with_description"], f"prompt_type must be 'simple' or 'with_description', got {prompt_type}"
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.mode = mode
        self.max_length = max_length
        self.prompt_type = prompt_type
        
        # Calculate prefix length for proper label masking
        self.prefix_len = _get_chat_template_prefix_len(tokenizer)
        
        with open(self.jsonl_path, 'r') as fh:
            self.examples = [json.loads(line) for line in fh]
        logger.info(f"Loaded {len(self.examples)} examples from {self.jsonl_path}")
        logger.info(f"Prompt type: {prompt_type}")

    def __len__(self) -> int:
        return len(self.examples)

    def _get_prompt(self, item: Dict) -> str:
        """Build prompt based on prompt_type and mode.
        
        Args:
            item: Example dict with 'description' key (if available)
            
        Returns:
            Prompt text
        """
        if self.prompt_type == "simple":
            if self.mode == "action":
                return "What action should the agent take to reach the goal? Reply with exactly one of: turn_left, turn_right, forward."
            elif self.mode == "text_action":
                return "Describe what you observe and what action the agent should take. End your reply with the action name."
        
        elif self.prompt_type == "with_description":
            description = item.get("description", "")
            description_text = f"{description}\n" if description else ""
            
            if self.mode == "action":
                return f"{description_text}What action should the agent take? Reply with exactly one of: turn_left, turn_right, forward."
            elif self.mode == "text_action":
                return f"{description_text}Describe your reasoning and what action the agent should take. End your reply with the action name."
        
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
    
    def _prepare_inputs_and_loss_mask(self, messages: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input IDs with proper loss mask.
        
        This approach:
        1. Applies chat template to full conversation
        2. Tokenizes the assistant response separately to find its tokens
        3. Searches for those tokens in the full sequence to find the response positions
        4. Marks only the response tokens as trainable
        5. Rolls labels for causal LM training
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (input_ids, labels, attention_mask) tensors
        """
        # Get full conversation tokens
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        input_ids = conv_ids["input_ids"]
        
        # Find assistant messages and extract their token representations
        labels = [IGNORE_INDEX] * len(input_ids)
        
        # Find all assistant messages and tokenize them
        for msg in messages:
            if msg["role"] == "assistant":
                # Tokenize just the assistant's content
                response_tokens = self.tokenizer.encode(
                    msg["content"],
                    add_special_tokens=False
                )
                
                logger.debug(f"Assistant response: {msg['content'][:50]}")
                logger.debug(f"Response tokens: {response_tokens}")
                
                if not response_tokens:
                    continue
                
                # Search for this sequence in the full input_ids
                # We search from the end since assistant messages come toward the end
                for i in range(len(input_ids) - len(response_tokens), -1, -1):
                    # Compare as lists directly
                    if list(input_ids[i:i+len(response_tokens)]) == response_tokens:
                        logger.debug(f"Found assistant response at position {i}-{i+len(response_tokens)}")
                        # Found it! Mark these tokens as trainable
                        for j in range(i, i + len(response_tokens)):
                            labels[j] = input_ids[j]
                        break
                else:
                    logger.warning(f"Could not find assistant tokens {response_tokens} in input_ids")
        
        # Check if we found any assistant tokens
        num_trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        logger.debug(f"Trainable tokens: {num_trainable} / {len(labels)}")
        

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(conv_ids["attention_mask"], dtype=torch.long)
        )

    def _process_item(self, item: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single example into model inputs.
        
        Returns dictionary with:
        - input_ids: Tokenized input sequence
        - attention_mask: Attention mask (1 for real tokens, 0 for padding)
        - labels: Labels for loss computation (IGNORE_INDEX for user prompts, token IDs for responses)
        - images: List of processed image patches
        
        Returns None if example is invalid (missing required fields).
        """
  
        image_path = item.get("image")
        if not image_path:
            logger.warning("Example missing 'image' key")
            return None
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        processed_patches, split_ratio = self.image_processor(image)
        
        # Remove global patch if tokenizer doesn't support it but processor generated it
        if not hasattr(self.tokenizer, "global_image_token") and split_ratio[0] * split_ratio[1] == len(processed_patches) - 1:
            processed_patches = processed_patches[1:]
        
        # Build image string
        image_str = get_image_string(
            self.tokenizer,
            [split_ratio],
            self.mp_image_token_length
        )
        
        action_name = item.get("target")
        if not action_name:
            logger.warning("Example missing 'target' key or value is empty")
            return None
        
        action_token = f"<{action_name}>"
        
        # Build messages based on mode
        prompt = self._get_prompt(item)
        
        if self.mode == "text_action":
            description = item.get("description", "The agent needs to navigate to the goal.")
            assistant_content = f"{description} Action: {action_token}"
        else:
            assistant_content = action_token
        
        messages = [
            {"role": "user", "content": image_str + prompt},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Prepare inputs and labels with proper masking
        input_ids, labels, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        
        # Truncate if necessary - keep last tokens for causal LM (we want to keep the assistant response at the end)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": [processed_patches],
        }
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return self._process_item(item)


class MiniGridSFTDataset(BaseMiniGridDataset):
    """Supervised Fine-Tuning dataset for MiniGrid action prediction. """
    pass  
