from typing import Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoConfig
from PIL import Image
import numpy as np

from processors import get_image_processor, get_image_string, get_tokenizer

# MiniGrid action space
ACTIONS = {
    0: "turn_left",
    1: "turn_right", 
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done"
}

ACTION_TO_ID = {v: k for k, v in ACTIONS.items()}


class NanoVLMActionPredictor(nn.Module):
    """
    Wrapper for NanoVLM to predict actions in MiniGrid.
    
    Training modes:
    - mode='action': Directly predict action from image
    - mode='text_action': Generate text description then action
    """
    
    def __init__(
        self,
        model_name: str = "qnguyen3/nanoLLaVA-1.5",
        mode: str = "action", 
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_img_size: int = 32,
        splitted_image_size: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.mode = mode
        self.max_img_size = max_img_size
        self.splitted_image_size = splitted_image_size  
   
        special_tokens = {
            "image_token": "<image>",
            "global_image_token": "<global_image>",
            **{f"r{i}c{j}": f"<r{i}c{j}>" for i in range(1, 9) for j in range(1, 9)},
        }
        
        # Add action tokens to dictionary
        for action in ACTIONS.values():
            special_tokens[f"<{action}>"] = f"<{action}>"
     
        self.tokenizer = get_tokenizer(
            model_name,
            extra_special_tokens=special_tokens,
            trust_remote_code=True
        )
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            config=config,
        )
        
        # Resize model embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Use custom image processor for training
        self.image_processor = get_image_processor(
            max_img_size=max_img_size,
            splitted_image_size=splitted_image_size
        )
        
        # Store action token IDs for later extraction
        self.action_token_ids = {
            action: self.tokenizer.convert_tokens_to_ids(f"<{action}>")
            for action in ACTIONS.values()
        }
        
        if use_lora:
            self._apply_lora(lora_r, lora_alpha)
    
    def _apply_lora(self, r: int, alpha: int):
        """Apply LoRA adapters for efficient fine-tuning."""
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)
        print(f"Applied LoRA with r={r}, alpha={alpha}")
    
    def prepare_prompt(self, mode: Optional[str] = None) -> str:
        """Prepare text prompt based on training mode."""
        if self.mode == "action":
            return "What action should the agent take to reach the goal?"
        elif self.mode == "text_action":
            return "Describe what you see and what action the agent should take."
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def forward(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: List of images (PIL Images)
            prompts: Optional text prompts (default uses self.prepare_prompt())
            labels: Ground truth token IDs for computing loss
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        if prompts is None:
            batch_size = len(images)
            prompts = [self.prepare_prompt()] * batch_size
        
        text_inputs = self.tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True
        )
    
        all_patches = []
        for img in images:
            patches, grid_info = self.image_processor(img)
            all_patches.append(patches)

        pixel_values = torch.cat(all_patches, dim=0)
        
        inputs = {**text_inputs, "pixel_values": pixel_values}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(
            **inputs,
            labels=labels,
            return_dict=True
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None
        }
    
    def predict_action(
        self,
        image: Union[Image.Image, np.ndarray],
        return_text: bool = False,
        max_new_tokens: int = 50,
    ) -> Union[int, Tuple[int, str]]:
        """
        Predict action from a single observation.
        
        Args:
            image: Single observation image
            return_text: If True, also return generated text
            max_new_tokens: Max tokens to generate
        
        Returns:
            action_id: Integer action ID (0-6)
            text (optional): Generated text if return_text=True
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        prompt = self.prepare_prompt()
        patches, grid_info = self.image_processor(image)

        # Build image token string 
        image_str = get_image_string(self.tokenizer, [grid_info], 2)
        text = image_str + prompt

        text_inputs = self.tokenizer(
            text=[text],
            return_tensors="pt",
            padding=True
        )
        
        input_ids = text_inputs["input_ids"].to(self.model.device)
        attention_mask = text_inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    return_dict=True,
                    use_cache=False
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                # Sample greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        # TODO: implement for batch > 1
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False
        )
        
        action_id = self._extract_action(generated_text)
        
        if return_text:
            return action_id, generated_text
        return action_id
    
    def _extract_action(self, text: str) -> int:
        """Extract action ID from generated text."""
        # Look for action tokens in the text
        for action_name, action_id in ACTION_TO_ID.items():
            if f"<{action_name}>" in text.lower():
                return action_id
        
        # Fallback: look for action names without brackets
        for action_name, action_id in ACTION_TO_ID.items():
            if action_name.replace("_", " ") in text.lower():
                return action_id
        
        # if no action found, return 'forward' action
        return 2
    
    def save_pretrained(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load fine-tuned model."""
        instance = cls(model_name=path, **kwargs)
        return instance


def prepare_training_data(
    trajectories: List[List[Dict]],
    mode: str = "action"
) -> List[Dict[str, any]]:
    """
    Convert trajectories to training examples.
    
    Args:
        trajectories: List of trajs, each episode is list of steps
        mode: 'action' or 'text_action'
    
    Returns:
        List of training examples with 'image', 'prompt', 'target'
    """
    examples = []
    
    for traj in trajectories:
        for step in traj:
            obs = step['obs']  
            action = step['action']
            image = obs['image']
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Prepare target text
            action_name = ACTIONS[action]
            if mode == "action":
                target = f"<{action_name}>"
            else: 
                # TODO use GPT to generate descriptions
                target = f"Move to reach the goal. <{action_name}>"
            
            examples.append({
                "image": image,
                "prompt": None,  # Will use default prompt
                "target": target,
                "action": action
            })
    
    return examples



if __name__ == "__main__":
    print("Testing NanoVLM action predictor...")
    dummy_image = Image.new("RGB", (32, 32), color=(73, 109, 137))
    
    model = NanoVLMActionPredictor(mode="action", use_lora=False, max_img_size = 32, splitted_image_size = 8)  # Use smaller sizes for testing with dummy image
    print(f"Model loaded: {model.model_name}")
    print(f"Mode: {model.mode}")
    print(f"Action tokens: {model.action_token_ids}")
    
    # Test prediction
    action = model.predict_action(dummy_image, return_text=True)
    print(f"Predicted action: {action}")
