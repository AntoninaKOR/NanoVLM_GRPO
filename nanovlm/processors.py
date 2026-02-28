import torchvision.transforms as transforms
from transformers import AutoTokenizer
import logging

from custom_transforms import DynamicResize, SplitImage, GlobalAndSplitImages

logger = logging.getLogger(__name__)

# Cache tokenizers to avoid reloading
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str, extra_special_tokens=None, chat_template=None, trust_remote_code=True):
    """Get or load a tokenizer from cache. """
    cache_key = (model_name, tuple(extra_special_tokens.items() if isinstance(extra_special_tokens, dict) else (extra_special_tokens or [])), chat_template)
    
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]
    
    try:
        tokenizer_kwargs = {
            "use_fast": True,
            "trust_remote_code": trust_remote_code
        }
        if extra_special_tokens:
            tokenizer_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template:
            tokenizer_kwargs["chat_template"] = chat_template
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token for {model_name}")
        
        _TOKENIZER_CACHE[cache_key] = tokenizer
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise ValueError(f"Cannot load tokenizer from {model_name}") from e


def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False):
    """Create a custom image processor that resizes and splits images into patches.
    
    Pipeline:
        1. DynamicResize: Resize image such that longer side ≤ max_img_size and divisible by patch_size
        2. ToTensor: Convert PIL Image to torch tensor in [0, 1]
        3. GlobalAndSplitImages: Generate global patch + individual patches
    
    Args:
        max_img_size: Maximum side length for resizing
        splitted_image_size: Patch size for image splitting
        resize_to_max_side_len: If True, always resize to max_img_size; if False, resize to smaller of actual/max
    
    Returns:
        Transform pipeline that processes images and returns (patches, grid_size)
        - patches: (N, 3, patch_size, patch_size) tensor where N = 1 + n_h*n_w if grid_size != (1,1) else 1
        - grid_size: (n_h, n_w) tuple indicating patch grid dimensions
    """
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])


def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    """
    Generate image token string for multimodal model input.

    Format example for a 2x2 grid with mp_image_token_length=256:
        <image: 0><global_image_token><image_token * 256><r1c1><image_token * 256><r1c2><image_token * 256>
        ...
    
    Args:
        tokenizer: Tokenizer with image tokens defined (must have attributes like:
                   global_image_token, image_token, r1c1, r1c2, etc.)
        splitted_image_counts: List of tuples (n_h, n_w) for each image's patch grid dimensions
        mp_image_token_length: Number of times to repeat image_token for each patch
    
    Returns:
        str: Concatenated image token string  
    """
    image_string = ""
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        
        # Add global image token if tokenizer supports it
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            
            # Skip individual patches if single patch
            if n_h == 1 and n_w == 1:
                continue
        
        # Add region tokens for each patch
        for i in range(n_h):
            for j in range(n_w):
                region_token_name = f'r{i+1}c{j+1}'
                if not hasattr(tokenizer, region_token_name):
                    raise AttributeError(f"Tokenizer missing required token: {region_token_name}")
                
                image_string += getattr(tokenizer, region_token_name)
                image_string += tokenizer.image_token * mp_image_token_length
    
    return image_string
