"""
Device management utilities for NanoVLM.

Provides utilities for selecting and managing compute devices (CUDA, MPS, CPU)
with proper fallback and optimization strategies.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device_type: str = "auto") -> torch.device:
    """
    Get appropriate compute device with GPU acceleration if available.
    
    Args:
        device_type: Device type to use
            - 'auto': CUDA if available, else MPS if available, else CPU
            - 'cuda': CUDA GPU
            - 'mps': Apple Metal GPU (macOS)
            - 'cpu': CPU only
    
    Returns:
        torch.device instance
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✓ Apple Metal Performance Shaders (MPS) available")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no GPU acceleration available)")
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            logger.info(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
            logger.info("✓ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device(device_type)
        logger.info(f"Using device: {device}")
    
    return device


def setup_device(device: torch.device):
    """
    Setup device-specific configurations and optimizations.
    
    Args:
        device: torch.device instance
    """
    if device.type == "cuda":
        # CUDA-specific optimizations
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("✓ CUDA optimizations enabled (TF32, benchmarking)")
    
    elif device.type == "mps":
        # MPS-specific configurations
        # Fallback for MPS operations not yet supported
        logger.info("✓ MPS device configured")
        logger.info("  Note: Some operations may fallback to CPU if not supported on MPS")
    
    elif device.type == "cpu":
        logger.info("Using CPU - training may be slow")


def get_device_info(device: torch.device) -> str:
    """
    Get human-readable device information.
    
    Args:
        device: torch.device instance
    
    Returns:
        Formatted device information string
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA ({name}, {memory:.1f}GB memory)"
    elif device.type == "mps":
        return "Apple Metal Performance Shaders (MPS)"
    else:
        return "CPU"


def move_to_device(data, device: torch.device):
    """
    Move data to specified device with MPS-aware handling.
    
    Args:
        data: Data to move (tensor, dict of tensors, or nested structure)
        device: Target device
    
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def supports_amp(device: torch.device) -> bool:
    """
    Check if Automatic Mixed Precision (AMP) is supported on device.
    
    Args:
        device: torch.device instance
    
    Returns:
        True if AMP is supported, False otherwise
    """
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # MPS supports AMP in recent PyTorch versions
        return True
    else:
        return False


def print_device_stats(device: torch.device):
    """Print detailed device statistics and capabilities."""
    print(f"\n{'='*60}")
    print(f"Device Information")
    print(f"{'='*60}")
    print(f"Device: {get_device_info(device)}")
    
    if device.type == "cuda":
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        props = torch.cuda.get_device_properties(device)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1e9:.1f}GB")
        print(f"Max Threads per Block: {props.max_threads_per_block}")
    elif device.type == "mps":
        print(f"PyTorch Version: {torch.__version__}")
        print(f"MPS Fallback Enabled: {torch.backends.mps.is_available()}")
    
    if hasattr(torch, "__version__"):
        print(f"PyTorch: {torch.__version__}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test device detection
    print("Testing device detection...\n")
    
    device = get_device("auto")
    setup_device(device)
    print_device_stats(device)
    
    # Test with different device types
    for dev_type in ["cuda", "mps", "cpu"]:
        print(f"\nTesting {dev_type}...")
        try:
            dev = get_device(dev_type)
            print(f"  Device: {dev}")
            setup_device(dev)
        except Exception as e:
            print(f"  Error: {e}")
