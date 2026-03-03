"""
Vision Transformer (ViT) encoder for NanoVLM.

Migrated from example_nanovlm/models/vision_transformer.py and modality_projector.py.
Uses SigLIP-compatible architecture with proper pretrained weight loading.

Pipeline: ViT(image) -> [B, num_patches, D_vit] -> ModalityProjector -> [B, mp_image_token_length, D_lm]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ViTConfig:
    """ViT configuration. Populated from SiglipVisionConfig when loading pretrained."""
    hidden_dim: int = 768
    inter_dim: int = 3072
    patch_size: int = 16
    img_size: int = 512
    n_heads: int = 12
    dropout: float = 0.0
    n_blocks: int = 12
    ln_eps: float = 1e-6
    cls_flag: bool = False
    model_type: str = "google/siglip2-base-patch16-512"


# --- ViT Building Blocks (from example_nanovlm) ---

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.cls_flag
        self.embd_dim = cfg.hidden_dim

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Interpolate position embeddings if input resolution differs from pretrained
        pos_embd = self.position_embedding
        if pos_embd.size(1) != x.size(1):
            # Reshape to 2D grid, interpolate, flatten back
            cls_offset = 1 if self.cls_flag else 0
            patch_pos = pos_embd[:, cls_offset:, :]  # [1, N_pretrain, D]
            grid_size_pretrain = int(math.sqrt(patch_pos.size(1)))
            grid_size_actual = int(math.sqrt(x.size(1) - cls_offset))
            patch_pos = patch_pos.reshape(1, grid_size_pretrain, grid_size_pretrain, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(grid_size_actual, grid_size_actual), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size_actual * grid_size_actual, -1)
            if self.cls_flag:
                pos_embd = torch.cat([pos_embd[:, :1, :], patch_pos], dim=1)
            else:
                pos_embd = patch_pos

        x = x + pos_embd
        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.embd_dim = cfg.hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class ViTMLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.inter_dim)
        self.fc2 = nn.Linear(cfg.inter_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer encoder (SigLIP-compatible architecture).

    Input:  (B, 3, img_size, img_size)
    Output: (B, num_patches, hidden_dim) when cls_flag=False
            (B, hidden_dim)              when cls_flag=True (CLS token only)
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.cls_flag
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
        return x

    @classmethod
    def from_pretrained(cls, vit_model_type: str = "google/siglip2-base-patch16-512") -> "ViT":
        """Load pretrained SigLIP weights into our ViT architecture."""
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(vit_model_type)

        cfg = ViTConfig(
            dropout=hf_config.attention_dropout,
            hidden_dim=hf_config.hidden_size,
            img_size=hf_config.image_size,
            inter_dim=hf_config.intermediate_size,
            ln_eps=hf_config.layer_norm_eps,
            n_heads=hf_config.num_attention_heads,
            n_blocks=hf_config.num_hidden_layers,
            patch_size=hf_config.patch_size,
            cls_flag=False,
            model_type=vit_model_type,
        )
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=vit_model_type, filename="model.safetensors")

        sd = model.state_dict()

        # Build weight mapping: HuggingFace key -> our key
        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }
        for i in range(cfg.n_blocks):
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = f"blocks.{i}.ln1.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = f"blocks.{i}.ln1.bias"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = f"blocks.{i}.ln2.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = f"blocks.{i}.ln2.bias"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = f"blocks.{i}.attn.out_proj.weight"
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = f"blocks.{i}.attn.out_proj.bias"

        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                assert hf_key in f.keys(), f"Key {hf_key} not found in safetensors file"
                assert our_key in sd, f"Key {our_key} not found in model state dict"
                tensor = f.get_tensor(hf_key)
                if tensor.shape == sd[our_key].shape:
                    sd[our_key].copy_(tensor)
                elif "position_embedding" in hf_key:
                    sd[our_key].copy_(tensor.unsqueeze(0))
                else:
                    raise ValueError(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")

            # QKV weights are separate in HuggingFace but concatenated in our model
            for i in range(cfg.n_blocks):
                q_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight")
                k_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight")
                v_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight")
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(torch.cat((q_weight, k_weight, v_weight), dim=0))

                q_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias")
                k_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias")
                v_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias")
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(torch.cat((q_bias, k_bias, v_bias), dim=0))

        model.load_state_dict(sd)
        logger.info(f"Loaded {vit_model_type} weights ({sum(p.numel() for p in model.parameters()):,} params)")
        return model


# --- Modality Projector (from example_nanovlm) ---

class ModalityProjector(nn.Module):
    """Project vision features to language model space via pixel shuffle + linear.

    Pixel shuffle compresses spatial dimensions while expanding channels:
        (B, H*W, C) -> (B, H/f * W/f, C * f^2) -> linear -> (B, mp_token_len, D_lm)

    Input:  (B, num_patches, vision_hidden_dim)
    Output: (B, num_patches // pixel_shuffle_factor^2, language_hidden_dim)
    
    Note: num_patches must be divisible by (pixel_shuffle_factor^2).
          Validation should happen at model initialization.
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        language_hidden_size: int = 576,
        pixel_shuffle_factor: int = 4,
    ):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.language_hidden_size = language_hidden_size
        self.scale_factor = pixel_shuffle_factor
        self.scale_factor_sq = pixel_shuffle_factor ** 2

        # Projection layer is created lazily since input dim depends on scale factor
        self._proj = None
        self._proj_input_dim = None

    def _compute_effective_scale_factor(self, num_patches: int) -> int:
        """Compute largest scale factor that divides num_patches evenly.
        
        Falls back to smaller factors if requested scale_factor is too large (e.g., in test configs).
        """
        # Try requested factor first
        if num_patches % self.scale_factor_sq == 0:
            return self.scale_factor
        
        # Find largest factor that works
        for factor in range(self.scale_factor - 1, 0, -1):
            if num_patches % (factor ** 2) == 0:
                return factor
        
        # No shuffling possible
        logger.warning(
            f"Cannot apply pixel shuffle: num_patches={num_patches} has no suitable "
            f"divisors. Skipping shuffle (scale_factor=1)."
        )
        return 1

    def _get_proj(self, input_dim: int, device, dtype) -> nn.Linear:
        """Lazily create or recreate projection layer matching actual input dim."""
        if self._proj is None or self._proj_input_dim != input_dim:
            self._proj = nn.Linear(input_dim, self.language_hidden_size, bias=False).to(device=device, dtype=dtype)
            nn.init.normal_(self._proj.weight, mean=0.0, std=0.02)
            self._proj_input_dim = input_dim
        return self._proj

    def output_token_count(self, num_patches: int) -> int:
        """Compute how many tokens the projector outputs for a given number of input patches."""
        effective_factor = self._compute_effective_scale_factor(num_patches)
        return num_patches // (effective_factor ** 2)

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H*W, C) -> (B, H/f * W/f, C * f^2).
        
        Adaptively reduces scale factor if needed for small sequences.
        """
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq ** 0.5)
        assert seq_root ** 2 == seq, f"Sequence length {seq} must be a perfect square"
        
        factor_root = self._compute_effective_scale_factor(seq)
        
        if factor_root <= 1:
            return x  # No shuffling possible

        height = width = seq_root
        
        if height % factor_root != 0 or width % factor_root != 0:
            raise ValueError(
                f"Grid dimensions ({height}x{width}) must be divisible by "
                f"scale_factor ({factor_root})."
            )

        x = x.view(bsz, height, width, embed_dim)
        h_out = height // factor_root
        w_out = width // factor_root

        x = x.reshape(bsz, h_out, factor_root, w_out, factor_root, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * factor_root ** 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(x)
        proj = self._get_proj(x.size(-1), x.device, x.dtype)
        x = proj(x)
        return x
