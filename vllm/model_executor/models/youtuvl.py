# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
YoutuVL: Native vLLM Unified Multimodal Model
=============================================
Architecture: Siglip2 (vision encoder) + UTUV1 (LLM with MLA attention)

This is a unified multimodal model that combines:
- Vision Encoder: Siglip2 with window attention and 2x2 patch merge
- Language Model: UTUV1 with Multi-head Latent Attention (MLA)

Image Processing:
- Images are converted to patches of size (patch_size * patch_size * 3)
- Patches are arranged in 2x2 merge order for efficient processing
- Each image produces (height/patch_size * width/patch_size) patches
- After 2x2 merge in vision encoder, tokens = patches / 4
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import islice
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear,
    ReplicatedLinear, RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict, MultiModalFieldConfig, MultiModalKwargsItems
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor, BaseProcessingInfo,
    PromptReplacement, PromptUpdateDetails
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader, PPMissingLayer, WeightsMapper,
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix, merge_multimodal_embeddings
)

logger = init_logger(__name__)

# Flash attention import
try:
    from flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None
    apply_rotary_emb = None

# =============================================================================
# Constants
# =============================================================================

DEFAULT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2
DEFAULT_MAX_IMAGE_PATCHES = 4096 * 6 * 2  # 49152 patches max
# SigLIP2 normalization parameters (match Sean's Siglip2ImageProcessorFast)
# Sean uses image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
# with rescale_factor=1/255.
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


# =============================================================================
# Image Processor (Native vLLM implementation)
# =============================================================================

def get_image_size_for_max_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    merge_size: int = 2,
) -> Tuple[int, int]:
    """Calculate target image size that respects max_num_patches limit."""
    effective_patch = patch_size * merge_size
    
    def get_scaled_size(scale: float, size: int) -> int:
        scaled = size * scale
        return max(effective_patch, int(math.ceil(scaled / effective_patch) * effective_patch))
    
    scale = 1.0
    while scale > 0.1:
        target_h = get_scaled_size(scale, image_height)
        target_w = get_scaled_size(scale, image_width)
        num_patches = (target_h // patch_size) * (target_w // patch_size)
        if num_patches <= max_num_patches:
            return target_h, target_w
        scale -= 0.02
    
    # Fallback to minimum size
    return effective_patch, effective_patch


def convert_image_to_patches(
    image: torch.Tensor,
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Convert image tensor to patches arranged in 2x2 merge order.
    
    Args:
        image: [C, H, W] tensor
        patch_size: Size of each patch
        merge_size: Merge size (2 for 2x2 merge)
    
    Returns:
        [num_patches, patch_size * patch_size * C] tensor
    """
    num_channels, image_height, image_width = image.shape
    num_patches_h = image_height // patch_size
    num_patches_w = image_width // patch_size
    
    # Reshape to group patches for 2x2 merge
    patched = image.reshape(
        num_channels,
        num_patches_h // merge_size, merge_size, patch_size,
        num_patches_w // merge_size, merge_size, patch_size
    )
    # Permute to arrange 2x2 blocks together
    patched = patched.permute(1, 4, 2, 5, 3, 6, 0)
    # Flatten to [num_patches, features]
    patched = patched.reshape(num_patches_h * num_patches_w, -1)
    return patched


class YoutuVLImageProcessor:
    """Native vLLM image processor for YoutuVL."""
    
    def __init__(
        self,
        patch_size: int = DEFAULT_PATCH_SIZE,
        merge_size: int = DEFAULT_MERGE_SIZE,
        max_num_patches: int = DEFAULT_MAX_IMAGE_PATCHES,
        image_mean: Tuple[float, ...] = DEFAULT_IMAGE_MEAN,
        image_std: Tuple[float, ...] = DEFAULT_IMAGE_STD,
    ):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_num_patches = max_num_patches
        self.image_mean = torch.tensor(image_mean).view(3, 1, 1)
        self.image_std = torch.tensor(image_std).view(3, 1, 1)
    
    def _to_tensor(self, image) -> torch.Tensor:
        """Convert image to tensor [C, H, W]."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                return image.float()
            elif image.dim() == 3 and image.shape[2] == 3:
                return image.permute(2, 0, 1).float()
        
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).float()
            return image
        
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            return torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def __call__(
        self,
        images,
        max_num_patches: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        """Process images to patches.
        
        Returns:
            BatchFeature with:
            - pixel_values: [total_patches, patch_size*patch_size*3]
            - pixel_attention_mask: [total_patches]
            - spatial_shapes: [num_images, 2] (height, width in patches)
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        if max_num_patches is None:
            max_num_patches = self.max_num_patches
        
        all_patches = []
        all_masks = []
        spatial_shapes = []
        
        for image in images:
            img_tensor = self._to_tensor(image)
            _, orig_h, orig_w = img_tensor.shape
            
            # Calculate target size
            target_h, target_w = get_image_size_for_max_patches(
                orig_h, orig_w, self.patch_size, max_num_patches, self.merge_size
            )
            
            # Resize
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Normalize
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - self.image_mean) / self.image_std
            
            # Convert to patches
            patches = convert_image_to_patches(img_tensor, self.patch_size, self.merge_size)
            mask = torch.ones(patches.shape[0], dtype=torch.int32)
            
            num_patches_h = target_h // self.patch_size
            num_patches_w = target_w // self.patch_size
            
            all_patches.append(patches)
            all_masks.append(mask)
            spatial_shapes.append([num_patches_h, num_patches_w])
        
        pixel_values = torch.cat(all_patches, dim=0)
        pixel_attention_mask = torch.cat(all_masks, dim=0)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long)
        
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )
    
    def get_num_image_tokens(self, image_width: int, image_height: int) -> int:
        """Calculate number of tokens for an image after 2x2 merge."""
        target_h, target_w = get_image_size_for_max_patches(
            image_height, image_width, self.patch_size, self.max_num_patches, self.merge_size
        )
        num_patches = (target_h // self.patch_size) * (target_w // self.patch_size)
        return num_patches // (self.merge_size ** 2)


@lru_cache(maxsize=1)
def _get_youtuvl_image_processor(
    patch_size: int = DEFAULT_PATCH_SIZE,
    merge_size: int = DEFAULT_MERGE_SIZE,
    max_num_patches: int = DEFAULT_MAX_IMAGE_PATCHES,
) -> YoutuVLImageProcessor:
    """Get cached image processor instance."""
    return YoutuVLImageProcessor(
        patch_size=patch_size,
        merge_size=merge_size,
        max_num_patches=max_num_patches,
    )


# =============================================================================
# Vision Encoder Components (Siglip2 architecture)
# =============================================================================

class YoutuVLPatchMerger(nn.Module):
    """Patch merger: merge 2x2 patches into 1 token."""
    
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        # MLP with bias - matches weight file structure
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, bias=True),
        )
    
    def forward(self, x: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Merge patches.
        
        Args:
            x: [num_patches, vision_hidden_size] - patches already in 2x2 order
        
        Returns:
            [num_patches/4, llm_hidden_size]
        """
        x = self.ln_q(x)
        # Reshape to group 4 consecutive patches (already in 2x2 order)
        x = x.view(-1, self.hidden_size)
        x = self.mlp(x)
        return x


class YoutuVLVisionEmbeddings(nn.Module):
    """Vision embeddings without positional encoding."""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        
        if hasattr(config, 'in_features') and config.in_features > 0:
            in_features = config.in_features
        else:
            in_features = config.num_channels * self.patch_size * self.patch_size
        
        self.patch_embedding = nn.Linear(in_features, self.embed_dim)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed patches.
        
        Args:
            pixel_values: [num_patches, patch_size*patch_size*3]
        
        Returns:
            [num_patches, embed_dim]
        """
        target_dtype = self.patch_embedding.weight.dtype
        return self.patch_embedding(pixel_values.to(dtype=target_dtype))


class YoutuVLVisionAttention(nn.Module):
    """Vision attention with flash attention and rotary embeddings."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        
        self.q_proj = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.q_proj")
        self.k_proj = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.k_proj")
        self.v_proj = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.v_proj")
        self.out_proj = ReplicatedLinear(dim, dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.out_proj")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        
        q = q.reshape(seq_length, self.num_heads, self.head_dim)
        k = k.reshape(seq_length, self.num_heads, self.head_dim)
        v = v.reshape(seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings
        # Optimization: apply_rotary_emb's Triton kernel requires same dtype for input and cos/sin.
        # We convert cos/sin to match input dtype to avoid separate float conversion kernels.
        if position_embeddings is not None and apply_rotary_emb is not None:
            cos, sin = position_embeddings
            cos = cos.chunk(2, dim=-1)[0].contiguous()
            sin = sin.chunk(2, dim=-1)[0].contiguous()
            # Convert cos/sin to match q/k dtype (bf16) - Triton kernel requires same dtype
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            q = apply_rotary_emb(q.unsqueeze(0), cos, sin).squeeze(0)
            k = apply_rotary_emb(k.unsqueeze(0), cos, sin).squeeze(0)
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(seq_length, -1)
        
        output, _ = self.out_proj(attn_output)
        return output


class YoutuVLVisionMLP(nn.Module):
    """Vision MLP module."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ReplicatedLinear(config.hidden_size, config.intermediate_size, bias=True, quant_config=quant_config, prefix=f"{prefix}.fc1")
        self.fc2 = ReplicatedLinear(config.intermediate_size, config.hidden_size, bias=True, quant_config=quant_config, prefix=f"{prefix}.fc2")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class FusedLayerNorm(nn.Module):
    """Fused LayerNorm using Triton kernel for better performance.
    
    This avoids separate kernel launches for mean, variance, and normalization.
    Falls back to PyTorch native implementation if Triton is not available.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Try to import Triton LayerNorm
        self._use_triton = False
        try:
            from vllm.model_executor.layers.fla.ops.layernorm_guard import LayerNormFn
            self._LayerNormFn = LayerNormFn
            self._use_triton = True
        except ImportError:
            self._LayerNormFn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_triton and x.is_cuda:
            # Use Triton fused kernel - works directly with bf16
            # LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)
            return self._LayerNormFn.apply(x, self.weight, self.bias, None, self.eps, None, True, False)
        else:
            # Fallback to PyTorch native - also avoids unnecessary float conversion
            return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)


class YoutuVLVisionEncoderLayer(nn.Module):
    """Single vision encoder layer with fused LayerNorm for better performance."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Use fused LayerNorm for better performance (avoids float32 conversion overhead)
        self.layer_norm1 = FusedLayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = YoutuVLVisionAttention(config, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.layer_norm2 = FusedLayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = YoutuVLVisionMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, position_embeddings)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class YoutuVLVisionRoPE(nn.Module):
    """Rotary position embedding for vision."""
    
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class YoutuVLVisionEncoder(nn.Module):
    """Vision encoder with window attention."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            YoutuVLVisionEncoderLayer(config, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        
        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.patch_size = config.patch_size
        self.window_size = self.patch_size * 2 * 8
        
        assert config.hidden_size % (config.num_attention_heads * 2) == 0
        self.rotary_pos_emb = YoutuVLVisionRoPE(config.hidden_size // config.num_attention_heads // 2)
    
    def _compute_rotary_pos_emb(self, spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings."""
        pos_ids = []
        for h, w in spatial_shapes:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
        
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = spatial_shapes.max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
    
    def _get_window_index(self, spatial_shapes: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Compute window indices for window attention."""
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        
        for grid_h, grid_w in spatial_shapes:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            
            index = torch.arange(llm_grid_h * llm_grid_w).reshape(1, llm_grid_h, llm_grid_w)
            pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size
            pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(1, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(1, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size)
            
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (llm_grid_h * llm_grid_w).item()
        
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        
        # Handle empty input case
        if hidden_states.shape[0] == 0:
            return hidden_states
        
        # Validate spatial_shapes
        if spatial_shapes.numel() == 0 or spatial_shapes.dim() < 2:
            logger.warning(f"Invalid spatial_shapes: {spatial_shapes.shape}, returning input unchanged")
            return hidden_states
        
        # Compute rotary position embeddings and window indices
        rotary_pos_emb = self._compute_rotary_pos_emb(spatial_shapes)
        window_index, cu_window_seqlens = self._get_window_index(spatial_shapes)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        # Reshape hidden states for window attention
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        # Prepare position embeddings
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # Compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(spatial_shapes[:, 0] * spatial_shapes[:, 1], 1).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        # Process through encoder layers
        for layer_num, encoder_layer in enumerate(self.layers):
            # Use global attention every 8 layers and at the last layer
            if (1 + layer_num) % 8 == 0 or layer_num == len(self.layers) - 1:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            
            hidden_states = encoder_layer(hidden_states, cu_seqlens_now, position_embeddings)
        
        # Reverse window indexing
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        return hidden_states


class YoutuVLVisionModel(nn.Module):
    """Complete vision model with encoder and patch merger."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "vision_model",
    ):
        super().__init__()
        vision_config = config.vision_config
        self.embeddings = YoutuVLVisionEmbeddings(vision_config)
        self.encoder = YoutuVLVisionEncoder(vision_config, quant_config=quant_config, prefix=f"{prefix}.encoder")
        # Use fused LayerNorm for better performance
        self.post_layernorm = FusedLayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.merger = YoutuVLPatchMerger(
            dim=config.hidden_size,
            context_dim=vision_config.hidden_size,
            spatial_merge_size=2,
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Process image patches through vision encoder.
        
        Args:
            pixel_values: [num_patches, patch_features]
            spatial_shapes: [num_images, 2] (height, width in patches)
        
        Returns:
            [num_tokens, llm_hidden_size] where num_tokens = num_patches / 4
        """
        # Handle empty input case
        if pixel_values.shape[0] == 0 or spatial_shapes.numel() == 0:
            # Return empty tensor with correct hidden size
            return pixel_values.new_empty(0, self.merger.mlp[-1].out_features)
        
        hidden_states = self.embeddings(pixel_values)
        
        hidden_states = self.encoder(hidden_states, spatial_shapes)
        
        hidden_states = self.post_layernorm(hidden_states)
        
        hidden_states = self.merger(hidden_states, spatial_shapes)
        
        return hidden_states


# =============================================================================
# Language Model Components (UTUV1 with MLA attention)
# =============================================================================

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """Get mscale for yarn rope scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YoutuVLMLP(nn.Module):
    """MLP module for language model."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class YoutuVLAttention(nn.Module):
    """MLA attention module for language model."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # Use fused_qkv_a_proj when q_lora_rank is set (weights are merged in load_weights)
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj")
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj")
        
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = False
        self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj if self.q_lora_rank is not None else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if self.q_lora_rank is None else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            is_sparse=self.is_v32,
            topk_indices_buffer=None,
        )

        self.mla_attn = MultiHeadLatentAttention(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class YoutuVLDecoderLayer(nn.Module):
    """Single decoder layer for language model."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: Optional[PretrainedConfig] = None
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = YoutuVLAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn"
        )
        
        self.mlp = YoutuVLMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class YoutuVLLanguageModel(nn.Module):
    """Language model backbone (UTUV1 architecture)."""
    
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        
        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: YoutuVLDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# =============================================================================
# Multimodal Processing
# =============================================================================

class YoutuVLProcessingInfo(BaseProcessingInfo):
    """Processing configuration for YoutuVL."""
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}  # Support multiple images
    
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # Max patches = 49152, after 4x merge = 12288 tokens
        return {"image": DEFAULT_MAX_IMAGE_PATCHES // 4}
    
    def get_hf_processor(self, **kwargs) -> YoutuVLImageProcessor:
        return _get_youtuvl_image_processor()
    
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        processor = self.get_hf_processor()
        return processor.get_num_image_tokens(image_width, image_height)
    
    def get_image_size_with_most_features(self) -> ImageSize:
        # Return a reasonable size for profiling
        # Using 1024x1024 which produces 64x64=4096 patches, 1024 tokens after merge
        return ImageSize(width=1024, height=1024)


class YoutuVLDummyInputsBuilder(BaseDummyInputsBuilder[YoutuVLProcessingInfo]):
    """Build dummy inputs for memory profiling."""
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        # Use the full vision token format to match chat template
        return "<|vision_start|><|image_pad|><|vision_end|>" * num_images
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=target_size.width,
                height=target_size.height,
                num_images=num_images,
            )
        }


class YoutuVLMultiModalProcessor(BaseMultiModalProcessor[YoutuVLProcessingInfo]):
    """Multimodal processor for YoutuVL."""
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process prompt and images."""
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_hf_processor()
        
        # Note: mm_data uses "images" key (plural) from ImageProcessorItems
        # but may also use "image" key (singular) from dummy inputs
        images = mm_data.get("images", mm_data.get("image", []))
        if not isinstance(images, list):
            images = [images]
        
        if not images:
            # No images, just tokenize
            prompt_ids = tokenizer.encode(prompt)
            return BatchFeature(
                dict(input_ids=torch.tensor([prompt_ids], dtype=torch.long)),
                tensor_type="pt"
            )
        
        # Process images
        max_num_patches = mm_kwargs.get("max_image_patches", DEFAULT_MAX_IMAGE_PATCHES)
        image_inputs = image_processor(images, max_num_patches=max_num_patches)
        
        # Expand image tokens in prompt
        # Support both formats:
        # 1. <|vision_start|><|image_pad|><|vision_end|> (Sean's format)
        # 2. <|image_pad|> (simple format)
        image_token = "<|image_pad|>"
        vision_start = "<|vision_start|>"
        vision_end = "<|vision_end|>"
        merge_length = 4
        spatial_shapes = image_inputs["spatial_shapes"]
        
        text = prompt
        for i in range(len(spatial_shapes)):
            num_tokens = int(spatial_shapes[i][0] * spatial_shapes[i][1]) // merge_length
            
            # Try to replace <|vision_start|><|image_pad|><|vision_end|> first
            full_pattern = f"{vision_start}{image_token}{vision_end}"
            if full_pattern in text:
                replacement = f"{vision_start}" + "<|placeholder|>" * num_tokens + f"{vision_end}"
                text = text.replace(full_pattern, replacement, 1)
            else:
                # Fallback to simple <|image_pad|> replacement
                text = text.replace(image_token, "<|placeholder|>" * num_tokens, 1)
        
        text = text.replace("<|placeholder|>", image_token)
        
        # Tokenize - ensure return_tensors="pt" for proper tensor output
        text_inputs = tokenizer(text, return_tensors="pt", **tok_kwargs)
        
        return BatchFeature(data={**text_inputs, **image_inputs})
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Configure how multimodal fields are batched."""
        if "spatial_shapes" not in hf_inputs:
            return {}
        
        spatial_shapes = hf_inputs["spatial_shapes"]
        if not torch.is_tensor(spatial_shapes):
            spatial_shapes = torch.tensor(spatial_shapes)
        
        # Number of patches per image (before merge)
        num_patches_per_image = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(torch.int64)
        
        # Number of tokens per image (after 4x merge)
        merge_length = 4
        num_tokens_per_image = num_patches_per_image // merge_length
        
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", num_patches_per_image),
            "pixel_attention_mask": MultiModalFieldConfig.flat_from_sizes("image", num_patches_per_image),
            "spatial_shapes": MultiModalFieldConfig.batched("image"),
        }
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Define how to replace <|image_pad|> tokens."""
        # Check if there are any images
        if "image" not in out_mm_kwargs or len(out_mm_kwargs["image"]) == 0:
            return []
        
        hf_config = self.info.ctx.get_hf_config()
        image_token_id = getattr(hf_config, "image_token_id", 128264)
        merge_length = 4
        
        def get_replacement(item_idx: int):
            """Get replacement tokens for one image."""
            out_item = out_mm_kwargs["image"][item_idx]
            spatial_shapes = out_item["spatial_shapes"].data
            
            # Fix: spatial_shapes is [[h, w]] for single image, so we need [0][0] and [0][1]
            if isinstance(spatial_shapes, torch.Tensor):
                if spatial_shapes.dim() == 2:
                    h, w = spatial_shapes[0][0].item(), spatial_shapes[0][1].item()
                else:
                    h, w = spatial_shapes[0].item(), spatial_shapes[1].item()
            else:
                h, w = spatial_shapes[0], spatial_shapes[1]
            
            num_tokens = int(h * w) // merge_length
            return [image_token_id] * num_tokens
        
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]
    
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Our processor already expands image tokens
        return True


# =============================================================================
# Main Model
# =============================================================================

@MULTIMODAL_REGISTRY.register_processor(
    YoutuVLMultiModalProcessor,
    info=YoutuVLProcessingInfo,
    dummy_inputs=YoutuVLDummyInputsBuilder,
)
class YoutuVLForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    """
    YoutuVL: Native vLLM unified multimodal model.
    
    Architecture: Siglip2 (vision encoder) + UTUV1 (LLM with MLA attention)
    
    This is a unified model that:
    1. Encodes images through vision_model (Siglip2)
    2. Merges image embeddings with text embeddings
    3. Generates text through language_model (UTUV1)
    """
    
    # Enable new multimodal kwargs merging
    merge_by_field_config = True
    
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        
        # Vision encoder (Siglip2)
        self.vision_model = YoutuVLVisionModel(
            config, 
            quant_config=quant_config, 
            prefix="vision_model"
        )
        
        # Language model backbone (UTUV1)
        self.model = YoutuVLLanguageModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        
        # LM head
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                config.vocab_size,
                logit_scale)
        else:
            self.lm_head = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        
        # Special token IDs
        self.image_token_id = getattr(config, "image_token_id", 128264)
    
    def _parse_and_validate_image_input(
        self, **kwargs
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Parse image inputs from kwargs."""
        pixel_values = kwargs.get("pixel_values")
        spatial_shapes = kwargs.get("spatial_shapes")
        
        if pixel_values is None:
            return None
        
        if spatial_shapes is None:
            logger.warning("spatial_shapes is None but pixel_values is provided")
            return None
        
        # Handle spatial_shapes dimensions
        if isinstance(spatial_shapes, torch.Tensor) and spatial_shapes.dim() == 3:
            spatial_shapes = spatial_shapes.squeeze(1)
        
        return pixel_values, spatial_shapes
    
    def _process_image_input(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Process image input through vision encoder.
        
        Args:
            pixel_values: [total_patches, patch_features]
            spatial_shapes: [num_images, 2]
        
        Returns:
            Tuple of tensors, one per image, each [num_tokens, hidden_size]
        """
        # Ensure correct dtype
        pixel_values = pixel_values.to(dtype=self.vision_model.embeddings.patch_embedding.weight.dtype)
        
        # Get embeddings from vision encoder
        # Output: [total_tokens, hidden_size] where total_tokens = total_patches / 4
        image_embeds = self.vision_model(pixel_values, spatial_shapes)
        
        # Split by image
        merge_length = 4
        num_images = spatial_shapes.shape[0]
        
        if num_images == 1:
            return (image_embeds,)
        
        # Calculate tokens per image
        token_counts = ((spatial_shapes[:, 0] * spatial_shapes[:, 1]) // merge_length).tolist()
        return tuple(image_embeds.split(token_counts))
    
    def get_multimodal_embeddings(self, **kwargs) -> Optional[Tuple[torch.Tensor, ...]]:
        """Get multimodal embeddings from image input.
        
        Returns:
            Tuple of 2D tensors, one per image item.
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        
        pixel_values, spatial_shapes = image_input
        result = self._process_image_input(pixel_values, spatial_shapes)
        return result
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        """Get input embeddings with multimodal fusion."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            total_mm_tokens = sum(t.shape[0] for t in multimodal_embeddings)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.image_token_id,
            )
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass."""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
        
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights with proper mapping.
        
        Handles weights from:
        1. Unified model (vision_model.*, model.*, lm_head.*)
        2. Separate encoder/decoder (siglip2.*, merger.*, model.*, lm_head.*)
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            
            # Map weight names to model parameter names
            mapped_name = name
            
            # Handle legacy encoder weights (siglip2.vision_model.* -> vision_model.*)
            if name.startswith("siglip2.vision_model."):
                mapped_name = name.replace("siglip2.vision_model.", "vision_model.")
            elif name.startswith("siglip2."):
                mapped_name = "vision_model." + name[len("siglip2."):]
            # Handle legacy merger weights (merger.* -> vision_model.merger.*)
            elif name.startswith("merger."):
                mapped_name = "vision_model." + name
            # vision_model.* weights are already correctly named
            
            # Handle KV cache quantization scales
            if self.quant_config is not None:
                scale_name = self.quant_config.get_cache_scale(mapped_name)
                if scale_name:
                    if scale_name in params_dict:
                        param = params_dict[scale_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                        weight_loader(param, loaded_weight)
                        loaded_params.add(scale_name)
                    continue
            
            # Remap FP8 kv-scale names
            if "scale" in mapped_name:
                remapped_name = maybe_remap_kv_scale_name(mapped_name, params_dict)
                if remapped_name is None:
                    continue
                mapped_name = remapped_name
            
            # Handle stacked parameters (gate_up_proj, fused_qkv_a_proj)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                stacked_name = mapped_name.replace(weight_name, param_name)
                
                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue
                
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                
                if stacked_name in params_dict:
                    param = params_dict[stacked_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(stacked_name)
                    is_stacked = True
                break
            
            if is_stacked:
                continue
            
            # Regular parameter loading
            if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                continue
            
            if is_pp_missing_parameter(mapped_name, self):
                continue
            
            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
        # Debug: log loading summary
        vision_loaded = len([p for p in loaded_params if 'vision_model' in p])
        model_loaded = len([p for p in loaded_params if p.startswith('model.')])
        return loaded_params


# =============================================================================
# Layout Parser for Two-Stage Decoding
# =============================================================================

import json
import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class LayoutElement:
    """Layout element data class."""
    type: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "bbox": {"x1": self.bbox[0], "y1": self.bbox[1],
                     "x2": self.bbox[2], "y2": self.bbox[3]},
            "text": self.text
        }


class YoutuVLLayoutParser:
    """YoutuVL Layout Parser for two-stage decoding.
    
    Parses model output layout strings in format:
    <x_10><y_20><x_100><y_50><LAYOUT_TEXT>
    <x_200><y_30><x_400><y_150><LAYOUT_TABLE>
    """

    # Regex pattern for layout output（放宽空白与分隔符）
    LAYOUT_PATTERN = re.compile(
        r"<x_\s*(\d+)\s*>\s*<y_\s*(\d+)\s*>\s*<x_\s*(\d+)\s*>\s*<y_\s*(\d+)\s*>\s*<\s*(LAYOUT_\w+)\s*>"
    )

    # Supported layout types
    LAYOUT_TYPES = {
        "LAYOUT_TEXT", "LAYOUT_TITLE", "LAYOUT_HEADER",
        "LAYOUT_FOOTER", "LAYOUT_FIGURE", "LAYOUT_FORMULA",
        "LAYOUT_TABLE", "LAYOUT_CODE", "LAYOUT_CAPTION"
    }

    # Type mapping for OCR (some types use TEXT recognition)
    TYPE_MAPPING_FOR_OCR = {
        "LAYOUT_CODE": "LAYOUT_TEXT",
        "LAYOUT_FOOTER": "LAYOUT_TEXT",
        "LAYOUT_FORMULA": "LAYOUT_TEXT",
        "LAYOUT_HEADER": "LAYOUT_TEXT",
        "LAYOUT_CAPTION": "LAYOUT_TEXT",
    }

    # Prompt templates（与原始 SDK 保持一致）
    # 原始 SDK infer_vllm_async.py 第 248 行
    LAYOUT_PROMPT = '检测文档中的所有布局元素，并将其分类为：<LAYOUT_TEXT>、<LAYOUT_TITLE>、<LAYOUT_HEADER>、<LAYOUT_FOOTER>、<LAYOUT_FIGURE>、<LAYOUT_FORMULA>、<LAYOUT_TABLE>、<LAYOUT_CODE>、<LAYOUT_CAPTION>。以“\n”区分各个区域边界'
    
    # OCR prompt prefix（原始 SDK: '根据输入框给出对应的文字内容：'）
    OCR_PROMPT_PREFIX = "根据输入框给出对应的文字内容："

    @classmethod
    def parse_layout_output(cls, output: str) -> List[LayoutElement]:
        """Parse model layout output string.
        
        Args:
            output: Raw model output string
            
        Returns:
            List of parsed layout elements
        """
        elements = []
        if not output:
            return elements

        # 1) 直接匹配宽松正则
        for match in cls.LAYOUT_PATTERN.finditer(output):
            x1, y1, x2, y2, label = match.groups()
            if label in cls.LAYOUT_TYPES:
                elements.append(LayoutElement(
                    type=label,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                ))

        # 2) 若未匹配到，尝试按管道或换行切分再匹配（模型可能输出分隔符）
        if not elements:
            parts = re.split(r"[|\n]+", output)
            for part in parts:
                match = cls.LAYOUT_PATTERN.search(part)
                if match:
                    x1, y1, x2, y2, label = match.groups()
                    if label in cls.LAYOUT_TYPES:
                        elements.append(LayoutElement(
                            type=label,
                            bbox=(int(x1), int(y1), int(x2), int(y2))
                        ))

        # 3) 若仍为空，尝试解析 JSON 数组 [{"type":..., "bbox":[x1,y1,x2,y2]}]
        if not elements:
            try:
                data = json.loads(output)
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        label = item.get("type")
                        bbox = item.get("bbox", [])
                        if label in cls.LAYOUT_TYPES and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            elements.append(LayoutElement(
                                type=label,
                                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                            ))
            except Exception:
                pass

        return elements

    @classmethod
    def format_bbox_query(cls, element: LayoutElement, for_ocr: bool = True) -> str:
        """Format single bbox query string.
        
        Args:
            element: Layout element
            for_ocr: Whether for OCR (applies type mapping)
            
        Returns:
            Formatted query string like <x_10><y_20><x_100><y_50><LAYOUT_TEXT>
        """
        x1, y1, x2, y2 = element.bbox
        layout_type = element.type
        if for_ocr:
            layout_type = cls.TYPE_MAPPING_FOR_OCR.get(layout_type, layout_type)
        return f"<x_{x1}><y_{y1}><x_{x2}><y_{y2}><{layout_type}>"

    @classmethod
    def format_ocr_prompt(cls, elements: List[LayoutElement]) -> str:
        """Format OCR query prompt.
        
        Args:
            elements: List of layout elements
            
        Returns:
            Complete OCR query prompt
        """
        if len(elements) == 1:
            return cls.OCR_PROMPT_PREFIX + cls.format_bbox_query(elements[0])
        queries = [cls.format_bbox_query(e) for e in elements]
        return cls.OCR_PROMPT_PREFIX + "|".join(queries)

    @classmethod
    def parse_ocr_output(cls, output: str, num_elements: int) -> List[str]:
        """Parse OCR output (may be batched).
        
        Args:
            output: Model OCR output
            num_elements: Expected number of elements
            
        Returns:
            List of text for each element
        """
        if num_elements == 1:
            return [output.strip()]

        # 严格对齐 Sean 的 SDK：批量 OCR 仅按 "<sep>" 分隔。
        # 如果模型没有输出 "<sep>"，则会导致结果不足，按空串补齐（与 Sean 一致）。
        texts = output.split("<sep>")

        while len(texts) < num_elements:
            texts.append("")
        return [t.strip() for t in texts[:num_elements]]

    @classmethod
    def filter_by_types(
        cls,
        elements: List[LayoutElement],
        types: Optional[List[str]]
    ) -> List[LayoutElement]:
        """Filter elements by type.
        
        Args:
            elements: List of layout elements
            types: Types to keep, None means keep all
            
        Returns:
            Filtered element list
        """
        if not types:
            return elements
        type_set = set(types)
        return [e for e in elements if e.type in type_set]

    @classmethod
    def sort_elements(cls, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Sort elements by reading order (top to bottom, left to right).
        
        Args:
            elements: List of layout elements
            
        Returns:
            Sorted element list
        """
        return sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))