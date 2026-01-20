# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YoutuVL model configuration for vLLM native multimodal support."""

from typing import Optional

from transformers import AutoConfig, PretrainedConfig


class Siglip2VisionConfig(PretrainedConfig):
    """Siglip2 Vision Encoder configuration."""
    
    model_type = "siglip2_vision_model"
    
    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        num_patches: int = 4096,
        patch_size: int = 16,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        out_hidden_size: int = 2048,
        vision_use_head: bool = False,
        tokens_per_second: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.out_hidden_size = out_hidden_size
        self.vision_use_head = vision_use_head
        self.tokens_per_second = tokens_per_second


class YoutuVLConfig(PretrainedConfig):
    """
    YoutuVL unified multimodal model configuration.
    
    Combines Siglip2 vision encoder with UTUV1 language model (MLA attention).
    """
    
    model_type = "youtu_vl"
    sub_configs = {"vision_config": Siglip2VisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        # Language model config
        vocab_size: int = 182641,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 16384,
        rms_norm_eps: float = 1e-6,
        # MLA parameters
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        rope_theta: float = 100000.0,
        rope_scaling: Optional[dict] = None,
        rope_interleave: bool = True,
        # Multimodal config
        vision_config: Optional[dict] = None,
        image_token_id: int = 128264,
        vision_start_token_id: int = 128262,
        vision_end_token_id: int = 128263,
        video_token_id: int = 128265,
        # Other config
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        mlp_bias: bool = False,
        use_cache: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        **kwargs,
    ):
        # Initialize vision config
        if isinstance(vision_config, dict):
            self.vision_config = Siglip2VisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Siglip2VisionConfig()
        else:
            self.vision_config = vision_config
        
        # Language model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        
        # MLA parameters
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        
        # Multimodal parameters
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.video_token_id = video_token_id
        
        # Other parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.mlp_bias = mlp_bias
        self.use_cache = use_cache
        
        # Validate rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register custom config with AutoConfig
AutoConfig.register("youtu_vl", YoutuVLConfig)