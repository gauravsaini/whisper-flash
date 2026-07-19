"""DFlash draft model for Whisper speculative decoding.

A lightweight block-diffusion Transformer that predicts B tokens in a single
parallel forward pass, conditioned on:
  1. Target decoder hidden states (from chosen layers)
  2. Audio summary (mean-pooled encoder output)

The draft model uses the target model's embedding and lm_head for token I/O,
keeping its own parameter count small (~25M).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_target_layer_ids


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WhisperDFlashConfig:
    """Configuration for the Whisper DFlash draft model."""

    # Draft model dimensions
    d_draft: int = 512
    num_layers: int = 5
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.0

    # Target model dimensions (Whisper-large-v3)
    d_target: int = 1280
    num_target_layers: int = 32
    vocab_size: int = 51866
    max_target_positions: int = 448

    # DFlash-specific
    block_size: int = 8
    mask_token_id: int = 50257  # Whisper's <|endoftext|> + 1, or use pad token
    target_layer_ids: Optional[list[int]] = None  # Auto-computed if None
    num_context_layers: int = 3  # Number of target layers to tap

    def __post_init__(self):
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(
                self.num_target_layers, self.num_context_layers
            )


# ---------------------------------------------------------------------------
# Attention layer: prefix-attention for DFlash
# ---------------------------------------------------------------------------

class DFlashAttention(nn.Module):
    """Prefix attention for the DFlash draft model.

    Queries come from the draft block (noise embeddings).
    Keys and values come from [audio_ctx || target_hidden || draft_block],
    allowing each draft position to attend to all context plus the block itself.
    Non-causal within the block (safe because block inputs are mask tokens).
    """

    def __init__(self, config: WhisperDFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_draft = config.d_draft
        self.num_heads = config.num_heads
        self.head_dim = config.d_draft // config.num_heads

        self.q_proj = nn.Linear(config.d_draft, config.d_draft)
        self.k_proj = nn.Linear(config.d_draft, config.d_draft)
        self.v_proj = nn.Linear(config.d_draft, config.d_draft)
        self.out_proj = nn.Linear(config.d_draft, config.d_draft)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        audio_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Draft block sequence, (batch, B, d_draft).
            target_hidden: Projected target context, (batch, ctx_len, d_draft).
            audio_ctx: Projected audio summary, (batch, 1, d_draft).

        Returns:
            Output of shape (batch, B, d_draft).
        """
        bsz, q_len, _ = hidden_states.shape

        # Queries from draft block
        q = self.q_proj(hidden_states)

        # Keys and values from [audio_ctx || target_hidden || hidden_states]
        kv_input = torch.cat([audio_ctx, target_hidden, hidden_states], dim=1)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head attention
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_len = kv_input.shape[1]
        k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (non-causal)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.d_draft)
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DFlashDecoderLayer(nn.Module):
    """Single DFlash decoder layer: attention + LayerNorm + FFN."""

    def __init__(self, config: WhisperDFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.attn_norm = nn.LayerNorm(config.d_draft)
        self.ffn_norm = nn.LayerNorm(config.d_draft)
        self.fc1 = nn.Linear(config.d_draft, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_draft)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        audio_ctx: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, audio_ctx)
        hidden_states = self.dropout(hidden_states) + residual

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Full draft model
# ---------------------------------------------------------------------------

class WhisperDFlashDraftModel(nn.Module):
    """DFlash draft model for Whisper speculative decoding.

    Takes target hidden states and audio summary as context, processes a block
    of noise/mask embeddings, and outputs hidden states that can be projected
    to logits via the target model's lm_head (proj_out).
    """

    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config

        # Projection layers between target dim and draft dim
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.output_proj = nn.Linear(config.d_draft, config.d_target, bias=False)

        # Context projection: concatenated target hidden states → d_draft
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)

        # Audio summary projection: pooled encoder output → d_draft
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)

        # Positional encoding (learned, matching Whisper style)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)

        # Decoder layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_draft)

        # Store layer ids and block size for easy access
        self.target_layer_ids = config.target_layer_ids
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        audio_summary: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            noise_embedding: Target model's token embeddings for the block,
                shape (batch, B, d_target). These are embeddings of
                [accepted_token, mask, mask, ..., mask].
            target_hidden: Concatenated hidden states from tapped target layers,
                shape (batch, ctx_len, num_taps * d_target).
            audio_summary: Mean-pooled encoder output,
                shape (batch, 1, d_target).
            position_ids: Position indices for the block,
                shape (batch, B).

        Returns:
            Hidden states of shape (batch, B, d_target), ready for target's
            proj_out (lm_head) to produce logits.
        """
        # Project noise embedding to draft dimension
        hidden_states = self.input_proj(noise_embedding)

        # Add positional encoding
        hidden_states = hidden_states + self.pos_embed(position_ids)

        # Project context features
        ctx = self.hidden_norm(self.fc(target_hidden))

        # Project audio summary
        audio_ctx = self.audio_proj(audio_summary)

        # Run through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, ctx, audio_ctx)

        # Final norm and project back to target dimension
        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
