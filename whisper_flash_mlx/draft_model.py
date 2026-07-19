"""DFlash draft model for Whisper speculative decoding — MLX version.

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

import mlx.core as mx
import mlx.nn as nn

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
        self.d_draft = config.d_draft
        self.num_heads = config.num_heads
        self.head_dim = config.d_draft // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_draft, config.d_draft)
        self.k_proj = nn.Linear(config.d_draft, config.d_draft)
        self.v_proj = nn.Linear(config.d_draft, config.d_draft)
        self.out_proj = nn.Linear(config.d_draft, config.d_draft)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        audio_ctx: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: Draft block sequence, (batch, B, d_draft).
            target_hidden: Projected target context, (batch, ctx_len, d_draft).
            audio_ctx: Projected audio summary, (batch, 1, d_draft).
            mask: Optional attention mask.

        Returns:
            Output of shape (batch, B, d_draft).
        """
        bsz, q_len, _ = hidden_states.shape

        # Queries from draft block
        q = self.q_proj(hidden_states)

        # Keys and values from [audio_ctx || target_hidden || hidden_states]
        kv_input = mx.concatenate([audio_ctx, target_hidden, hidden_states], axis=1)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head attention
        kv_len = kv_input.shape[1]
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        attn_output = weights @ v

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.d_draft)
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

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        audio_ctx: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, audio_ctx, mask=mask)
        hidden_states = hidden_states + residual

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Full draft model
# ---------------------------------------------------------------------------

class WhisperDFlashDraftModel(nn.Module):
    """DFlash draft model for Whisper speculative decoding (MLX).

    Takes target hidden states and audio summary as context, processes a block
    of noise/mask embeddings, and outputs hidden states that can be projected
    to logits via the target model's lm_head (token_embedding.as_linear).
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
        self.layers = [
            DFlashDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_draft)

        # Store layer ids and block size for easy access
        self.target_layer_ids = config.target_layer_ids
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        audio_summary: mx.array,
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
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
            mask: Optional attention mask, shape (batch, 1, B, kv_len).

        Returns:
            Hidden states of shape (batch, B, d_target), ready for target's
            token_embedding.as_linear to produce logits.
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
            hidden_states = layer(hidden_states, ctx, audio_ctx, mask=mask)

        # Final norm and project back to target dimension
        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states

    def count_params(self) -> int:
        """Return total number of parameters."""
        from mlx.utils import tree_flatten
        nparams = sum(
            x.size for k, x in tree_flatten(self.parameters())
        )
        return nparams

class BottleneckFreeDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        
        # Context consists of:
        # 1. Target hidden states from tapped layers (d_target * num_taps)
        # 2. Audio summary (d_target)
        # We project them down to d_draft
        num_taps = len(config.target_layer_ids)
        self.ctx_proj = nn.Linear(num_taps * config.d_target + config.d_target, config.d_draft)
        
        # We want to predict `block_size` continuous states.
        # We can use a simple MLP that maps d_draft -> block_size * d_draft
        self.mlp = nn.Sequential(
            nn.Linear(config.d_draft, config.d_draft * 2),
            nn.GELU(),
            nn.Linear(config.d_draft * 2, config.block_size * config.d_draft)
        )
        
        self.output_proj = nn.Linear(config.d_draft, config.d_target, bias=False)
        self.norm = nn.LayerNorm(config.d_draft)
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id
        
    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array, # (bsz, seq_len, taps * d_target)
        audio_summary: mx.array, # (bsz, 1, d_target)
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # For an MLP chunk predictor, we only need the LAST step of the target_hidden!
        # target_hidden is shape (bsz, seq_len, taps*d_target)
        last_target = target_hidden[:, -1:, :]
        
        # audio_summary is (bsz, 1, d_target)
        ctx_input = mx.concatenate([last_target, audio_summary], axis=-1)
        
        # Project down
        hidden = self.ctx_proj(ctx_input) # (bsz, 1, d_draft)
        
        # Predict block_size states
        predicted_states = self.mlp(hidden) # (bsz, 1, block_size * d_draft)
        
        # Reshape to (bsz, block_size, d_draft)
        bsz = predicted_states.shape[0]
        predicted_states = predicted_states.reshape(bsz, self.config.block_size, self.config.d_draft)
        
        predicted_states = self.norm(predicted_states)
        predicted_states = self.output_proj(predicted_states) # (bsz, block_size, d_target)
        
        return predicted_states


class ContinuousDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        
        # We don't need input embedding projection because we input continuous noise/states
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # KEY DIFFERENCE: Instead of outputting to vocab, we output to target hidden dimension
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)
        
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
            
        x = self.norm(x)
        # Directly predict the target hidden state
        predicted_hidden = self.continuous_head(x) 
        return predicted_hidden

class LayerSkipDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        
        # Context consists of:
        # 1. Target hidden states from tapped layers (d_target * num_taps)
        # We DO NOT use audio summary (no cross-attention path) to make it pure layer skip
        num_taps = len(config.target_layer_ids)
        self.ctx_proj = nn.Linear(num_taps * config.d_target, config.d_draft)
        
        # Simple MLP that maps the early layer state to future final layer states
        self.mlp = nn.Sequential(
            nn.Linear(config.d_draft, config.d_draft * 2),
            nn.GELU(),
            nn.Linear(config.d_draft * 2, config.block_size * config.d_draft)
        )
        
        self.output_proj = nn.Linear(config.d_draft, config.d_target, bias=False)
        self.norm = nn.LayerNorm(config.d_draft)
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id
        
    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array, # (bsz, seq_len, taps * d_target)
        audio_summary: mx.array, # (bsz, 1, d_target)
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # target_hidden is shape (bsz, seq_len, taps*d_target)
        last_target = target_hidden[:, -1:, :]
        
        # Project down
        hidden = self.ctx_proj(last_target) # (bsz, 1, d_draft)
        
        # Predict block_size states
        predicted_states = self.mlp(hidden) # (bsz, 1, block_size * d_draft)
        
        # Reshape to (bsz, block_size, d_draft)
        bsz = predicted_states.shape[0]
        predicted_states = predicted_states.reshape(bsz, self.config.block_size, self.config.d_draft)
        
        predicted_states = self.norm(predicted_states)
        predicted_states = self.output_proj(predicted_states) # (bsz, block_size, d_target)
        
        return predicted_states

