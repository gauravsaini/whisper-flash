"""Wrapper around the mlx-whisper Whisper model for DFlash integration.

The standard mlx-whisper TextDecoder doesn't expose intermediate hidden states.
This module provides a thin wrapper that runs the decoder layer-by-layer and
collects the hidden states needed by the draft model, without duplicating
the full model code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Re-export for convenience
from mlx_whisper.load_models import load_model as _load_mlx_whisper
from mlx_whisper.whisper import Whisper, ModelDimensions
import mlx_whisper.whisper as whisper_module

# --- Monkey-patch mlx-whisper's qkv_attention to support speculative verification ---
# mlx-whisper assumes q_len == kv_len (prefill) or q_len == 1 (decode). 
# For speculative verification, we pass q_len = B > 1 with an existing kv_cache.
original_qkv_attention = whisper_module.MultiHeadAttention.qkv_attention

def patched_qkv_attention(self, q, k, v, mask=None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
    k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
    v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

    qk = q @ k
    if mask is not None:
        kv_len = k.shape[-1]
        if kv_len > n_ctx:
            offset = kv_len - n_ctx
            causal = mask[:n_ctx, :n_ctx]
            pad = mx.zeros((n_ctx, offset), dtype=causal.dtype)
            full_mask = mx.concatenate([pad, causal], axis=1)
            qk = qk + full_mask
        else:
            qk = qk + mask[:n_ctx, :n_ctx]

    w = mx.softmax(qk, axis=-1, precise=True)
    out = (w @ v).transpose(0, 2, 1, 3)
    out = out.reshape(n_batch, n_ctx, n_state)
    return out, qk

whisper_module.MultiHeadAttention.qkv_attention = patched_qkv_attention
# ----------------------------------------------------------------------------------



def load_target_model(
    path_or_hf_repo: str = "mlx-community/whisper-large-v3-mlx",
    dtype: mx.Dtype = mx.float16,
) -> Whisper:
    """Load the frozen mlx-whisper target model.

    Args:
        path_or_hf_repo: HuggingFace repo id or local path. Recommended MLX
            models from mlx-community: whisper-large-v3-mlx, etc.
        dtype: Model dtype (float16 recommended for Apple Silicon).

    Returns:
        A frozen mlx_whisper.Whisper model.
    """
    model = _load_mlx_whisper(path_or_hf_repo, dtype=dtype)
    model.eval()
    # Freeze all parameters
    model.freeze()
    return model


def encoder_forward(model: Whisper, mel: mx.array) -> mx.array:
    """Run the Whisper encoder and return hidden states.

    Args:
        model: The frozen Whisper model.
        mel: Mel spectrogram, shape (batch, n_frames, n_mels) or (n_frames, n_mels).
            mlx-whisper uses (frames, mels) layout. Pad to 3000 frames for 30s.

    Returns:
        Encoder hidden states, shape (batch, T_enc, d_model).
    """
    if mel.ndim == 2:
        mel = mel[None]
    return model.encoder(mel)



def decoder_forward_with_hidden_states(
    model: Whisper,
    tokens: mx.array,
    audio_features: mx.array,
    kv_cache: Optional[list] = None,
    collect_hidden_states: bool = True,
    return_cross_attention: bool = False,
    offset: Optional[int] = None,
) -> tuple:
    """Run the Whisper decoder and optionally collect per-layer hidden states and cross-attentions.

    This reimplements the TextDecoder forward pass to intercept hidden states
    at each layer, which the standard mlx-whisper code doesn't expose.

    Args:
        model: The frozen Whisper model.
        tokens: Token ids, shape (batch, seq_len).
        audio_features: Encoder output, shape (batch, T_enc, d_model).
        kv_cache: Optional KV cache from previous steps.
        collect_hidden_states: Whether to collect per-layer hidden states.
        return_cross_attention: Whether to return cross-attention weights.

    Returns:
        Tuple of:
        - logits: shape (batch, seq_len, vocab_size)
        - kv_cache: Updated KV cache
        - hidden_states: List of hidden states (embedding + per-layer outputs)
            if collect_hidden_states is True, else empty list.
        - cross_attns: List of cross-attention weights, shape (batch, heads, n_ctx, T_enc)
            for each layer, returned only if return_cross_attention is True.
    """
    decoder = model.decoder

    # Compute offset from KV cache (true token position). When the self-attn
    # cache has been compressed (entries merged), its length no longer equals
    # the true token position, so an explicit offset must be supplied.
    if offset is None:
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0

    # Embedding + positional encoding
    x = (
        decoder.token_embedding(tokens)
        + decoder.positional_embedding[offset: offset + tokens.shape[-1]]
    )

    hidden_states = []
    if collect_hidden_states:
        hidden_states.append(x)  # Index 0: embedding output

    cross_attns = []

    if kv_cache is None:
        kv_cache = [None] * len(decoder.blocks)

    # Run through decoder blocks, collecting hidden states
    for e, block in enumerate(decoder.blocks):
        x, kv_cache[e], attn = block(
            x, audio_features, mask=decoder._mask, kv_cache=kv_cache[e]
        )
        if collect_hidden_states:
            hidden_states.append(x)  # Index e+1: layer e output
        if return_cross_attention:
            cross_attns.append(attn)

    x = decoder.ln(x)

    # Project to logits using tied embedding weights
    logits = decoder.token_embedding.as_linear(x)

    if return_cross_attention:
        return logits, kv_cache, hidden_states, cross_attns
    return logits, kv_cache, hidden_states


def get_token_embedding(model: Whisper, token_ids: mx.array) -> mx.array:
    """Get token embeddings from the target model's embedding table.

    Args:
        model: The Whisper model.
        token_ids: Token ids, shape (batch, seq_len).

    Returns:
        Embeddings of shape (batch, seq_len, d_model).
    """
    return model.decoder.token_embedding(token_ids)


def project_to_logits(model: Whisper, hidden_states: mx.array) -> mx.array:
    """Project hidden states to logits using the target model's lm_head.

    The mlx-whisper model uses tied weights: token_embedding.as_linear().

    Args:
        model: The Whisper model.
        hidden_states: Shape (batch, seq_len, d_model).

    Returns:
        Logits of shape (batch, seq_len, vocab_size).
    """
    return model.decoder.token_embedding.as_linear(hidden_states)


def select_encoder_frames(
    cross_attn_weights: list[mx.array],
    margin: int = 25,
    min_window: int = 30,
    max_window: int = 200,
) -> tuple[int, int]:
    """Select encoder frame range from cross-attention weights (peak + margin).

    Args:
        cross_attn_weights: List of cross-attention score arrays from each decoder
            layer, shape (batch, heads, seq_len, T_enc). We use the last layer's
            last-query-position scores, averaged over heads.
        margin: Number of frames to include on each side of the peak.
        min_window: Minimum window size (overrides margin if needed).
        max_window: Maximum window size.

    Returns:
        (start_frame, end_frame) 1-indexed slice indices into encoder_hidden.
    """
    attn = cross_attn_weights[-1]  # (batch, heads, seq_len, T_enc)
    profile = attn[:, :, -1:, :].mean(axis=(0, 1, 2))  # (T_enc,)
    peak_idx = mx.argmax(profile).item()
    T_enc = profile.shape[0]

    start = max(0, peak_idx - margin)
    end = min(T_enc, peak_idx + margin)

    if end - start < min_window:
        mid = (start + end) // 2
        half = min_window // 2
        start = max(0, mid - half)
        end = min(T_enc, start + min_window)

    if end - start > max_window:
        mid = (start + end) // 2
        half = max_window // 2
        start = max(0, mid - half)
        end = min(T_enc, start + max_window)

    return start, end


def slice_cross_attention_cache(
    kv_cache: list,
    start_frame: int,
    end_frame: int,
) -> list:
    """Slice cross-attention K/V cache to a window of encoder frames.

    Self-attention cache is returned unchanged.  Each layer's cross-attn
    entry ``(k, v)`` has shape ``(1, T_enc, d_model)`` and is sliced along
    axis 1.
    """
    new_cache = []
    for self_kv, cross_kv in kv_cache:
        if cross_kv is not None:
            k, v = cross_kv
            cross_kv = (k[:, start_frame:end_frame, :], v[:, start_frame:end_frame, :])
        new_cache.append((self_kv, cross_kv))
    return new_cache


def extract_cross_cache(kv_cache: list) -> list:
    """Return a shallow copy of every layer's cross-attention (k, v)."""
    return [cross_kv for _, cross_kv in kv_cache]


def build_sparse_working_cache(
    kv_cache: list,
    full_cross: list,
    start_frame: int,
    end_frame: int,
) -> list:
    """Build a working kv_cache that uses a sliced cross-attention window.

    ``full_cross`` is the original (full T_enc) cross cache captured after
    a probe step.  Each element is ``(k_full, v_full)`` with shape
    ``(1, T_enc, d_model)``.
    """
    working = []
    for i in range(len(kv_cache)):
        self_kv = kv_cache[i][0]
        if full_cross[i] is not None:
            kf, vf = full_cross[i]
            working.append((self_kv, (kf[:, start_frame:end_frame, :], vf[:, start_frame:end_frame, :])))
        else:
            working.append((self_kv, None))
    return working


def crop_self_attention_cache(
    kv_cache: list,
    max_length: int,
) -> list:
    """Crop the self-attention part of the KV cache to max_length.

    In the mlx-whisper model, each cache entry is:
        (self_attn_kv, cross_attn_kv)
    where self_attn_kv = (k, v) and cross_attn_kv = (k, v).

    We only crop self-attention (the cross-attention cache is static).

    Args:
        kv_cache: The KV cache list.
        max_length: Maximum sequence length to keep.

    Returns:
        Updated KV cache with self-attention cropped.
    """
    new_cache = []
    for self_kv, cross_kv in kv_cache:
        if self_kv is not None:
            k, v = self_kv
            self_kv = (k[:, :max_length, :], v[:, :max_length, :])
        new_cache.append((self_kv, cross_kv))
    return new_cache
