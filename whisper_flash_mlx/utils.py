"""Shared utilities for whisper-flash-mlx."""

from __future__ import annotations

import mlx.core as mx


def build_target_layer_ids(
    num_target_layers: int,
    num_draft_layers: int,
) -> list[int]:
    """Select evenly-spaced decoder layer indices to tap for context features.

    For 32 target layers and 3 taps, returns approximately [8, 16, 24].
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[mx.array],
    layer_ids: list[int],
) -> mx.array:
    """Extract and concatenate decoder hidden states from chosen layers.

    Args:
        hidden_states: List of all decoder hidden states.
            Index 0 is the embedding output, index i+1 is layer i's output.
        layer_ids: Which decoder layer outputs to extract.

    Returns:
        Concatenated features of shape (batch, seq_len, len(layer_ids) * d_model).
    """
    offset = 1  # hidden_states[0] is embedding, hidden_states[1] is layer 0
    selected = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return mx.concatenate(selected, axis=-1)


def sample(logits: mx.array, temperature: float = 0.0) -> mx.array:
    """Sample tokens from logits.

    Args:
        logits: Shape (batch, seq_len, vocab_size).
        temperature: 0 for greedy, >0 for multinomial sampling.

    Returns:
        Token ids of shape (batch, seq_len).
    """
    if temperature < 1e-5:
        return mx.argmax(logits, axis=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size) / temperature
    # MLX categorical sampling: mx.random.categorical expects log-probs
    tokens = mx.random.categorical(logits_flat)  # (bsz * seq_len,)
    return tokens.reshape(bsz, seq_len)
