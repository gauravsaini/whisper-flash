"""Shared utilities for whisper-flash."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """Extract and concatenate decoder hidden states from chosen layers.

    Args:
        hidden_states: Tuple of all decoder hidden states.
            Index 0 is the embedding output, index i+1 is layer i's output.
        layer_ids: Which decoder layer outputs to extract.

    Returns:
        Concatenated features of shape (batch, seq_len, len(layer_ids) * d_model).
    """
    offset = 1  # hidden_states[0] is embedding, hidden_states[1] is layer 0
    selected = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Sample tokens from logits.

    Args:
        logits: Shape (batch, seq_len, vocab_size).
        temperature: 0 for greedy, >0 for multinomial sampling.

    Returns:
        Token ids of shape (batch, seq_len).
    """
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)
