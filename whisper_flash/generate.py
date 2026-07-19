"""DFlash speculative decoding loop for Whisper.

Adapts the DFlash draft-then-verify loop for Whisper's encoder-decoder
architecture, handling:
  - Encoder output computation (once, cached)
  - Cross-attention KV caching in the target decoder
  - Audio summary extraction for the draft model
  - Proper cache management (only crop self-attention, not cross-attention)
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache, EncoderDecoderCache

from .draft_model import WhisperDFlashDraftModel
from .utils import extract_context_feature, sample


def _sync_time(device: torch.device) -> float:
    """Synchronized time measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _crop_encoder_decoder_cache(cache: EncoderDecoderCache, max_length: int):
    """Crop only the self-attention part of an EncoderDecoderCache.

    The cross-attention cache is static (encoder outputs don't change)
    and must NOT be cropped.
    """
    cache.self_attention_cache.crop(max_length)


@torch.inference_mode()
def whisper_dflash_generate(
    draft_model: WhisperDFlashDraftModel,
    target: nn.Module,
    input_features: torch.FloatTensor,
    max_new_tokens: int = 448,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
    decoder_start_ids: Optional[torch.LongTensor] = None,
    stop_token_ids: Optional[list[int]] = None,
    return_stats: bool = False,
) -> torch.LongTensor | SimpleNamespace:
    """Speculative decoding for Whisper using DFlash draft model.

    Args:
        draft_model: Trained WhisperDFlashDraftModel.
        target: WhisperForConditionalGeneration (frozen target model).
        input_features: Mel spectrogram, shape (1, n_mels, n_frames).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        block_size: Override draft model's block size.
        decoder_start_ids: Initial decoder token ids (SOT + language + task).
            If None, uses target.config.decoder_start_token_id.
        stop_token_ids: Token ids that signal end of generation.
        return_stats: If True, return detailed statistics.

    Returns:
        Generated token ids, or SimpleNamespace with stats if return_stats=True.
    """
    device = input_features.device
    block_size = draft_model.block_size if block_size is None else block_size
    mask_token_id = draft_model.mask_token_id

    # Default stop tokens
    if stop_token_ids is None:
        stop_token_ids = [target.config.eos_token_id]

    # Default decoder start
    if decoder_start_ids is None:
        decoder_start_ids = torch.tensor(
            [[target.config.decoder_start_token_id]], device=device
        )

    num_prompt_tokens = decoder_start_ids.shape[1]
    max_length = num_prompt_tokens + max_new_tokens

    # Pre-allocate output buffer
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id,
        dtype=torch.long, device=device,
    )
    output_ids[:, :num_prompt_tokens] = decoder_start_ids

    # ----------------------------------------------------------------
    # Step 1: Encode audio (once)
    # ----------------------------------------------------------------
    prefill_start = _sync_time(device) if return_stats else None

    encoder_outputs = target.model.encoder(input_features)
    encoder_hidden = encoder_outputs.last_hidden_state  # (1, T_enc, 1280)
    audio_summary = encoder_hidden.mean(dim=1, keepdim=True)  # (1, 1, 1280)

    # ----------------------------------------------------------------
    # Step 2: Prefill — run target decoder on prompt tokens
    # ----------------------------------------------------------------
    target_output = target(
        encoder_outputs=(encoder_hidden,),
        decoder_input_ids=decoder_start_ids,
        use_cache=True,
        output_hidden_states=block_size > 1,
    )

    # First predicted token
    output_ids[:, num_prompt_tokens] = sample(
        target_output.logits[:, -1:, :], temperature
    ).squeeze(-1)

    # Extract target hidden states for draft conditioning
    if block_size > 1:
        target_hidden = extract_context_feature(
            target_output.decoder_hidden_states, draft_model.target_layer_ids
        )

    # Set up target cache
    past_key_values_target = target_output.past_key_values

    time_to_first_token = _sync_time(device) - prefill_start if return_stats else None

    # ----------------------------------------------------------------
    # Step 3: Speculative decode loop
    # ----------------------------------------------------------------
    decode_start = _sync_time(device) if return_stats else None
    acceptance_lengths = []
    start = num_prompt_tokens
    draft_prefill = True

    while start < max_length:
        block_ids = output_ids[:, start: start + block_size].clone()
        block_positions = torch.arange(
            start, start + block_size, device=device
        ).unsqueeze(0)

        # --- DRAFT STEP ---
        if block_size > 1:
            # Embed block tokens using target's embedding
            noise_embedding = target.model.decoder.embed_tokens(block_ids)

            # Run draft model
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                audio_summary=audio_summary,
                position_ids=block_positions,
            )

            # Project through target's lm_head and sample
            draft_logits = target.proj_out(draft_hidden[:, :-1, :])
            block_ids[:, 1:] = sample(draft_logits, temperature)

            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _sync_time(device)

        # --- VERIFY STEP ---
        target_output = target(
            encoder_outputs=(encoder_hidden,),
            decoder_input_ids=block_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )

        past_key_values_target = target_output.past_key_values
        posterior = sample(target_output.logits, temperature)

        # Find longest accepted prefix
        acceptance_length = (
            (block_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )

        # Accept tokens
        output_ids[:, start: start + acceptance_length + 1] = block_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[
            :, acceptance_length
        ]
        start += acceptance_length + 1

        # Crop target cache (self-attention only, keep cross-attention)
        if isinstance(past_key_values_target, EncoderDecoderCache):
            _crop_encoder_decoder_cache(past_key_values_target, start)
        else:
            past_key_values_target.crop(start)

        acceptance_lengths.append(acceptance_length + 1)

        # Update target hidden for next draft step
        if block_size > 1:
            target_hidden = extract_context_feature(
                target_output.decoder_hidden_states,
                draft_model.target_layer_ids,
            )[:, : acceptance_length + 1, :]

        # Check for stop tokens
        if stop_token_ids is not None and any(
            sid in output_ids[0, num_prompt_tokens: start + 1]
            for sid in stop_token_ids
        ):
            break

    # ----------------------------------------------------------------
    # Post-process
    # ----------------------------------------------------------------
    output_ids = output_ids[:, : min(start + 1, max_length)]

    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=device)
        generated = output_ids[0, num_prompt_tokens:]
        stop_indices = torch.isin(generated, stop_tensor).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_prompt_tokens + stop_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_prompt_tokens
    total_decode_time = _sync_time(device) - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_prompt_tokens=num_prompt_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(
            total_decode_time / max(num_output_tokens, 1)
        ),
        acceptance_lengths=acceptance_lengths,
    )
