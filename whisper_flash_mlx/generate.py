"""DFlash speculative decoding loop for Whisper — MLX version.

Adapts the DFlash draft-then-verify loop for Whisper's encoder-decoder
architecture, running entirely on Apple Silicon via MLX, handling:
  - Encoder output computation (once)
  - Cross-attention KV caching in the target decoder
  - Audio summary extraction for the draft model
  - Proper cache management (only crop self-attention, not cross-attention)
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import mlx.core as mx

from .draft_model import WhisperDFlashDraftModel
from .target_model import (
    crop_self_attention_cache,
    decoder_forward_with_hidden_states,
    encoder_forward,
    get_token_embedding,
    project_to_logits,
)
from .utils import extract_context_feature, sample

from mlx_whisper.whisper import Whisper


def whisper_dflash_generate(
    draft_model: WhisperDFlashDraftModel,
    target: Whisper,
    mel: mx.array,
    max_new_tokens: int = 448,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
    decoder_start_ids: Optional[mx.array] = None,
    stop_token_ids: Optional[list[int]] = None,
    return_stats: bool = False,
) -> mx.array | SimpleNamespace:
    """Speculative decoding for Whisper using DFlash draft model (MLX).

    Args:
        draft_model: Trained WhisperDFlashDraftModel (MLX).
        target: Frozen mlx_whisper.Whisper model.
        mel: Mel spectrogram, shape (1, n_mels, n_frames) or (n_mels, n_frames).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        block_size: Override draft model's block size.
        decoder_start_ids: Initial decoder token ids (SOT + language + task).
            If None, uses SOT token (50258).
        stop_token_ids: Token ids that signal end of generation.
        return_stats: If True, return detailed statistics.

    Returns:
        Generated token ids, or SimpleNamespace with stats if return_stats=True.
    """
    if draft_model is not None:
        block_size = draft_model.config.block_size if block_size is None else block_size
        mask_token_id = draft_model.mask_token_id
    else:
        block_size = 1
        mask_token_id = None

    # Default stop tokens (Whisper EOT = 50257)
    if stop_token_ids is None:
        stop_token_ids = [50257]

    # Default decoder start (Whisper SOT = 50258)
    if decoder_start_ids is None:
        decoder_start_ids = mx.array([[50258]])

    num_prompt_tokens = decoder_start_ids.shape[1]
    max_length = num_prompt_tokens + max_new_tokens

    # Pre-allocate output buffer as a Python list (MLX arrays are immutable)
    output_list = [mask_token_id] * (max_length + block_size)
    # Set prompt tokens
    prompt_list = decoder_start_ids.tolist()[0]
    for i, t in enumerate(prompt_list):
        output_list[i] = t


    # ----------------------------------------------------------------
    # Step 1: Encode audio (once)
    # ----------------------------------------------------------------
    prefill_start = time.perf_counter() if return_stats else None

    encoder_hidden = encoder_forward(target, mel)  # (1, T_enc, d_model)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)  # (1, 1, d_model)
    mx.eval(encoder_hidden, audio_summary)

    # ----------------------------------------------------------------
    # Step 2: Prefill — run target decoder on prompt tokens
    # ----------------------------------------------------------------
    logits, kv_cache, all_hidden = decoder_forward_with_hidden_states(
        target, decoder_start_ids, encoder_hidden,
        kv_cache=None,
        collect_hidden_states=(block_size > 1),
    )

    # First predicted token
    first_token = sample(logits[:, -1:, :], temperature)  # (1, 1)
    mx.eval(first_token)

    # Update output buffer with first predicted token
    output_list[num_prompt_tokens] = first_token.item()


    # Extract target hidden states for draft conditioning
    if block_size > 1:
        target_hidden = extract_context_feature(
            all_hidden, draft_model.target_layer_ids
        )

    time_to_first_token = (time.perf_counter() - prefill_start) if return_stats else None

    # ----------------------------------------------------------------
    # Step 3: Speculative decode loop
    # ----------------------------------------------------------------
    decode_start = time.perf_counter() if return_stats else None
    acceptance_lengths = []
    block_sizes = []
    start = num_prompt_tokens
    draft_prefill = True
    current_block_size = block_size
    ngram_cache = {}
    is_grafted = False
    grafted_tokens = []

    while start < max_length:
        if draft_model is not None:
            # Get block token ids
            block_ids_list = output_list[start: start + current_block_size]
            # Pad if needed
            while len(block_ids_list) < current_block_size:
                block_ids_list.append(mask_token_id)
            block_ids = mx.array([block_ids_list], dtype=mx.int32)
            block_positions = mx.arange(start, start + current_block_size)[None]  # (1, B)
        else:
            block_ids = mx.array([[output_list[start - 1]]], dtype=mx.int32)
            block_positions = mx.arange(start - 1, start)[None]

        # --- DRAFT STEP ---
        if current_block_size > 1:
            if is_grafted:
                # Use retrieved grafted tokens instead of draft model
                for i, t in enumerate(grafted_tokens):
                    block_ids_list[i + 1] = t
                block_ids = mx.array([block_ids_list], dtype=mx.int32)
            else:
                # Embed block tokens using target's embedding
                noise_embedding = get_token_embedding(target, block_ids)

                # Run draft model
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    audio_summary=audio_summary,
                    position_ids=block_positions,
                )

                # Project through target's lm_head and sample
                draft_logits = project_to_logits(target, draft_hidden[:, :-1, :])
                draft_tokens = sample(draft_logits, temperature)  # (1, B-1)
                mx.eval(draft_tokens)

                # Fill in drafted tokens
                draft_tokens_list = draft_tokens.tolist()[0]
                for i, t in enumerate(draft_tokens_list):
                    block_ids_list[i + 1] = t
                block_ids = mx.array([block_ids_list], dtype=mx.int32)

                if draft_prefill and return_stats:
                    draft_prefill = False
                    decode_start = time.perf_counter()

        # --- VERIFY STEP ---
        logits, kv_cache, all_hidden_verify = decoder_forward_with_hidden_states(
            target, block_ids, encoder_hidden,
            kv_cache=kv_cache,
            collect_hidden_states=(draft_model is not None),
        )

        posterior = sample(logits, temperature)  # (1, B)
        mx.eval(posterior)

        # Find longest accepted prefix
        posterior_list = posterior.tolist()[0]
        acceptance_length = 0
        for i in range(1, current_block_size):
            if i < len(block_ids_list) and block_ids_list[i] == posterior_list[i - 1]:
                acceptance_length += 1
            else:
                break

        # Accept tokens
        for i in range(acceptance_length + 1):
            idx = start + i
            if draft_model is not None:
                output_list[idx] = block_ids_list[i] if i < acceptance_length else posterior_list[i]
            else:
                output_list[idx] = posterior_list[i]
            # TRICK 4a: Update ngram cache (using N=2 bigrams)
            if idx >= num_prompt_tokens + 2:
                key = (output_list[idx - 2], output_list[idx - 1])
                ngram_cache[key] = output_list[idx]

        output_list[start + acceptance_length + 1] = posterior_list[acceptance_length]
        # Update cache for the target fallback token too
        idx = start + acceptance_length + 1
        if idx >= num_prompt_tokens + 2:
            key = (output_list[idx - 2], output_list[idx - 1])
            ngram_cache[key] = output_list[idx]
        
        # TRICK 3 & 4: BlockPilot Adaptive Block Size & Grafting
        # If the target token we're falling back to is unconfident, shrink block size
        target_token_logits = logits[0, acceptance_length]
        max_prob = mx.max(mx.softmax(target_token_logits, axis=-1)).item()
        
        is_grafted = False
        grafted_tokens = []
        if max_prob < 0.4 and block_size > 1:
            current_block_size = 1
            
            # TRICK 4b: Grafting (Try to retrieve tokens when confident draft is impossible)
            ctx_idx = start + acceptance_length + 1
            if ctx_idx >= num_prompt_tokens + 2:
                key = (output_list[ctx_idx - 2], output_list[ctx_idx - 1])
                curr_key = key
                while len(grafted_tokens) < block_size - 1 and curr_key in ngram_cache:
                    next_tok = ngram_cache[curr_key]
                    grafted_tokens.append(next_tok)
                    curr_key = (curr_key[1], next_tok)
                
                if grafted_tokens:
                    is_grafted = True
                    current_block_size = len(grafted_tokens) + 1

        elif max_prob < 0.7 and block_size > 1:
            current_block_size = max(2, block_size // 2)
        else:
            current_block_size = block_size

        start += acceptance_length + 1

        # Crop target cache (self-attention only, keep cross-attention)
        kv_cache = crop_self_attention_cache(kv_cache, start)

        acceptance_lengths.append(acceptance_length + 1)
        block_sizes.append(current_block_size)

        # Update target hidden for next draft step
        if draft_model is not None:
            valid_hidden = extract_context_feature(
                all_hidden_verify,
                draft_model.target_layer_ids,
            )[:, : acceptance_length + 1, :]
            target_hidden = mx.concatenate([target_hidden, valid_hidden], axis=1)

        # Check for stop tokens
        if stop_token_ids is not None:
            generated = output_list[num_prompt_tokens: start + 1]
            if any(sid in generated for sid in stop_token_ids):
                break

    # ----------------------------------------------------------------
    # Post-process
    # ----------------------------------------------------------------
    end_idx = min(start + 1, max_length)
    final_ids = output_list[:end_idx]

    if stop_token_ids is not None:
        generated = final_ids[num_prompt_tokens:]
        for i, tid in enumerate(generated):
            if tid in stop_token_ids:
                final_ids = final_ids[: num_prompt_tokens + i + 1]
                break

    output_ids = mx.array([final_ids], dtype=mx.int32)

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_prompt_tokens
    total_decode_time = time.perf_counter() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_prompt_tokens=num_prompt_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(total_decode_time / max(num_output_tokens, 1)),
        acceptance_lengths=acceptance_lengths,
        block_sizes=block_sizes,
    )
