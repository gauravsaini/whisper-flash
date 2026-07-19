#!/usr/bin/env python3
"""
P6-Lite: Tree Verifier Prototype.

Build a tree of K draft candidate sequences, verify all in one batched
target forward pass, measure overhead vs single-sequence DFlash.

Tree structure:
  - Root: anchor token (last verified)
  - Each level: K candidates per node (fan-out K)
  - Depth: block_size - 1

For K=5, B=8 with FULL tree: K^(B-1) = 78K sequences (too many).
Practical approach: generate K independent sequences (each from top-K at
each position, greedily extended). This gives K × (B-1) tokens in
K sequences, verified in a batch of size K.

Metrics:
  - Latency: tree verify vs single verify
  - Memory: KV cache growth with K
  - Acceptance: max across branches vs single branch
"""

import argparse
import time
from typing import Optional

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.whisper import Whisper

from whisper_flash_mlx.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash_mlx.target_model import (
    crop_self_attention_cache,
    decoder_forward_with_hidden_states,
    encoder_forward,
    get_token_embedding,
    load_target_model,
    project_to_logits,
)
from whisper_flash_mlx.utils import extract_context_feature, sample

EOS_ID, SOT_ID = 50257, 50258


def load_or_init_draft(target: Whisper, block_size: int) -> WhisperDFlashDraftModel:
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state,
        num_target_layers=target.dims.n_text_layer,
        vocab_size=target.dims.n_vocab,
        max_target_positions=target.dims.n_text_ctx,
        block_size=block_size,
    )
    return WhisperDFlashDraftModel(config)


def build_tree_candidates(draft_logits: mx.array, K: int, B: int) -> list[list[int]]:
    """Build K candidate sequences from draft logits.

    For each position i (1..B-1):
      - Take top-K tokens from draft_logits[:, i-1, :]
      - But to build sequences: for seq k, at position i use the
        k-th best token (ranked by draft_logits[:, i-1, :]).
      - This gives K sequences where seq k uses the k-th best token
        at each position independently.

    Returns K sequences, each as [anchor, d_1, ..., d_{B-1}].
    """
    # draft_logits: (1, B-1, vocab)
    B_minus_1 = draft_logits.shape[1]
    candidates = mx.argsort(-draft_logits, axis=-1)[:, :, :K]  # (1, B-1, K)

    sequences = []
    for k in range(K):
        seq = []
        for pos in range(B_minus_1):
            seq.append(candidates[0, pos, k].item())
        sequences.append(seq)

    return sequences


def verify_tree(
    target: Whisper,
    anchor: int,
    candidates: list[list[int]],
    encoder_hidden: mx.array,
    kv_cache_self: list,
    kv_cache_cross: list,
) -> tuple[int, mx.array, int]:
    """Verify all candidate sequences in one batched forward pass.

    Args:
        target: Whisper model.
        anchor: anchor token id.
        candidates: list of K sequences, each is [d_1, ..., d_{B-1}].
        encoder_hidden: (1, T_enc, d_model).
        kv_cache_self: current self-attention KV cache from anchor position.
        kv_cache_cross: current cross-attention KV cache.

    Returns:
        (acceptance_length, updated_self_kv_cache, best_sequence_idx)
    """
    K = len(candidates)
    B = len(candidates[0]) + 1  # +1 for anchor

    # Build batch of sequences: all share the same anchor
    # Pad shorter sequences with EOS (though all should be same length)
    batch = []
    for seq in candidates:
        batch.append([anchor] + seq)

    batch_tokens = mx.array(batch, dtype=mx.int32)  # (K, B)

    # Fan-out KV cache: duplicate self-attention cache K times
    # Each cache entry has shape (n_heads, seq_len, head_dim) for k and v
    K_kv_cache = []
    for layer_kv in kv_cache_self:
        if layer_kv is not None:
            k, v = layer_kv
            # (n_heads, seq_len, head_dim) → tile K times
            k_fanned = mx.tile(k, (K, 1, 1))
            v_fanned = mx.tile(v, (K, 1, 1))
            K_kv_cache.append((k_fanned, v_fanned))
        else:
            K_kv_cache.append(None)

    # Fan-out cross-attention KV cache
    K_cross_kv = []
    for layer_kv in kv_cache_cross:
        if layer_kv is not None:
            k, v = layer_kv
            K_cross_kv.append((mx.tile(k, (K, 1, 1)), mx.tile(v, (K, 1, 1))))
        else:
            K_cross_kv.append(None)

    # Combine self + cross
    kv_cache_batch = list(zip(K_kv_cache, K_cross_kv))

    # Tile encoder hidden to batch
    enc_batch = mx.tile(encoder_hidden, (K, 1, 1))

    # Single batched target forward
    logits, updated_kv, _ = decoder_forward_with_hidden_states(
        target, batch_tokens, enc_batch,
        kv_cache=kv_cache_batch, collect_hidden_states=False,
    )
    # logits: (K, B, vocab)
    posterior = sample(logits, 0.0)  # (K, B)

    # Find best sequence: longest accepted prefix
    best_len = -1
    best_seq = 0
    for k in range(K):
        acceptance_length = 0
        for i in range(1, B):
            if batch[k][i] == posterior[k, i - 1].item():
                acceptance_length += 1
            else:
                break
        if acceptance_length > best_len:
            best_len = acceptance_length
            best_seq = k

    # Extract the first branch's updated KV cache for the next iteration
    # (use best_seq's cache)
    single_kv_self = [(k[best_seq:best_seq+1], v[best_seq:best_seq+1])
                      for k, v in [x[0] for x in updated_kv]]
    single_kv_cross = [(k[best_seq:best_seq+1], v[best_seq:best_seq+1])
                       for k, v in [x[1] for x in updated_kv]]

    return best_len, (single_kv_self, single_kv_cross), best_seq


def run_benchmark(
    model_name: str,
    audio_path: str,
    block_size: int = 8,
    K: int = 5,
    max_new_tokens: int = 100,
    use_tree: bool = True,
):
    """Run DFlash decoding and benchmark verify step time.

    Returns timing stats and acceptance lengths.
    """
    target = load_target_model(model_name)
    draft = load_or_init_draft(target, block_size)

    # Audio
    arr, sr = sf.read(audio_path)
    if arr.ndim == 2: arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    mel = log_mel_spectrogram(arr, n_mels=target.dims.n_mels, padding=16000*30-len(arr))
    mel = mx.array(mel)[None]
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    mx.eval(enc, audio_summary)

    # Prefill with SOT
    dec_ids = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, all_h = decoder_forward_with_hidden_states(target, dec_ids, enc, kv_cache=None, collect_hidden_states=True)
    first_tok = sample(logits[:, -1:, :], 0.0)
    mx.eval(first_tok)
    dec_ids = mx.concatenate([dec_ids, first_tok], axis=1)
    output_list = dec_ids[0].tolist()
    target_hidden = extract_context_feature(all_h, draft.target_layer_ids)
    current_block_size = block_size
    tokens_produced = len(output_list)

    # Extract self/cross KV caches
    self_kv = [kv[0] for kv in kv_cache]
    cross_kv = [kv[1] for kv in kv_cache]

    verify_times = []
    acceptance_lengths = []

    while tokens_produced < max_new_tokens:
        if current_block_size <= 1:
            inp = mx.array([[output_list[-1]]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(target, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0).item()
            output_list.append(tok)
            tokens_produced += 1
            if tok == EOS_ID: break
            target_hidden_new = extract_context_feature(
                decoder_forward_with_hidden_states(target, inp, enc, kv_cache=None, collect_hidden_states=True)[2],
                draft.target_layer_ids)[:, :1, :]
            target_hidden = mx.concatenate([target_hidden, target_hidden_new], axis=1)
            kv_cache = crop_self_attention_cache(kv_cache, tokens_produced)
            continue

        # Draft step
        anchor_tok = output_list[-1]
        block_ids = mx.array([[anchor_tok] + [EOS_ID] * (current_block_size - 1)], dtype=mx.int32)
        noise = get_token_embedding(target, block_ids)
        pos_ids = mx.arange(noise.shape[1], dtype=mx.int32)[None]
        draft_hidden = draft(noise, target_hidden, audio_summary, pos_ids, mask=None)
        mx.eval(draft_hidden)

        n_draft = draft_hidden.shape[1] - 1
        draft_logits = project_to_logits(target, draft_hidden[:, 1:, :])
        draft_tokens = sample(draft_logits, 0.0)
        mx.eval(draft_logits, draft_tokens)

        # Build tree candidates or use single
        if use_tree:
            candidates = build_tree_candidates(draft_logits, K, current_block_size)
        else:
            candidates = [draft_tokens[0].tolist()]

        # Measure verify time
        t0 = time.perf_counter()

        if use_tree:
            best_len, (self_kv, cross_kv), best_seq = verify_tree(
                target, anchor_tok, candidates, enc,
                self_kv, cross_kv,
            )
            # Reconstruct kv_cache for the next iteration from self_kv and cross_kv
            # The verify_tree returns updated KV caches already
            kv_cache = list(zip(self_kv, cross_kv))
            # Accepted tokens from the best sequence
            accepted = [anchor_tok] + candidates[best_seq]
            posterior_check = sample(
                decoder_forward_with_hidden_states(
                    target, mx.array([accepted], dtype=mx.int32), enc,
                    kv_cache=None, collect_hidden_states=False,
                )[0], 0.0)[0]
            acceptance_length = min(best_len, current_block_size - 1)
        else:
            # Single-sequence verify
            draft_ids = mx.concatenate([block_ids[:, :1], draft_tokens], axis=1)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                target, draft_ids, enc, kv_cache=kv_cache, collect_hidden_states=False)
            posterior = sample(logits, 0.0)
            mx.eval(logits, posterior)
            self_kv = [kv[0] for kv in kv_cache]
            cross_kv = [kv[1] for kv in kv_cache]

            acceptance_length = 0
            for i in range(1, draft_ids.shape[1]):
                if draft_ids[0, i].item() == posterior[0, i - 1].item():
                    acceptance_length += 1
                else:
                    break

        t1 = time.perf_counter()
        verify_times.append((t1 - t0) * 1000)
        acceptance_lengths.append(acceptance_length)

        # Accept tokens
        if use_tree:
            seq = candidates[best_seq]
            accepted_tokens = [anchor_tok] + seq[:acceptance_length + 1]
            # The last accepted token is posterior at position acceptance_length
            # which should be from the best sequence
            posterior_check = sample(
                decoder_forward_with_hidden_states(
                    target, mx.array([accepted_tokens[:acceptance_length+2]], dtype=mx.int32), enc,
                    kv_cache=None, collect_hidden_states=False,
                )[0], 0.0)[0]
            new_tokens = []
            for i in range(acceptance_length + 1):
                if i == 0:
                    continue  # anchor already accepted
                new_tokens.append(accepted_tokens[i])
            # The fallback
            new_tokens.append(posterior_check[acceptance_length].item())
        else:
            new_tokens = []
            for i in range(1, acceptance_length + 1):
                new_tokens.append(draft_ids[0, i].item())
            if acceptance_length < logits.shape[1]:
                new_tokens.append(mx.argmax(logits[0, acceptance_length]).item())

        for tok in new_tokens:
            output_list.append(tok)
            tokens_produced += 1
            if tok == EOS_ID:
                break

        if output_list[-1] == EOS_ID:
            break

        # Update draft context
        # Re-run just the accepted tokens for hidden states
        accepted_prefix = output_list[-(acceptance_length + 2):]
        if len(accepted_prefix) > 1:
            inp = mx.array([accepted_prefix], dtype=mx.int32)
            _, _, all_h_v = decoder_forward_with_hidden_states(
                target, inp, enc, kv_cache=None, collect_hidden_states=True)
            valid_h = extract_context_feature(all_h_v, draft.target_layer_ids)[:, -1:, :]
            target_hidden = mx.concatenate([target_hidden, valid_h], axis=1)

        # Adaptive block size
        current_block_size = block_size

    return {
        "verify_times": verify_times,
        "acceptance_lengths": acceptance_lengths,
        "total_tokens": tokens_produced,
        "n_verify_steps": len(verify_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="/tmp/jfk_16k.wav")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--tree-k", type=int, default=5, help="Tree width (K candidates)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    for use_tree, label in [(False, "Single"), (True, f"Tree(K={args.tree_k})")]:
        print(f"\n{'='*60}")
        print(f"  {label} Verification Benchmark")
        print(f"{'='*60}")

        stats = run_benchmark(
            args.model, args.audio,
            block_size=args.block_size,
            K=args.tree_k if use_tree else 1,
            max_new_tokens=args.max_new_tokens,
            use_tree=use_tree,
        )

        times = stats["verify_times"]
        accs = stats["acceptance_lengths"]
        print(f"  Total tokens:    {stats['total_tokens']}")
        print(f"  Verify steps:    {stats['n_verify_steps']}")
        if times:
            print(f"  Avg verify:      {np.mean(times):.2f} ms")
            print(f"  Median verify:   {np.median(times):.2f} ms")
            print(f"  Max verify:      {max(times):.2f} ms")
            print(f"  Min verify:      {min(times):.2f} ms")
        if accs:
            print(f"  Avg acceptance:  {np.mean(accs):.1f} tokens (of {args.block_size - 1})")
            print(f"  Acceptance > 0:  {sum(1 for a in accs if a > 0)}/{len(accs)} steps")


if __name__ == "__main__":
    main()
