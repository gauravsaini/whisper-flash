#!/usr/bin/env python3
"""
P6: Parallel Tree Drafting for ASR — acceptance-rate upper bound.

Measures the improvement of tree drafting (top-K candidates per position)
over single-candidate drafting by using the TARGET model to simulate a
well-trained draft model's top-K distribution.

Method:
  1. Greedy-decode to get ground-truth token sequence.
  2. For each block-size window over the sequence:
     a. The "draft" (target-simulated) predicts tokens for positions 1..B-1.
     b. "Standard accept": does draft's top-1 match the ground truth?
     c. "Tree accept": does draft's top-K contain the ground truth?
  3. The gap topK - top1 = the maximum tree-drafting benefit with a perfect
     draft model that captures the target's true distribution.

Key distinction from the earlier entropy probe:
  - The draft model in DFlash sees ONLY the prefix (anchor + verified history).
  - Its prediction for position k (1..B-1) is conditioned on the same prefix,
    NOT on k ground-truth tokens.
  - We simulate this by running the target on JUST [anchor] to predict position 1,
    [anchor, pred_1] to predict position 2, etc., where pred_k is the target's
    OWN greedy token. This gives the UPPER BOUND on what a perfect draft
    could achieve.
"""

import argparse
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_whisper.audio import log_mel_spectrogram

from whisper_flash_mlx.target_model import (
    decoder_forward_with_hidden_states,
    encoder_forward,
    load_target_model,
)
from whisper_flash_mlx.utils import sample


# ---------------------------------------------------------------------------
# Simulated tree-drafting measurement
# ---------------------------------------------------------------------------

def measure_tree_acceptance(
    target,
    audio_path: str,
    block_size: int = 8,
    top_k: int = 5,
    max_new_tokens: int = 200,
) -> dict:
    # ── Audio → encoder hidden states ──
    arr, sr = sf.read(audio_path)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    mel = log_mel_spectrogram(arr, n_mels=target.dims.n_mels,
                               padding=16000 * 30 - len(arr))
    mel = mx.array(mel)[None]
    encoder_hidden = encoder_forward(target, mel)
    mx.eval(encoder_hidden)

    # ── Greedy decode to get ground-truth sequence ──
    decoder_ids = mx.array([[50258]], dtype=mx.int32)
    kv_cache = None
    eos_id, sot_id = 50257, 50258

    for _ in range(max_new_tokens):
        inp = decoder_ids[:, -1:] if kv_cache is not None else decoder_ids
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            target, inp, encoder_hidden,
            kv_cache=kv_cache, collect_hidden_states=False,
        )
        tok = sample(logits[:, -1:, :], 0.0)
        mx.eval(tok)
        decoder_ids = mx.concatenate([decoder_ids, tok], axis=1)
        if tok.item() == eos_id:
            break

    ground_truth = decoder_ids[0].tolist()  # starts with SOT
    n = len(ground_truth) - 1  # tokens after SOT
    print(f"  Ground truth length: {n} tokens (excl. SOT)")

    if n < 2:
        return {"error": "sequence too short"}

    # ── Simulate DFlash blocks ──
    # For each block:
    #   anchor = gt[pos]  (the last verified token)
    #   "Draft" predicts positions 1..B-1 sequentially using a
    #   separate target forward that sees only the prefix.
    #
    #   The draft's top-K at each position tells us its
    #   candidate set.  The ground truth continuation is gt[pos+k].
    #
    #   Standard accept: draft_top1[pos+k] == gt[pos+k]
    #   Tree accept:     draft_topK[pos+k] contains gt[pos+k]

    n_positions = 0
    n_standard = 0
    n_tree = 0

    pos = 0  # index into ground_truth (0 = SOT, 1 = first real token)
    while pos < n:
        anchor = pos + 1  # ground_truth[anchor] is the last verified token

        # Simulate drafting B-1 tokens, recording top-1 and top-K at each step
        draft_context = mx.array([ground_truth[: anchor + 1]], dtype=mx.int32)

        for k in range(1, block_size):
            target_pos = anchor + k
            if target_pos >= len(ground_truth):
                break

            # The "draft model" predicts the next token given the prefix.
            # For a perfect draft: use the target's own logits when processing
            # the prefix.  Since draft_context contains ground-truth tokens,
            # the last position's logits predict the next ground-truth token.
            logits, _, _ = decoder_forward_with_hidden_states(
                target, draft_context, encoder_hidden,
                kv_cache=None, collect_hidden_states=False,
            )
            # last position predicts the next token after draft_context
            next_logits = logits[:, -1, :]  # (1, vocab)

            # Draft's top-1 and top-K predictions
            draft_top1 = mx.argmax(next_logits, axis=-1).item()
            draft_topk = mx.argsort(-next_logits, axis=-1)[:, :top_k][0].tolist()

            actual_next = ground_truth[target_pos]

            # Record acceptance
            n_positions += 1
            if draft_top1 == actual_next:
                n_standard += 1
            if actual_next in draft_topk:
                n_tree += 1

            # Extend the draft context with the target's GREEDY choice
            # (simulates what the draft model would generate step-by-step)
            greedy_tok = mx.array([[draft_top1]], dtype=mx.int32)
            draft_context = mx.concatenate([draft_context, greedy_tok], axis=1)

            # --- FIX: after concatenating, the total length must stay ≤
            #     model's max positional embedding (max_target_positions).
            #     Whisper tiny = 448, large = 448 (or 1500 for encoder).
            #     We just need it ≤ 448.
            if draft_context.shape[1] > 400:
                break

        # Advance to next block (simulate full acceptance of block)
        pos += block_size - 1

    if n_positions == 0:
        return {"error": "no positions measured"}

    return {
        "n_positions": n_positions,
        "standard_accept_rate": n_standard / n_positions,
        f"tree_top{top_k}_accept_rate": n_tree / n_positions,
        "improvement_pp": (n_tree - n_standard) / n_positions * 100,
        "block_size": block_size,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P6: Tree-drafting acceptance-rate measurement"
    )
    parser.add_argument("--audio", default="/tmp/jfk_16k.wav")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    target = load_target_model(args.model)

    print(f"Model:       {args.model}")
    print(f"Block size:  {args.block_size}")
    print(f"Top-K:       {args.top_k}")

    results = measure_tree_acceptance(
        target, args.audio,
        block_size=args.block_size,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    if "error" in results:
        print(f"\n❌ {results['error']}")
        return results

    print(f"\n{'='*65}")
    print(f"  TREE DRAFTING UPPER BOUND — {args.model}")
    print(f"{'='*65}")
    print(f"  Positions:            {results['n_positions']}")
    print(f"  Block size:           {results['block_size']}")
    print(f"  Top-K:                {results['top_k']}")
    print(f"  Standard accept rate: {results['standard_accept_rate']*100:.1f}%")
    print(f"  Tree (top-{args.top_k}) accept rate:   {results[f'tree_top{args.top_k}_accept_rate']*100:.1f}%")
    print(f"  Improvement:          {results['improvement_pp']:+.1f} pp")
    print(f"{'='*65}")

    if results['improvement_pp'] > 5:
        print(f"\n✅  Tree drafting viable: +{results['improvement_pp']:.1f}pp improvement")
    elif results['improvement_pp'] > 2:
        print(f"\n⚠️  Marginal benefit ({results['improvement_pp']:.1f}pp)."
              " Worth profiling overhead.")
    else:
        print(f"\n❌  Tree draft improvement < 2pp."
              " Target is too deterministic — pivot to P8 (MLX fused kernels).")

    return results


if __name__ == "__main__":
    main()
