#!/usr/bin/env python3
"""
Experiment #4: Branching / multi-modality probe.

Tests Phase 2's founding premise: that the target hidden-state manifold is a
single smooth surface where alternate high-probability continuations stay close.

Methodology:
- For each held-out sample, at each step t:
  1. Get target logits at step t
  2. Take top-k (k=5) tokens by probability
  3. For each, force-decode one more step with that token as input
  4. Measure cosine similarity between the top-1 branch and each alternate branch
     at positions t+1, t+2, t+3
- If alternate branches stay >0.9 cosine-aligned → smooth manifold (premise supported)
- If they diverge to <0.7 by step +2 → manifold has discontinuities (premise weakened)
- If they diverge to <0.5 → the ~0.5 cosine ceiling IS the branching ceiling
"""

import time, math, numpy as np
import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states

TOP_K = 5
BLOCK = 4
NUM_EVAL = 5  # evaluate on 5 held-out samples

def run():
    t0 = time.time()
    print("=" * 65)
    print("EXP #4: BRANCHING / MULTI-MODALITY PROBE")
    print("=" * 65)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    all_cosines = {k: [] for k in range(BLOCK)}  # step delta → list of cosine values
    branch_counts = 0
    step_counts = 0

    for sample_idx in range(10, 10 + NUM_EVAL):
        sample = ds[sample_idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)

        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)

        encoder_hidden = encoder_forward(target, mel_mx)

        print(f"\n  Sample [{sample_idx}] ({labels.shape[1]-1} tokens)")

        for t in range(1, max(2, labels.shape[1] - BLOCK - 1), 3):
            inp = labels[:, :t+1]

            # Get target hidden states and logits at position t
            logits, _, hidden_all = decoder_forward_with_hidden_states(
                target, inp, encoder_hidden, collect_hidden_states=True, return_cross_attention=False)

            logits_t = logits[0, -1, :]
            probs = mx.softmax(logits_t)
            sorted_indices = mx.argsort(-probs)
            topk_tokens = sorted_indices[:TOP_K]
            topk_vals = probs[topk_tokens]

            topk_tokens_np = np.array([int(t) for t in topk_tokens])
            probs_np = np.array(topk_vals)

            # For each of the top-k branches, decode BLOCK more steps
            branch_hiddens = []

            for b_idx, branch_tok in enumerate(topk_tokens_np):
                branch_input = mx.concatenate([
                    labels[:, :t+1],
                    mx.array([[int(branch_tok)]], dtype=mx.int32)
                ], axis=1)

                # Extend input by BLOCK-1 more tokens using greedy decoding
                branch_tokens_list = [int(branch_tok)]
                for step in range(BLOCK - 1):
                    logits_b, _, hidden_b = decoder_forward_with_hidden_states(
                        target, branch_input, encoder_hidden,
                        collect_hidden_states=False, return_cross_attention=False)
                    next_tok = mx.argmax(logits_b[:, -1:, :], axis=-1).item()
                    branch_tokens_list.append(next_tok)
                    branch_input = mx.concatenate([
                        branch_input,
                        mx.array([[next_tok]], dtype=mx.int32)
                    ], axis=1)

                # Get hidden states at the original position t for the continuation
                full_input = mx.concatenate([
                    labels[:, :t+1],
                    mx.array([branch_tokens_list], dtype=mx.int32)
                ], axis=1)

                _, _, h_branch = decoder_forward_with_hidden_states(
                    target, full_input, encoder_hidden,
                    collect_hidden_states=True, return_cross_attention=False)

                # Collect hidden states for positions t+1 through t+1+BLOCK
                h_future = np.array(h_branch[-1][0, t+1:t+1+BLOCK, :])
                branch_hiddens.append(h_future)

            # Compare each alternate branch (idx 1..k-1) to the top-1 branch (idx 0)
            for b_idx in range(1, len(branch_hiddens)):
                branch_counts += 1
                for step_delta in range(BLOCK):
                    v1 = branch_hiddens[0][step_delta]
                    v2 = branch_hiddens[b_idx][step_delta]
                    n1 = np.linalg.norm(v1) + 1e-9
                    n2 = np.linalg.norm(v2) + 1e-9
                    cs = float(np.dot(v1, v2) / (n1 * n2))
                    all_cosines[step_delta].append(cs)

        step_counts += 1

    # Report results
    print(f"\n{'='*65}")
    print("  BRANCHING PROBE RESULTS")
    print(f"{'='*65}")
    print(f"  Evaluated {step_counts} steps, {branch_counts} branch-pairs across {NUM_EVAL} samples")
    print()

    print(f"  {'Step Δ':>8} | {'Mean Cos':>9} | {'Std':>7} | {'Min':>6} | {'% >0.9':>8} | {'% <0.7':>8} | {'% <0.5':>8}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for k in range(BLOCK):
        vals = all_cosines.get(k, [])
        if vals:
            mu = np.mean(vals)
            sd = np.std(vals)
            mi = np.min(vals)
            p90 = 100 * np.mean(np.array(vals) > 0.9)
            p70 = 100 * np.mean(np.array(vals) < 0.7)
            p50 = 100 * np.mean(np.array(vals) < 0.5)
            print(f"  {f'+{k+1}':>8} | {mu:>9.4f} | {sd:>7.4f} | {mi:>6.3f} | {p90:>7.1f}% | {p70:>7.1f}% | {p50:>7.1f}%")

    # Summary verdict
    cos_by_step = {k: np.mean(all_cosines.get(k, [0])) for k in range(BLOCK)}
    cos_0 = cos_by_step.get(0, 1.0)
    cos_3 = cos_by_step.get(BLOCK-1, 1.0)
    decay = cos_0 - cos_3

    print(f"\n  {'='*65}")
    print(f"  VERDICT: Δcos({BLOCK}steps) = {decay:.4f}")
    if decay < 0.05:
        print(f"  ✅ SMOOTH MANIFOLD — alternate continuations stay aligned")
    elif decay < 0.2:
        print(f"  ⚠️  MODERATE DIVERGENCE — manifold has some branching")
    elif decay < 0.4:
        print(f"  ❌ SIGNIFICANT DIVERGENCE — branches pull apart by step +{BLOCK}")
    else:
        print(f"  🔴 CATASTROPHIC DIVERGENCE — the manifold hypothesis itself is weakened")
    print(f"  (Branching decay of {decay:.3f} would explain a maximum cosine ceiling of ~{max(0.0, 1.0 - decay):.3f})")

    print(f"Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    run()
