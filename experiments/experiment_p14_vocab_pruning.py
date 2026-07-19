"""P14: Dynamic Vocabulary Pruning for ASR Decoding.

Key insight: Whisper's vocab is 51,865 tokens but at any given step,
only a tiny fraction are plausible. The final lm_head projection
(d_model × vocab) is one of the most expensive single operations.

Approach:
  1. Run a "pilot" greedy decode (or use first-pass statistics)
  2. Collect active token vocabulary (unique tokens ever produced)
  3. Build a pruned lm_head with only the active vocab slice
  4. Decode with pruned head — map back to full vocab for output

For whisper-tiny (d=384), lm_head = 384×51865 = 19.9M params.
If we prune to top-500 tokens: 384×500 = 192K params (99% reduction).

This is composable with Q8 + KV cache + stride-2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ══════════════════════════════════════════════════════════════════
# Vocabulary usage analysis
# ══════════════════════════════════════════════════════════════════

def analyze_vocab_usage(model, samples: list) -> dict:
    """Analyze which vocab tokens are actually used during greedy decode."""
    from collections import Counter
    token_counts = Counter()
    total_tokens = 0
    
    for mel, ref, idx in samples:
        enc = encoder_forward(model, mel)
        mx.eval(enc)
        
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, dec, enc, kv_cache=None, collect_hidden_states=False)
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        ids = [SOT_ID, first.item()]
        
        while len(ids) < 448:
            inp = mx.array([[ids[-1]]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            
            # Record the full logit distribution shape and top-k usage
            logits_np = np.array(logits[0, -1, :])
            top_k_indices = np.argsort(logits_np)[-100:][::-1]  # top-100
            for idx_tok in top_k_indices:
                token_counts[int(idx_tok)] += 1
            
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            tid = tok.item()
            ids.append(tid)
            total_tokens += 1
            if tid == EOS_ID:
                break
    
    # Analyze coverage
    unique_tokens = len(token_counts)
    coverage = {}
    sorted_tokens = token_counts.most_common()
    cumsum = 0
    for rank, (tok_id, count) in enumerate(sorted_tokens):
        cumsum += count
        pct = cumsum / sum(token_counts.values()) * 100
        if rank + 1 in [50, 100, 200, 500, 1000, 2000, 5000]:
            coverage[rank + 1] = round(pct, 2)
    
    return {
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
        "vocab_size": 51865,
        "coverage": coverage,
        "top_100_tokens": [tok_id for tok_id, _ in sorted_tokens[:100]],
    }


# ══════════════════════════════════════════════════════════════════
# Pruned lm_head decoder
# ══════════════════════════════════════════════════════════════════

def build_pruned_lm_head(model, active_token_ids: list[int]) -> tuple[mx.array, dict]:
    """Build a pruned embedding matrix for fast vocab projection.
    
    Returns:
        - pruned_weight: shape (len(active_token_ids), d_model)
        - id_map: {pruned_idx: original_token_id}
    """
    # Get the full embedding weight
    full_weight = model.decoder.token_embedding.weight  # (vocab, d_model)
    
    # Ensure we include critical tokens
    critical = {SOT_ID, EOS_ID}
    active_set = set(active_token_ids) | critical
    active_sorted = sorted(active_set)
    
    # Build pruned weight matrix
    indices = mx.array(active_sorted, dtype=mx.int32)
    pruned_weight = full_weight[indices]  # (n_active, d_model)
    
    # Build mapping
    id_map = {i: orig for i, orig in enumerate(active_sorted)}
    reverse_map = {orig: i for i, orig in enumerate(active_sorted)}
    
    return pruned_weight, id_map, reverse_map


def generate_with_pruned_head(
    model, mel: mx.array,
    pruned_weight: mx.array, id_map: dict, reverse_map: dict, *,
    use_kv_cache: bool = True,
    encoder_stride: int = 1,
    max_tokens: int = 448,
) -> tuple[list[int], float]:
    """Greedy decode using pruned vocabulary projection."""
    t0 = time.perf_counter()
    
    # Encode
    enc = encoder_forward(model, mel)
    mx.eval(enc)
    
    if encoder_stride > 1:
        B, T, D = enc.shape
        T_trim = (T // encoder_stride) * encoder_stride
        enc = mx.mean(enc[:, :T_trim, :].reshape(
            B, T_trim // encoder_stride, encoder_stride, D), axis=2)
        mx.eval(enc)
    
    # Prefill SOT (use full head for first token — handles SOT correctly)
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, hidden_states = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=True)
    
    # Use pruned head for first prediction
    h = hidden_states[-1]  # last layer output, before ln
    h = model.decoder.ln(h)
    pruned_logits = h[:, -1:, :] @ pruned_weight.T  # (1, 1, n_active)
    mx.eval(pruned_logits)
    pruned_idx = mx.argmax(pruned_logits[:, -1, :], axis=-1).item()
    
    # Map back to original token id
    if pruned_idx in id_map:
        first_tok = id_map[pruned_idx]
    else:
        # Fallback to full head
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        first_tok = first.item()
    
    output_ids = [SOT_ID, first_tok]
    
    while len(output_ids) < max_tokens:
        last_tok = output_ids[-1]
        if last_tok == EOS_ID:
            break
        
        inp = mx.array([[last_tok]], dtype=mx.int32)
        if use_kv_cache:
            _, kv_cache, hidden_states = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=True)
        else:
            full_seq = mx.array([output_ids], dtype=mx.int32)
            _, _, hidden_states = decoder_forward_with_hidden_states(
                model, full_seq, enc, kv_cache=None, collect_hidden_states=True)
        
        # Pruned projection
        h = hidden_states[-1]
        h = model.decoder.ln(h)
        pruned_logits = h[:, -1:, :] @ pruned_weight.T  # (1, 1, n_active)
        mx.eval(pruned_logits)
        pruned_idx = mx.argmax(pruned_logits[:, -1, :], axis=-1).item()
        
        if pruned_idx in id_map:
            tid = id_map[pruned_idx]
        else:
            tid = EOS_ID  # safety fallback
        
        output_ids.append(tid)
    
    t1 = time.perf_counter()
    return output_ids, t1 - t0


# ══════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════

def load_dataset(n_samples: int = 20):
    from datasets import load_dataset as hf_load
    from mlx_whisper.audio import log_mel_spectrogram

    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation")
    samples = []
    for i in range(min(n_samples, len(ds))):
        audio = ds[i]["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        mel = log_mel_spectrogram(arr, n_mels=80, padding=16000 * 30 - len(arr))
        mel = mx.array(mel)[None]
        ref = ds[i].get("text", ds[i].get("transcription", ""))
        samples.append((mel, ref, i))
    return samples


def decode_tokens(model, token_ids: list[int]) -> str:
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)
    text_tokens = [t for t in token_ids[1:] if t < tokenizer.eot]
    return tokenizer.decode(text_tokens).strip()


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import wer
    return wer([r.strip().lower() for r in refs], [h.strip().lower() for h in hyps])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P14: Dynamic Vocabulary Pruning")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--n-train", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--prune-sizes", type=str, default="100,200,500,1000,2000",
                        help="Comma-separated list of pruned vocab sizes to test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    prune_sizes = [int(x) for x in args.prune_sizes.split(",")]
    
    print(f"\n{'#'*60}")
    print(f"  P14: Dynamic Vocabulary Pruning")
    print(f"  Model: {args.model}")
    print(f"  Prune sizes: {prune_sizes}")
    print(f"{'#'*60}")
    
    model = load_target_model(args.model, dtype=mx.float16)
    
    total_samples = args.n_train + args.n_eval
    all_samples = load_dataset(total_samples)
    train_samples = all_samples[:args.n_train]
    eval_samples = all_samples[args.n_train:args.n_train + args.n_eval]
    
    # Phase 1: Analyze vocab usage
    print(f"\n--- Analyzing vocab usage on {len(train_samples)} train samples ---")
    usage = analyze_vocab_usage(model, train_samples)
    print(f"  Unique tokens in top-100 logits: {usage['unique_tokens']}")
    print(f"  Total token predictions: {usage['total_tokens']}")
    print(f"  Coverage:")
    for k, v in usage['coverage'].items():
        print(f"    Top-{k}: {v}%")
    
    # Phase 2: Greedy baseline
    print(f"\n--- Greedy baseline on {len(eval_samples)} eval samples ---")
    greedy_refs, greedy_hyps = [], []
    greedy_total_time = 0
    greedy_total_tokens = 0
    
    for mel, ref, idx in eval_samples:
        enc = encoder_forward(model, mel)
        mx.eval(enc)
        t0 = time.perf_counter()
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, dec, enc, kv_cache=None, collect_hidden_states=False)
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        ids = [SOT_ID, first.item()]
        while len(ids) < 448:
            inp = mx.array([[ids[-1]]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            tid = tok.item()
            ids.append(tid)
            if tid == EOS_ID:
                break
        wall = time.perf_counter() - t0
        text = decode_tokens(model, ids)
        greedy_refs.append(ref)
        greedy_hyps.append(text)
        greedy_total_time += wall
        greedy_total_tokens += len(ids) - 1
        print(f"  Sample {idx}: {len(ids)-1:3d} tok, {wall:.3f}s | {text[:60]}")
    
    greedy_wer = compute_wer(greedy_refs, greedy_hyps)
    print(f"  Greedy WER: {greedy_wer:.4f}, Time: {greedy_total_time:.3f}s, "
          f"Tok/s: {greedy_total_tokens/greedy_total_time:.1f}")
    
    # Phase 3: Pruned head evaluation
    results_all = []
    
    for prune_size in prune_sizes:
        print(f"\n--- Pruned vocab: {prune_size} tokens ---")
        
        active_tokens = usage['top_100_tokens'][:prune_size]
        # If we need more than 100, just use a range
        if prune_size > len(active_tokens):
            # Include common ASCII range + special tokens
            active_tokens = list(range(min(prune_size, 51865)))
        
        pruned_w, id_map, rev_map = build_pruned_lm_head(model, active_tokens)
        mx.eval(pruned_w)
        
        d_model = pruned_w.shape[1]
        full_params = 51865 * d_model
        pruned_params = pruned_w.shape[0] * d_model
        
        print(f"  lm_head: {full_params:,} → {pruned_params:,} params "
              f"({(1-pruned_params/full_params)*100:.1f}% savings)")
        
        pruned_refs, pruned_hyps = [], []
        pruned_total_time = 0
        pruned_total_tokens = 0
        
        for mel, ref, idx in eval_samples:
            ids, wall = generate_with_pruned_head(
                model, mel, pruned_w, id_map, rev_map,
                use_kv_cache=True, encoder_stride=1,
            )
            text = decode_tokens(model, ids)
            pruned_refs.append(ref)
            pruned_hyps.append(text)
            pruned_total_time += wall
            pruned_total_tokens += len(ids) - 1
            print(f"  Sample {idx}: {len(ids)-1:3d} tok, {wall:.3f}s | {text[:60]}")
        
        pruned_wer = compute_wer(pruned_refs, pruned_hyps)
        speedup = greedy_total_time / pruned_total_time if pruned_total_time > 0 else 0
        
        result = {
            "prune_size": prune_size,
            "wer": round(pruned_wer, 6),
            "wer_delta": round(pruned_wer - greedy_wer, 6),
            "total_time_s": round(pruned_total_time, 4),
            "tokens_per_sec": round(pruned_total_tokens / pruned_total_time, 2) if pruned_total_time > 0 else 0,
            "speedup": round(speedup, 3),
            "lm_head_params_full": full_params,
            "lm_head_params_pruned": pruned_params,
            "param_savings_pct": round((1 - pruned_params / full_params) * 100, 1),
        }
        results_all.append(result)
        
        print(f"  WER: {pruned_wer:.4f} (Δ={pruned_wer-greedy_wer:+.4f})")
        print(f"  Time: {pruned_total_time:.3f}s, Speedup: {speedup:.3f}×")
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"  RESULTS — P14 Dynamic Vocabulary Pruning ({args.model})")
    print(f"{'='*80}")
    print(f"{'Vocab':<8} {'WER':>8} {'ΔWER':>8} {'Time(s)':>8} {'Tok/s':>8} {'Speedup':>8} {'Savings':>8}")
    print("-" * 80)
    print(f"{'Full':<8} {greedy_wer:>8.4f} {'—':>8} {greedy_total_time:>8.3f} "
          f"{greedy_total_tokens/greedy_total_time:>8.1f} {'1.000':>8} {'0%':>8}")
    for r in results_all:
        print(f"{r['prune_size']:<8} {r['wer']:>8.4f} {r['wer_delta']:>+8.4f} "
              f"{r['total_time_s']:>8.3f} {r['tokens_per_sec']:>8.1f} "
              f"{r['speedup']:>8.3f}× {r['param_savings_pct']:>7.1f}%")
    print("=" * 80)
    
    # Save
    out_path = args.output or f"results/p14_vocab_pruning_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P14: Dynamic Vocabulary Pruning",
            "model": args.model,
            "vocab_usage": usage,
            "greedy_wer": round(greedy_wer, 6),
            "greedy_time_s": round(greedy_total_time, 4),
            "results": results_all,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
