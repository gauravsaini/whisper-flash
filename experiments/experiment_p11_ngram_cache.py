"""P11: N-gram / Phrase Cache Speculation for ASR.

Discrete token speculation that avoids the branching ceiling entirely.
Instead of predicting continuous hidden states, we build a token n-gram
cache from previously decoded tokens and speculate the next N tokens
by looking up the cache. Only tokens the model has already committed to
producing are speculated.

Key insight: ASR transcription has extremely repetitive patterns:
  - Common bigrams: "the", "of the", "in the", "it is", etc.
  - Within a single audio, many subsequences repeat (names, terms)
  - Cross-audio: language model priors are stable

Architecture:
  1. Build n-gram cache from training data OR from the current decoding stream
  2. At each step, look up the last K tokens in the cache
  3. If there's a match, speculate the next M tokens from the cache
  4. Verify via single-forward target decoder pass (batch verify M candidates)
  5. Accept longest matching prefix

This is composable with Q8, KV cache, and stride-2 (all validated).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
    crop_self_attention_cache,
)
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ══════════════════════════════════════════════════════════════════
# N-gram Cache
# ══════════════════════════════════════════════════════════════════

class NgramCache:
    """Token n-gram cache for speculation.
    
    Stores mappings from token n-grams to their most common continuations.
    Built from a corpus of decoded token sequences.
    """
    
    def __init__(self, context_len: int = 3, max_speculation: int = 5):
        """
        Args:
            context_len: Number of preceding tokens to use as context key.
            max_speculation: Maximum tokens to speculate from cache.
        """
        self.context_len = context_len
        self.max_speculation = max_speculation
        # {(tok_1, ..., tok_k): [(next_tok_1, next_tok_2, ..., next_tok_m), count]}
        self.cache: dict[tuple, list[tuple[tuple, int]]] = defaultdict(list)
    
    def add_sequence(self, token_ids: list[int]):
        """Add a decoded sequence to the cache."""
        # Filter out special tokens
        tokens = [t for t in token_ids if t < EOS_ID]
        
        for ctx_len in range(1, self.context_len + 1):
            for i in range(len(tokens) - ctx_len):
                context = tuple(tokens[i:i + ctx_len])
                # Store continuations of varying lengths
                for spec_len in range(1, min(self.max_speculation + 1, len(tokens) - i - ctx_len + 1)):
                    continuation = tuple(tokens[i + ctx_len:i + ctx_len + spec_len])
                    self._add_continuation(context, continuation)
    
    def _add_continuation(self, context: tuple, continuation: tuple):
        """Add or increment a continuation for a context."""
        entries = self.cache[context]
        for j, (cont, count) in enumerate(entries):
            if cont == continuation:
                entries[j] = (cont, count + 1)
                return
        entries.append((continuation, 1))
        # Keep sorted by count (descending), limit size
        entries.sort(key=lambda x: -x[1])
        if len(entries) > 10:
            self.cache[context] = entries[:10]
    
    def lookup(self, recent_tokens: list[int]) -> list[int] | None:
        """Look up the most likely continuation for recent tokens.
        
        Tries longest context match first, falls back to shorter.
        Returns the best continuation tokens or None.
        """
        tokens = [t for t in recent_tokens if t < EOS_ID]
        
        for ctx_len in range(min(self.context_len, len(tokens)), 0, -1):
            context = tuple(tokens[-ctx_len:])
            if context in self.cache:
                entries = self.cache[context]
                if entries:
                    # Return the longest high-confidence continuation
                    best = max(entries, key=lambda x: (len(x[0]), x[1]))
                    if best[1] >= 1:  # minimum count threshold
                        return list(best[0])
        return None
    
    def stats(self) -> dict:
        """Return cache statistics."""
        total_entries = sum(len(v) for v in self.cache.values())
        return {
            "n_contexts": len(self.cache),
            "n_continuations": total_entries,
            "context_len": self.context_len,
            "max_speculation": self.max_speculation,
        }


# ══════════════════════════════════════════════════════════════════
# Speculative decoder with n-gram cache
# ══════════════════════════════════════════════════════════════════

def downsample_encoder(enc: mx.array, stride: int) -> mx.array:
    if stride <= 1:
        return enc
    B, T, D = enc.shape
    T_trim = (T // stride) * stride
    return mx.mean(enc[:, :T_trim, :].reshape(B, T_trim // stride, stride, D), axis=2)


def generate_ngram_speculative(
    model, mel: mx.array, cache: NgramCache, *,
    use_kv_cache: bool = True,
    encoder_stride: int = 1,
    max_tokens: int = 448,
) -> tuple[list[int], float, dict]:
    """Greedy decode with n-gram speculation.
    
    Returns (token_ids, wall_time_s, stats).
    """
    t0 = time.perf_counter()
    
    stats = {
        "n_speculation_attempts": 0,
        "n_speculation_hits": 0,
        "total_tokens_speculated": 0,
        "total_tokens_accepted": 0,
        "decoder_calls": 0,
    }

    # Encode
    enc = encoder_forward(model, mel)
    mx.eval(enc)
    if encoder_stride > 1:
        enc = downsample_encoder(enc, encoder_stride)
        mx.eval(enc)

    # Prefill SOT
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]
    stats["decoder_calls"] += 1

    while len(output_ids) < max_tokens:
        last_tok = output_ids[-1]
        if last_tok == EOS_ID:
            break

        # Try n-gram speculation
        draft_tokens = cache.lookup(output_ids[-cache.context_len:])
        
        if draft_tokens and len(draft_tokens) > 0:
            stats["n_speculation_attempts"] += 1
            stats["total_tokens_speculated"] += len(draft_tokens)
            
            # Verify draft tokens via batched forward pass
            # Build input: [last_tok, draft_0, draft_1, ..., draft_{M-1}]
            verify_input = [last_tok] + draft_tokens
            inp = mx.array([verify_input], dtype=mx.int32)
            
            if use_kv_cache:
                logits_block, new_kv, _ = decoder_forward_with_hidden_states(
                    model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            else:
                full_seq = mx.array([output_ids + draft_tokens], dtype=mx.int32)
                logits_block, _, _ = decoder_forward_with_hidden_states(
                    model, full_seq, enc, kv_cache=None, collect_hidden_states=False)
                # Only take the last len(verify_input) logits
                logits_block = logits_block[:, -len(verify_input):, :]
            
            stats["decoder_calls"] += 1
            mx.eval(logits_block)
            
            # Verify: check each speculated token against target's greedy output
            n_accepted = 0
            for k in range(len(draft_tokens)):
                # logits_block[:, k, :] predicts the token at position k+1 of verify_input
                target_tok = mx.argmax(logits_block[:, k, :], axis=-1).item()
                if target_tok == draft_tokens[k]:
                    n_accepted += 1
                else:
                    break
            
            if n_accepted > 0:
                stats["n_speculation_hits"] += 1
                stats["total_tokens_accepted"] += n_accepted
                
                # Accept the matched prefix
                output_ids.extend(draft_tokens[:n_accepted])
                
                if use_kv_cache:
                    # Crop KV cache to only include accepted tokens
                    # Total self-attn length should be: original + 1 (last_tok) + n_accepted
                    target_len = len(output_ids)  # includes SOT + all accepted
                    kv_cache = crop_self_attention_cache(new_kv, target_len)
                
                # Get the token after the last accepted draft token
                # It's at position n_accepted in logits_block
                next_tok = mx.argmax(logits_block[:, n_accepted, :], axis=-1).item()
                output_ids.append(next_tok)
                
                if next_tok == EOS_ID:
                    break
                continue
            else:
                # All draft tokens rejected — fall back to greedy
                # Use the logits at position 0 (predicts next token after last_tok)
                next_tok = mx.argmax(logits_block[:, 0, :], axis=-1).item()
                output_ids.append(next_tok)
                
                if use_kv_cache:
                    # Crop KV cache to only first token
                    target_len = len(output_ids)
                    kv_cache = crop_self_attention_cache(new_kv, target_len)
                
                if next_tok == EOS_ID:
                    break
                continue
        
        # No n-gram match — standard greedy step
        inp = mx.array([[last_tok]], dtype=mx.int32)
        if use_kv_cache:
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        else:
            full_seq = mx.array([output_ids], dtype=mx.int32)
            logits, _, _ = decoder_forward_with_hidden_states(
                model, full_seq, enc, kv_cache=None, collect_hidden_states=False)
        
        tok = sample(logits[:, -1:, :], 0.0)
        mx.eval(tok)
        stats["decoder_calls"] += 1
        tid = tok.item()
        output_ids.append(tid)
        if tid == EOS_ID:
            break

    t1 = time.perf_counter()
    return output_ids, t1 - t0, stats


# ══════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════

def load_dataset(n_samples: int = 20):
    from datasets import load_dataset as hf_load
    from mlx_whisper.audio import log_mel_spectrogram

    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation", )
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
    parser = argparse.ArgumentParser(description="P11: N-gram Cache Speculation")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--n-train", type=int, default=10, help="Samples to build cache from")
    parser.add_argument("--n-eval", type=int, default=10, help="Samples to evaluate on")
    parser.add_argument("--context-len", type=int, default=3, help="N-gram context length")
    parser.add_argument("--max-spec", type=int, default=5, help="Max speculation length")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--encoder-stride", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  P11: N-gram Cache Speculation")
    print(f"  Model: {args.model}")
    print(f"  Context: {args.context_len}, Max Spec: {args.max_spec}")
    print(f"{'#'*60}")

    # Load model
    model = load_target_model(args.model, dtype=mx.float16)
    if args.quantize:
        quantize_model(model, encoder_bits=8, decoder_bits=8, group_size=64)

    total_samples = args.n_train + args.n_eval
    all_samples = load_dataset(total_samples)
    train_samples = all_samples[:args.n_train]
    eval_samples = all_samples[args.n_train:args.n_train + args.n_eval]

    print(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

    # Phase 1: Build n-gram cache from training samples (greedy decode)
    print(f"\n--- Building n-gram cache from {len(train_samples)} training samples ---")
    ngram_cache = NgramCache(context_len=args.context_len, max_speculation=args.max_spec)
    
    for mel, ref, idx in train_samples:
        # Greedy decode to get token sequence
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, dec, encoder_forward(model, mel), kv_cache=None, collect_hidden_states=False)
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        ids = [SOT_ID, first.item()]
        enc = encoder_forward(model, mel)
        mx.eval(enc)
        
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
        
        ngram_cache.add_sequence(ids)
        text = decode_tokens(model, ids)
        print(f"  Train {idx}: {len(ids)} tokens | {text[:60]}")

    cache_stats = ngram_cache.stats()
    print(f"\nCache built: {cache_stats}")

    # Phase 2: Evaluate with and without n-gram speculation
    print(f"\n--- Evaluating on {len(eval_samples)} samples ---")
    
    # Greedy baseline
    print(f"\n  [Greedy Baseline]")
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
        print(f"    Sample {idx}: {len(ids)-1:3d} tok, {wall:.3f}s | {text[:60]}")
    
    greedy_wer = compute_wer(greedy_refs, greedy_hyps)
    print(f"  Greedy WER: {greedy_wer:.4f}, Time: {greedy_total_time:.3f}s, Tok/s: {greedy_total_tokens/greedy_total_time:.1f}")
    
    # N-gram speculative
    print(f"\n  [N-gram Speculation]")
    spec_refs, spec_hyps = [], []
    spec_total_time = 0
    spec_total_tokens = 0
    total_stats = {
        "n_speculation_attempts": 0,
        "n_speculation_hits": 0,
        "total_tokens_speculated": 0,
        "total_tokens_accepted": 0,
        "decoder_calls": 0,
    }
    
    for mel, ref, idx in eval_samples:
        ids, wall, stats = generate_ngram_speculative(
            model, mel, ngram_cache,
            use_kv_cache=True,
            encoder_stride=args.encoder_stride,
        )
        text = decode_tokens(model, ids)
        spec_refs.append(ref)
        spec_hyps.append(text)
        spec_total_time += wall
        spec_total_tokens += len(ids) - 1
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"    Sample {idx}: {len(ids)-1:3d} tok, {wall:.3f}s, "
              f"spec_attempts={stats['n_speculation_attempts']}, "
              f"hits={stats['n_speculation_hits']}, "
              f"accepted={stats['total_tokens_accepted']} | {text[:50]}")

    spec_wer = compute_wer(spec_refs, spec_hyps)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"  RESULTS — P11 N-gram Speculation")
    print(f"{'='*70}")
    print(f"  Greedy:     WER={greedy_wer:.4f}, Time={greedy_total_time:.3f}s, "
          f"Tok/s={greedy_total_tokens/greedy_total_time:.1f}")
    print(f"  N-gram:     WER={spec_wer:.4f}, Time={spec_total_time:.3f}s, "
          f"Tok/s={spec_total_tokens/spec_total_time:.1f}")
    print(f"  WER Delta:  {spec_wer - greedy_wer:+.4f}")
    print(f"  Speedup:    {greedy_total_time/spec_total_time:.3f}×")
    print(f"  Speculation stats: {total_stats}")
    hit_rate = (total_stats['n_speculation_hits'] / total_stats['n_speculation_attempts'] * 100 
                if total_stats['n_speculation_attempts'] > 0 else 0)
    print(f"  Hit rate: {hit_rate:.1f}%")
    accept_rate = (total_stats['total_tokens_accepted'] / total_stats['total_tokens_speculated'] * 100
                   if total_stats['total_tokens_speculated'] > 0 else 0)
    print(f"  Token accept rate: {accept_rate:.1f}%")
    print(f"  Decoder calls saved: {greedy_total_tokens - total_stats['decoder_calls']}"
          f" ({(1 - total_stats['decoder_calls']/greedy_total_tokens)*100:.1f}%)")
    print(f"{'='*70}")

    # Save results
    out_path = args.output or f"results/p11_ngram_spec_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P11: N-gram Cache Speculation",
            "model": args.model,
            "context_len": args.context_len,
            "max_speculation": args.max_spec,
            "n_train": len(train_samples),
            "n_eval": len(eval_samples),
            "cache_stats": cache_stats,
            "greedy": {
                "wer": round(greedy_wer, 6),
                "total_time_s": round(greedy_total_time, 4),
                "tokens_per_sec": round(greedy_total_tokens / greedy_total_time, 2),
            },
            "ngram": {
                "wer": round(spec_wer, 6),
                "total_time_s": round(spec_total_time, 4),
                "tokens_per_sec": round(spec_total_tokens / spec_total_time, 2),
                "wer_delta": round(spec_wer - greedy_wer, 6),
                "speedup": round(greedy_total_time / spec_total_time, 3),
            },
            "speculation_stats": total_stats,
            "hit_rate_pct": round(hit_rate, 2),
            "token_accept_rate_pct": round(accept_rate, 2),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
