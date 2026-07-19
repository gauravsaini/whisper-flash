#!/usr/bin/env python3
"""
Benchmark KV cache speedup for the adaptive multi-path gate.

Compares:
  - generate_adaptive(use_kv_cache=False)  — full-sequence pass every step
  - generate_adaptive(use_kv_cache=True)   — single-token pass with KV cache

Measures:
  - Wall-clock time per sample
  - Token generation rate (tokens/sec)
  - WER equivalence (both modes must produce identical results)
"""

import time
import numpy as np
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward
from experiment_adaptive_multipath import (
    CorrectionDrafter, generate_adaptive, compute_pca_basis,
    to_pca, generate_greedy, norm, run_model
)

N_SAMPLES = 5   # quick correctness check
MAX_TOKENS = 150

def benchmark():
    model_name = "mlx-community/whisper-tiny"
    pca_rank = 64
    d_draft = 256
    n_train = 5

    print("=" * 60)
    print(f"KV CACHE BENCHMARK — {model_name}")
    print("=" * 60)

    # Load model
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Quick train on 5 samples
    print(f"\nTraining on {n_train} samples...")
    train_data = []
    for i in range(n_train):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(s["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        labels = mx.concatenate([
            mx.array([[tokenizer.sol]], dtype=mx.int32),
            mx.array([text_tokens], dtype=mx.int32)], axis=1) if target.is_multilingual else \
            mx.concatenate([
                mx.array([[tokenizer.sot]], dtype=mx.int32),
                mx.array([tokenizer.sot_prev] if hasattr(tokenizer, 'sot_prev') else [],
                         dtype=mx.int32),
                mx.array([text_tokens], dtype=mx.int32)], axis=1)

    print(f"\n{'='*60}")
    print(f"WALL-CLOCK COMPARISON — {N_SAMPLES} eval samples")
    print(f"{'='*60}")

    times_no_cache = []
    times_with_cache = []
    wer_no_cache = []
    wer_with_cache = []

    for i in range(N_SAMPLES):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        ref_text = s["text"]

        # --- No KV cache ---
        t0 = time.perf_counter()
        text_no, _, _ = generate_adaptive(
            target, drafter, tokenizer, mel_mx,
            pca_mean=pca_mean, pca_V=pca_V,
            static_k=1, use_kv_cache=False)
        dt_no = time.perf_counter() - t0

        # --- With KV cache ---
        t0 = time.perf_counter()
        text_yes, _, _ = generate_adaptive(
            target, drafter, tokenizer, mel_mx,
            pca_mean=pca_mean, pca_V=pca_V,
            static_k=1, use_kv_cache=True)
        dt_yes = time.perf_counter() - t0

        times_no_cache.append(dt_no)
        times_with_cache.append(dt_yes)

        w_no = jiwer.wer(norm(ref_text), norm(text_no)) if ref_text.strip() else 1.0
        w_yes = jiwer.wer(norm(ref_text), norm(text_yes)) if ref_text.strip() else 1.0
        wer_no_cache.append(w_no)
        wer_with_cache.append(w_yes)

        match = "✓" if text_no == text_yes else "✗ MISMATCH"
        print(f"  [{i}] no-cache={dt_no:.3f}s  kv-cache={dt_yes:.3f}s  "
              f"speedup={dt_no/dt_yes:.1f}x  WER={w_no:.4f}/{w_yes:.4f}  {match}")

    # Summary
    mean_no = np.mean(times_no_cache)
    mean_yes = np.mean(times_with_cache)
    print(f"\n{'='*60}")
    print(f"RESULTS — {N_SAMPLES} samples")
    print(f"{'='*60}")
    print(f"  No KV cache : {mean_no:.4f}s/sample  ({N_SAMPLES/mean_no:.1f} samples/sec)")
    print(f"  With KV cache: {mean_yes:.4f}s/sample  ({N_SAMPLES/mean_yes:.1f} samples/sec)")
    print(f"  Speedup     : {mean_no/mean_yes:.1f}x")
    print(f"  WER no cache: {np.mean(wer_no_cache):.4f}")
    print(f"  WER w cache : {np.mean(wer_with_cache):.4f}")
    print(f"  Match rate  : {sum(1 for a,b in zip(times_no_cache, times_with_cache) if a==b)}/{N_SAMPLES}")


if __name__ == "__main__":
    import mlx.core as mx
    import jiwer
    benchmark()
