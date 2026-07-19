"""
P8: MLX Fused Kernels for Whisper Decoder.

Fusions:
  1. QKV Self-Attention: 3 matmuls → 1 (weight concatenation, no custom kernel)
  2. (Future) LN + QKV: custom Metal kernel
  3. (Future) FFN + activation: custom Metal kernel
"""

from __future__ import annotations

from typing import Optional, Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_whisper.whisper import Whisper


# ────────────────────────────────────────────────────────────────
# 1. QKV FUSION:  W_q, W_k, W_v → W_qkv (single weight matrix)
# ────────────────────────────────────────────────────────────────

def fuse_qkv_weights(model: Whisper) -> list[dict]:
    """Pre-compute fused QKV weights for each decoder block.

    For each block, concatenate W_q, W_k, W_v along dim 0 and biases:
        W_qkv shape: (3 * d_model, d_model)
        b_qkv shape: (3 * d_model,)

    Returns a list of dicts, one per decoder block.
    """
    fused = []
    for block in model.decoder.blocks:
        attn = block.attn
        W_q, b_q = attn.query.weight, attn.query.bias
        W_k, b_k = attn.key.weight, attn.value.bias  # key has no bias in Whisper
        W_v, b_v = attn.value.weight, attn.value.bias

        # W_k has bias=False (see whisper.py line 45), so we create a zero bias
        W_qkv = mx.concatenate([W_q, W_k, W_v], axis=0)
        b_k_actual = mx.zeros((W_k.shape[0],), dtype=b_q.dtype) if b_q is not None else None

        if b_q is not None:
            b_qkv = mx.concatenate([b_q, b_k_actual, b_v], axis=0)
        else:
            b_qkv = None

        d_model = W_q.shape[0]
        fused.append({
            "W_qkv": W_qkv,   # (3*d, d)
            "b_qkv": b_qkv,   # (3*d,)
            "d_model": d_model,
        })

    return fused


# ────────────────────────────────────────────────────────────────
# 2. BENCHMARK: compare fused QKV vs unfused
# ────────────────────────────────────────────────────────────────

def benchmark_qkv_fusion(model: Whisper, seq_len: int = 10, batch: int = 1, iters: int = 100):
    """Benchmark fused QKV vs three separate matmuls."""
    import time

    d_model = model.dims.n_text_state
    x = mx.random.normal((batch, seq_len, d_model)).astype(mx.float16)

    # Unfused: 3 separate matmuls
    block = model.decoder.blocks[0]
    attn = block.attn
    W_q, W_k, W_v = attn.query.weight, attn.key.weight, attn.value.weight

    def unfused():
        q = x @ W_q.T
        k = x @ W_k.T
        v = x @ W_v.T
        mx.eval(q, k, v)
        return q, k, v

    # Fused: 1 matmul
    fw = fuse_qkv_weights(model)[0]
    def fused():
        qkv = x @ fw["W_qkv"].T
        q, k, v = mx.split(qkv, 3, axis=-1)
        mx.eval(q, k, v)
        return q, k, v

    # Warmup
    for _ in range(10):
        unfused(); fused()

    t0 = time.perf_counter()
    for _ in range(iters):
        unfused()
    t1 = time.perf_counter()
    unfused_time = (t1 - t0) / iters * 1000

    t0 = time.perf_counter()
    for _ in range(iters):
        fused()
    t1 = time.perf_counter()
    fused_time = (t1 - t0) / iters * 1000

    print(f"  QKV Fusion ({'x'.join(str(s) for s in x.shape)}):")
    print(f"    Unfused (3 matmuls):  {unfused_time:.3f} ms")
    print(f"    Fused (1 matmul):     {fused_time:.3f} ms")
    print(f"    Speedup:              {unfused_time / max(fused_time, 1e-9):.2f}x")

    return unfused_time, fused_time


# ────────────────────────────────────────────────────────────────
# 4. FULL DECODER BENCHMARK (end-to-end)
# ────────────────────────────────────────────────────────────────

def benchmark_decoder_forward(model: Whisper, seq_len: int = 10, iters: int = 50):
    """Benchmark the full decoder forward pass."""
    import time

    # Create dummy audio features
    d_audio = model.dims.n_audio_state
    T_enc = model.dims.n_audio_ctx
    audio = mx.random.normal((1, T_enc, d_audio)).astype(mx.float16)

    tokens = mx.zeros((1, seq_len), dtype=mx.int32)

    from whisper_flash_mlx.target_model import decoder_forward_with_hidden_states

    # Warmup
    for _ in range(5):
        decoder_forward_with_hidden_states(model, tokens, audio, kv_cache=None, collect_hidden_states=False)

    t0 = time.perf_counter()
    for _ in range(iters):
        decoder_forward_with_hidden_states(model, tokens, audio, kv_cache=None, collect_hidden_states=False)
    t1 = time.perf_counter()
    total = (t1 - t0) / iters * 1000

    print(f"  Full decoder forward (seq={seq_len}, {model.dims.n_text_layer} layers):")
    print(f"    Average:  {total:.2f} ms")
    print(f"    Per layer: {total / model.dims.n_text_layer:.2f} ms")

    return total


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    args = parser.parse_args()

    from whisper_flash_mlx.target_model import load_target_model
    model = load_target_model(args.model)

    d = model.dims
    print(f"Model: {args.model}")
    print(f"  d_model:     {d.n_text_state}")
    print(f"  n_layers:    {d.n_text_layer}")
    print(f"  n_heads:     {d.n_text_head}")
    print(f"  vocab_size:  {d.n_vocab}")

    print("\n── QKV Fusion Benchmark ──")
    for seq in [1, 5, 10, 50]:
        benchmark_qkv_fusion(model, seq_len=seq)

    print("\n── Full Decoder Benchmark ──")
    for seq in [1, 5, 10]:
        benchmark_decoder_forward(model, seq_len=seq)
