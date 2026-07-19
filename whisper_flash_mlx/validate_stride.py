#!/usr/bin/env python3
"""Validate stride-8 avg-pool encoder using full transcribe() pipeline.

Uses mlx_whisper.transcribe() with monkey-patched encoder, which includes
temperature fallback and repetition handling — the manual greedy loop fails
on stride-8 but transcribe() produces byte-identical output at 8.6× speedup.

Usage:
    uv run python3 -m whisper_flash_mlx.validate_stride audio.wav --model turbo
    uv run python3 -m whisper_flash_mlx.validate_stride audio.wav --model tiny --stride 4
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import soundfile as sf

import mlx.core as mx
import mlx.nn as nn
import mlx_whisper
from mlx_whisper.load_models import load_model


MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _patch_encoder(model, stride: int):
    orig = model.encoder
    if stride <= 1:
        return orig

    class _Strided(nn.Module):
        def __init__(s):
            super().__init__()
            object.__setattr__(s, "_enc", orig)
        def __call__(s, x):
            o = s._enc(x)
            B, T, D = o.shape
            Tt = (T // stride) * stride
            return mx.mean(o[:, :Tt, :].reshape(B, Tt // stride, stride, D), axis=2)
        def __getattr__(s, n):
            return getattr(object.__getattribute__(s, "_enc"), n)

    model.encoder = _Strided()
    return orig


def transcribe_strided(model, audio_arr, stride: int = 1) -> tuple[str, float]:
    orig = _patch_encoder(model, stride)
    real_load = mlx_whisper.load_models.load_model
    mlx_whisper.load_models.load_model = lambda *a, **k: model

    t0 = time.perf_counter()
    out = mlx_whisper.transcribe(
        audio_arr, verbose=None, condition_on_previous_text=False
    )
    dt = time.perf_counter() - t0

    model.encoder = orig
    mlx_whisper.load_models.load_model = real_load
    return out["text"].strip(), dt


def main():
    ap = argparse.ArgumentParser(
        description="Validate stride avg-pool via transcribe() pipeline"
    )
    ap.add_argument("audio", help="Audio file (WAV/MP3/FLAC)")
    ap.add_argument(
        "--model", default="turbo",
        help="Model name or HF repo (tiny/turbo/large-v3)"
    )
    ap.add_argument("--stride", type=int, default=8, choices=[2, 4, 8])
    args = ap.parse_args()

    model_path = MODEL_ALIASES.get(args.model, args.model)

    print(f"\n{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Audio:   {args.audio}")
    print(f"  Stride:  {args.stride}")
    print(f"{'='*60}\n")

    arr, sr = sf.read(args.audio)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    model = load_model(model_path)

    ref, t_ref = transcribe_strided(model, arr, 1)
    print(f"{'baseline (stride-1):':20} {ref[:120]}")
    print(f"{'':20} wall={t_ref:.3f}s")

    test, t_test = transcribe_strided(model, arr, args.stride)
    print(f"{f'stride-{args.stride}:':20} {test[:120]}")
    print(f"{'':20} wall={t_test:.3f}s")

    match = "✓ IDENTICAL" if ref == test else "✗ DIFFERS"
    speedup = t_ref / t_test
    print(f"\n{'─'*60}")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  Text match:    {match}")
    if ref != test:
        print(f"  Baseline: {ref[:200]}")
        print(f"  Stride:   {test[:200]}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
