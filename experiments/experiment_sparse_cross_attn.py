#!/usr/bin/env python3
"""
P9: Frame-sparsified cross-attention (alignment-aware) — single-file eval.

Hypothesis: Whisper's decoder cross-attends to all 1500 encoder frames per
token, but each token only needs ~20-40 frames (its aligned acoustic segment).
Dynamically selecting relevant frames via monotonic frame-window sliding
should give a 1.5-3× speedup at zero WER loss.

Strategy:
  1. Warm-up phase (first N tokens): full cross-attention; estimate alignment
     rate (frames-per-token) from cross-attention peak positions.
  2. Sparse phase: sliding window of K frames centred at a linearly
     extrapolated position; slice the cross-attn KV cache so the decoder
     matmuls only K × d_model instead of 1500 × d_model.
  3. Periodic probes (every M steps): full cross-attention to re-estimate
     alignment rate and correct drift.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from whisper_flash_mlx.target_model import (
    build_sparse_working_cache,
    decoder_forward_with_hidden_states,
    encoder_forward,
    extract_cross_cache,
    load_target_model,
)
from whisper_flash_mlx.utils import sample


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_audio_file(path: str) -> np.ndarray:
    """Load a single audio file and return float32 mono array at 16 kHz."""
    import soundfile as sf
    arr, sr = sf.read(path)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Sparse baseline generator
# ---------------------------------------------------------------------------

def rebuild_with_full_cross(kv_cache, saved_full_cross_cache):
    """Merge current self-attn cache with saved full cross-attn cache."""
    working = []
    for i in range(len(kv_cache)):
        self_kv = kv_cache[i][0]
        if saved_full_cross_cache is not None and saved_full_cross_cache[i] is not None:
            kf, vf = saved_full_cross_cache[i]
            working.append((self_kv, (kf, vf)))
        else:
            working.append((self_kv, None))
    return working


def sparse_baseline_generate(
    target,
    mel: mx.array,
    max_new_tokens: int = 448,
    temperature: float = 0.0,
    warmup_tokens: int = 3,
    window_margin: int = 25,
    probe_interval: int | None = 20,
    min_window: int = 30,
    max_window: int = 200,
) -> dict:
    t0 = time.perf_counter()

    encoder_hidden = encoder_forward(target, mel)
    mx.eval(encoder_hidden)
    T_enc = encoder_hidden.shape[1]

    decoder_ids = mx.array([[50258]], dtype=mx.int32)
    kv_cache = None
    saved_full_cross_cache = None
    eos_token_id = 50257

    t_first = None
    n_full_steps = 0
    n_sparse_steps = 0
    total_frames_attended = 0
    # Heuristic alignment:  whisper's encoder produces ~1500 frames for 30 s,
    # and we expect ~200 output tokens → ~7.5 frames per token of speech.
    # The centre of the cross-attention window advances by this many frames
    # at each sparse step.  At every full/probe step we overwrite it with
    # the actual centre-of-mass from the cross-attention weights.
    fpt: float = 10.0   # default frames-per-token (dynamic below)
    frame_centre: int = T_enc // 8
    steps_since_full = 0
    prev_probe_com: float | None = None  # for dynamic fpt estimation

    for step in range(max_new_tokens):
        if kv_cache is None:
            input_tokens = decoder_ids
        else:
            input_tokens = decoder_ids[:, -1:]
        is_warmup = step < warmup_tokens
        is_probe = (not is_warmup
                     and probe_interval is not None
                     and step % probe_interval == 0)
        is_full_step = is_warmup or is_probe

        if is_full_step:
            # ── Full (or probe) step ──────────────────────────────────
            if is_probe and saved_full_cross_cache is not None:
                full_working = rebuild_with_full_cross(kv_cache, saved_full_cross_cache)
            else:
                full_working = kv_cache

            logits, new_cache, _, cross_attns = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden,
                kv_cache=full_working,
                collect_hidden_states=False,
                return_cross_attention=True,
            )
            kv_cache = new_cache
            n_full_steps += 1
            steps_since_full = 0

            if is_warmup and saved_full_cross_cache is None:
                saved_full_cross_cache = extract_cross_cache(kv_cache)

            # Reset frame centre from cross-attention centre-of-mass.
            # Warmup tokens (SOT / language / transcribe) have scattered
            # COM; we only accept updates once the COM stabilises
            # (magnitude < 100-frame jump from previous).
            if cross_attns:
                profile = cross_attns[-1][0, :, -1, :].mean(axis=0)
                weights = mx.softmax(profile, axis=-1)
                indices = mx.arange(profile.shape[0], dtype=mx.float32)
                com = mx.sum(weights * indices).item()

                # Reject updates that jump > 100 frames (non-speech tokens)
                jump = abs(com - frame_centre)
                if jump < 100 or not is_warmup:
                    new_centre = int(0.7 * com + 0.3 * frame_centre)
                    frame_centre = max(0, min(T_enc - 1, new_centre))

                # Dynamic fpt:  distance from previous probe / steps elapsed
                if prev_probe_com is not None and not is_warmup:
                    steps = probe_interval if probe_interval else 1
                    delta_com = com - prev_probe_com
                    if delta_com > 0:
                        inferred = delta_com / steps
                        fpt = max(3.0, min(60.0, 0.5 * fpt + 0.5 * inferred))
                prev_probe_com = com
        else:
            # ── Sparse step ───────────────────────────────────────────
            steps_since_full += 1
            # Advance centre forward at the estimated frames-per-token rate
            centre = int(frame_centre + steps_since_full * fpt)
            centre = max(0, min(T_enc - 1, centre))
            start = max(0, centre - window_margin)
            end = min(T_enc, centre + window_margin)
            if end - start < min_window:
                mid = (start + end) // 2
                half = min_window // 2
                start = max(0, mid - half)
                end = min(T_enc, start + min_window)
            if end - start > max_window:
                mid = (start + end) // 2
                half = max_window // 2
                start = max(0, mid - half)
                end = min(T_enc, start + max_window)

            n_sparse_steps += 1
            total_frames_attended += (end - start)
            working_cache = build_sparse_working_cache(
                kv_cache, saved_full_cross_cache, start, end
            )
            logits, new_cache, _, _ = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden,
                kv_cache=working_cache,
                collect_hidden_states=False,
                return_cross_attention=True,
            )
            kv_cache = new_cache

        next_token = sample(logits[:, -1:, :], temperature)
        mx.eval(next_token)
        decoder_ids = mx.concatenate([decoder_ids, next_token], axis=1)
        if t_first is None:
            t_first = time.perf_counter() - t0
        if next_token.item() == eos_token_id:
            break

    total_time = time.perf_counter() - t0
    num_tokens = decoder_ids.shape[1] - 1
    return {
        "output_ids": decoder_ids,
        "num_tokens": num_tokens,
        "total_time": total_time,
        "time_to_first_token": t_first,
        "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
        "n_full_steps": n_full_steps,
        "n_sparse_steps": n_sparse_steps,
        "avg_frames_attended": (total_frames_attended / max(n_sparse_steps, 1) if n_sparse_steps else 0),
    }


def baseline_generate(target, mel, max_new_tokens=448, temperature=0.0):
    t0 = time.perf_counter()
    encoder_hidden = encoder_forward(target, mel)
    mx.eval(encoder_hidden)
    decoder_ids = mx.array([[50258]], dtype=mx.int32)
    t_first = None
    kv_cache = None
    eos_token_id = 50257
    for step in range(max_new_tokens):
        if kv_cache is None:
            input_tokens = decoder_ids
        else:
            input_tokens = decoder_ids[:, -1:]
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            target, input_tokens, encoder_hidden,
            kv_cache=kv_cache,
            collect_hidden_states=False,
        )
        next_token = sample(logits[:, -1:, :], temperature)
        mx.eval(next_token)
        decoder_ids = mx.concatenate([decoder_ids, next_token], axis=1)
        if t_first is None:
            t_first = time.perf_counter() - t0
        if next_token.item() == eos_token_id:
            break
    total_time = time.perf_counter() - t0
    num_tokens = decoder_ids.shape[1] - 1
    return {
        "output_ids": decoder_ids,
        "num_tokens": num_tokens,
        "total_time": total_time,
        "time_to_first_token": t_first,
        "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Evaluate on a single audio file
# ---------------------------------------------------------------------------

def evaluate_file(
    audio_path: str,
    model_name: str = "mlx-community/whisper-tiny-mlx",
    temperature: float = 0.0,
    warmup_tokens: int = 3,
    window_margin: int = 25,
    probe_interval: int | None = None,
    min_window: int = 30,
    max_window: int = 200,
):
    from mlx_whisper.audio import log_mel_spectrogram
    from mlx_whisper.tokenizer import get_tokenizer

    print(f"Model:  {model_name}")
    print(f"Audio:  {audio_path}")
    print(f"Sparse: margin={window_margin} warmup={warmup_tokens} probe={probe_interval}")

    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)

    audio = load_audio_file(audio_path)
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
    mel = mx.array(mel)[None]

    # Baseline
    bl = baseline_generate(target, mel, temperature=temperature)
    bl_text = tokenizer.decode(np.array(bl["output_ids"][0]).tolist())

    # Sparse
    sp = sparse_baseline_generate(
        target, mel,
        temperature=temperature,
        warmup_tokens=warmup_tokens,
        window_margin=window_margin,
        probe_interval=probe_interval,
        min_window=min_window,
        max_window=max_window,
    )
    sp_text = tokenizer.decode(np.array(sp["output_ids"][0]).tolist())

    print(f"\n{'─'*60}")
    print(f"BASELINE     : {bl_text}")
    print(f"SPARSE       : {sp_text}")
    print(f"{'─'*60}")
    print(f"Baseline:   {bl['tokens_per_second']:.1f} tok/s  ({bl['total_time']*1000:.0f} ms)")
    print(f"Sparse:     {sp['tokens_per_second']:.1f} tok/s  ({sp['total_time']*1000:.0f} ms)")
    print(f"Speedup:    {sp['tokens_per_second']/bl['tokens_per_second']:.3f}x")
    print(f"Sparse:     {sp['n_full_steps']} full + {sp['n_sparse_steps']} sparse steps")
    print(f"  Avg {sp['avg_frames_attended']:.0f} frames/sparse step")

    exact = bl_text == sp_text
    print(f"Exact match: {exact}")
    return {"exact_match": exact, "baseline": bl, "sparse": sp}


# ---------------------------------------------------------------------------
# Sweep over margins
# ---------------------------------------------------------------------------

def sweep_file(audio_path: str, model_name: str = "mlx-community/whisper-tiny-mlx"):
    margins = [10, 25, 50, 100, 250, 500]
    results = []
    for m in margins:
        print(f"\n{'='*60}")
        print(f"  MARGIN = {m}  (window = {m*2} frames)")
        print(f"{'='*60}")
        r = evaluate_file(audio_path, model_name=model_name, window_margin=m, probe_interval=None)
        results.append({"margin": m, **r})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P9: Frame-sparsified cross-attention")
    parser.add_argument("--audio", default="/tmp/jfk.flac")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-tokens", type=int, default=3)
    parser.add_argument("--window-margin", type=int, default=25)
    parser.add_argument("--probe-interval", type=int, default=0)
    parser.add_argument("--min-window", type=int, default=30)
    parser.add_argument("--max-window", type=int, default=200)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()
    probe_interval = args.probe_interval if args.probe_interval and args.probe_interval > 0 else None

    if args.sweep:
        results = sweep_file(args.audio, model_name=args.model)
        if args.output_json:
            simple = []
            for r in results:
                simple.append({
                    "margin": r["margin"],
                    "exact_match": r["exact_match"],
                    "baseline_tps": r["baseline"]["tokens_per_second"],
                    "sparse_tps": r["sparse"]["tokens_per_second"],
                    "speedup": r["sparse"]["tokens_per_second"] / r["baseline"]["tokens_per_second"],
                    "sparse_steps": r["sparse"]["n_sparse_steps"],
                    "avg_frames": r["sparse"]["avg_frames_attended"],
                })
            with open(args.output_json, "w") as f:
                json.dump(simple, f, indent=2)
            print(f"\nSweep saved to {args.output_json}")
    else:
        r = evaluate_file(
            args.audio, model_name=args.model,
            temperature=args.temperature,
            warmup_tokens=args.warmup_tokens,
            window_margin=args.window_margin,
            probe_interval=probe_interval,
            min_window=args.min_window,
            max_window=args.max_window,
        )
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(r, f, indent=2, default=str)
            print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
