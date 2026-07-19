"""P10: Combined Stack Benchmark — All 4 Production Levers.

Measures wall-clock speedup of composable optimisation combinations:
  - Baseline (fp16, no cache, stride-1)
  - Q8 only
  - KV cache only
  - Stride-2 only
  - Q8 + KV cache
  - Q8 + stride-2
  - KV cache + stride-2
  - Q8 + KV cache + stride-2  ← FULL STACK

Reports WER, tokens/sec, wall-clock time, and multiplicative speedup for each.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

# ── Load model utilities ──────────────────────────────────────────
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
# Greedy decoder with composable levers
# ══════════════════════════════════════════════════════════════════

def downsample_encoder(enc: mx.array, stride: int) -> mx.array:
    """Average-pool encoder output along time axis."""
    if stride <= 1:
        return enc
    B, T, D = enc.shape
    T_trim = (T // stride) * stride
    enc_trimmed = enc[:, :T_trim, :]
    return mx.mean(enc_trimmed.reshape(B, T_trim // stride, stride, D), axis=2)


def generate_greedy(
    model, mel: mx.array, *,
    use_kv_cache: bool = False,
    encoder_stride: int = 1,
    max_tokens: int = 448,
) -> tuple[list[int], float]:
    """Greedy decode with composable levers. Returns (token_ids, wall_time_s)."""
    t0 = time.perf_counter()

    # Encode
    enc = encoder_forward(model, mel)
    mx.eval(enc)

    # Downsample encoder frames
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

    if not use_kv_cache:
        # No KV cache: pass full token sequence each step
        dec = mx.concatenate([dec, first], axis=1)
        while len(output_ids) < max_tokens:
            logits, _, _ = decoder_forward_with_hidden_states(
                model, dec, enc, kv_cache=None, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            tid = tok.item()
            output_ids.append(tid)
            if tid == EOS_ID:
                break
            dec = mx.concatenate([dec, tok], axis=1)
    else:
        # KV cache: pass only last token + cached context
        while len(output_ids) < max_tokens:
            last_tok = output_ids[-1]
            inp = mx.array([[last_tok]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            tid = tok.item()
            output_ids.append(tid)
            if tid == EOS_ID:
                break

    t1 = time.perf_counter()
    return output_ids, t1 - t0


# ══════════════════════════════════════════════════════════════════
# Benchmark harness
# ══════════════════════════════════════════════════════════════════

def load_dataset(n_samples: int = 10):
    """Load LibriSpeech dummy clean validation."""
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


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Compute aggregate WER."""
    from jiwer import wer
    # Normalize
    refs_clean = [r.strip().lower() for r in refs]
    hyps_clean = [h.strip().lower() for h in hyps]
    return wer(refs_clean, hyps_clean)


def decode_tokens(model, token_ids: list[int]) -> str:
    """Decode token ids to text."""
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)
    text_tokens = [t for t in token_ids[1:] if t < tokenizer.eot]  # skip SOT, skip special
    return tokenizer.decode(text_tokens).strip()


def run_configuration(
    model_path: str,
    samples: list,
    *,
    quantize: bool = False,
    use_kv_cache: bool = False,
    encoder_stride: int = 1,
    warmup: int = 1,
    runs: int = 1,
) -> dict:
    """Run a single configuration and return metrics."""
    # Load fresh model for each config (quantization is in-place)
    model = load_target_model(model_path, dtype=mx.float16)
    if quantize:
        quantize_model(model, encoder_bits=8, decoder_bits=8, group_size=64)

    config_name = []
    if quantize:
        config_name.append("Q8")
    if use_kv_cache:
        config_name.append("KV")
    if encoder_stride > 1:
        config_name.append(f"S{encoder_stride}")
    name = "+".join(config_name) if config_name else "Baseline"

    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"  Q8={quantize}, KV={use_kv_cache}, Stride={encoder_stride}")
    print(f"{'='*60}")

    # Warmup
    for _ in range(warmup):
        mel, _, _ = samples[0]
        generate_greedy(model, mel, use_kv_cache=use_kv_cache,
                       encoder_stride=encoder_stride, max_tokens=50)

    # Benchmark runs
    all_times = []
    all_tokens = []
    refs = []
    hyps = []

    for run_idx in range(runs):
        run_times = []
        run_token_counts = []
        for mel, ref, idx in samples:
            ids, wall = generate_greedy(
                model, mel,
                use_kv_cache=use_kv_cache,
                encoder_stride=encoder_stride,
            )
            text = decode_tokens(model, ids)
            run_times.append(wall)
            run_token_counts.append(len(ids) - 1)  # exclude SOT
            if run_idx == 0:
                refs.append(ref)
                hyps.append(text)
                print(f"  Sample {idx:2d}: {len(ids)-1:3d} tokens, {wall:.3f}s | {text[:60]}")

        all_times.append(sum(run_times))
        all_tokens.append(sum(run_token_counts))

    wer_val = compute_wer(refs, hyps)
    avg_total_time = sum(all_times) / len(all_times)
    avg_total_tokens = sum(all_tokens) / len(all_tokens)
    tps = avg_total_tokens / avg_total_time if avg_total_time > 0 else 0

    result = {
        "config": name,
        "quantize": quantize,
        "kv_cache": use_kv_cache,
        "encoder_stride": encoder_stride,
        "wer": round(wer_val, 6),
        "total_time_s": round(avg_total_time, 4),
        "tokens_per_sec": round(tps, 2),
        "total_tokens": int(avg_total_tokens),
        "n_samples": len(samples),
        "n_runs": runs,
    }

    print(f"\n  WER: {wer_val:.4f}")
    print(f"  Total time: {avg_total_time:.3f}s")
    print(f"  Tokens/sec: {tps:.1f}")

    return result


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P10: Combined Stack Benchmark")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx",
                        help="Model path (mlx-community/whisper-tiny-mlx or whisper-large-v3-mlx)")
    parser.add_argument("--samples", type=int, default=10, help="Number of eval samples")
    parser.add_argument("--runs", type=int, default=2, help="Benchmark runs per config")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  P10: Combined Stack Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.samples}, Runs: {args.runs}")
    print(f"{'#'*60}")

    samples = load_dataset(args.samples)
    print(f"Loaded {len(samples)} samples")

    # All 2^3 combinations of {Q8, KV, Stride-2}
    configs = [
        # Baseline
        dict(quantize=False, use_kv_cache=False, encoder_stride=1),
        # Single levers
        dict(quantize=True,  use_kv_cache=False, encoder_stride=1),
        dict(quantize=False, use_kv_cache=True,  encoder_stride=1),
        dict(quantize=False, use_kv_cache=False, encoder_stride=2),
        # Double combos
        dict(quantize=True,  use_kv_cache=True,  encoder_stride=1),
        dict(quantize=True,  use_kv_cache=False, encoder_stride=2),
        dict(quantize=False, use_kv_cache=True,  encoder_stride=2),
        # FULL STACK
        dict(quantize=True,  use_kv_cache=True,  encoder_stride=2),
    ]

    results = []
    for cfg in configs:
        r = run_configuration(
            args.model, samples,
            warmup=args.warmup, runs=args.runs,
            **cfg,
        )
        results.append(r)

    # Compute speedups relative to baseline
    baseline_time = results[0]["total_time_s"]
    for r in results:
        r["speedup"] = round(baseline_time / r["total_time_s"], 3) if r["total_time_s"] > 0 else 0
        r["wer_delta"] = round(r["wer"] - results[0]["wer"], 6)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  RESULTS SUMMARY — {args.model}")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'WER':>8} {'ΔWER':>8} {'Time(s)':>8} {'Tok/s':>8} {'Speedup':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['config']:<25} {r['wer']:>8.4f} {r['wer_delta']:>+8.4f} "
              f"{r['total_time_s']:>8.3f} {r['tokens_per_sec']:>8.1f} {r['speedup']:>8.3f}×")
    print("=" * 80)

    # Save results
    out_path = args.output or f"results/p10_combined_stack_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P10: Combined Stack Benchmark",
            "model": args.model,
            "n_samples": len(samples),
            "n_runs": args.runs,
            "results": results,
            "baseline_time_s": baseline_time,
            "best_speedup": max(r["speedup"] for r in results),
            "best_config": max(results, key=lambda r: r["speedup"])["config"],
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
