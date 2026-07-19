"""E2: Encoder Downsampling — Halve the 1500 frames before the decoder sees them.

Hypothesis: Adjacent encoder frames are highly redundant (30ms hop, ~33 frames/sec).
Average-pooling encoder outputs from 1500 → 750 (or 500) frames will:
  - Halve cross-attention KV cache memory
  - Halve cross-attention compute per decoder step
  - Preserve WER because adjacent frames carry near-identical acoustic information

Usage:
    uv run python experiments/experiment_encoder_downsample.py
    uv run python experiments/experiment_encoder_downsample.py --model mlx-community/whisper-large-v3-mlx
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# ── Project imports ──────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ════════════════════════════════════════════════════════════════
# Encoder downsampling functions
# ════════════════════════════════════════════════════════════════

def downsample_encoder_output(enc: mx.array, stride: int) -> mx.array:
    """Average-pool encoder output along the time axis.

    Args:
        enc: Encoder hidden states, shape (batch, T_enc, d_model).
        stride: Pooling stride. 2 → 1500→750, 3 → 1500→500, 4 → 1500→375.

    Returns:
        Downsampled encoder output, shape (batch, T_enc // stride, d_model).
    """
    if stride <= 1:
        return enc

    B, T, D = enc.shape
    # Trim to multiple of stride
    T_trim = (T // stride) * stride
    enc_trimmed = enc[:, :T_trim, :]
    # Reshape and average-pool
    enc_reshaped = enc_trimmed.reshape(B, T_trim // stride, stride, D)
    return mx.mean(enc_reshaped, axis=2)


def downsample_encoder_strided(enc: mx.array, stride: int) -> mx.array:
    """Select every Nth encoder frame (strided selection, no averaging).

    Args:
        enc: Encoder hidden states, shape (batch, T_enc, d_model).
        stride: Selection stride.

    Returns:
        Strided encoder output.
    """
    if stride <= 1:
        return enc
    return enc[:, ::stride, :]


# ════════════════════════════════════════════════════════════════
# Greedy decode (clean, with optional encoder downsampling)
# ════════════════════════════════════════════════════════════════

def greedy_decode(
    model,
    mel: mx.array,
    max_new_tokens: int = 448,
    encoder_stride: int = 1,
    downsample_method: str = "avg_pool",  # "avg_pool" or "strided"
    use_q8: bool = False,
) -> tuple[list[int], float]:
    """Clean greedy decode with optional encoder downsampling.

    Returns:
        (token_ids, wall_time_seconds)
    """
    t0 = time.perf_counter()

    # Encode
    enc = encoder_forward(model, mel)
    mx.eval(enc)

    # Downsample encoder output
    if encoder_stride > 1:
        if downsample_method == "avg_pool":
            enc = downsample_encoder_output(enc, encoder_stride)
        else:
            enc = downsample_encoder_strided(enc, encoder_stride)
        mx.eval(enc)

    # Prefill with SOT
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]

    # Autoregressive decode
    while len(output_ids) < max_new_tokens:
        inp = mx.array([[output_ids[-1]]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        tok = sample(logits[:, -1:, :], 0.0)
        mx.eval(tok)
        token_id = tok.item()
        output_ids.append(token_id)
        if token_id == EOS_ID:
            break

    wall = time.perf_counter() - t0
    return output_ids, wall


# ════════════════════════════════════════════════════════════════
# Evaluation harness
# ════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens_to_text(tokenizer, token_ids: list[int]) -> str:
    """Decode tokens to text, skipping special tokens."""
    text_tokens = [t for t in token_ids[1:] if t < 50257]  # skip SOT and special
    return tokenizer.decode(text_tokens).strip()


def load_dataset(n_samples: int = 10):
    """Load LibriSpeech dummy dataset."""
    from datasets import load_dataset as hf_load
    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return ds.select(range(min(n_samples, len(ds))))


def audio_to_mel(model, audio_array: np.ndarray, sr: int = 16000) -> mx.array:
    """Convert audio array to mel spectrogram."""
    from mlx_whisper.audio import log_mel_spectrogram

    arr = np.ascontiguousarray(audio_array, dtype=np.float32)
    if len(arr) > 16000 * 30:
        arr = arr[:16000 * 30]

    mel = log_mel_spectrogram(arr, n_mels=model.dims.n_mels,
                               padding=16000 * 30 - len(arr))
    return mx.array(mel)[None]


def compute_wer(ref: str, hyp: str) -> float:
    """Simple WER calculation."""
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def run_experiment(args):
    """Run the encoder downsampling experiment."""
    print(f"\n{'='*70}")
    print(f"  E2: Encoder Downsampling Experiment")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Strides: {args.strides}")
    print(f"  Q8: {args.q8}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    model = load_target_model(args.model)
    if args.q8:
        print("Applying Q8 quantization...")
        quantize_model(model, encoder_bits=8, decoder_bits=8, group_size=64)

    # Get tokenizer
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)

    # Load dataset
    print(f"Loading {args.n_samples} LibriSpeech samples...")
    ds = load_dataset(args.n_samples)

    # ── Baseline: stride=1 (no downsampling) ──
    strides_to_test = [1] + [int(s) for s in args.strides.split(",")]
    strides_to_test = sorted(set(strides_to_test))

    all_results = {}

    for stride in strides_to_test:
        for method in (["avg_pool", "strided"] if stride > 1 else ["none"]):
            config_name = f"stride-{stride}" if stride == 1 else f"stride-{stride}-{method}"
            print(f"\n── Config: {config_name} ──")

            sample_results = []
            total_time = 0.0

            for i, sample_data in enumerate(ds):
                audio = sample_data["audio"]["array"]
                sr = sample_data["audio"]["sampling_rate"]
                reference = sample_data["text"]

                mel = audio_to_mel(model, audio, sr)

                # Warm up on first sample
                if i == 0 and stride == strides_to_test[0]:
                    _ = greedy_decode(model, mel, encoder_stride=1)

                token_ids, wall_time = greedy_decode(
                    model, mel,
                    encoder_stride=stride,
                    downsample_method=method if stride > 1 else "avg_pool",
                )

                hypothesis = tokens_to_text(tokenizer, token_ids)
                wer = compute_wer(reference, hypothesis)

                sample_results.append({
                    "sample_idx": i,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer,
                    "n_tokens": len(token_ids) - 1,
                    "wall_time": wall_time,
                })
                total_time += wall_time

                if i < 3 or wer > 0.5:  # Print first 3 + any bad ones
                    print(f"  [{i:2d}] WER={wer:.4f} | {len(token_ids)-1} tokens | "
                          f"{wall_time:.3f}s | hyp='{hypothesis[:60]}...'")

            # Aggregate
            mean_wer = sum(r["wer"] for r in sample_results) / len(sample_results)
            mean_time = total_time / len(sample_results)
            total_tokens = sum(r["n_tokens"] for r in sample_results)
            tps = total_tokens / total_time if total_time > 0 else 0

            all_results[config_name] = {
                "stride": stride,
                "method": method if stride > 1 else "none",
                "mean_wer": mean_wer,
                "mean_time_s": mean_time,
                "total_time_s": total_time,
                "tokens_per_sec": tps,
                "total_tokens": total_tokens,
                "n_samples": len(sample_results),
                "enc_frames": 1500 // stride if stride > 0 else 1500,
                "per_sample": sample_results,
            }

            print(f"  ▶ Mean WER: {mean_wer:.4f} | Mean time: {mean_time:.3f}s | "
                  f"TPS: {tps:.1f} | Enc frames: {1500 // stride}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    baseline_time = all_results.get("stride-1", {}).get("mean_time_s", 1.0)
    baseline_wer = all_results.get("stride-1", {}).get("mean_wer", 0.0)

    print(f"\n  {'Config':<25} {'Frames':>6} {'WER':>8} {'ΔWER':>8} "
          f"{'Time(s)':>8} {'Speedup':>8} {'TPS':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for config_name, res in sorted(all_results.items()):
        delta_wer = res["mean_wer"] - baseline_wer
        speedup = baseline_time / res["mean_time_s"] if res["mean_time_s"] > 0 else 0
        print(f"  {config_name:<25} {res['enc_frames']:>6} "
              f"{res['mean_wer']:>8.4f} {delta_wer:>+8.4f} "
              f"{res['mean_time_s']:>8.3f} {speedup:>8.2f}x "
              f"{res['tokens_per_sec']:>8.1f}")

    # ── Token-level comparison ──
    if "stride-1" in all_results:
        print(f"\n  Token-level comparison vs baseline:")
        baseline_hyps = [r["hypothesis"] for r in all_results["stride-1"]["per_sample"]]
        for config_name, res in sorted(all_results.items()):
            if config_name == "stride-1":
                continue
            hyps = [r["hypothesis"] for r in res["per_sample"]]
            exact_match = sum(1 for a, b in zip(baseline_hyps, hyps)
                            if normalize_text(a) == normalize_text(b))
            print(f"  {config_name:<25}: {exact_match}/{len(baseline_hyps)} "
                  f"exact matches ({100*exact_match/len(baseline_hyps):.0f}%)")

    # ── Also measure encoder frame redundancy ──
    print(f"\n  Encoder frame redundancy analysis:")
    sample_data = ds[0]
    mel = audio_to_mel(model, sample_data["audio"]["array"])
    enc = encoder_forward(model, mel)
    mx.eval(enc)

    # Cosine similarity between adjacent frames
    enc_np = np.array(enc[0])  # (T, D)
    adjacent_cos = []
    for i in range(len(enc_np) - 1):
        a, b = enc_np[i], enc_np[i+1]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        adjacent_cos.append(cos)
    adjacent_cos = np.array(adjacent_cos)

    print(f"  Adjacent frame cosine similarity (sample 0):")
    print(f"    Mean:   {adjacent_cos.mean():.4f}")
    print(f"    Std:    {adjacent_cos.std():.4f}")
    print(f"    Min:    {adjacent_cos.min():.4f}")
    print(f"    >0.99:  {(adjacent_cos > 0.99).sum()}/{len(adjacent_cos)} "
          f"({100*(adjacent_cos > 0.99).mean():.1f}%)")
    print(f"    >0.95:  {(adjacent_cos > 0.95).sum()}/{len(adjacent_cos)} "
          f"({100*(adjacent_cos > 0.95).mean():.1f}%)")
    print(f"    >0.90:  {(adjacent_cos > 0.90).sum()}/{len(adjacent_cos)} "
          f"({100*(adjacent_cos > 0.90).mean():.1f}%)")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "encoder_downsample.json"

    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                    for k, v in all_results.items()}
    save_results["frame_redundancy"] = {
        "adjacent_cos_mean": float(adjacent_cos.mean()),
        "adjacent_cos_std": float(adjacent_cos.std()),
        "adjacent_cos_min": float(adjacent_cos.min()),
        "pct_above_099": float((adjacent_cos > 0.99).mean()),
        "pct_above_095": float((adjacent_cos > 0.95).mean()),
        "pct_above_090": float((adjacent_cos > 0.90).mean()),
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2: Encoder Downsampling Experiment")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx",
                        help="MLX Whisper model path")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of LibriSpeech samples to evaluate")
    parser.add_argument("--strides", default="2,3,4",
                        help="Comma-separated list of downsampling strides to test")
    parser.add_argument("--q8", action="store_true",
                        help="Apply Q8 quantization before testing")
    run_experiment(parser.parse_args())
