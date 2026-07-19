"""E1: Mixed-Precision Q4 — Per-layer quantization sensitivity sweep.

Hypothesis: Whisper's layers have vastly different quantization sensitivity.
Cross-attention projections (Q, K, V for audio) are likely the most sensitive.
Self-attention and FFN layers are likely robust to Q4.

By sweeping each layer individually, we build a precision map and achieve
near-Q4 compression with near-Q8 quality.

Usage:
    uv run python experiments/experiment_mixed_precision.py
    uv run python experiments/experiment_mixed_precision.py --model mlx-community/whisper-large-v3-mlx --bits 4
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ════════════════════════════════════════════════════════════════
# Layer enumeration + surgical quantization
# ════════════════════════════════════════════════════════════════

@dataclass
class LinearLayerInfo:
    """Metadata for a single Linear layer in the model."""
    path: str           # e.g. "encoder.blocks.3.attn.key"
    component: str      # "encoder" or "decoder"
    block_idx: int      # block index (-1 for non-block layers)
    layer_type: str     # "attn_query", "attn_key", "attn_value", "attn_out",
                        # "cross_query", "cross_key", "cross_value", "cross_out",
                        # "mlp1", "mlp2", "embed", "ln", "other"
    param_count: int    # number of parameters


def enumerate_linear_layers(model) -> list[LinearLayerInfo]:
    """Walk the model and enumerate all nn.Linear layers with their paths."""
    layers = []

    def _classify_layer(path: str) -> str:
        """Classify a layer by its path."""
        if "mlp.0" in path or "mlp_ln" in path:
            return "mlp1"
        if "mlp.2" in path:
            return "mlp2"
        if "cross_attn" in path or "cross" in path:
            if "query" in path:
                return "cross_query"
            if "key" in path:
                return "cross_key"
            if "value" in path:
                return "cross_value"
            if "out" in path:
                return "cross_out"
            return "cross_other"
        if "attn" in path:
            if "query" in path:
                return "attn_query"
            if "key" in path:
                return "attn_key"
            if "value" in path:
                return "attn_value"
            if "out" in path:
                return "attn_out"
            return "attn_other"
        return "other"

    def _extract_block_idx(path: str) -> int:
        import re
        m = re.search(r"blocks\.(\d+)", path)
        return int(m.group(1)) if m else -1

    def _walk(module, prefix: str):
        if isinstance(module, nn.Linear):
            component = "encoder" if prefix.startswith("encoder") else "decoder"
            layers.append(LinearLayerInfo(
                path=prefix,
                component=component,
                block_idx=_extract_block_idx(prefix),
                layer_type=_classify_layer(prefix),
                param_count=module.weight.size,
            ))
            return

        if hasattr(module, "children") and callable(module.children):
            try:
                children = module.children()
                if isinstance(children, dict):
                    for name, child in children.items():
                        child_prefix = f"{prefix}.{name}" if prefix else name
                        if isinstance(child, nn.Module):
                            _walk(child, child_prefix)
                        elif isinstance(child, list):
                            for i, item in enumerate(child):
                                if isinstance(item, nn.Module):
                                    _walk(item, f"{child_prefix}.{i}")
            except Exception:
                pass

    _walk(model.encoder, "encoder")
    _walk(model.decoder, "decoder")
    return layers


def quantize_single_layer(model, layer_path: str, bits: int, group_size: int):
    """Quantize a SINGLE linear layer by its path, in-place.

    Args:
        model: The Whisper model.
        layer_path: e.g. "encoder.blocks.3.attn.key"
        bits: Quantization bits (4, 8, etc.)
        group_size: Group size for quantization.
    """
    parts = layer_path.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    attr_name = parts[-1]
    lin = getattr(obj, attr_name)
    if not isinstance(lin, nn.Linear):
        raise ValueError(f"{layer_path} is not nn.Linear, got {type(lin)}")

    q = nn.QuantizedLinear.from_linear(lin, group_size, bits, mode="affine")
    setattr(obj, attr_name, q)


def dequantize_single_layer(model, layer_path: str, original_linear: nn.Linear):
    """Restore a layer to its original nn.Linear (undo quantization)."""
    parts = layer_path.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], original_linear)


def get_layer_module(model, layer_path: str):
    """Get the module at a given path."""
    parts = layer_path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


# ════════════════════════════════════════════════════════════════
# Decode + WER
# ════════════════════════════════════════════════════════════════

def greedy_decode(model, mel: mx.array, max_new_tokens: int = 448) -> list[int]:
    """Clean greedy decode."""
    enc = encoder_forward(model, mel)
    mx.eval(enc)

    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]

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
    return output_ids


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
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


# ════════════════════════════════════════════════════════════════
# Main experiment
# ════════════════════════════════════════════════════════════════

def run_experiment(args):
    print(f"\n{'='*70}")
    print(f"  E1: Mixed-Precision Q4 Sensitivity Sweep")
    print(f"  Model: {args.model}")
    print(f"  Target bits: {args.bits}")
    print(f"  Samples: {args.n_samples}")
    print(f"{'='*70}\n")

    # Load model + dataset
    print("Loading model...")
    model = load_target_model(args.model)

    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)

    from datasets import load_dataset as hf_load
    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.select(range(min(args.n_samples, len(ds))))

    from mlx_whisper.audio import log_mel_spectrogram

    def make_mel(audio_arr):
        arr = np.ascontiguousarray(audio_arr, dtype=np.float32)
        if len(arr) > 16000 * 30:
            arr = arr[:16000 * 30]
        mel = log_mel_spectrogram(arr, n_mels=model.dims.n_mels,
                                   padding=16000 * 30 - len(arr))
        return mx.array(mel)[None]

    def tokens_to_text(tids):
        text_tokens = [t for t in tids[1:] if t < 50257]
        return tokenizer.decode(text_tokens).strip()

    # ── Step 1: FP16 baseline ──
    print("Running FP16 baseline...")
    baseline_results = []
    for i, sample_data in enumerate(ds):
        mel = make_mel(sample_data["audio"]["array"])
        tids = greedy_decode(model, mel)
        hyp = tokens_to_text(tids)
        ref = sample_data["text"]
        wer = compute_wer(ref, hyp)
        baseline_results.append({"hyp": hyp, "ref": ref, "wer": wer, "tids": tids})

    baseline_wer = sum(r["wer"] for r in baseline_results) / len(baseline_results)
    print(f"  FP16 baseline WER: {baseline_wer:.4f}\n")

    # ── Step 2: Enumerate all linear layers ──
    layers = enumerate_linear_layers(model)
    print(f"  Found {len(layers)} linear layers:")

    # Group by type
    type_counts = {}
    for layer in layers:
        key = f"{layer.component}/{layer.layer_type}"
        type_counts[key] = type_counts.get(key, 0) + 1
    for k, v in sorted(type_counts.items()):
        print(f"    {k}: {v} layers")

    # ── Step 3: Per-layer sensitivity sweep ──
    print(f"\n  Sweeping {len(layers)} layers at Q{args.bits}...")
    print(f"  {'Layer Path':<50} {'Type':<15} {'Params':>10} {'ΔWER':>10} {'Sensitive?':>10}")
    print(f"  {'-'*50} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    layer_sensitivities = []

    for li, layer_info in enumerate(layers):
        # Save original
        original = get_layer_module(model, layer_info.path)

        # Quantize this single layer
        try:
            quantize_single_layer(model, layer_info.path, args.bits, args.group_size)
        except Exception as e:
            print(f"  [{li:3d}] SKIP {layer_info.path}: {e}")
            layer_sensitivities.append({
                "path": layer_info.path,
                "component": layer_info.component,
                "layer_type": layer_info.layer_type,
                "block_idx": layer_info.block_idx,
                "param_count": layer_info.param_count,
                "delta_wer": None,
                "error": str(e),
            })
            continue

        # Evaluate
        total_wer = 0.0
        n_mismatch = 0
        for i, sample_data in enumerate(ds):
            mel = make_mel(sample_data["audio"]["array"])
            tids = greedy_decode(model, mel)
            hyp = tokens_to_text(tids)
            ref = sample_data["text"]
            wer = compute_wer(ref, hyp)
            total_wer += wer
            if normalize_text(hyp) != normalize_text(baseline_results[i]["hyp"]):
                n_mismatch += 1

        mean_wer = total_wer / len(ds)
        delta_wer = mean_wer - baseline_wer
        is_sensitive = abs(delta_wer) > 0.01  # >1% WER change

        layer_sensitivities.append({
            "path": layer_info.path,
            "component": layer_info.component,
            "layer_type": layer_info.layer_type,
            "block_idx": layer_info.block_idx,
            "param_count": layer_info.param_count,
            "delta_wer": delta_wer,
            "mean_wer": mean_wer,
            "n_mismatch": n_mismatch,
            "is_sensitive": is_sensitive,
        })

        marker = "🔴" if is_sensitive else "🟢"
        short_path = layer_info.path[-48:] if len(layer_info.path) > 48 else layer_info.path
        print(f"  [{li:3d}] {short_path:<50} {layer_info.layer_type:<15} "
              f"{layer_info.param_count:>10,} {delta_wer:>+10.4f} {marker}")

        # Restore original
        dequantize_single_layer(model, layer_info.path, original)

    # ── Step 4: Build mixed-precision profile ──
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY RANKING")
    print(f"{'='*70}\n")

    valid = [s for s in layer_sensitivities if s["delta_wer"] is not None]
    valid.sort(key=lambda x: abs(x["delta_wer"]), reverse=True)

    sensitive = [s for s in valid if s.get("is_sensitive", False)]
    insensitive = [s for s in valid if not s.get("is_sensitive", False)]

    print(f"  Sensitive layers ({len(sensitive)} — keep at Q8):")
    for s in sensitive[:20]:
        print(f"    {s['path']:<50} ΔWER={s['delta_wer']:+.4f}")

    print(f"\n  Insensitive layers ({len(insensitive)} — safe for Q{args.bits}):")
    for s in insensitive[:20]:
        print(f"    {s['path']:<50} ΔWER={s['delta_wer']:+.4f}")

    # ── Step 5: Apply mixed-precision profile and measure ──
    print(f"\n  Applying mixed-precision profile (insensitive → Q{args.bits}, sensitive → Q8)...")

    # Quantize all insensitive layers to Q4
    for s in insensitive:
        try:
            quantize_single_layer(model, s["path"], args.bits, args.group_size)
        except Exception:
            pass

    # Quantize sensitive layers to Q8
    for s in sensitive:
        try:
            quantize_single_layer(model, s["path"], 8, args.group_size)
        except Exception:
            pass

    # Evaluate mixed-precision
    print(f"  Evaluating mixed-precision profile...")
    mixed_wers = []
    t0 = time.perf_counter()
    for i, sample_data in enumerate(ds):
        mel = make_mel(sample_data["audio"]["array"])
        tids = greedy_decode(model, mel)
        hyp = tokens_to_text(tids)
        ref = sample_data["text"]
        wer = compute_wer(ref, hyp)
        mixed_wers.append(wer)
    t1 = time.perf_counter()

    mixed_wer = sum(mixed_wers) / len(mixed_wers)
    mixed_time = (t1 - t0) / len(ds)

    # Also measure full Q4 and full Q8 for comparison
    # Reload model for full Q4
    print(f"  Evaluating full Q{args.bits} for comparison...")
    model_q4 = load_target_model(args.model)
    from whisper_flash_mlx.quantization import quantize_model as full_quantize
    full_quantize(model_q4, encoder_bits=args.bits, decoder_bits=args.bits, group_size=args.group_size)

    q4_wers = []
    t0 = time.perf_counter()
    for i, sample_data in enumerate(ds):
        mel = make_mel(sample_data["audio"]["array"])
        tids = greedy_decode(model_q4, mel)
        hyp = tokens_to_text(tids)
        ref = sample_data["text"]
        wer = compute_wer(ref, hyp)
        q4_wers.append(wer)
    t1 = time.perf_counter()
    q4_wer = sum(q4_wers) / len(q4_wers)
    q4_time = (t1 - t0) / len(ds)

    # Full Q8
    print(f"  Evaluating full Q8 for comparison...")
    model_q8 = load_target_model(args.model)
    full_quantize(model_q8, encoder_bits=8, decoder_bits=8, group_size=args.group_size)

    q8_wers = []
    t0 = time.perf_counter()
    for i, sample_data in enumerate(ds):
        mel = make_mel(sample_data["audio"]["array"])
        tids = greedy_decode(model_q8, mel)
        hyp = tokens_to_text(tids)
        ref = sample_data["text"]
        wer = compute_wer(ref, hyp)
        q8_wers.append(wer)
    t1 = time.perf_counter()
    q8_wer = sum(q8_wers) / len(q8_wers)
    q8_time = (t1 - t0) / len(ds)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    total_params = sum(s["param_count"] for s in valid)
    q4_params = sum(s["param_count"] for s in insensitive)

    print(f"\n  {'Config':<30} {'WER':>8} {'ΔWER':>8} {'Time(s)':>8} "
          f"{'Q4 Params':>12}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    print(f"  {'FP16 baseline':<30} {baseline_wer:>8.4f} {'—':>8} {'—':>8} "
          f"{'0%':>12}")
    print(f"  {'Full Q8':<30} {q8_wer:>8.4f} {q8_wer-baseline_wer:>+8.4f} "
          f"{q8_time:>8.3f} {'0%':>12}")
    print(f"  {'Full Q' + str(args.bits):<30} {q4_wer:>8.4f} "
          f"{q4_wer-baseline_wer:>+8.4f} {q4_time:>8.3f} {'100%':>12}")
    print(f"  {'Mixed (sensitive=Q8,rest=Q'+str(args.bits)+')':<30} "
          f"{mixed_wer:>8.4f} {mixed_wer-baseline_wer:>+8.4f} "
          f"{mixed_time:>8.3f} "
          f"{100*q4_params/total_params:.0f}%")

    # Save
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "mixed_precision.json"

    save_data = {
        "model": args.model,
        "target_bits": args.bits,
        "group_size": args.group_size,
        "n_samples": args.n_samples,
        "baseline_wer": baseline_wer,
        "q8_wer": q8_wer,
        f"q{args.bits}_wer": q4_wer,
        "mixed_wer": mixed_wer,
        "n_sensitive": len(sensitive),
        "n_insensitive": len(insensitive),
        "pct_q4_params": q4_params / total_params if total_params > 0 else 0,
        "layer_sensitivities": layer_sensitivities,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: Mixed-Precision Q4 Sensitivity Sweep")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx",
                        help="MLX Whisper model path")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of LibriSpeech samples")
    parser.add_argument("--bits", type=int, default=4,
                        help="Target quantization bits for insensitive layers")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size")
    run_experiment(parser.parse_args())
