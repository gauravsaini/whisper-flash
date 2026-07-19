#!/usr/bin/env python3
"""P49: Colab Q8 Quantization Universality Validation.

Tests whether Q8 quantization speedup (~1.2-1.3× on MLX) is universal
across frameworks and hardware, or MLX-specific.

Three Q8 approaches:
  1. PyTorch dynamic quantization (torch.ao.quantization.quantize_dynamic)
  2. bitsandbytes LLM.int8() (load_in_8bit=True)
  3. Custom group-wise dequant (from colab_quantization.py)

Test matrix: 3 models × 3 methods + baseline = 12 configs
  Models: whisper-tiny, whisper-small, whisper-large-v3-turbo

Each config: WER on 20 LibriSpeech dummy samples + avg decode time

Usage (colab): colab run --gpu T4 experiments/p49_colab_q8_validate.py
"""

import subprocess, sys, os, json, time, math, gc

# ── Install dependencies ──────────────────────────────────────────────
_deps = [
    "torch", "transformers", "soundfile", "librosa",
    "accelerate", "datasets", "jiwer", "bitsandbytes",
]
for _pkg in _deps:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", _pkg],
        capture_output=True,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from jiwer import wer as jiwer_wer

torch.set_grad_enabled(False)

# ── Config ─────────────────────────────────────────────────────────────

MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-large-v3-turbo",
]

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 150
NUM_EVAL = 20
NUM_WARMUP = 3
GROUP_SIZE = 64


# ── Data loading ───────────────────────────────────────────────────────

def load_eval_data(n=NUM_EVAL):
    """Load LibriSpeech dummy samples for evaluation."""
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean", split="validation",
    )
    samples = []
    for i in range(min(n, len(ds))):
        audio = np.array(ds[i]["audio"]["array"], dtype=np.float32)
        ref = ds[i]["text"].strip()
        samples.append((audio, ref))
    return samples


# ── Benchmark helper ───────────────────────────────────────────────────

def benchmark(model, processor, samples, device, label=""):
    """Run greedy decoding, return {wer, avg_time_ms, mem_mb}."""
    model.eval()
    preds = []
    refs = []
    times = []

    # Warmup
    for i in range(min(NUM_WARMUP, len(samples))):
        audio, _ = samples[i]
        inputs = processor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device)
        if inputs.dtype != model.dtype:
            inputs = inputs.to(model.dtype)
        _ = model.generate(inputs, max_new_tokens=16)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    for audio, ref in samples:
        refs.append(ref.lower())
        inputs = processor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device)
        if inputs.dtype != model.dtype:
            inputs = inputs.to(model.dtype)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        text = processor.tokenizer.decode(
            generated[0], skip_special_tokens=True
        ).strip().lower()
        preds.append(text)

    w = jiwer_wer(refs, preds)
    avg_ms = np.mean(times) * 1000

    # Memory usage
    mem_mb = 0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "wer": round(w, 6),
        "avg_time_ms": round(avg_ms, 1),
        "mem_mb": round(mem_mb, 1),
    }


# ── Method 1: FP16 baseline ───────────────────────────────────────────

def test_baseline_fp16(model_id, processor, samples, device):
    """FP16 baseline — no quantization."""
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = benchmark(model, processor, samples, device, "fp16")
    result["method"] = "fp16_baseline"

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ── Method 2: PyTorch dynamic quantization ─────────────────────────────

def test_torch_dynamic_q8(model_id, processor, samples, device):
    """PyTorch native dynamic quantization (int8 GEMM).

    Note: torch.ao.quantization.quantize_dynamic primarily benefits CPU.
    On CUDA, it may fall back to fp32 ops. We test both to see.
    """
    from transformers import WhisperForConditionalGeneration

    # Dynamic quant works on CPU (int8 GEMM kernels are CPU-optimized)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32,
    )
    model.eval()

    # Apply dynamic quantization to all Linear layers
    model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # Dynamic quant models stay on CPU (CUDA int8 support is limited)
    test_device = "cpu"
    model = model.to(test_device)

    result = benchmark(model, processor, samples, test_device, "torch_dynamic_q8")
    result["method"] = "torch_dynamic_q8"
    result["note"] = "CPU-only (PyTorch dynamic quant has limited CUDA support)"

    del model
    gc.collect()

    return result


# ── Method 3: bitsandbytes LLM.int8() ─────────────────────────────────

def test_bitsandbytes_int8(model_id, processor, samples, device):
    """bitsandbytes 8-bit quantization (LLM.int8()).

    Uses mixed-precision decomposition: most weights in int8,
    outlier features in fp16. Designed for memory savings on GPU.
    """
    if not torch.cuda.is_available():
        return {
            "method": "bitsandbytes_int8",
            "wer": None,
            "avg_time_ms": None,
            "mem_mb": None,
            "error": "CUDA required for bitsandbytes",
        }

    try:
        from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = benchmark(model, processor, samples, device, "bnb_int8")
        result["method"] = "bitsandbytes_int8"

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        return {
            "method": "bitsandbytes_int8",
            "wer": None,
            "avg_time_ms": None,
            "mem_mb": None,
            "error": str(e),
        }


# ── Method 4: Custom group-wise dequant Q8 ─────────────────────────────

def quantize_weight(w, bits, group_size=GROUP_SIZE):
    """Quantize weight tensor to N bits with group-wise scaling."""
    w = w.float()
    orig_shape = w.shape
    w_flat = w.flatten()
    n = w_flat.shape[0]
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        w_flat = F.pad(w_flat, (0, pad))
        n = w_flat.shape[0]
    w_groups = w_flat.reshape(-1, group_size)
    w_max = w_groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    w_norm = w_groups / w_max
    q_max = 2 ** (bits - 1) - 1
    w_q = torch.round(w_norm * q_max).clamp(-q_max - 1, q_max).to(torch.int8)
    w_q = w_q.reshape(-1)[:orig_shape.numel()].reshape(orig_shape)
    scale = w_max.reshape(-1).flatten()[:math.ceil(orig_shape.numel() / group_size)]
    return w_q, scale


def dequantize_weight(w_q, scale, bits, group_size=GROUP_SIZE, n_orig=None):
    """Dequantize N-bit weight back to float."""
    q_max = 2 ** (bits - 1) - 1
    w_q_flat = w_q.float().flatten()
    n = w_q_flat.shape[0]
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        w_q_flat = F.pad(w_q_flat, (0, pad))
        n = w_q_flat.shape[0]
    w_groups = w_q_flat.reshape(-1, group_size)
    scale_g = scale[:w_groups.shape[0]].unsqueeze(1)
    w_deq = (w_groups.float() / q_max) * scale_g
    if n_orig is not None and n > n_orig:
        w_deq = w_deq.flatten()[:n_orig]
    return w_deq


class CustomQuantizedLinear(nn.Module):
    """Linear with int8 weights, dequantized on forward."""

    def __init__(self, in_f, out_f, bits=8, group_size=GROUP_SIZE, bias=True):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.bits = bits
        self.group_size = group_size
        self.register_buffer("weight_q", torch.empty(out_f, in_f, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(math.ceil(out_f * in_f / group_size)))
        self._n_orig = out_f * in_f
        if bias:
            self.bias = nn.Parameter(torch.empty(out_f), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        w = dequantize_weight(
            self.weight_q, self.scale, self.bits, self.group_size, self._n_orig,
        ).reshape(self.weight_q.shape)
        w = w.to(x.dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(x.dtype)
        return F.linear(x, w, self.bias)


def replace_with_custom_q8(module, bits=8, group_size=GROUP_SIZE, depth=0):
    """Recursively replace nn.Linear with CustomQuantizedLinear."""
    if depth > 20:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            q = CustomQuantizedLinear(
                child.in_features, child.out_features,
                bits, group_size, bias=child.bias is not None,
            )
            w_q, scale = quantize_weight(child.weight.data, bits, group_size)
            q.weight_q.data = w_q
            q.scale.data = scale
            if child.bias is not None:
                q.bias.data = child.bias.data
            setattr(module, name, q)
        else:
            replace_with_custom_q8(child, bits, group_size, depth + 1)


def test_custom_dequant_q8(model_id, processor, samples, device):
    """Custom group-wise Q8 with dequant on forward (from colab_quantization.py)."""
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    )
    model.eval()

    # Apply custom Q8 to encoder and decoder
    replace_with_custom_q8(model.model.encoder, bits=8, group_size=GROUP_SIZE)
    replace_with_custom_q8(model.model.decoder, bits=8, group_size=GROUP_SIZE)

    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = benchmark(model, processor, samples, device, "custom_q8")
    result["method"] = "custom_dequant_q8"

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ── Main ───────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"[p49] device={device} gpu={gpu_name}", flush=True)

    samples = load_eval_data(NUM_EVAL)
    print(f"[p49] loaded {len(samples)} eval samples", flush=True)

    all_results = []

    for model_id in MODELS:
        print(f"\n{'='*70}", flush=True)
        print(f"  MODEL: {model_id}", flush=True)
        print(f"{'='*70}", flush=True)

        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(model_id)

        model_results = {"model": model_id, "device": device, "gpu": gpu_name}

        # 1. FP16 baseline
        print(f"\n  [1/4] FP16 baseline...", flush=True)
        r_base = test_baseline_fp16(model_id, processor, samples, device)
        print(f"    WER={r_base['wer']:.4f}  time={r_base['avg_time_ms']:.0f}ms  mem={r_base['mem_mb']:.0f}MB", flush=True)
        model_results["fp16_baseline"] = r_base

        # 2. PyTorch dynamic Q8
        print(f"\n  [2/4] PyTorch dynamic Q8 (CPU)...", flush=True)
        r_torch = test_torch_dynamic_q8(model_id, processor, samples, device)
        if r_torch.get("error"):
            print(f"    FAILED: {r_torch['error']}", flush=True)
        else:
            speedup = r_base["avg_time_ms"] / r_torch["avg_time_ms"] if r_torch["avg_time_ms"] > 0 else 0
            print(f"    WER={r_torch['wer']:.4f}  time={r_torch['avg_time_ms']:.0f}ms  speedup={speedup:.2f}×", flush=True)
            r_torch["speedup_vs_fp16"] = round(speedup, 3)
        model_results["torch_dynamic_q8"] = r_torch

        # 3. bitsandbytes int8
        print(f"\n  [3/4] bitsandbytes LLM.int8()...", flush=True)
        r_bnb = test_bitsandbytes_int8(model_id, processor, samples, device)
        if r_bnb.get("error"):
            print(f"    FAILED: {r_bnb['error']}", flush=True)
        else:
            speedup = r_base["avg_time_ms"] / r_bnb["avg_time_ms"] if r_bnb["avg_time_ms"] > 0 else 0
            print(f"    WER={r_bnb['wer']:.4f}  time={r_bnb['avg_time_ms']:.0f}ms  mem={r_bnb['mem_mb']:.0f}MB  speedup={speedup:.2f}×", flush=True)
            r_bnb["speedup_vs_fp16"] = round(speedup, 3)
        model_results["bitsandbytes_int8"] = r_bnb

        # 4. Custom dequant Q8
        print(f"\n  [4/4] Custom dequant Q8...", flush=True)
        r_custom = test_custom_dequant_q8(model_id, processor, samples, device)
        if r_custom.get("error"):
            print(f"    FAILED: {r_custom['error']}", flush=True)
        else:
            speedup = r_base["avg_time_ms"] / r_custom["avg_time_ms"] if r_custom["avg_time_ms"] > 0 else 0
            print(f"    WER={r_custom['wer']:.4f}  time={r_custom['avg_time_ms']:.0f}ms  mem={r_custom['mem_mb']:.0f}MB  speedup={speedup:.2f}×", flush=True)
            r_custom["speedup_vs_fp16"] = round(speedup, 3)
        model_results["custom_dequant_q8"] = r_custom

        all_results.append(model_results)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*90}", flush=True)
    print(f"  SUMMARY: Q8 Quantization Universality (device={device}, gpu={gpu_name})", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"{'Model':<30} {'Method':<20} {'WER':>8} {'Time(ms)':>10} {'Speedup':>8} {'Mem(MB)':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for mr in all_results:
        model_short = mr["model"].split("/")[-1]
        for method_key in ["fp16_baseline", "torch_dynamic_q8", "bitsandbytes_int8", "custom_dequant_q8"]:
            r = mr.get(method_key, {})
            method_name = r.get("method", method_key)
            wer_s = f"{r['wer']:.4f}" if r.get("wer") is not None else "FAIL"
            time_s = f"{r['avg_time_ms']:.0f}" if r.get("avg_time_ms") is not None else "N/A"
            sp_s = f"{r.get('speedup_vs_fp16', 1.0):.2f}×" if r.get("wer") is not None else "N/A"
            mem_s = f"{r['mem_mb']:.0f}" if r.get("mem_mb") is not None else "N/A"
            print(f"{model_short:<30} {method_name:<20} {wer_s:>8} {time_s:>10} {sp_s:>8} {mem_s:>8}", flush=True)
        print(flush=True)

    # ── Save results ───────────────────────────────────────────────────
    out_path = "/content/p49_q8_results.json" if os.path.isdir("/content") else "results/p49_q8_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[p49] Results saved to {out_path}", flush=True)

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  VERDICT", flush=True)
    print(f"{'='*70}", flush=True)
    for mr in all_results:
        model_short = mr["model"].split("/")[-1]
        base_t = mr["fp16_baseline"]["avg_time_ms"]
        for mk in ["torch_dynamic_q8", "bitsandbytes_int8", "custom_dequant_q8"]:
            r = mr.get(mk, {})
            if r.get("avg_time_ms") and r.get("wer") is not None:
                sp = base_t / r["avg_time_ms"]
                wer_delta = r["wer"] - mr["fp16_baseline"]["wer"]
                verdict = "✅ FASTER" if sp > 1.05 else "⚠️ SLOWER" if sp < 0.95 else "➡️ NEUTRAL"
                lossless = "lossless" if abs(wer_delta) < 0.01 else f"WER delta={wer_delta:+.4f}"
                print(f"  {model_short:25s} {r['method']:20s} → {sp:.2f}× ({verdict}, {lossless})", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
