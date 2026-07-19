#!/usr/bin/env python3
"""
P5: Low-bit Whisper Quantization. Test Q2/Q3/Q4/Q8 on encoder vs decoder.
Measures WER, model size, and speed for each config.

Usage:
  colab run --gpu T4 colab/colab_quantization.py --model openai/whisper-tiny --train 10 --eval 10
  colab run --gpu T4 colab/colab_quantization.py --model openai/whisper-large-v3-turbo --train 30 --eval 30
"""

import subprocess, sys, json, os, argparse, time, math
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], capture_output=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from copy import deepcopy
from collections import OrderedDict

SAMPLE_RATE = 16000
MAX_TOKENS = 150
EOS_ID = 50257

# ─── Quantization helpers ──────────────────────────────────────────────

def quantize_weight(w, bits, group_size=64):
    """Quantize weight tensor to N bits per element with group-wise scaling."""
    w = w.float()
    orig_shape = w.shape
    w_flat = w.flatten()
    n = w_flat.shape[0]
    if n % group_size != 0:
        pad = group_size - (n % group_size)
        w_flat = F.pad(w_flat, (0, pad))
        n = w_flat.shape[0]
    w_groups = w_flat.reshape(-1, group_size)
    w_max = w_groups.abs().max(dim=1, keepdim=True).values
    w_max = w_max.clamp(min=1e-8)
    w_norm = w_groups / w_max
    q_max = 2 ** (bits - 1) - 1
    w_q = torch.round(w_norm * q_max).clamp(-q_max - 1, q_max).to(torch.int8)
    w_q = w_q.reshape(-1)
    w_q = w_q[:orig_shape.numel()].reshape(orig_shape)
    scale = w_max.reshape(-1).flatten()
    scale = scale[:math.ceil(orig_shape.numel() / group_size)]
    return w_q, scale

def dequantize_weight(w_q, scale, bits, group_size=64, n_orig=None):
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

def compute_model_size(model):
    """Compute total model parameter size in MB."""
    total_bytes = 0
    for p in model.parameters():
        if hasattr(p, '_quantized'):
            scale_size = p._scale.numel() * p._scale.element_size()
            q_bytes = p._q_bytes
            total_bytes += q_bytes + scale_size
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)

def model_size_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * (p.element_size() if hasattr(p, 'element_size') else 4)
    for b in model.buffers():
        total += b.numel() * (b.element_size() if hasattr(b, 'element_size') else 4)
    return total

# ─── Quantized Linear ──────────────────────────────────────────────────

class QuantizedLinear(nn.Module):
    """Replacement Linear layer with N-bit quantized weights."""
    def __init__(self, in_features, out_features, bits, group_size=64, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.register_buffer('weight_q', torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.empty(out_features, math.ceil(in_features / group_size)))
        self._n_orig = out_features * in_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w = dequantize_weight(self.weight_q, self.scale, self.bits, self.group_size, self._n_orig)
        w = w.reshape(self.weight_q.shape)
        return F.linear(x, w, self.bias)

def replace_with_quantized(module, bits, group_size=64, name='', max_depth=20, depth=0):
    """Recursively replace nn.Linear with QuantizedLinear in module."""
    if depth > max_depth:
        return
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear):
            q = QuantizedLinear(
                child.in_features, child.out_features, bits, group_size,
                bias=child.bias is not None
            )
            w = child.weight.data
            w_q, scale = quantize_weight(w, bits, group_size)
            q.weight_q.data = w_q
            q.scale.data = scale
            if child.bias is not None:
                q.bias.data = child.bias.data
            setattr(module, child_name, q)
        else:
            replace_with_quantized(child, bits, group_size, full_name, max_depth, depth + 1)

def count_linear(model):
    return sum(1 for _ in model.modules() if isinstance(_, nn.Linear))

def measure_parameters(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total

# ─── Quantization configs ──────────────────────────────────────────────

CONFIGS = [
    # (name, encoder_bits, decoder_bits, group_size)
    ("fp16_baseline", None, None, None),
    ("enc_q8_dec_fp16", 8, None, 64),
    ("enc_q4_dec_fp16", 4, None, 64),
    ("enc_q3_dec_fp16", 3, None, 64),
    ("enc_q2_dec_fp16", 2, None, 64),
    ("enc_fp16_dec_q8", None, 8, 64),
    ("enc_fp16_dec_q4", None, 4, 64),
    ("enc_fp16_dec_q3", None, 3, 64),
    ("enc_fp16_dec_q2", None, 2, 64),
    ("enc_q4_dec_q8", 4, 8, 64),
    ("enc_q4_dec_q4", 4, 4, 64),
    ("enc_q3_dec_q4", 3, 4, 64),
    ("enc_q2_dec_q4", 2, 4, 64),
    ("enc_q3_dec_q3", 3, 3, 64),
    ("enc_q2_dec_q2", 2, 2, 64),
]

# ─── Evaluation ─────────────────────────────────────────────────────────

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, processor, dataset, max_samples=10):
    """Run greedy decoding on dataset, return WER and speed."""
    model.eval()
    model = model.to(_DEVICE)
    texts = []
    refs = []
    times = []
    for i in range(min(max_samples, len(dataset))):
        audio = np.array(dataset[i]["audio"]["array"], dtype=np.float32)
        ref = dataset[i]["text"]
        refs.append(ref)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(_DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(inputs, max_new_tokens=MAX_TOKENS)
        t = time.perf_counter() - t0
        times.append(t)
        text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        texts.append(text)
    import jiwer
    refs_norm = [jiwer.RemoveMultipleSpaces()(jiwer.Strip()(jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(r)))) for r in refs]
    texts_norm = [jiwer.RemoveMultipleSpaces()(jiwer.Strip()(jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t)))) for t in texts]
    wer = jiwer.wer(refs_norm, texts_norm)
    avg_time = np.mean(times)
    return wer, avg_time, texts

def apply_quantization(model, encoder_bits, decoder_bits, group_size=64):
    """Apply quantization to encoder and/or decoder submodules."""
    encoder = model.model.encoder
    decoder = model.model.decoder
    if encoder_bits is not None:
        replace_with_quantized(encoder, encoder_bits, group_size)
    if decoder_bits is not None:
        replace_with_quantized(decoder, decoder_bits, group_size)
    return model

# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--eval", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Device: {_DEVICE}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})", flush=True)
    print(f"Model: {args.model}", flush=True)

    # Load processor
    processor = WhisperProcessor.from_pretrained(args.model)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    eval_ds = ds.select(range(min(args.eval, len(ds))))

    results = []
    for cfg_name, enc_bits, dec_bits, gs in CONFIGS:
        if args.group_size != 64 and gs is not None:
            gs = args.group_size
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {cfg_name}  (enc={enc_bits}, dec={dec_bits}, gs={gs})", flush=True)
        print(f"{'='*60}", flush=True)

        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        fp16_size = model_size_bytes(model)

        if cfg_name != "fp16_baseline":
            model = apply_quantization(model, enc_bits, dec_bits, gs)

        q_size = model_size_bytes(model)
        model = model.to(_DEVICE)

        try:
            wer, avg_time, texts = evaluate_model(model, processor, eval_ds, args.eval)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}", flush=True)
            results.append({
                "config": cfg_name, "enc_bits": enc_bits, "dec_bits": dec_bits,
                "group_size": gs, "wer": None, "avg_time": None, "size_mb": None,
                "error": str(e)
            })
            continue

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        size_mb = q_size / (1024 * 1024)
        speedup = None
        if len(results) > 0 and results[0]["avg_time"] is not None and cfg_name != "fp16_baseline":
            base_time = results[0]["avg_time"]
            if base_time > 0:
                speedup = base_time / avg_time

        print(f"  WER: {wer*100:.2f}%  |  Avg time: {avg_time*1000:.1f}ms  |  Size: {size_mb:.1f}MB  |  Speedup: {speedup:.2f}x" if speedup else f"  WER: {wer*100:.2f}%  |  Avg time: {avg_time*1000:.1f}ms  |  Size: {size_mb:.1f}MB", flush=True)

        results.append({
            "config": cfg_name, "enc_bits": enc_bits, "dec_bits": dec_bits,
            "group_size": gs, "wer": round(wer, 6), "avg_time": round(avg_time, 4),
            "size_mb": round(size_mb, 2), "speedup": round(speedup, 4) if speedup else None,
        })

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY - {args.model}", flush=True)
    print(f"{'='*80}", flush=True)
    header = f"{'Config':<25} {'WER':>8} {'Time(ms)':>10} {'Size(MB)':>10} {'Speedup':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        wer_str = f"{r['wer']*100:.2f}%" if r['wer'] is not None else "FAILED"
        time_str = f"{r['avg_time']*1000:.0f}" if r['avg_time'] else "N/A"
        size_str = f"{r['size_mb']:.1f}" if r['size_mb'] else "N/A"
        sp_str = f"{r['speedup']:.2f}x" if r.get('speedup') else "1.00x" if r['config'] == 'fp16_baseline' else "N/A"
        print(f"{r['config']:<25} {wer_str:>8} {time_str:>10} {size_str:>10} {sp_str:>8}", flush=True)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}", flush=True)
    else:
        out_path = f"/content/quantization_results_{args.model.replace('/', '_')}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
