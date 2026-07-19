"""P41: Cross-Attention Sparing (Alternating Cross-Attention Updates)

This experiment tests if we can statically skip cross-attention computations
on alternating steps (e.g. every N-th step) during autoregressive decoding,
reusing the cached cross-attention output from the previous step.

This cuts the cross-attention FLOPs (which is the main decoder bottleneck
on large models) without requiring any CPU-GPU sync.
"""
import time
import json
import sys
import os
import tempfile
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_whisper
from mlx_whisper.load_models import load_model
import pyarrow as pa
from jiwer import wer as _wer

DUMMY_ARROW = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"
LONG_WAV = "/tmp/flashbench/long.wav"

# --- Encoder Stride Wrapper (P38/P40) ---
class EncoderStrideWrapper(nn.Module):
    def __init__(self, enc, s):
        super().__init__()
        object.__setattr__(self, "_enc", enc)
        object.__setattr__(self, "_stride", s)

    def __call__(self, x):
        out = self._enc(x)
        if self._stride > 1:
            B, T, D = out.shape
            Tt = (T // self._stride) * self._stride
            out = out[:, :Tt, :]
            out = mx.mean(out.reshape(B, Tt // self._stride, self._stride, D), axis=2)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_enc"), name)

# --- Q8 Quantization (P38/P40) ---
def quantize(model, bits):
    def _apply(name, module):
        if isinstance(module, nn.Linear):
            gs = 64 if module.weight.shape[-1] % 64 == 0 else 32
            return nn.QuantizedLinear.from_linear(module, gs, bits, "affine")
        return module
    model.apply_to_modules(_apply)

# --- Monkey-Patching ResidualAttentionBlock for Sparing ---
import mlx_whisper.whisper as whisper_module
original_block_call = whisper_module.ResidualAttentionBlock.__call__

def patched_block_call(self, x, xa=None, mask=None, kv_cache=None):
    kv, cross_kv = kv_cache if kv_cache else (None, None)
    
    # Self-attention
    y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
    x = x + y
    
    cross_qk = None
    if self.cross_attn:
        sparing_n = getattr(self, "sparing_n", 1)
        q_len = x.shape[1]
        
        # We only spare if:
        # 1. q_len == 1 (autoregressive generation)
        # 2. cross_kv is not None (prefilled cross-attention KV exists)
        # 3. sparing_n > 1
        if q_len == 1 and cross_kv is not None and sparing_n > 1:
            step = kv[0].shape[1] if kv is not None else 0
            
            # If step % sparing_n == 0 or we haven't computed y_cross yet
            if len(cross_kv) < 3 or step % sparing_n == 0:
                k_v_cache = (cross_kv[0], cross_kv[1])
                y_cross, k_v, cross_qk = self.cross_attn(
                    self.cross_attn_ln(x), xa, kv_cache=k_v_cache
                )
                x = x + y_cross
                cross_kv = (k_v[0], k_v[1], y_cross)
            else:
                y_cross = cross_kv[2]
                x = x + y_cross
        else:
            # Prefill or sparing disabled
            y_cross, k_v, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x = x + y_cross
            cross_kv = k_v
            
    x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
    return x, (kv, cross_kv), cross_qk

whisper_module.ResidualAttentionBlock.__call__ = patched_block_call

# --- Helper functions ---
def set_sparing(model, sparing_n):
    for block in model.decoder.blocks:
        block.sparing_n = sparing_n

def load_dummy_rows(n=None):
    with pa.memory_map(DUMMY_ARROW) as src:
        try:
            t = pa.ipc.open_file(src).read_all()
        except Exception:
            t = pa.ipc.open_stream(src).read_all()
    rows = t.to_pylist()
    return rows if n is None else rows[:n]

_GLOBAL_MODEL = None

def transcribe_with_model(model, sparing_n, audio_bytes):
    set_sparing(model, sparing_n)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        out = mlx_whisper.transcribe(path, fp16=True, verbose=False, condition_on_previous_text=False)
        text = out["text"]
    finally:
        os.unlink(path)
    return text

def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny-mlx"
    n_wer = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    print(f"Using model: {repo}")
    print(f"Using {n_wer} samples for WER evaluation")
    
    # Pre-load dummy dataset
    rows = load_dummy_rows(n_wer)
    refs = [r["text"].strip().lower() for r in rows]
    
    # Configurations to evaluate: (stride, q8, sparing_n)
    # We sweep sparing_n across 1 (none), 2, 4, 8
    configs = [
        ("s1_fp16_sp1", (1, False, 1)),
        ("s1_fp16_sp2", (1, False, 2)),
        ("s1_fp16_sp3", (1, False, 3)),
        ("s1_fp16_sp4", (1, False, 4)),
        ("s4_fp16_sp1", (4, False, 1)),
        ("s4_fp16_sp2", (4, False, 2)),
        ("s4_fp16_sp4", (4, False, 4)),
        ("s4_q8_sp2",   (4, True, 2)),
        ("s4_q8_sp4",   (4, True, 4)),
    ]
    
    # If using tiny, we don't run stride-4 as it hurts tiny. We do stride-2.
    if "tiny" in repo:
        configs = [
            ("s1_fp16_sp1", (1, False, 1)),
            ("s1_fp16_sp2", (1, False, 2)),
            ("s1_fp16_sp3", (1, False, 3)),
            ("s1_fp16_sp4", (1, False, 4)),
            ("s2_fp16_sp1", (2, False, 1)),
            ("s2_fp16_sp2", (2, False, 2)),
            ("s1_q8_sp1",   (1, True, 1)),
            ("s1_q8_sp2",   (1, True, 2)),
            ("s1_q8_sp4",   (1, True, 4)),
        ]

    # Check if LONG_WAV exists. If not, we will construct it from the first 25 samples to make a long wav.
    if not os.path.exists(LONG_WAV):
        print(f"LONG_WAV not found at {LONG_WAV}. Constructing temporary long clip from dataset...")
        os.makedirs(os.path.dirname(LONG_WAV), exist_ok=True)
        import soundfile as sf
        long_audio_data = []
        for r in load_dummy_rows(25): # concatenate 25 samples (approx 150-200s)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(r["audio"]["bytes"])
                path = f.name
            data, sr = sf.read(path)
            long_audio_data.append(data)
            os.unlink(path)
        concatenated = np.concatenate(long_audio_data)
        sf.write(LONG_WAV, concatenated, sr)
        print(f"Created concatenated long clip: {len(concatenated)/sr:.2f}s")
        
    results = {}
    wers = {}
    base_time = None
    
    for name, (s, q, sp) in configs:
        print(f"\nEvaluating configuration: {name}")
        
        # Load and configure model ONCE per config
        model = load_model(repo, mx.float16)
        if s > 1:
            model.encoder = EncoderStrideWrapper(model.encoder, s)
        if q:
            quantize(model, 8)
        mx.eval(model.parameters())
        
        # Monkey-patch mlx_whisper load_model to return this instance
        global _GLOBAL_MODEL
        _GLOBAL_MODEL = model
        real_load = mlx_whisper.load_models.load_model
        mlx_whisper.load_models.load_model = lambda *a, **k: _GLOBAL_MODEL
        
        try:
            # 1. Run WER evaluation
            hyps = []
            dts = []
            for r in rows:
                t0 = time.perf_counter()
                h = transcribe_with_model(model, sp, r["audio"]["bytes"])
                dts.append(time.perf_counter() - t0)
                hyps.append(h.lower())
            w = float(np.mean([_wer(refs[i], hyps[i]) for i in range(len(rows))]))
            wers[name] = w
            print(f"  WER: {w:.4f} (avg sample time={np.mean(dts):.3f}s)")
            
            # 2. Run Long-Audio timing
            t0 = time.perf_counter()
            set_sparing(model, sp)
            mlx_whisper.transcribe(LONG_WAV, fp16=True, verbose=False, condition_on_previous_text=False)
            dt = time.perf_counter() - t0
            
            if base_time is None:
                base_time = dt
            speedup = base_time / dt
            results[name] = {
                "wer": round(w, 4),
                "lossless": bool(w <= wers[configs[0][0]] + 0.01),
                "long_time_s": round(dt, 3),
                "speedup": round(speedup, 3)
            }
            print(f"  Long timing: {dt:.3f}s (Speedup: {speedup:.3f}x)")
        finally:
            mlx_whisper.load_models.load_model = real_load
            del model
            _GLOBAL_MODEL = None
            mx.clear_cache()
        
    out = {
        "experiment": "P41: Cross-Attention Sparing (Alternating Cross-Attention Updates)",
        "model": repo,
        "n_wer_samples": n_wer,
        "results": results
    }
    
    print("\n" + json.dumps(out, indent=2))
    
    # Save results to file
    out_path = "/Users/ektasaini/Desktop/whisper-flash/results/p41_cross_attn_sparing.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
