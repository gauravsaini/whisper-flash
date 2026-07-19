"""P37+P35+P38: True composable speedup stack via native mlx_whisper path.

Measures the REAL lossless speedup of composable levers on whisper-tiny and
whisper-large-v3-turbo using mlx_whisper.transcribe (the vectorized path that
gave the original 3.5x KV claim — NOT the slow custom per-step loop).

Levers:
  - Q8   : native nn.QuantizedLinear
  - STRIDE: avg-pool encoder output along time (lossless per E2 on tiny)
  - KV    : inherent to mlx_whisper.transcribe (no-op flag, but we confirm)

WER computed against ground-truth dummy transcripts.
"""

import time
import json
import io
import tempfile
import os
import soundfile as sf
import numpy as np
import mlx.core as mx
import mlx.nn as nn

import mlx_whisper
from mlx_whisper.load_models import load_model
import mlx_whisper.whisper as whisper_module

STRIDE = 1  # set per-run

# ── Monkey-patch encoder to support stride-2 downsample ──
_orig_encoder_call = None


def make_encoder_wrapper(orig_encoder, stride):
    class _Wrapper(nn.Module):
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

    return _Wrapper(orig_encoder, stride)


def quantize_linear_modules(model, bits):
    def _apply(name, module):
        if isinstance(module, nn.Linear):
            gs = 64 if module.weight.shape[-1] % 64 == 0 else 32
            return nn.QuantizedLinear.from_linear(module, gs, bits, "affine")
        return module
    model.apply_to_modules(_apply)


def load(repo, quantize=False, stride=1):
    model = load_model(repo, mx.float16)
    if stride > 1:
        model.encoder = make_encoder_wrapper(model.encoder, stride)
    if quantize:
        quantize_linear_modules(model, 8)
    mx.eval(model.parameters())
    return model


_GLOBAL_MODEL = None


def transcribe(model, audio_bytes):
    global _GLOBAL_MODEL
    _GLOBAL_MODEL = model
    real_load = mlx_whisper.load_models.load_model

    def _patched(repo_id, dtype=None, **kw):
        return _GLOBAL_MODEL

    mlx_whisper.load_models.load_model = _patched
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        t0 = time.perf_counter()
        out = mlx_whisper.transcribe(path, fp16=True, verbose=False,
                                     condition_on_previous_text=False)
        t1 = time.perf_counter()
        mlx_whisper.load_models.load_model = real_load
        return out["text"].strip(), t1 - t0
    finally:
        mlx_whisper.load_models.load_model = real_load
        os.unlink(path)


def load_dummy(n=10):
    import pyarrow as pa
    p = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"
    with pa.memory_map(p) as src:
        try:
            t = pa.ipc.open_file(src).read_all()
        except Exception:
            t = pa.ipc.open_stream(src).read_all()
    rows = t.to_pylist()[:n]
    return rows


def main():
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny-mlx"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = load_dummy(n)
    refs = [r["text"].strip().lower() for r in rows]

    def wer(ref, hyp):
        from jiwer import wer as _wer
        return _wer(ref, hyp)

    extra_strides = []
    if len(sys.argv) > 3:
        try:
            extra_strides = [int(x) for x in sys.argv[3].split(",")]
        except ValueError:
            pass
    configs = {
        "Baseline(fp16,stride1)": dict(quantize=False, stride=1),
        "Q8": dict(quantize=True, stride=1),
        "STRIDE2": dict(quantize=False, stride=2),
        "Q8+STRIDE2": dict(quantize=True, stride=2),
    }
    for s in extra_strides:
        if s != 1:
            configs[f"STRIDE{s}"] = dict(quantize=False, stride=s)
            configs[f"Q8+STRIDE{s}"] = dict(quantize=True, stride=s)

    results = {}
    base_time = None
    base_wer = None
    for name, cfg in configs.items():
        m = load(repo, **cfg)
        times, hyps = [], []
        for r in rows:
            hyp, dt = transcribe(m, r["audio"]["bytes"])
            times.append(dt)
            hyps.append(hyp.lower())
        del m
        mx.clear_cache()
        w = np.mean([wer(refs[i], hyps[i]) for i in range(len(rows))])
        mt = np.mean(times)
        if base_time is None:
            base_time = mt
            base_wer = float(w)
        sp = base_time / mt
        results[name] = dict(mean_time_s=round(mt, 4), mean_wer=round(float(w), 4),
                             speedup=round(sp, 3), lossless=bool(float(w) <= base_wer + 0.01))
        print(f"{name:24s} time={mt:.3f}s wer={w:.4f} speedup={sp:.3f}x")
        mx.clear_cache()

    out = dict(experiment="P37/P35/P38 composable stack native path", model=repo,
               n_samples=n, baseline_time_s=round(base_time, 4), results=results)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
