"""P38: True composable speedup stack on a long (481s) realistic audio clip.

Uses mlx_whisper.transcribe (native vectorized + native KV cache path) with
modular levers applied via monkey-patched model:
  - STRIDE : avg-pool encoder frames (lossless per E2 on tiny; measure on large)
  - Q8     : native nn.QuantizedLinear

Measures wall-clock speedup vs fp16 baseline. WER is spot-checked on the
8-sample dummy set (short, ground-truth available).
"""

import time, json, os, io, sys
import numpy as np
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn
import mlx_whisper
from mlx_whisper.load_models import load_model

LONG_WAV = "/tmp/flashbench/long.wav"
DUMMY_ARROW = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"


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


def quantize(model, bits):
    def _apply(name, module):
        if isinstance(module, nn.Linear):
            gs = 64 if module.weight.shape[-1] % 64 == 0 else 32
            return nn.QuantizedLinear.from_linear(module, gs, bits, "affine")
        return module
    model.apply_to_modules(_apply)


_GLOBAL = None


def run(repo, stride, q8, audio=LONG_WAV):
    global _GLOBAL
    model = load_model(repo, mx.float16)
    if stride > 1:
        model.encoder = EncoderStrideWrapper(model.encoder, stride)
    if q8:
        quantize(model, 8)
    mx.eval(model.parameters())
    _GLOBAL = model
    real = mlx_whisper.load_models.load_model

    def _patched(*a, **k):
        return _GLOBAL
    mlx_whisper.load_models.load_model = _patched
    try:
        t0 = time.perf_counter()
        out = mlx_whisper.transcribe(audio, fp16=True, verbose=False,
                                     condition_on_previous_text=False)
        t1 = time.perf_counter()
    finally:
        mlx_whisper.load_models.load_model = real
        del model
        mx.clear_cache()
    return out["text"], t1 - t0


def dummy_wer_rows(n=8):
    import pyarrow as pa
    with pa.memory_map(DUMMY_ARROW) as src:
        try:
            t = pa.ipc.open_file(src).read_all()
        except Exception:
            t = pa.ipc.open_stream(src).read_all()
    return t.to_pylist()[:n]


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-large-v3-turbo"
    configs = {
        "fp16_s1": (1, False),
        "q8_s1": (1, True),
        "s2": (2, False),
        "q8_s2": (2, True),
        "s3": (3, False),
        "q8_s3": (3, True),
        "s4": (4, False),
        "q8_s4": (4, True),
    }
    results = {}
    base = None
    # timing on long audio
    for name, (s, q) in configs.items():
        text, dt = run(repo, s, q)
        if base is None:
            base = dt
        sp = base / dt
        results[name] = dict(long_time_s=round(dt, 3), speedup=round(sp, 3))
        print(f"{name:10s} long_time={dt:.3f}s speedup={sp:.3f}x")
    # WER spot-check (short dummy, 8 samples)
    from jiwer import wer as _wer
    rows = dummy_wer_rows(8)
    refs = [r["text"].strip().lower() for r in rows]
    wers = {}
    for name, (s, q) in configs.items():
        hyps = []
        for r in rows:
            h, _ = run_bytes(repo, s, q, r["audio"]["bytes"])
            hyps.append(h.lower())
        w = float(np.mean([_wer(refs[i], hyps[i]) for i in range(len(rows))]))
        wers[name] = round(w, 4)
        print(f"{name:10s} wer={w:.4f}")
    for name in configs:
        results[name]["wer"] = wers[name]
        results[name]["lossless"] = bool(wers[name] <= wers["fp16_s1"] + 0.01)
    out = dict(experiment="P38 composable stack (long audio + WER)",
               model=repo, long_audio_sec=481, n_wer_samples=8, results=results)
    print(json.dumps(out, indent=2))


def run_bytes(repo, stride, q8, audio_bytes):
    global _GLOBAL
    model = load_model(repo, mx.float16)
    if stride > 1:
        model.encoder = EncoderStrideWrapper(model.encoder, stride)
    if q8:
        quantize(model, 8)
    mx.eval(model.parameters())
    _GLOBAL = model
    real = mlx_whisper.load_models.load_model

    def _patched(*a, **k):
        return _GLOBAL
    mlx_whisper.load_models.load_model = _patched
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        t0 = time.perf_counter()
        out = mlx_whisper.transcribe(path, fp16=True, verbose=False,
                                     condition_on_previous_text=False)
        t1 = time.perf_counter()
    finally:
        mlx_whisper.load_models.load_model = real
        os.unlink(path)
        del model
        mx.clear_cache()
    return out["text"], t1 - t0


if __name__ == "__main__":
    main()
