"""P31: Q4 Quantization and Full Stack on whisper-large-v3-turbo.

Runs configurations on the SAME 10 dense LibriSpeech WAV samples:
  (a) Baseline (fp16, no KV)
  (b) Q8 only (no KV)
  (c) Q4 only (no KV)
  (d) Q8 + KV cache + stride-2 (FULL Q8)
  (e) Q4 + KV cache + stride-2 (FULL Q4)
"""

import json
import time
import argparse
from pathlib import Path
import mlx.core as mx
import numpy as np

from whisper_flash_mlx.production import ProductionConfig, GreedyDecoder
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.target_model import load_target_model

MODEL = "mlx-community/whisper-large-v3-turbo"
N_SAMPLES = 10
RUNS = 2
AUDIO_DIR = Path("experiments/p15_audio")

def get_audio_paths():
    paths = []
    for i in range(N_SAMPLES):
        p = AUDIO_DIR / f"sample_{i:02d}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing sample file: {p}. Run p15 first.")
        paths.append(str(p))
    return paths

def run_custom_config(quantize_bits: int | None, use_kv: bool, stride: int, audio_paths: list[str]) -> dict:
    # Set up production config
    cfg = ProductionConfig(
        model_path=MODEL,
        quantize=False,  # We will quantize manually if quantize_bits is set
        encoder_stride=stride,
        use_kv_cache=use_kv,
    )
    
    # Load decoder
    dec = GreedyDecoder(cfg)
    
    # Quantize manually if requested
    if quantize_bits is not None:
        print(f"Quantizing model to {quantize_bits}-bits...")
        quantize_model(dec.model, encoder_bits=quantize_bits, decoder_bits=quantize_bits, group_size=64)
        
    times, tps, texts = [], [], []
    for ap in audio_paths:
        run_times = []
        last_text = None
        for _ in range(RUNS):
            r = dec.decode(ap)
            run_times.append(r.wall_time_s)
            last_text = r.text
        mean_t = sum(run_times) / len(run_times)
        times.append(mean_t)
        tps.append(r.tokens_per_sec)
        texts.append(last_text)
        
    return {
        "mean_wall_s": sum(times) / len(times),
        "mean_tps": sum(tps) / len(tps),
        "texts": texts,
    }

def wer_lossless(baseline_texts, texts):
    from jiwer import wer
    w = max(wer(b, h) for b, h in zip(baseline_texts, texts))
    exact = all(b == h for b, h in zip(baseline_texts, texts))
    return exact, w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/p31_large_q4.json")
    args = parser.parse_args()

    audio_paths = get_audio_paths()

    configs = {
        "Baseline (fp16,noKV)":      (None, False, 1),
        "Q8 only (noKV)":            (8, False, 1),
        "Q4 only (noKV)":            (4, False, 1),
        "FULL Q8 (Q8+KV+stride2)":   (8, True, 2),
        "FULL Q4 (Q4+KV+stride2)":   (4, True, 2),
    }

    results = {}
    for name, (q_bits, use_kv, stride) in configs.items():
        print(f"\n{'='*70}\n  Running: {name}\n{'='*70}")
        res = run_custom_config(q_bits, use_kv, stride, audio_paths)
        results[name] = res

    base = results["Baseline (fp16,noKV)"]
    base_texts = base["texts"]
    base_time = base["mean_wall_s"]

    # Calculate exact matching and WER against baseline
    loss = {}
    for name in configs:
        if name == "Baseline (fp16,noKV)":
            continue
        exact, w = wer_lossless(base_texts, results[name]["texts"])
        loss[name] = (exact, w)

    print(f"\n\n{'='*80}")
    print(f"  P31 LARGE Q4 RESULTS — {MODEL}  (10 samples, {RUNS} runs each)")
    print(f"{'='*80}")
    print(f"{'Config':<28}{'Time(s)':>9}{'Tok/s':>9}{'Speedup':>9}{'Lossless':>14}")
    print("-" * 80)
    for name, res in results.items():
        sp = base_time / res["mean_wall_s"] if res["mean_wall_s"] > 0 else 0
        if name == "Baseline (fp16,noKV)":
            ident = "n/a (ref)"
        else:
            exact, w = loss[name]
            ident = "YES" if (exact and w == 0.0) else f"NO (WER={w:.4f})"
        print(f"{name:<28}{res['mean_wall_s']:>9.3f}{res['mean_tps']:>9.1f}{sp:>8.3f}x{ident:>14}")
    print("=" * 80)

    out = {
        "experiment": "P31: Q4 Quantization and Full Stack on whisper-large-v3-turbo",
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "runs": RUNS,
        "baseline_time_s": base_time,
        "results": {n: {"time_s": r["mean_wall_s"], "tps": r["mean_tps"],
                        "speedup": base_time / r["mean_wall_s"] if r["mean_wall_s"] > 0 else 0,
                        "lossless_exact": (loss[n][0] if n != "Baseline (fp16,noKV)" else None),
                        "wer_vs_baseline": (loss[n][1] if n != "Baseline (fp16,noKV)" else 0.0)}
                    for n, r in results.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
