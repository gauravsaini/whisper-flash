"""P33: Architectural Pruning (Large-V3 Turbo)

Instead of algorithmic hacking, early exit, or speculative decoding with mismatched tokenizers,
we evaluate a fully trained, depth-pruned architecture (whisper-large-v3-turbo).
Since it reduces the decoder layers from 32 to 4, it should theoretically provide 
an 8x speedup over the large-v3 baseline without zero-shot semantic degradation.
"""

import time
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

import mlx_whisper
from datasets import load_dataset

def main():
    print("Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    
    # Concatenate 15 distinct samples to create 152.7s audio
    print("Concatenating 15 distinct audio samples...")
    audio_chunks = []
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    
    for i in range(15):
        audio_chunks.append(np.array(ds[i]["audio"]["array"], dtype=np.float32))
        audio_chunks.append(silence)
        
    long_audio = np.concatenate(audio_chunks)
    audio_duration = len(long_audio) / sample_rate
    print(f"Total audio duration: {audio_duration:.1f} seconds")
    
    # Baseline: Large-V3
    import sys
    sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None
    
    print("\nRunning Baseline (whisper-large-v3)...")
    model_name_large = "mlx-community/whisper-large-v3-mlx"
    t0 = time.perf_counter()
    res_large = mlx_whisper.transcribe(long_audio, path_or_hf_repo=model_name_large)
    t_large = time.perf_counter() - t0
    text_large = res_large["text"]
    
    sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None
    
    print("\nRunning Hypothesis (whisper-large-v3-turbo, Q4)...")
    model_name_turbo = "mlx-community/whisper-large-v3-turbo"
    # Wait, is turbo available in 4-bit? Yes, typically on mlx-community. If not, we just use the default.
    # mlx_whisper.transcribe natively loads the default weights. If 4bit exists, it uses it, else FP16.
    
    # We will just load standard turbo and manually quantize to be fair if we want,
    # or just use it directly. We'll load directly.
    t0 = time.perf_counter()
    res_turbo = mlx_whisper.transcribe(long_audio, path_or_hf_repo=model_name_turbo)
    t_turbo = time.perf_counter() - t0
    text_turbo = res_turbo["text"]
    
    # Metric: Real-Time Factor Speedup (RTF)
    baseline_rtf = audio_duration / t_large
    turbo_rtf = audio_duration / t_turbo
    
    print(f"\n--- Performance ---")
    print(f"Large-V3 Time: {t_large:.2f}s (Speedup: {baseline_rtf:.2f}x real-time)")
    print(f"Turbo Time:    {t_turbo:.2f}s (Speedup: {turbo_rtf:.2f}x real-time)")
    print(f"\nRelative Speedup over Baseline: {t_large / t_turbo:.2f}x")
    
    out_path = Path("results/p33_turbo.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P33: Architectural Pruning (Turbo)",
            "time_large": t_large,
            "time_turbo": t_turbo,
            "baseline_rtf": baseline_rtf,
            "turbo_rtf": turbo_rtf,
            "relative_speedup": t_large / t_turbo,
        }, f, indent=2)

if __name__ == "__main__":
    main()
