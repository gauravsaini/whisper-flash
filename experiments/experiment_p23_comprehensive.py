"""P23 Comprehensive Validation

Goal: The initial P23 result showed an 8.7x speedup and identical text when disabling 
`condition_on_previous_text`. However, because we duplicated the same audio sample,
Whisper's fallback mechanism likely triggered due to repetition penalty, artificially
inflating the latency of the Conditioned mode.

To comprehensively validate this, we will concatenate 15 distinct, sequential samples
from the Librispeech dummy dataset to form a ~2-minute long contiguous audio file.
We will then test Conditioned vs Unconditioned transcription to accurately measure:
1. Realistic Latency difference (Speedup).
2. Semantic/Textual difference (Similarity ratio).
"""

import time
import json
import difflib
from pathlib import Path
import mlx.core as mx
import numpy as np
import mlx_whisper
from datasets import load_dataset

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    
    print("Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    
    # Concatenate first 15 distinct samples to create realistic long-form audio
    print("Concatenating 15 distinct audio samples...")
    audio_chunks = []
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    
    for i in range(15):
        audio_chunks.append(np.array(ds[i]["audio"]["array"], dtype=np.float32))
        audio_chunks.append(silence)
        
    long_audio = np.concatenate(audio_chunks)
    audio_duration = len(long_audio) / sample_rate
    print(f"Total audio duration: {audio_duration:.1f} seconds")
    
    # 1. Conditioned (Sequential)
    print("\nRunning Conditioned (Sequential) Transcription...")
    t0 = time.perf_counter()
    res_cond = mlx_whisper.transcribe(
        long_audio, 
        path_or_hf_repo=model_name,
        condition_on_previous_text=True
    )
    t_cond = time.perf_counter() - t0
    
    # 2. Unconditioned (Parallelizable)
    print("\nRunning Unconditioned (Parallelizable) Transcription...")
    t0 = time.perf_counter()
    res_uncond = mlx_whisper.transcribe(
        long_audio, 
        path_or_hf_repo=model_name,
        condition_on_previous_text=False
    )
    t_uncond = time.perf_counter() - t0
    
    text_cond = res_cond["text"]
    text_uncond = res_uncond["text"]
    
    print(f"\n--- Performance ---")
    print(f"Conditioned Time:   {t_cond:.2f}s")
    print(f"Unconditioned Time: {t_uncond:.2f}s")
    
    speedup = t_cond / t_uncond
    print(f"Speedup: {speedup:.2f}x")
    
    print(f"\n--- Accuracy ---")
    matcher = difflib.SequenceMatcher(None, text_cond, text_uncond)
    similarity = matcher.ratio()
    print(f"Text Similarity: {similarity * 100:.2f}%")
    
    out_path = Path("results/p23_comprehensive.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P23: Comprehensive Audio Speculation",
            "audio_duration_sec": audio_duration,
            "time_cond": t_cond,
            "time_uncond": t_uncond,
            "speedup": speedup,
            "similarity": similarity,
            "text_cond": text_cond,
            "text_uncond": text_uncond
        }, f, indent=2)

if __name__ == "__main__":
    main()
