"""P30: 4-Bit Extreme Quantization

Goal: The 3.94x baseline was achieved via Q8 quantization and Flash Attention.
To beat it without algorithmic hacks (which have uniformly failed), we will
push the hardware limits by applying 4-bit quantization (Q4) to the entire model.
We will test if Q4 doubles the speed of Q8 while maintaining textual fidelity on
long-form continuous audio.
"""

import time
import json
import difflib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_whisper
from datasets import load_dataset

def main():
    model_name = "mlx-community/whisper-large-v3-mlx"
    
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
    
    # We will use mlx_whisper.transcribe for the easiest pipeline
    
    # 1. Float16 Baseline
    print("\nRunning FP16 Baseline...")
    t0 = time.perf_counter()
    model_fp16 = mlx_whisper.load_models.load_model(model_name)
    res_fp16 = mlx_whisper.transcribe(long_audio, path_or_hf_repo=model_name)
    t_fp16 = time.perf_counter() - t0
    text_fp16 = res_fp16["text"]
    

    # We can quantize the model manually:
    model = mlx_whisper.load_models.load_model(model_name)
    nn.quantize(model, group_size=64, bits=8)
    
    import sys
    
    print("\nRunning Q8 (Baseline in paper)...")
    t0 = time.perf_counter()
    sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None
    
    # Pre-load and quantize so transcribe picks it up? No, transcribe just loads it!
    # Wait, if transcribe loads it, how do we quantize it before transcribe uses it?
    # We can't unless we patch load_model!
    
    # Better: just use our own decode loop for speed test, or patch load_model!
    # Patch load_model!
    original_load = mlx_whisper.load_models.load_model
    
    def mock_load_q8(*args, **kwargs):
        m = original_load(*args, **kwargs)
        nn.quantize(m, group_size=64, bits=8)
        return m
        
    mlx_whisper.load_models.load_model = mock_load_q8
    res_q8 = mlx_whisper.transcribe(long_audio, path_or_hf_repo=model_name)
    t_q8 = time.perf_counter() - t0
    text_q8 = res_q8["text"]
    
    def mock_load_q4(*args, **kwargs):
        m = original_load(*args, **kwargs)
        nn.quantize(m, group_size=64, bits=4)
        return m
        
    sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None
    mlx_whisper.load_models.load_model = mock_load_q4
    
    print("\nRunning Q4...")
    t0 = time.perf_counter()
    res_q4 = mlx_whisper.transcribe(long_audio, path_or_hf_repo=model_name)
    t_q4 = time.perf_counter() - t0
    text_q4 = res_q4["text"]
    
    
    print(f"\n--- Performance ---")
    print(f"FP16 Time: {t_fp16:.2f}s")
    print(f"Q8 Time:   {t_q8:.2f}s (Speedup over FP16: {t_fp16/t_q8:.2f}x)")
    print(f"Q4 Time:   {t_q4:.2f}s (Speedup over FP16: {t_fp16/t_q4:.2f}x, over Q8: {t_q8/t_q4:.2f}x)")
    
    print(f"\n--- Accuracy (Similarity to FP16) ---")
    sim_q8 = difflib.SequenceMatcher(None, text_fp16, text_q8).ratio()
    sim_q4 = difflib.SequenceMatcher(None, text_fp16, text_q4).ratio()
    print(f"Q8 Text Similarity: {sim_q8 * 100:.2f}%")
    print(f"Q4 Text Similarity: {sim_q4 * 100:.2f}%")
    
    out_path = Path("results/p30_q4.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P30: 4-bit Quantization",
            "time_fp16": t_fp16,
            "time_q8": t_q8,
            "time_q4": t_q4,
            "sim_q8": sim_q8,
            "sim_q4": sim_q4,
            "text_fp16": text_fp16,
            "text_q4": text_q4
        }, f, indent=2)

if __name__ == "__main__":
    main()
