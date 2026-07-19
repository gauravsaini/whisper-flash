"""P23: Input-Audio Speculative Streaming

Goal: In long-form decoding (e.g., streaming), Whisper processes 30s chunks sequentially.
The text output of Chunk A is passed as the `<|startofprev|>` prompt to Chunk B.
If this prompt does not significantly alter the output of Chunk B, we can process
all chunks in parallel (or without waiting for Chunk A's text), breaking the sequential
text bottleneck.
"""

import time
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

import mlx_whisper

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    sample_rate = ds[0]["audio"]["sampling_rate"]
    
    # Create a 45-second audio by duplicating the dummy audio with 2-second silent gaps
    silence = np.zeros(2 * sample_rate, dtype=np.float32)
    long_audio = np.concatenate([audio, silence, audio, silence, audio])
    
    print(f"Testing long-form transcription on {len(long_audio)/sample_rate:.1f}s audio.")
    
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
    
    print(f"\nConditioned Time:   {t_cond:.2f}s")
    print(f"Unconditioned Time: {t_uncond:.2f}s")
    
    # Is it exactly the same?
    match = (text_cond.strip() == text_uncond.strip())
    print(f"Text Match: {match}")
    
    if not match:
        print(f"\nConditioned:   {text_cond}")
        print(f"Unconditioned: {text_uncond}")
        
    out_path = Path("results/p23_audio_speculation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P23: Audio Speculation",
            "time_cond": t_cond,
            "time_uncond": t_uncond,
            "match": match,
            "text_cond": text_cond,
            "text_uncond": text_uncond
        }, f, indent=2)

if __name__ == "__main__":
    main()
