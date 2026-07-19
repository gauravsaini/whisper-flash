"""P13: Batched Multi-Segment Decoding

Goal: Accelerate transcription of long audio by processing independent 30s chunks in parallel.
Whisper typically processes 30s chunks sequentially. By batching them at the encoder
and decoder stages, we can achieve near-linear speedups on high-memory-bandwidth hardware (Apple M-series).

This experiment:
1. Simulates a long audio stream by concatenating multiple LibriSpeech samples.
2. Segments the audio into 30s chunks.
3. Decodes sequentially (baseline).
4. Decodes in a single batch (B=chunks).
5. Compares wall-clock throughput.
"""

import time
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


def get_dummy_audio(num_chunks: int = 4) -> np.ndarray:
    """Load and concatenate samples to simulate a long audio file."""
    from datasets import load_dataset as hf_load
    import librosa
    
    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_segments = []
    
    for i in range(min(num_chunks, len(ds))):
        audio = ds[i]["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        
        # Pad slightly if needed to make it interesting, or just append
        audio_segments.append(arr)
        
    # We want exactly 30s chunks for simplicity, so let's just create synthetic 30s chunks
    # by repeating the audio if necessary, to ensure we have exactly `num_chunks` 30s chunks.
    
    chunk_samples = 16000 * 30
    full_audio = np.concatenate(audio_segments)
    
    # Repeat audio until we have enough for num_chunks * 30s
    target_length = num_chunks * chunk_samples
    while len(full_audio) < target_length:
        full_audio = np.concatenate([full_audio, full_audio])
        
    return full_audio[:target_length]


def prepare_mels(audio: np.ndarray) -> mx.array:
    """Convert audio to (B, 3000, 80) mel spectrograms."""
    from mlx_whisper.audio import log_mel_spectrogram
    
    chunk_samples = 16000 * 30
    n_chunks = len(audio) // chunk_samples
    
    mels = []
    for i in range(n_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        mel = log_mel_spectrogram(chunk, n_mels=80)
        # Pad to exactly 3000 frames (30 seconds)
        if mel.shape[0] < 3000:
            pad_widths = [(0, 3000 - mel.shape[0]), (0, 0)]
            mel = np.pad(mel, pad_widths)
        else:
            mel = mel[:3000, :]
        mels.append(mel)
        
    # Stack into (B, 80, 3000)
    return mx.array(np.stack(mels, axis=0))


def decode_sequential(model, mels: mx.array, max_tokens: int = 100) -> tuple[float, list[list[int]]]:
    """Decode each chunk sequentially (Standard approach)."""
    B = mels.shape[0]
    total_time = 0.0
    all_outputs = []
    
    for i in range(B):
        # Add batch dimension: (1, 80, 3000)
        mel_slice = mels[i:i+1]
        
        t0 = time.perf_counter()
        enc = encoder_forward(model, mel_slice)
        mx.eval(enc)
        
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, dec, enc, kv_cache=None, collect_hidden_states=False)
        
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        output_ids = [SOT_ID, first.item()]
        
        while len(output_ids) < max_tokens:
            last_tok = output_ids[-1]
            if last_tok == EOS_ID:
                break
                
            inp = mx.array([[last_tok]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            output_ids.append(tok.item())
            
        total_time += (time.perf_counter() - t0)
        all_outputs.append(output_ids)
        
    return total_time, all_outputs


def decode_batched(model, mels: mx.array, max_tokens: int = 100) -> tuple[float, list[list[int]]]:
    """Decode all chunks in parallel within a single batch."""
    B = mels.shape[0]
    t0 = time.perf_counter()
    
    enc = encoder_forward(model, mels)
    mx.eval(enc)
    
    # Initialize decoder input for B sequences
    dec = mx.full((B, 1), SOT_ID, dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    
    # Greedy sample for all B sequences
    first = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(first)
    
    output_ids = [[SOT_ID, first[i].item()] for i in range(B)]
    done = [first[i].item() == EOS_ID for i in range(B)]
    
    step = 0
    while not all(done) and step < max_tokens - 2:
        step += 1
        
        # Prepare input for next step (B, 1)
        last_toks = [[out[-1]] for out in output_ids]
        inp = mx.array(last_toks, dtype=mx.int32)
        
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        
        toks = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(toks)
        
        for i in range(B):
            if not done[i]:
                tid = toks[i].item()
                output_ids[i].append(tid)
                if tid == EOS_ID:
                    done[i] = True
                    
    wall_time = time.perf_counter() - t0
    return wall_time, output_ids


def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    model = load_target_model(model_name, dtype=mx.float16)
    
    # Simulate a 2-minute audio file (4 chunks of 30s)
    num_chunks = 4
    audio = get_dummy_audio(num_chunks)
    mels = prepare_mels(audio)
    
    print(f"Prepared {num_chunks} chunks of 30s audio. Total duration: {num_chunks * 30}s")
    
    # Run once to warmup
    print("Warming up...")
    _ = decode_batched(model, mels[:1], max_tokens=10)
    
    print("\n--- Running Sequential Decoding ---")
    seq_time, seq_outs = decode_sequential(model, mels, max_tokens=150)
    seq_tokens = sum(len(o) for o in seq_outs)
    print(f"Sequential Time: {seq_time:.3f}s")
    print(f"Sequential TPS:  {seq_tokens / seq_time:.1f}")
    
    print("\n--- Running Batched Decoding ---")
    bat_time, bat_outs = decode_batched(model, mels, max_tokens=150)
    bat_tokens = sum(len(o) for o in bat_outs)
    print(f"Batched Time:    {bat_time:.3f}s")
    print(f"Batched TPS:     {bat_tokens / bat_time:.1f}")
    
    speedup = seq_time / bat_time
    print(f"\nOverall Speedup: {speedup:.2f}x")
    
    # Verify outputs match
    matches = sum(1 for s, b in zip(seq_outs, bat_outs) if s == b)
    print(f"Output parity: {matches}/{num_chunks} matches")
    
    # Save results
    results = {
        "experiment": "P13: Batched Multi-Segment Decoding",
        "model": model_name,
        "num_chunks": num_chunks,
        "sequential_time_s": seq_time,
        "sequential_tps": seq_tokens / seq_time,
        "batched_time_s": bat_time,
        "batched_tps": bat_tokens / bat_time,
        "speedup": speedup,
        "parity_matched": matches == num_chunks
    }
    
    out_path = Path("results/p13_batched_decoding.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
