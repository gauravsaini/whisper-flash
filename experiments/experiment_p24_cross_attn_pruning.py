"""P24: Dynamic Monotonic Cross-Attention Pruning

Goal: The cross-attention mechanism reads the entire 1500-frame encoder cache
at every decoding step. This consumes massive memory bandwidth: O(1500 * D) per layer.
Since speech is monotonic, once Whisper has transcribed a word, it never needs to
look at those audio frames again.

By dynamically pruning the `cross_kv` cache based on the trailing cross-attention
argmax, we can shrink the 1500 frames down to a rolling window of ~100 frames,
theoretically yielding massive speedups on memory-bound Apple Silicon.
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
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
MAX_TOKENS = 50
WINDOW_SIZE = 200  # Keep 200 frames around the peak

def decode_standard(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    
    t0 = time.perf_counter()
    for _ in range(MAX_TOKENS):
        logits, kv_cache, _, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache, return_cross_attention=True
        )
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache) 
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def decode_pruned(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    
    # enc is (1, 1500, D). We don't prune `enc` directly because cross_kv is already built
    # inside kv_cache. We will prune the cross_kv inside kv_cache.
    
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        logits, kv_cache, _, cross_qk = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache, return_cross_attention=True
        )
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
        # cross_qk is a list of layers, each (B, heads, 1, current_frames)
        # Find the peak attention across all layers and heads
        # We can just use the last layer's average head attention
        last_layer_attn = cross_qk[-1] # (1, heads, 1, current_frames)
        avg_attn = mx.mean(last_layer_attn, axis=1)[0, 0] # (current_frames,)
        peak_idx = mx.argmax(avg_attn).item()
        
        # We want to keep [peak_idx - 50 : peak_idx + 150]
        # But wait! If we slice `cross_kv`, the length shrinks. Next step, `cross_qk` will have a smaller length.
        # This means `peak_idx` will be relative to the ALREADY PRUNED window!
        # If we just keep slicing, we might drop future frames before we reach them.
        # It's better to just mask `cross_kv`? No, masking doesn't save memory bandwidth!
        # Slicing saves memory bandwidth, but we must slice `enc` instead of `cross_kv`!
        # Actually, if we slice `cross_kv`, it's totally fine, but we must be careful not to drop FUTURE frames.
        # Since speech goes forward, we can just drop past frames!
        # Keep [peak_idx - 50 : END]. This shrinks the cache continuously from the left.
        
        drop_left = max(10, peak_idx - 50)
        
        if drop_left > 10:
            new_cache = []
            for self_kv, cross_kv in kv_cache:
                c_k, c_v = cross_kv
                # Keep sink [0:10] and window [drop_left:]
                c_k_sink = c_k[:, :10, :]
                c_v_sink = c_v[:, :10, :]
                c_k_window = c_k[:, drop_left:, :]
                c_v_window = c_v[:, drop_left:, :]
                
                c_k = mx.concatenate([c_k_sink, c_k_window], axis=1)
                c_v = mx.concatenate([c_v_sink, c_v_window], axis=1)
                new_cache.append((self_kv, (c_k, c_v)))
            kv_cache = new_cache
            
            # We MUST ALSO slice `enc` so that if any layer recalculates cross_kv from `enc`, it uses the sliced one.
            # Wait, `decoder_forward` does NOT recalculate cross_kv if it already exists in kv_cache!
            # So slicing `cross_kv` in place is perfect.
            
    mx.eval(kv_cache)
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel = mx.array(mel)[None]
    
    enc = encoder_forward(whisper_model, mel)
    
    print("\nRunning Standard Decoding...")
    std_ids, std_time = decode_standard(whisper_model, enc)
    std_tps = len(std_ids) / std_time
    print(f"Standard TPS: {std_tps:.1f} ({std_time:.3f}s for {len(std_ids)} tokens)")
    
    print("\nRunning Cross-Attention Pruned Decoding...")
    pruned_ids, pruned_time = decode_pruned(whisper_model, enc)
    pruned_tps = len(pruned_ids) / pruned_time
    print(f"Pruned TPS:   {pruned_tps:.1f} ({pruned_time:.3f}s for {len(pruned_ids)} tokens)")
    
    speedup = pruned_tps / std_tps
    print(f"\nSpeedup: {speedup:.2f}x")
    
    tokenizer = get_tokenizer(multilingual=False)
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot])
    pruned_text = tokenizer.decode([t for t in pruned_ids if t < tokenizer.eot])
    
    print(f"\nStandard Text: {std_text}")
    print(f"Pruned Text:   {pruned_text}")
    
    out_path = Path("results/p24_cross_attn_pruning.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P24: Cross-Attn Pruning",
            "std_tps": std_tps,
            "pruned_tps": pruned_tps,
            "speedup": speedup,
            "std_text": std_text,
            "pruned_text": pruned_text
        }, f, indent=2)

if __name__ == "__main__":
    main()
