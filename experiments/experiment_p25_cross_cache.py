"""P25: Static Cross-Attention Caching

Goal: Cut cross-attention memory bandwidth by freezing the cross-attention output `y`
and reusing it for N steps. Since acoustic targets move slowly, adjacent tokens 
might share the exact same acoustic context. 

We monkey-patch ResidualAttentionBlock to cache and reuse the cross-attention output.
"""

import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
MAX_TOKENS = 50
REUSE_FACTOR = 2  # Compute once, reuse for (REUSE_FACTOR - 1) steps

# We will monkey patch ResidualAttentionBlock
import mlx_whisper.whisper as whisper_module
original_call = whisper_module.ResidualAttentionBlock.__call__

def patched_call(self, x, xa=None, mask=None, kv_cache=None):
    kv, cross_kv = kv_cache if kv_cache else (None, None)
    y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
    x += y
    cross_qk = None
    
    if self.cross_attn:
        reuse = getattr(self, "reuse_cross_attn", False)
        
        # Extract actual cross_kv and cached_y if it exists
        cached_y = None
        actual_cross_kv = cross_kv
        if cross_kv is not None and isinstance(cross_kv, tuple) and len(cross_kv) == 2:
            if isinstance(cross_kv[0], tuple):
                actual_cross_kv, cached_y = cross_kv
                
        if reuse and cached_y is not None:
            # Reuse the cached output! Skip cross_attn entirely!
            y = cached_y
            cross_kv = (actual_cross_kv, cached_y)
        else:
            y, actual_cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=actual_cross_kv
            )
            cross_kv = (actual_cross_kv, y)
            
        x += y
        
    x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
    return x, (kv, cross_kv), cross_qk

def set_reuse_flag(model, reuse: bool):
    for block in model.decoder.blocks:
        block.reuse_cross_attn = reuse

def decode(model, enc, use_cache=False):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        
        if use_cache:
            # We compute normally on step % REUSE_FACTOR == 0, else we reuse
            do_reuse = (step % REUSE_FACTOR != 0)
            set_reuse_flag(model, do_reuse)
        else:
            set_reuse_flag(model, False)
            
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache
        )
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache) 
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def main():
    whisper_module.ResidualAttentionBlock.__call__ = patched_call
    
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
    std_ids, std_time = decode(whisper_model, enc, use_cache=False)
    std_tps = len(std_ids) / std_time
    print(f"Standard TPS: {std_tps:.1f} ({std_time:.3f}s for {len(std_ids)} tokens)")
    
    print("\nRunning Static Cross-Attention Caching (Reuse=2)...")
    cached_ids, cached_time = decode(whisper_model, enc, use_cache=True)
    cached_tps = len(cached_ids) / cached_time
    print(f"Cached TPS:   {cached_tps:.1f} ({cached_time:.3f}s for {len(cached_ids)} tokens)")
    
    speedup = cached_tps / std_tps
    print(f"\nSpeedup: {speedup:.2f}x")
    
    tokenizer = get_tokenizer(multilingual=False)
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot])
    cached_text = tokenizer.decode([t for t in cached_ids if t < tokenizer.eot])
    
    print(f"\nStandard Text: {std_text}")
    print(f"Cached Text:   {cached_text}")
    
    out_path = Path("results/p25_cross_cache.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P25: Static Cross-Attention Caching",
            "std_tps": std_tps,
            "cached_tps": cached_tps,
            "speedup": speedup,
            "std_text": std_text,
            "cached_text": cached_text
        }, f, indent=2)
        
    whisper_module.ResidualAttentionBlock.__call__ = original_call

if __name__ == "__main__":
    main()
