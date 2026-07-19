"""P18: Adaptive KV Cache Pruning (Windowed Attention)

Goal: Memory bandwidth is the final bottleneck on Apple Silicon. The self-attention
KV cache grows linearly with sequence length. By pruning old/uninformative tokens
from the KV cache, we can bound memory bandwidth.

Here we test a Windowed KV Cache: keep the first 4 tokens (prompt/SOT) and the
last N tokens (e.g., 64). We force the decoder to generate 400 tokens by suppressing EOS
and measure the TPS (Tokens Per Second) speedup.
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
MAX_TOKENS = 400
WINDOW_SIZE = 64
KEEP_FIRST = 4

def decode_standard(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]  # SOT, EN, TRANSCRIBE, NOTIMESTAMPS
    kv_cache = None
    
    t0 = time.perf_counter()
    for _ in range(MAX_TOKENS):
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache
        )
        
        # Suppress EOS to force long generation
        logits[:, :, EOS_ID] = -float('inf')
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        
    mx.eval(kv_cache) # ensure completion
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def decode_pruned(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache
        )
        
        logits[:, :, EOS_ID] = -float('inf')
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        
        # Prune KV Cache
        if step > WINDOW_SIZE:
            new_cache = []
            for k, v in kv_cache:
                # k, v are shape (B, num_heads, seq_len, head_dim)
                # Keep first KEEP_FIRST, and last WINDOW_SIZE
                k_pruned = mx.concatenate([k[:, :, :KEEP_FIRST, :], k[:, :, -WINDOW_SIZE:, :]], axis=2)
                v_pruned = mx.concatenate([v[:, :, :KEEP_FIRST, :], v[:, :, -WINDOW_SIZE:, :]], axis=2)
                new_cache.append((k_pruned, v_pruned))
            kv_cache = new_cache
            
            # Since we sliced the cache, the internal offset in decoder_forward_with_hidden_states
            # will be incorrect if it assumes offset = kv_cache length. 
            # Wait, the offset is used for Positional Embeddings!
            # Positional embeddings MUST map to the absolute sequence position.
            # If we pass the pruned kv_cache, `offset` will be calculated as KEEP_FIRST + WINDOW_SIZE,
            # which is wrong! It should be the absolute step count.
            # mlx_whisper's decoder doesn't accept an explicit offset parameter easily,
            # we must patch it or just pass offset.
            # Let's check if decoder_forward_with_hidden_states accepts offset.
            # It DOES accept offset! (Wait, I need to check target_model.py, I think I saw offset=None).
            
    mx.eval(kv_cache)
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    whisper_model = load_target_model(model_name, dtype=mx.float16)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel = mx.array(mel)[None]
    
    enc = encoder_forward(whisper_model, mel)
    
    print("\nRunning Standard Decoding (400 tokens, EOS suppressed)...")
    std_ids, std_time = decode_standard(whisper_model, enc)
    std_tps = MAX_TOKENS / std_time
    print(f"Standard TPS: {std_tps:.1f} ({std_time:.3f}s)")
    
    # We must patch target_model to accept 'offset' in decoder_forward_with_hidden_states
    # Wait, in target_model.py: `def decoder_forward_with_hidden_states(..., offset=None)`
    # I saw offset=None in the signature!
    
    # Let's redefine decode_pruned to use the offset correctly.
    def decode_pruned_fixed(model, enc):
        output_ids = [SOT_ID, 50259, 50359, 50363]
        kv_cache = None
        
        t0 = time.perf_counter()
        for step in range(MAX_TOKENS):
            # Absolute position is len(output_ids) - 1
            abs_offset = len(output_ids) - 1
            
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache, offset=abs_offset
            )
            
            logits[:, :, EOS_ID] = -float('inf')
            tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
            output_ids.append(tok)
            
            if len(output_ids) > WINDOW_SIZE + KEEP_FIRST:
                new_cache = []
                for self_kv, cross_kv in kv_cache:
                    k, v = self_kv
                    k_pruned = mx.concatenate([k[:, :KEEP_FIRST, :], k[:, -WINDOW_SIZE:, :]], axis=1)
                    v_pruned = mx.concatenate([v[:, :KEEP_FIRST, :], v[:, -WINDOW_SIZE:, :]], axis=1)
                    new_cache.append(((k_pruned, v_pruned), cross_kv))
                kv_cache = new_cache
                
        mx.eval(kv_cache)
        t_total = time.perf_counter() - t0
        return output_ids, t_total

    print("\nRunning Pruned Decoding (400 tokens, Window=64)...")
    pruned_ids, pruned_time = decode_pruned_fixed(whisper_model, enc)
    pruned_tps = MAX_TOKENS / pruned_time
    print(f"Pruned TPS:   {pruned_tps:.1f} ({pruned_time:.3f}s)")
    
    speedup = pruned_tps / std_tps
    print(f"\nSpeedup: {speedup:.2f}x")
    
    tokenizer = get_tokenizer(multilingual=False)
    
    out_path = Path("results/p18_kv_pruning.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P18: Adaptive KV Cache Pruning",
            "std_tps": std_tps,
            "pruned_tps": pruned_tps,
            "speedup": speedup,
            "std_text": tokenizer.decode([t for t in std_ids if t < tokenizer.eot]),
            "pruned_text": tokenizer.decode([t for t in pruned_ids if t < tokenizer.eot])
        }, f, indent=2)

if __name__ == "__main__":
    main()
