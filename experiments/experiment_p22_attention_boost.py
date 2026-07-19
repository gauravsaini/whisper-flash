"""P22: Attention-Boosted Prefix Injection

Goal: Standard text prompts provide weak context. If we want to strictly enforce
the transcription of specific domain vocabulary or names, can we do so by artificially
scaling the magnitude of their Key (K) vectors in the continuous KV cache?

This experiment tests if scaling the K vector of a prepended context word ("Hardy")
forces Whisper to transcribe it instead of the acoustically correct word ("Harvey").
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

def decode_with_prompt(model, enc, prompt_tokens, boost_alpha=1.0):
    output_ids = [SOT_ID, 50259, 50359, 50363]  # SOT, EN, TRANSCRIBE, NOTIMESTAMPS
    kv_cache = None
    
    # 1. Pre-fill KV cache with boosted prompt tokens
    if len(prompt_tokens) > 0:
        # We must process the prompt tokens to build the KV cache.
        # But we don't want to output anything yet.
        _, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([prompt_tokens]), enc, kv_cache=None
        )
        
        # Boost the Key vectors of the prompt
        if boost_alpha != 1.0:
            new_cache = []
            for self_kv, cross_kv in kv_cache:
                k, v = self_kv
                # k is (B, prompt_len, head_dim) or similar? 
                # Wait, mlx_whisper MultiHeadAttention returns 3D `(B, seq, n_state)`
                k = k * boost_alpha
                new_cache.append(((k, v), cross_kv))
            kv_cache = new_cache
            
        # The prompt is now in the cache. We start decoding from SOT.
        # Wait, if the prompt is in the cache, the offset for SOT is len(prompt_tokens).
        # We need to pass the correct offset to the decoder for the absolute positional embeddings.
        offset = len(prompt_tokens)
    else:
        offset = 0
        
    # 2. Decode the actual sequence
    for _ in range(50):
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache, offset=offset
        )
        offset += 1
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache)
    return output_ids

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    tokenizer = get_tokenizer(multilingual=False)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel = mx.array(mel)[None]
    
    enc = encoder_forward(whisper_model, mel)
    
    # The dummy audio says "Harvey". Let's try to inject " Hardy"
    target_word = " Hardy"
    prompt_tokens = tokenizer.encode(target_word)
    
    print("\nRunning Standard Prompt Injection...")
    std_ids = decode_with_prompt(whisper_model, enc, prompt_tokens, boost_alpha=1.0)
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot])
    
    print("\nRunning Boosted Prompt Injection (alpha=5.0)...")
    boosted_ids = decode_with_prompt(whisper_model, enc, prompt_tokens, boost_alpha=5.0)
    boosted_text = tokenizer.decode([t for t in boosted_ids if t < tokenizer.eot])
    
    print(f"\nTarget Word: '{target_word}'")
    print(f"Standard Prompt Output: {std_text}")
    print(f"Boosted Prompt Output:  {boosted_text}")
    
    out_path = Path("results/p22_attention_boost.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P22: Attention-Boosted Prefix Injection",
            "target_word": target_word,
            "std_text": std_text,
            "boosted_text": boosted_text
        }, f, indent=2)

if __name__ == "__main__":
    main()
