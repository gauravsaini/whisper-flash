"""P31: Standard Speculative Decoding (Tiny drafting for Large-v3)

We use mlx-community/whisper-tiny as a draft model to generate K tokens.
Then we use mlx-community/whisper-large-v3 to verify K+1 tokens in parallel.
Since Apple Silicon memory bandwidth is the bottleneck, running the large model
on K+1 tokens takes nearly the same time as 1 token.
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
    decoder_forward_with_hidden_states
)
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
MAX_TOKENS = 50

def decode_speculative(draft_model, target_model, enc_draft, enc_target, K=4):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    
    draft_kv = None
    target_kv = None
    
    t0 = time.perf_counter()
    step = 0
    accepts = 0
    total = 0
    
    while step < MAX_TOKENS:
        # 1. Draft K tokens using Tiny
        draft_ids = []
        curr_draft_kv = draft_kv
        curr_token = output_ids[-1]
        
        for _ in range(K):
            logits, curr_draft_kv, _ = decoder_forward_with_hidden_states(
                draft_model, mx.array([[curr_token]]), enc_draft, kv_cache=curr_draft_kv
            )
            curr_token = mx.argmax(logits[:, -1:, :], axis=-1).item()
            draft_ids.append(curr_token)
            if curr_token == EOS_ID:
                break
                
        # 2. Verify with Large model
        verify_tokens = [output_ids[-1]] + draft_ids
        
        logits_target, curr_target_kv, _ = decoder_forward_with_hidden_states(
            target_model, mx.array([verify_tokens]), enc_target, kv_cache=target_kv
        )
        
        # Verify tokens
        accepted_count = 0
        for i in range(len(draft_ids)):
            target_tok = mx.argmax(logits_target[:, i:i+1, :], axis=-1).item()
            if target_tok == draft_ids[i]:
                output_ids.append(draft_ids[i])
                accepted_count += 1
                if draft_ids[i] == EOS_ID:
                    break
            else:
                output_ids.append(target_tok)
                break
                
        # If we accepted all draft tokens, get bonus token
        if accepted_count == len(draft_ids) and output_ids[-1] != EOS_ID:
            bonus_tok = mx.argmax(logits_target[:, -1:, :], axis=-1).item()
            output_ids.append(bonus_tok)
            accepted_count += 1
            
        # 3. Update Caches
        # The number of new tokens added to the true sequence is accepted_count (or accepted_count + 1 if rejected)
        # Actually, output_ids has grown by `accepted_count + 1` (since we append the rejected target_tok too)
        # Wait, if accepted all, output_ids grew by K+1.
        # So we just keep the target_kv up to the new length!
        
        added_tokens = accepted_count
        if accepted_count < len(draft_ids):
            added_tokens += 1 # The corrected token
            
        def crop_kv(kv_cache, keep_len):
            if kv_cache is None: return None
            new_cache = []
            for self_kv, cross_kv in kv_cache:
                if self_kv is not None:
                    k, v = self_kv
                    k = k[:, :keep_len, :]
                    v = v[:, :keep_len, :]
                    self_kv = (k, v)
                new_cache.append((self_kv, cross_kv))
            return new_cache
            
        current_seq_len = (target_kv[0][0][0].shape[1] if target_kv else 0) + added_tokens
        target_kv = crop_kv(curr_target_kv, current_seq_len)
        
        # For the draft model, if we rejected, its KV cache is invalid for the new corrected token!
        # Instead of complex re-routing, let's just wipe the draft_kv and re-prefill it next step!
        # MLX is fast at prefilling small models.
        if accepted_count < len(draft_ids):
            draft_kv = None
            # Need to prefill draft_kv up to current output_ids (excluding the last one which will be passed in loop)
            # Actually, to make it perfectly correct, we can just prefill it right now:
            _, draft_kv, _ = decoder_forward_with_hidden_states(
                draft_model, mx.array([output_ids[:-1]]), enc_draft, kv_cache=None
            )
        else:
            draft_kv = crop_kv(curr_draft_kv, current_seq_len)
            
        accepts += accepted_count
        total += len(draft_ids)
        step += added_tokens
        
        if output_ids[-1] == EOS_ID:
            break
            
    mx.eval(draft_kv, target_kv) 
    t_total = time.perf_counter() - t0
    
    return output_ids, t_total, accepts, total

def decode_standard(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    kv_cache = None
    t0 = time.perf_counter()
    for step in range(MAX_TOKENS):
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache
        )
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
    mx.eval(kv_cache)
    return output_ids, time.perf_counter() - t0

def main():
    print("Loading target model (Large-V3, Q4)...")
    target_model = load_target_model("mlx-community/whisper-large-v3-mlx", dtype=mx.float16)
    nn.quantize(target_model, group_size=64, bits=4)
    
    print("Loading draft model (Turbo, Q4)...")
    draft_model = load_target_model("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    nn.quantize(draft_model, group_size=64, bits=4)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=128) # Large v3 uses 128 mels
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel_large = mx.array(mel)[None]
    
    mel_tiny = log_mel_spectrogram(audio, n_mels=80)
    mel_tiny = np.pad(mel_tiny, [(0, max(0, 3000 - mel_tiny.shape[0])), (0, 0)])[:3000, :]
    mel_tiny = mx.array(mel_tiny)[None]
    
    enc_target = encoder_forward(target_model, mel_large)
    enc_draft = encoder_forward(draft_model, mel_large)
    
    print("\nWarming up caches...")
    decode_standard(target_model, enc_target)
    
    print("\nRunning Standard Decoding (Large-V3)...")
    std_ids, std_time = decode_standard(target_model, enc_target)
    std_tps = len(std_ids) / std_time
    
    print("\nRunning Speculative Decoding (K=4)...")
    spec_ids, spec_time, accepts, total = decode_speculative(draft_model, target_model, enc_draft, enc_target, K=4)
    spec_tps = len(spec_ids) / spec_time
    
    tokenizer = get_tokenizer(multilingual=False)
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot])
    spec_text = tokenizer.decode([t for t in spec_ids if t < tokenizer.eot])
    
    print(f"\n--- Performance ---")
    print(f"Standard TPS: {std_tps:.1f}")
    print(f"Speculative TPS: {spec_tps:.1f} (Speedup: {spec_tps/std_tps:.2f}x)")
    print(f"Acceptance Rate: {accepts}/{total} ({accepts/total*100:.1f}%)")
    
    print(f"\n--- Output ---")
    print(f"Standard Text: {std_text}")
    print(f"Speculative Text: {spec_text}")
    
    out_path = Path("results/p31_speculative.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P31: Speculative Decoding",
            "std_tps": std_tps,
            "spec_tps": spec_tps,
            "acceptance": accepts / total,
            "std_text": std_text,
            "spec_text": spec_text
        }, f, indent=2)

if __name__ == "__main__":
    main()
