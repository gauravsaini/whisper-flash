"""P26: Layer-Level Early Exit (Depth Pruning)

Goal: Reduce decoder compute and memory bandwidth by skipping later blocks for tokens
that are highly confident early on. We evaluate two options:
1. Static Depth Pruning: Decode using only 1, 2, or 3 decoder layers out of 4.
2. Dynamic Early Exit: Exit early at block E if the maximum probability of the projected
   intermediate hidden state is above a confidence threshold.
"""

import time
import json
import difflib
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
)
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
MAX_TOKENS = 60

# ════════════════════════════════════════════════════════════════
# Custom Decoder forwards
# ════════════════════════════════════════════════════════════════

def custom_decoder_forward_static(
    model,
    tokens: mx.array,
    audio_features: mx.array,
    kv_cache: list = None,
    n_layers: int = 2,
    offset: int = 0,
) -> tuple:
    decoder = model.decoder
    x = (
        decoder.token_embedding(tokens)
        + decoder.positional_embedding[offset: offset + tokens.shape[-1]]
    )
    if kv_cache is None:
        kv_cache = [None] * n_layers
    for e in range(n_layers):
        block = decoder.blocks[e]
        x, kv_cache[e], _ = block(
            x, audio_features, mask=decoder._mask, kv_cache=kv_cache[e]
        )
    x_ln = decoder.ln(x)
    logits = decoder.token_embedding.as_linear(x_ln)
    return logits, kv_cache


def custom_decoder_forward_dynamic(
    model,
    tokens: mx.array,
    audio_features: mx.array,
    kv_cache: list,
    threshold: float,
    offset: int,
) -> tuple:
    decoder = model.decoder
    x = (
        decoder.token_embedding(tokens)
        + decoder.positional_embedding[offset: offset + tokens.shape[-1]]
    )
    
    exit_layer = len(decoder.blocks) - 1
    
    for e in range(len(decoder.blocks)):
        block = decoder.blocks[e]
        x, kv_cache[e], _ = block(
            x, audio_features, mask=decoder._mask, kv_cache=kv_cache[e]
        )
        
        # Check if we should exit early (only if we are not at the last block)
        if e < len(decoder.blocks) - 1:
            # We can project the intermediate x to logits to check confidence
            x_ln = decoder.ln(x)
            logits = decoder.token_embedding.as_linear(x_ln)
            probs = mx.softmax(logits[:, -1, :], axis=-1)
            max_prob = mx.max(probs).item()
            
            if max_prob >= threshold:
                exit_layer = e
                # Pad the KV cache for the remaining layers to keep sequence length aligned
                for re in range(e + 1, len(decoder.blocks)):
                    self_kv, cross_kv = kv_cache[re] if kv_cache[re] else (None, None)
                    if self_kv is not None:
                        k, v = self_kv
                        k_new = mx.concatenate([k, k[:, -1:, :]], axis=1)
                        v_new = mx.concatenate([v, v[:, -1:, :]], axis=1)
                        kv_cache[re] = ((k_new, v_new), cross_kv)
                break
    else:
        # Full evaluation
        x_ln = decoder.ln(x)
        logits = decoder.token_embedding.as_linear(x_ln)
        
    return logits, kv_cache, exit_layer


# ════════════════════════════════════════════════════════════════
# Decoding Loops
# ════════════════════════════════════════════════════════════════

def decode_standard(model, enc):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    decoder = model.decoder
    kv_cache = [None] * len(decoder.blocks)
    
    # SOT prefill
    prefill = mx.array([output_ids])
    x = decoder.token_embedding(prefill) + decoder.positional_embedding[:len(output_ids)]
    for e, block in enumerate(decoder.blocks):
        x, kv_cache[e], _ = block(x, enc, mask=decoder._mask, kv_cache=kv_cache[e])
    
    # Generate tokens
    t0 = time.perf_counter()
    steps = 0
    for _ in range(MAX_TOKENS):
        pos = len(output_ids)
        inp = mx.array([[output_ids[-1]]])
        x_tok = decoder.token_embedding(inp) + decoder.positional_embedding[pos:pos+1]
        for e, block in enumerate(decoder.blocks):
            x_tok, kv_cache[e], _ = block(x_tok, enc, mask=decoder._mask, kv_cache=kv_cache[e])
        logits = decoder.token_embedding.as_linear(decoder.ln(x_tok))
        tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        output_ids.append(tok)
        steps += 1
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache)
    t_total = time.perf_counter() - t0
    return output_ids, t_total, steps


def decode_static_pruning(model, enc, n_layers):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    decoder = model.decoder
    kv_cache = [None] * n_layers
    
    # SOT prefill
    prefill = mx.array([output_ids])
    x = decoder.token_embedding(prefill) + decoder.positional_embedding[:len(output_ids)]
    for e in range(n_layers):
        block = decoder.blocks[e]
        x, kv_cache[e], _ = block(x, enc, mask=decoder._mask, kv_cache=kv_cache[e])
        
    t0 = time.perf_counter()
    steps = 0
    for _ in range(MAX_TOKENS):
        pos = len(output_ids)
        inp = mx.array([[output_ids[-1]]])
        logits, kv_cache = custom_decoder_forward_static(
            model, inp, enc, kv_cache=kv_cache, n_layers=n_layers, offset=pos
        )
        tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        output_ids.append(tok)
        steps += 1
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache)
    t_total = time.perf_counter() - t0
    return output_ids, t_total, steps


def decode_dynamic_early_exit(model, enc, threshold):
    output_ids = [SOT_ID, 50259, 50359, 50363]
    decoder = model.decoder
    kv_cache = [None] * len(decoder.blocks)
    
    # SOT prefill
    prefill = mx.array([output_ids])
    x = decoder.token_embedding(prefill) + decoder.positional_embedding[:len(output_ids)]
    for e, block in enumerate(decoder.blocks):
        x, kv_cache[e], _ = block(x, enc, mask=decoder._mask, kv_cache=kv_cache[e])
        
    t0 = time.perf_counter()
    steps = 0
    total_exits = []
    
    for _ in range(MAX_TOKENS):
        pos = len(output_ids)
        inp = mx.array([[output_ids[-1]]])
        logits, kv_cache, exit_layer = custom_decoder_forward_dynamic(
            model, inp, enc, kv_cache=kv_cache, threshold=threshold, offset=pos
        )
        tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        output_ids.append(tok)
        total_exits.append(exit_layer)
        steps += 1
        if tok == EOS_ID:
            break
            
    mx.eval(kv_cache)
    t_total = time.perf_counter() - t0
    avg_exit_layer = sum(total_exits) / len(total_exits) if total_exits else 0
    return output_ids, t_total, steps, avg_exit_layer


# ════════════════════════════════════════════════════════════════
# Main Evaluation
# ════════════════════════════════════════════════════════════════

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading target model {model_name}...")
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    
    from datasets import load_dataset
    print("Loading local cached validation sample...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(audio, n_mels=80)
    mel = np.pad(mel, [(0, max(0, 3000 - mel.shape[0])), (0, 0)])[:3000, :]
    mel = mx.array(mel)[None]
    
    print("Running encoder forward...")
    enc = encoder_forward(whisper_model, mel)
    mx.eval(enc)
    
    tokenizer = get_tokenizer(multilingual=False)
    
    # 1. Standard baseline
    print("\n[Baseline] Running standard decoding...")
    std_ids, std_time, std_steps = decode_standard(whisper_model, enc)
    std_tps = std_steps / std_time
    std_text = tokenizer.decode([t for t in std_ids if t < tokenizer.eot]).strip()
    print(f"Standard: {std_tps:.1f} TPS | Text: '{std_text}'")
    
    results = {
        "standard": {
            "tps": std_tps,
            "text": std_text,
            "similarity": 1.0,
            "time_s": std_time
        },
        "static_pruning": {},
        "dynamic_exit": {}
    }
    
    # 2. Static Pruning sweep
    for layers in [3, 2, 1]:
        print(f"\n[Static Pruning] Running with {layers} layers...")
        ids, t_time, steps = decode_static_pruning(whisper_model, enc, layers)
        tps = steps / t_time
        text = tokenizer.decode([t for t in ids if t < tokenizer.eot]).strip()
        sim = difflib.SequenceMatcher(None, std_text, text).ratio()
        print(f"Static L={layers}: {tps:.1f} TPS (Speedup: {tps/std_tps:.2f}x) | Sim: {sim*100:.1f}% | Text: '{text}'")
        results["static_pruning"][f"L_{layers}"] = {
            "tps": tps,
            "speedup": tps / std_tps,
            "text": text,
            "similarity": sim,
            "time_s": t_time
        }
        
    # 3. Dynamic Early Exit sweep
    for thresh in [0.7, 0.8, 0.9, 0.95]:
        print(f"\n[Dynamic Exit] Running with threshold={thresh}...")
        ids, t_time, steps, avg_exit = decode_dynamic_early_exit(whisper_model, enc, thresh)
        tps = steps / t_time
        text = tokenizer.decode([t for t in ids if t < tokenizer.eot]).strip()
        sim = difflib.SequenceMatcher(None, std_text, text).ratio()
        print(f"Dynamic Th={thresh}: {tps:.1f} TPS (Speedup: {tps/std_tps:.2f}x) | Avg Exit Block: {avg_exit:.2f} | Sim: {sim*100:.1f}% | Text: '{text}'")
        results["dynamic_exit"][f"Th_{thresh}"] = {
            "tps": tps,
            "speedup": tps / std_tps,
            "avg_exit_block": avg_exit,
            "text": text,
            "similarity": sim,
            "time_s": t_time
        }
        
    # Save results
    out_path = Path("results/p26_early_exit.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
