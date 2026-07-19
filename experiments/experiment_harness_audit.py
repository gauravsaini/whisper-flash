#!/usr/bin/env python3
"""
Experiment #1: Canonical Harness Audit.

The WER across all "speculative loop" experiments (ID 57, 59, cross-attn ablation)
is consistently 6.7427 even for 0% draft acceptance (pure target fallback).
Expected whisper-tiny WER on LibriSpeech clean ~5-6% (0.05-0.06), but we get
674% (6.74). This means the harness is broken.

This script:
1. Runs standard greedy decoding on whisper-tiny
2. Runs 4 different "speculative loop" harnesses with 0% drafts
3. Compares WER across all
4. Root-causes the discrepancy

The 4 harnesses to compare:
  A) Standard greedy: no draft model, pure autoregressive
  B) Experiment_semantic_graph style: mask tokens, KV cache, SOT sequence
  C) experiment_id57 style: no KV cache, full forward pass each step
  D) Our experiment #2 style: simplified generate_speculative
"""

import time, numpy as np
import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel
import jiwer

def normalize_text(text):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(text))))

# ─── Harness A: Standard Greedy Decoding ───
def harness_greedy(target, tokenizer, mel, max_tokens=100):
    enc = encoder_forward(target, mel)
    tokens = [tokenizer.sot]
    for _ in range(max_tokens):
        inp = mx.array([tokens], dtype=mx.int32)
        logits, _, _ = decoder_forward_with_hidden_states(
            target, inp, enc, collect_hidden_states=False, return_cross_attention=False)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        tokens.append(next_tok)
        if next_tok == tokenizer.eot:
            break
    return tokenizer.decode(tokens)

# ─── Harness B: experiment_semantic_graph style (KV cache, mask tokens, SOT seq) ───
def harness_semantic_graph(target, tokenizer, mel, max_tokens=100, block_size=4):
    enc = encoder_forward(target, mel)
    mask_id = 50257
    
    output = [mask_id] * (max_tokens + block_size)
    output[0] = 50258  # SOT
    
    # Prefill
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        target, mx.array([[50258]], dtype=mx.int32), enc,
        kv_cache=None, collect_hidden_states=False, return_cross_attention=False)
    first = mx.argmax(logits[:, -1:, :], axis=-1).item()
    output[1] = first
    start = 1
    
    while start < max_tokens:
        block = mx.array([output[start:start+block_size]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            target, block, enc, kv_cache=kv_cache,
            collect_hidden_states=False, return_cross_attention=False)
        posterior = mx.argmax(logits, axis=-1)[0].tolist()
        
        # Accept 1 token (fallback)
        output[start] = posterior[0]
        start += 1
        
        if 50257 in output[:start] or tokenizer.eot in output[:start]:
            break
    
    return tokenizer.decode(output[:start])

# ─── Harness C: experiment_id57 style (no KV cache, full forward each step) ───
def harness_id57_style(target, tokenizer, mel, max_tokens=100):
    enc = encoder_forward(target, mel)
    tokens = [tokenizer.sot]
    
    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        _, _, h_all = decoder_forward_with_hidden_states(
            target, inp, enc,
            collect_hidden_states=True, return_cross_attention=False)
        target_hidden = mx.concatenate([h_all[1][:, -1:, :], h_all[2][:, -1:, :]], axis=-1)
        audio_summary = mx.mean(enc, axis=1, keepdims=True)
        
        # Use a dummy model to check: does the probe affect output?
        noise = target.decoder.token_embedding(mx.array([[50257] * 4]))
        pos = mx.arange(len(tokens), len(tokens) + 4, dtype=mx.int32)[None]
        
        # Pure target fallback: decode normally
        _, _, h_future = decoder_forward_with_hidden_states(
            target, inp, enc,
            collect_hidden_states=True, return_cross_attention=False)
        next_logits = target.decoder.token_embedding.as_linear(h_all[-1])
        next_tok = mx.argmax(next_logits[:, -1, :], axis=-1).item()
        tokens.append(next_tok)
        
        if next_tok == tokenizer.eot:
            break
    
    return tokenizer.decode(tokens)

# ─── Harness D: Our simplified generate_speculative (the broken one) ───
def harness_simple_spec(target, tokenizer, mel, max_tokens=100, block_size=4):
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    
    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        _, _, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc,
            collect_hidden_states=True, return_cross_attention=False)
        
        target_hidden = mx.concatenate([hidden_all[1][:, -1:, :], hidden_all[2][:, -1:, :]], axis=-1)
        noise = target.decoder.token_embedding(mx.array([[50257] * block_size]))
        pos = mx.arange(len(tokens), len(tokens) + block_size, dtype=mx.int32)[None]
        
        draft_hidden = target.decoder.token_embedding(
            mx.array([[50257]])).repeat(block_size, axis=1)
        draft_logits = target.decoder.token_embedding.as_linear(draft_hidden)
        draft_tokens = mx.argmax(draft_logits, axis=-1)[0].tolist()
        
        spec_tokens = tokens + draft_tokens
        spec_inp = mx.array([spec_tokens], dtype=mx.int32)
        _, _, true_all = decoder_forward_with_hidden_states(
            target, spec_inp, enc,
            collect_hidden_states=True, return_cross_attention=False)
        true_logits = target.decoder.token_embedding.as_linear(true_all[-1])
        true_next = mx.argmax(true_logits, axis=-1)[0].tolist()
        
        accepted_k = 0
        tokens.extend(draft_tokens[:accepted_k])
        tokens.append(true_next[len(tokens) - 1])
        
        if tokens[-1] == tokenizer.eot:
            break
    
    return tokenizer.decode(tokens)

def run():
    print("=" * 65)
    print("EXP #1: CANONICAL HARNESS AUDIT")
    print("=" * 65)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    print(f"\n{'='*65}")
    print("HARNESS COMPARISON ACROSS 10 HELD-OUT SAMPLES")
    print(f"{'='*65}")

    harnesses = [
        ("A: Standard Greedy", harness_greedy),
        ("B: Semantic Graph Style", harness_semantic_graph),
        ("C: ID57 Style", harness_id57_style),
        ("D: Simple Spec (broken)", harness_simple_spec),
    ]

    for name, fn in harnesses:
        print(f"\n  {name}")
        wers = []
        for i in range(10, 20):
            sample = ds[i]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels,
                                       padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)

            text = fn(target, tokenizer, mel_mx)
            text_norm = normalize_text(text)
            ref_norm = normalize_text(sample["text"])
            wer = jiwer.wer(ref_norm, text_norm) if ref_norm else 1.0
            wers.append(wer)
            print(f"    [{i}] WER={wer:.4f}  len={len(text.split())}")

        mean_wer = np.mean(wers)
        print(f"  -> {name}: mean WER={mean_wer:.4f}")

    # Debug: print token-level output for sample 10
    print(f"\n{'='*65}")
    print("DEBUG: Token-level output for sample 10")
    print(f"{'='*65}")
    sample = ds[10]
    print(f"  Reference: '{sample['text']}'")
    
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels,
                               padding=16000 * 30 - len(audio))
    mel_mx = mx.array(mel[None], dtype=mx.float32)
    
    for name, fn in harnesses:
        text = fn(target, tokenizer, mel_mx)
        print(f"  [{name}]: '{text.strip()}'")

    total = time.time()
    print(f"\nTotal: {total:.0f}s")

if __name__ == "__main__":
    run()
