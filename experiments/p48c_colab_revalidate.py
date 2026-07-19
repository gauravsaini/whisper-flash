#!/usr/bin/env python3
"""P48c Re-validation: stride-8 works on HuggingFace WITH temperature fallback.

Previous P48b failed because model.generate() uses greedy decode internally.
Stride-8 (188 frames) causes low-confidence repetition loops that only
temperature fallback can escape. This script tests stride-8 with
temperature fallback to prove the framework isn't the issue.

Usage (colab): colab run --gpu T4 p48c_colab_revalidate.py
"""

import subprocess, sys, os, json, time, math

_deps = ["torch", "transformers", "soundfile", "librosa", "accelerate", "datasets", "jiwer"]
for _pkg in _deps:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], capture_output=True)

import torch
import numpy as np
from datasets import load_dataset
from jiwer import wer as jiwer_wer

torch.set_grad_enabled(False)

MODEL_ID = "openai/whisper-large-v3-turbo"
DTYPE = torch.float16
CHUNK_SEC = 30


def stride8_hook_fn(_module, _input, output):
    hs = output.last_hidden_state
    B, T, D = hs.shape
    Tt = (T // 8) * 8
    pooled = hs[:, :Tt, :].reshape(B, Tt // 8, 8, D).mean(dim=2)
    return type(output)(last_hidden_state=pooled)


def install_stride8(encoder):
    return encoder.register_forward_hook(stride8_hook_fn)


def decode_with_fallback(model, processor, audio, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Manual temperature fallback matching mlx_whisper.transcribe() logic."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt",
                       padding="max_length", max_length=CHUNK_SEC * 16000)
    feat = inputs.input_features.to(model.device, dtype=DTYPE)

    gen_kwargs = dict(
        max_new_tokens=224,
        language="en",
        task="transcribe",
        return_dict_in_generate=True,
        output_scores=True,
    )

    for temp in temperatures:
        if temp == 0.0:
            out = model.generate(input_features=feat, do_sample=False, **gen_kwargs)
        else:
            out = model.generate(input_features=feat, do_sample=True, temperature=temp, **gen_kwargs)

        seq = out.sequences[0].tolist()
        scores = torch.stack(out.scores).cpu()

        # Check for repetition: compute avg logprob and repetition pattern
        logprobs = []
        for i, s in enumerate(scores):
            lp = torch.log_softmax(s[0], dim=-1)[seq[len(seq) - len(scores) + i]]
            logprobs.append(lp.item())

        avg_lp = sum(logprobs) / len(logprobs) if logprobs else -100
        min_lp = min(logprobs) if logprobs else -100

        # Check for repeating bigrams (mlx_whisper-style detection)
        rep_detected = False
        for i in range(1, len(seq)):
            for j in range(i + 2, len(seq)):
                if j + 1 < len(seq) and seq[i] == seq[j] and seq[i - 1] == seq[j - 1]:
                    rep_detected = True
                    break
            if rep_detected:
                break

        # Fallback condition: low avg logprob or repetition detected
        if avg_lp > -0.5 and not rep_detected:
            return processor.decode(seq, skip_special_tokens=True).strip()

    # Last resort: use the last temperature
    return processor.decode(seq, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] device={device} model={MODEL_ID}", flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model = model.to(device)
    model.eval()
    print(f"[bench] model loaded", flush=True)

    # Load 10 clean samples
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    samples = []
    for r in ds:
        arr = r["audio"]["array"].astype(np.float32)
        if len(arr) < 1: continue
        samples.append((arr, r["text"].strip()))
    print(f"[bench] loaded {len(samples)} samples", flush=True)

    # Test each sample with:
    # 1. Greedy (no stride)  - baseline
    # 2. Greedy (stride-8)   - should FAIL (repetition)
    # 3. Fallback (stride-8)  - should MATCH baseline

    wers_greedy, wers_fallback = [], []
    matches_greedy, matches_fallback = 0, 0
    total_time_greedy, total_time_fallback = 0.0, 0.0

    for idx, (audio, ref_text) in enumerate(samples):
        print(f"[bench] sample {idx+1}/{len(samples)}", flush=True)

        # 1. Baseline (greedy, no stride)
        t0 = time.perf_counter()
        text_norm = decode_with_fallback(model, processor, audio, temperatures=(0.0,))
        dt_norm = time.perf_counter() - t0

        # 2. Stride-8 greedy (should fail)
        handle = install_stride8(model.model.encoder)
        t0 = time.perf_counter()
        text_stride_greedy = decode_with_fallback(model, processor, audio, temperatures=(0.0,))
        dt_greedy = time.perf_counter() - t0

        # 3. Stride-8 with temperature fallback (should work)
        t0 = time.perf_counter()
        text_stride_fb = decode_with_fallback(model, processor, audio)
        dt_fb = time.perf_counter() - t0
        handle.remove()

        w_greedy = jiwer_wer(text_norm, text_stride_greedy)
        w_fb = jiwer_wer(text_norm, text_stride_fb)
        m_greedy = text_norm == text_stride_greedy
        m_fb = text_norm == text_stride_fb

        wers_greedy.append(w_greedy)
        wers_fallback.append(w_fb)
        if m_greedy: matches_greedy += 1
        if m_fb: matches_fallback += 1
        total_time_greedy += dt_greedy
        total_time_fallback += dt_fb

        print(f"  baseline:   {text_norm[:50]}")
        print(f"  stride-8 (greedy):   {text_stride_greedy[:50]}  WER={w_greedy:.4f}  match={m_greedy}")
        print(f"  stride-8 (fallback): {text_stride_fb[:50]}  WER={w_fb:.4f}  match={m_fb}")

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS ({len(samples)} samples on {device}):", flush=True)
    print(f"  Stride-8 greedy:  avg WER={np.mean(wers_greedy):.4f}  matches={matches_greedy}/{len(samples)}", flush=True)
    print(f"  Stride-8 fallback: avg WER={np.mean(wers_fallback):.4f}  matches={matches_fallback}/{len(samples)}", flush=True)
    print(f"  Speedup (fallback vs baseline): {total_time_greedy/total_time_fallback:.2f}x", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
