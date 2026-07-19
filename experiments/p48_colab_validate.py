#!/usr/bin/env python3
"""P48 Colab Double-Validation: stride-8 WER & parallel batch decode on PyTorch GPU.

Two tests:
1. WER equivalence: stride-8 avg-pool vs full encoder (is WER preserved?)
2. Speedup: parallel batch decode vs sequential (how much faster?)

Uses encoder forward hook to inject stride-8, then vanilla model.generate().

Usage (colab): colab run --gpu T4 p48_colab_validate.py
"""

import subprocess, sys, os, json, time

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
MAX_NEW = 224


def stride8_hook_fn(_module, _input, output):
    """Forward hook: avg-pool encoder output from T -> T//8 frames."""
    hs = output.last_hidden_state
    B, T, D = hs.shape
    Tt = (T // 8) * 8
    pooled = hs[:, :Tt, :].reshape(B, Tt // 8, 8, D).mean(dim=2)
    return type(output)(last_hidden_state=pooled)


def install_stride8(encoder):
    return encoder.register_forward_hook(stride8_hook_fn)


def build_long_audio():
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sigs, sr, texts = [], None, []
    for r in ds:
        s = r["audio"]["array"]
        rs = r["audio"]["sampling_rate"]
        if sr is None: sr = rs
        if rs != sr: continue
        sigs.append(s)
        texts.append(r["text"].strip())
    return np.concatenate(sigs, dtype=np.float32), sr, texts


def load_chunks(audio, sr, chunk_sec=CHUNK_SEC, max_chunks=0):
    chunk_n = chunk_sec * sr
    chunk_data = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_n, len(audio))
        seg = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        chunk_data.append(seg)
        start = end
        if max_chunks and len(chunk_data) >= max_chunks:
            break
    return chunk_data, sr


def decode_sequential(model, processor, chunk_data, max_new=MAX_NEW, batch_size=1):
    hyps = []
    t0 = time.perf_counter()
    for seg in chunk_data:
        inputs = processor(seg, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=CHUNK_SEC * 16000)
        feat = inputs.input_features.to(model.device, dtype=DTYPE)
        out = model.generate(
            input_features=feat,
            max_new_tokens=max_new,
            language="en",
            task="transcribe",
        )[0].tolist()
        hyps.append(out)
    t1 = time.perf_counter()
    return hyps, t1 - t0


def decode_parallel(model, processor, chunk_data, max_new=MAX_NEW):
    batch_feats = []
    for seg in chunk_data:
        inputs = processor(seg, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=CHUNK_SEC * 16000)
        batch_feats.append(inputs.input_features)
    feat = torch.cat(batch_feats, dim=0).to(model.device, dtype=DTYPE)
    # Create attention mask for batched input (all have same length, all ones)
    attn_mask = torch.ones(feat.shape[0], feat.shape[-1], dtype=torch.long, device=model.device)
    t0 = time.perf_counter()
    out = model.generate(
        input_features=feat,
        max_new_tokens=max_new,
        language="en",
        task="transcribe",
        attention_mask=attn_mask,
    )
    hyps = [seq.tolist() for seq in out]
    t1 = time.perf_counter()
    return hyps, t1 - t0


def decode_to_text(processor, token_ids_list):
    """Decode token IDs to text strings."""
    return [processor.decode(ids, skip_special_tokens=True).strip() for ids in token_ids_list]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] device={device} model={MODEL_ID}", flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model = model.to(device)
    model.eval()
    print(f"[bench] model loaded", flush=True)

    audio, sr, _ = build_long_audio()
    print(f"[bench] audio={len(audio)/sr:.1f}s", flush=True)
    all_chunks, _ = load_chunks(audio, sr, CHUNK_SEC, max_chunks=0)
    print(f"[bench] total_chunks={len(all_chunks)}", flush=True)

    # Test 1: WER equivalence (stride-8 vs no stride)
    print(f"[bench] === TEST 1: WER equivalence ===", flush=True)

    # Without stride-8
    seqs_norm, t_norm = decode_sequential(model, processor, all_chunks[:1], max_new=MAX_NEW)
    text_norm = decode_to_text(processor, seqs_norm)
    print(f"[bench] no-stride: {text_norm[0][:60]}", flush=True)

    # With stride-8
    handle = install_stride8(model.model.encoder)
    seqs_stride, t_stride = decode_sequential(model, processor, all_chunks[:1], max_new=MAX_NEW)
    text_stride = decode_to_text(processor, seqs_stride)
    handle.remove()
    print(f"[bench] stride-8:  {text_stride[0][:60]}", flush=True)

    w = jiwer_wer(text_norm[0], text_stride[0])
    exact = text_norm[0] == text_stride[0]
    print(f"[bench] WER(stride-8 vs no-stride): {w:.4f}  exact_match={exact}", flush=True)

    # If WER > 0, try with stride-4 (which P40 validated as lossless)
    if w > 0:
        print(f"[bench] stride-8 WER > 0, testing stride-4...", flush=True)
        # Custom hook for stride-4
        def stride4_hook_fn(_mod, _inp, out):
            hs = out.last_hidden_state
            B,T,D = hs.shape; Tt = (T//4)*4
            pooled = hs[:,:Tt,:].reshape(B,Tt//4,4,D).mean(dim=2)
            return type(out)(last_hidden_state=pooled)

        h4 = model.model.encoder.register_forward_hook(stride4_hook_fn)
        seqs_s4, _ = decode_sequential(model, processor, all_chunks[:1], max_new=MAX_NEW)
        text_s4 = decode_to_text(processor, seqs_s4)
        h4.remove()
        w4 = jiwer_wer(text_norm[0], text_s4[0])
        exact4 = text_norm[0] == text_s4[0]
        print(f"[bench] stride-4 WER: {w4:.4f}  exact_match={exact4}", flush=True)

    # Test 2: Parallel speedup
    print(f"[bench] === TEST 2: Parallel speedup ===", flush=True)
    # Use stride-8 hook for both sequential and parallel
    handle = install_stride8(model.model.encoder)

    for chunk_count in [2, 4, 8]:
        max_c = min(chunk_count, len(all_chunks))
        ck = all_chunks[:max_c]
        K = len(ck)
        print(f"[bench] K={K} chunks", flush=True)

        # Warmup
        _ = decode_sequential(model, processor, ck[:1], max_new=16)
        torch.cuda.empty_cache()
        _ = decode_parallel(model, processor, ck[:1], max_new=16)
        torch.cuda.empty_cache()

        seqs_s, t_seq = decode_sequential(model, processor, ck, max_new=MAX_NEW)
        torch.cuda.empty_cache()
        seqs_p, t_par = decode_parallel(model, processor, ck, max_new=MAX_NEW)
        torch.cuda.empty_cache()

        speedup = t_seq / t_par if t_par > 0 else 0

        # Check if text matches between sequential and parallel
        text_s = decode_to_text(processor, seqs_s)
        text_p = decode_to_text(processor, seqs_p)
        match_count = sum(1 for i in range(K) if text_s[i] == text_p[i])

        result = {
            "model": MODEL_ID,
            "chunks": K,
            "device": device,
            "t_seq": round(t_seq, 3),
            "t_par": round(t_par, 3),
            "speedup": round(speedup, 3),
            "text_match": f"{match_count}/{K}",
        }
        print(json.dumps(result), flush=True)

    handle.remove()
    print(f"[bench] done", flush=True)


if __name__ == "__main__":
    main()
