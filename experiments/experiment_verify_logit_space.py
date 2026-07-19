#!/usr/bin/env python3
"""
Test whether logit-space top-K agreement predicts draft correctness
better than hidden-state cosine similarity.

We simulate a draft model by taking target hidden states and adding
controlled noise (approximating a draft model's approximation error).
This generates a mix of "correct" and "incorrect" draft predictions.

For each position, we record:
  1. cos_sim:   cosine similarity between draft_h and target_h (node_sim)
  2. logit_ov:  top-K overlap between draft_logits and target_logits
  3. token_match: does draft argmax == target argmax? (label)

Then: which signal best separates correct from incorrect?
"""

import argparse
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_whisper.audio import log_mel_spectrogram

from whisper_flash_mlx.target_model import (
    decoder_forward_with_hidden_states,
    encoder_forward,
    load_target_model,
    project_to_logits,
)
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="/tmp/jfk_16k.wav")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--noise-scale", type=float, default=0.1,
                        help="Noise added to simulated draft hidden states")
    args = parser.parse_args()

    target = load_target_model(args.model)
    lm_head = target.decoder.token_embedding.as_linear
    print(f"Model:       {args.model}")
    print(f"Top-K:       {args.top_k}")
    print(f"Noise scale: {args.noise_scale}")

    # Audio
    arr, sr = sf.read(args.audio)
    if arr.ndim == 2: arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    mel = log_mel_spectrogram(arr, n_mels=target.dims.n_mels, padding=16000*30-len(arr))
    mel = mx.array(mel)[None]
    enc = encoder_forward(target, mel)
    mx.eval(enc)

    # Greedy decode
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    kv_cache = None
    for _ in range(args.max_new_tokens):
        inp = dec[:, -1:] if kv_cache is not None else dec
        l, kv_cache, _ = decoder_forward_with_hidden_states(target, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        tok = sample(l[:, -1:, :], 0.0); mx.eval(tok)
        dec = mx.concatenate([dec, tok], axis=1)
        if tok.item() == EOS_ID: break
    gt = dec[0].tolist()
    n_gt = len(gt)
    print(f"  Ground truth: {n_gt - 2} tokens")
    if n_gt < 3: print("Sequence too short."); return

    # Set random seed for reproducibility
    mx.random.seed(42)

    records = []
    for p in range(1, n_gt - 2):
        prefix = gt[:p+1]

        # Run target on prefix → collect hidden states + logits at last position
        inp = mx.array([prefix], dtype=mx.int32)
        _, _, all_h = decoder_forward_with_hidden_states(
            target, inp, enc, kv_cache=None, collect_hidden_states=True)

        # Target hidden state at last position (before lm_head)
        target_h = all_h[-1][0, -1]  # (d_target,)
        # Target logits at last position = lm_head(target_h)
        target_logits = lm_head(target_h[None, None, :])  # (1, 1, vocab)
        target_greedy = mx.argmax(target_logits, axis=-1).item()

        # Simulate draft hidden state: add noise to target hidden
        noise = mx.random.normal(target_h.shape) * args.noise_scale * mx.linalg.norm(target_h)
        draft_h = target_h + noise  # (d_target,)
        # Draft logits = lm_head(draft_h)
        draft_logits = lm_head(draft_h[None, None, :])  # (1, 1, vocab)
        draft_greedy = mx.argmax(draft_logits, axis=-1).item()

        is_correct = (draft_greedy == target_greedy)

        # 1. Cosine similarity (node_sim)
        cos_sim = mx.sum(draft_h * target_h) / (mx.linalg.norm(draft_h) * mx.linalg.norm(target_h) + 1e-9)

        # 2. Logit-space top-K agreement
        d_topk = set(mx.argsort(-draft_logits[0], axis=-1)[:, :args.top_k][0].tolist())
        t_topk = set(mx.argsort(-target_logits[0], axis=-1)[:, :args.top_k][0].tolist())
        logit_ov = len(d_topk & t_topk) / args.top_k

        # 3. KL divergence: draft → target
        d_probs = mx.softmax(draft_logits, axis=-1)
        t_probs = mx.softmax(target_logits, axis=-1)
        kl = mx.sum(t_probs * (mx.log(t_probs + 1e-9) - mx.log(d_probs + 1e-9)))

        records.append({
            "pos": p,
            "draft_greedy": draft_greedy,
            "target_greedy": target_greedy,
            "is_correct": is_correct,
            "cos_sim": cos_sim.item(),
            "logit_overlap": logit_ov,
            "kl": kl.item(),
        })

    print(f"\nCollected {len(records)} records\n")

    # Analysis
    correct = np.array([r["is_correct"] for r in records], dtype=float)
    cos_sims = np.array([r["cos_sim"] for r in records])
    logit_ov = np.array([r["logit_overlap"] for r in records])
    kls = np.array([r["kl"] for r in records])

    n_c = int(correct.sum())
    n_t = len(records)
    print(f"  Correct: {n_c}/{n_t} ({n_c/n_t*100:.1f}%)")

    sigs = [("cosine_sim (node_sim)", cos_sims, True),
            ("logit_topk_overlap", logit_ov, True),
            ("kl_draft_vs_target", kls, False)]

    print(f"\n{'─'*70}")
    print(f"  {'Signal':<28} {'Correct μ':<12} {'Wrong μ':<12} {'Sep':<8}")
    print(f"{'─'*70}")
    for name, vals, _ in sigs:
        mu_c = vals[correct == 1].mean() if n_c > 0 else 0
        mu_w = vals[correct == 0].mean() if n_c < n_t else 0
        print(f"  {name:<28} {mu_c:<12.4f} {mu_w:<12.4f} {abs(mu_c-mu_w):<8.4f}")

    print(f"\n{'─'*70}")
    print(f"  Best binary-predictor F1")
    print(f"{'─'*70}")
    for name, vals, higher_better in sigs:
        if higher_better:
            th = np.linspace(vals.min(), vals.max(), 50)
            preds = vals[:, None] > th[None, :]
        else:
            th = np.linspace(vals.min(), vals.max(), 50)
            preds = vals[:, None] < th[None, :]

        bf1 = bt = bp = br = 0
        for ti, t in enumerate(th):
            tp = ((correct == 1) & preds[:, ti]).sum()
            fp = ((correct == 0) & preds[:, ti]).sum()
            fn = ((correct == 1) & ~preds[:, ti]).sum()
            p = tp / max(tp+fp,1); r = tp / max(tp+fn,1)
            f1 = 2*p*r / max(p+r, 1e-9)
            if f1 > bf1: bf1, bt, bp, br = f1, t, p, r
        print(f"  {name:<28} F1={bf1:.3f} @ t={bt:.4f} (P={bp:.3f} R={br:.3f})")

    print(f"\n  Token identity (baseline): F1=1.000 @ t=0.50 (P=1.000 R=1.000)")
    print(f"\n  Noise scale = {args.noise_scale}")


if __name__ == "__main__":
    main()
