#!/usr/bin/env python3
"""
Evaluation: Δz Correction Quality (Representation Space + Matched-Baseline Speed).

Representation-space metrics:
  - CE(baseline) vs CE(corrected) for the true next token
  - True-token probability lift and rank improvement
  - Δz norm and correction quality distribution

Speed benchmark with matched baselines:
  A) HF model.generate(use_cache=True) — external reference
  B) Custom greedy (use_cache=True) — same decoder_step loop as adaptive, no Δz
  C) Adaptive v2 (use_cache=True) — Δz correction + Top-1 acceptance gate
  D) Custom greedy (use_cache=False) — KV benchmark baseline
  E) KV comparison: greedy_no_kv vs greedy_kv — pure KV cache speedup
  F) Parity: adaptive_v2 text vs custom_greedy text (sample-by-sample)

Usage:
  python3 colab_eval_correction.py --model openai/whisper-tiny --train 10 --eval 10
  colab run --gpu T4 colab_eval_correction.py --model openai/whisper-large-v3-turbo --train 50 --eval 50
"""

import subprocess, sys, json, os, argparse, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], capture_output=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ─── Constants ─────────────────────────────────────────────────────────
EPOCHS = 30
LAMBDA_CE = 0.1
MAX_TOKENS = 150
SAMPLE_RATE = 16000
N_SPEED_SAMPLES = 5
PCA_RANK_MAP = {
    "openai/whisper-tiny": 64,
    "openai/whisper-small": 64,
    "openai/whisper-base": 64,
    "openai/whisper-medium": 128,
    "openai/whisper-large": 128,
    "openai/whisper-large-v3": 128,
    "openai/whisper-large-v3-turbo": 128,
}


# ─── Helpers ───────────────────────────────────────────────────────────
def compute_pca_basis_torch(hidden_states_list, pca_rank, device):
    all_h = torch.cat([h.flatten(0, 1) for h in hidden_states_list], dim=0).float()
    mean = all_h.mean(dim=0, keepdim=True)
    X = all_h - mean
    n = X.shape[0]
    actual_rank = min(pca_rank, n)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    V = Vt[:actual_rank].T.contiguous()
    if actual_rank < pca_rank:
        pad = torch.zeros(X.shape[1], pca_rank - actual_rank, device=device)
        V = torch.cat([V, pad], dim=1)
    return mean.to(device), V.to(device)


def to_pca(h, mean, V):
    return (h.float() - mean) @ V


def from_pca(z, mean, V):
    return z @ V.T + mean


# ─── KV Cache Helpers ─────────────────────────────────────────────────
def decoder_step(model, input_ids, enc_out, past_key_values=None):
    """Single decoder forward with optional PKV. Returns (out, pkv)."""
    out = model.model.decoder(
        input_ids=input_ids,
        encoder_hidden_states=enc_out,
        output_hidden_states=True,
        use_cache=True,
        past_key_values=past_key_values,
    )
    return out, out.past_key_values


# ─── Generation Functions (matched loop structure) ─────────────────────
def generate_greedy_custom(target, enc_out, max_tokens=150):
    """
    Same decoder loop as generate_adaptive_v2, but:
    - No Δz correction computation
    - Always appends argmax from target's own logits
    - Uses PKV throughout
    Returns list of token IDs.

    This is the matched baseline for adaptive: identical loop structure,
    only difference is the absence of the correction head.
    """
    device = next(target.parameters()).device
    tokens = [target.config.decoder_start_token_id]
    pkv = None

    with torch.no_grad():
        for _ in range(max_tokens):
            if pkv is not None:
                inp = torch.tensor([[tokens[-1]]], device=device)
                out, pkv = decoder_step(target, inp, enc_out, pkv)
            else:
                inp = torch.tensor([tokens], device=device)
                out, pkv = decoder_step(target, inp, enc_out, None)

            logits = target.proj_out(out.last_hidden_state)
            next_tok = logits[0, -1].argmax().item()
            tokens.append(next_tok)
            if next_tok == target.config.eos_token_id:
                break

    return tokens


def generate_adaptive_v2(target, enc_out, pca_mean, pca_V, drafter,
                          model_dtype, max_tokens=150):
    """
    Same loop as generate_greedy_custom, but with Δz correction + Top-1 gate.
    - Computes draft token from corrected hidden state
    - Accepts draft only when draft_token == target_token
    - Falls back to target_token otherwise

    Returns (tokens, n_accepted, n_total_tokens).
    The validated controller collapses to Top-1 mirror of the custom loop.
    """
    device = next(target.parameters()).device
    audio_summary = enc_out.mean(dim=1, keepdim=True)
    tokens = [target.config.decoder_start_token_id]
    accepted = 0
    pkv = None

    with torch.no_grad():
        for step in range(max_tokens):
            if pkv is not None:
                inp = torch.tensor([[tokens[-1]]], device=device)
                out, pkv = decoder_step(target, inp, enc_out, pkv)
            else:
                inp = torch.tensor([tokens], device=device)
                out, pkv = decoder_step(target, inp, enc_out, None)

            L = len(out.hidden_states) - 1
            h1 = out.hidden_states[1][:, -1:, :] if L >= 1 else out.last_hidden_state[:, -1:, :]
            h2 = out.hidden_states[2][:, -1:, :] if L >= 2 else out.last_hidden_state[:, -1:, :]
            h_t = out.last_hidden_state[:, -1:, :]
            ctx_feats = torch.cat([h1, h2], dim=-1)
            logits = target.proj_out(out.last_hidden_state)
            target_token = logits[0, -1].argmax().item()

            # Δz correction
            z_t = to_pca(h_t, pca_mean, pca_V)
            delta_z = drafter(z_t, ctx_feats, audio_summary)
            h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V)
            draft_logits = target.proj_out(h_corrected.to(model_dtype))
            draft_token = draft_logits[0, -1].argmax().item()

            # Top-1 acceptance gate
            if draft_token == target_token:
                accepted += 1
                tokens.append(draft_token)
            else:
                tokens.append(target_token)

            if tokens[-1] == target.config.eos_token_id:
                break

    return tokens, accepted, len(tokens)


def proc_audio(processor, audio_array, model):
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    x = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    return x.to(dev, dtype=dtype)


# ─── Correction Head ──────────────────────────────────────────────────
class CorrectionDrafter(nn.Module):
    """Learns Δz to correct h_t for better next-token prediction."""
    def __init__(self, d_draft, pca_rank, d_target, d_audio, num_taps=2):
        super().__init__()
        ctx_dim = num_taps * d_target + d_audio + pca_rank
        self.ctx_proj = nn.Linear(ctx_dim, d_draft)
        self.layer1 = nn.Linear(d_draft, d_draft)
        self.layer1_norm = nn.LayerNorm(d_draft)
        self.layer2 = nn.Linear(d_draft, d_draft)
        self.layer2_norm = nn.LayerNorm(d_draft)
        self.output_head = nn.Linear(d_draft, pca_rank, bias=False)

    def forward(self, z_t, target_hidden, audio_summary):
        ctx = torch.cat([target_hidden, audio_summary, z_t], dim=-1)
        x = self.ctx_proj(ctx)
        r1 = x
        x = self.layer1_norm(x)
        x = F.gelu(self.layer1(x))
        x = x + r1
        r2 = x
        x = self.layer2_norm(x)
        x = F.gelu(self.layer2(x))
        x = x + r2
        return self.output_head(x)


# ─── Training Data ────────────────────────────────────────────────────
def extract_training_data(target, processor, ds_iter, n_samples, device, text_field="text"):
    """Extract (h_t, context, audio_summary, true_token) pairs for training.
    Works with both indexed and streaming (IterableDataset) inputs."""
    import itertools
    data = []
    for i, s in enumerate(itertools.islice(ds_iter, n_samples)):
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        input_features = proc_audio(processor, audio, target)

        text = s[text_field]
        labels = processor(text=text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        labels = torch.cat([torch.tensor([target.config.decoder_start_token_id]), labels]).tolist()

        with torch.no_grad():
            enc_out = target.model.encoder(input_features).last_hidden_state
        audio_summ = enc_out.mean(dim=1, keepdim=True)

        for t_idx in range(1, len(labels) - 1):
            inp_tok = torch.tensor([labels[:t_idx + 1]], device=device)
            dec_out = target.model.decoder(
                input_ids=inp_tok,
                encoder_hidden_states=enc_out,
                output_hidden_states=True,
            )
            L = len(dec_out.hidden_states) - 1
            h1 = dec_out.hidden_states[1][:, -1:, :] if L >= 1 else dec_out.last_hidden_state[:, -1:, :]
            h2 = dec_out.hidden_states[2][:, -1:, :] if L >= 2 else dec_out.last_hidden_state[:, -1:, :]
            ctx = torch.cat([h1, h2], dim=-1)
            h_t = dec_out.last_hidden_state[:, -1:, :]
            data.append({
                "h_t": h_t.detach().cpu(),
                "ctx": ctx.detach().cpu(),
                "audio": audio_summ.detach().cpu(),
                "true_token": labels[t_idx + 1],
            })

        if (i + 1) % 5 == 0:
            print(f"    Extracted {i+1}/{n_samples} samples ({len(data)} datapoints)", flush=True)
    return data


# ─── Correction Quality Evaluation (Representation Space) ─────────────
def evaluate_correction(target, processor, ds_iter, drafter, pca_mean, pca_V,
                        train_end, n_eval, text_field, device):
    """
    Evaluate correction quality per-token on ground-truth transcriptions.

    For each token position:
      - baseline: logits = proj_out(h_t)          [uncorrected]
      - corrected: logits = proj_out(h_t + Δz)     [after correction]

    Works with both indexed and streaming (IterableDataset) inputs.
    Returns per-token metrics dict.
    """
    import itertools
    model_dtype = next(target.parameters()).dtype
    all_metrics = []

    for i, s in enumerate(itertools.islice(ds_iter, train_end, train_end + n_eval)):
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        input_features = proc_audio(processor, audio, target)
        text = s[text_field]

        with torch.no_grad():
            enc_out = target.model.encoder(input_features).last_hidden_state
        audio_summary = enc_out.mean(dim=1, keepdim=True)

        labels = processor(text=text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        labels = torch.cat([torch.tensor([target.config.decoder_start_token_id]), labels]).to(device)

        for t_idx in range(1, len(labels) - 1):
            inp = labels[:t_idx + 1].unsqueeze(0)
            out = target.model.decoder(
                input_ids=inp,
                encoder_hidden_states=enc_out,
                output_hidden_states=True,
            )
            L = len(out.hidden_states) - 1
            h1 = out.hidden_states[1][:, -1:, :] if L >= 1 else out.last_hidden_state[:, -1:, :]
            h2 = out.hidden_states[2][:, -1:, :] if L >= 2 else out.last_hidden_state[:, -1:, :]
            h_t = out.last_hidden_state[:, -1:, :]
            ctx_feats = torch.cat([h1, h2], dim=-1)
            true_tok = labels[t_idx + 1].unsqueeze(0)

            # ── Baseline (uncorrected) ──
            baseline_logits = target.proj_out(h_t)                           # (1, 1, V)
            bl = baseline_logits[0, -1]                                      # (V,)
            baseline_ce = F.cross_entropy(bl.unsqueeze(0), true_tok).item()
            baseline_pred = bl.argmax().item()
            baseline_correct = int(baseline_pred == true_tok.item())
            baseline_probs = F.softmax(bl, dim=-1)
            baseline_true_prob = baseline_probs[true_tok].item()
            # Rank of true token (1 = best)
            baseline_rank = (baseline_probs > baseline_probs[true_tok]).sum().item() + 1

            # ── Corrected ──
            z_t = to_pca(h_t, pca_mean, pca_V)
            delta_z = drafter(z_t, ctx_feats, audio_summary)                 # Δz ∈ R^R
            h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V).to(model_dtype)
            corrected_logits = target.proj_out(h_corrected)                  # (1, 1, V)
            cl = corrected_logits[0, -1]
            corrected_ce = F.cross_entropy(cl.unsqueeze(0), true_tok).item()
            corrected_pred = cl.argmax().item()
            corrected_correct = int(corrected_pred == true_tok.item())
            corrected_probs = F.softmax(cl, dim=-1)
            corrected_true_prob = corrected_probs[true_tok].item()
            corrected_rank = (corrected_probs > corrected_probs[true_tok]).sum().item() + 1

            # ── Δz analysis ──
            dz_norm = delta_z.norm(dim=-1).squeeze().item()
            # Normalised Δz direction (unit vector)
            if dz_norm > 1e-8:
                dz_unit = delta_z / dz_norm
            else:
                dz_unit = torch.zeros_like(delta_z)

            # What is the "ideal" Δz direction? (z_{t+1} - z_t) in PCA space
            # Only if we have the next token's hidden state
            # This measures if the correction moves toward the next state

            all_metrics.append({
                "sample_idx": i,
                "token_pos": t_idx,
                "true_token": true_tok.item(),
                "baseline_ce": baseline_ce,
                "corrected_ce": corrected_ce,
                "ce_delta": baseline_ce - corrected_ce,          # positive = correction helps
                "baseline_correct": baseline_correct,
                "corrected_correct": corrected_correct,
                "baseline_true_prob": baseline_true_prob,
                "corrected_true_prob": corrected_true_prob,
                "prob_lift": corrected_true_prob - baseline_true_prob,
                "baseline_rank": baseline_rank,
                "corrected_rank": corrected_rank,
                "rank_delta": baseline_rank - corrected_rank,    # positive = rank improved
                "dz_norm": dz_norm,
            })

    return all_metrics


# ─── Matched-Baseline Speed & WER Benchmark ──────────────────────────
def benchmark_speed(target, processor, ds_iter, drafter, pca_mean, pca_V,
                    train_end, n_samples, text_field, device):
    """
    Speed and WER comparison with matched baselines.
    Every variant uses the same decoder_step() + PKV path.

    Variants:
      A) HF model.generate(use_cache=True)        — external reference only
      B) Custom greedy (use_cache=True)             — matched baseline (no Dz)
      C) Adaptive v2 (use_cache=True)               — Dz correction + Top-1 gate
      D) Custom greedy (use_cache=False)            — KV benchmark baseline
      E) KV: greedy_no_kv vs greedy_kv              — pure KV speedup
      F) Parity: adaptive_v2 text matches greedy_custom exactly?

    Returns list of per-sample results + computed WERs.
    """
    import itertools
    import jiwer
    model_dtype = next(target.parameters()).dtype
    results = []

    for i, s in enumerate(itertools.islice(ds_iter, train_end, train_end + n_samples)):
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        input_features = proc_audio(processor, audio, target)
        ref_text = s[text_field] if text_field in s else ""  # not needed for WER (we compare methods)

        # ── Encode once ──
        with torch.no_grad():
            enc_out = target.model.encoder(input_features).last_hidden_state

        # ── A) HF Greedy (reference) ──
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            hf_tokens = target.generate(
                input_features, max_length=MAX_TOKENS, num_beams=1, use_cache=True,
            )
        torch.cuda.synchronize()
        t_hf = time.time() - t0
        hf_text = processor.decode(hf_tokens[0], skip_special_tokens=True)

        # ── B) Custom Greedy (use_cache=True) — matched baseline ──
        torch.cuda.synchronize()
        t0 = time.time()
        greedy_tokens = generate_greedy_custom(target, enc_out, max_tokens=MAX_TOKENS)
        torch.cuda.synchronize()
        t_greedy_kv = time.time() - t0
        greedy_text = processor.decode(greedy_tokens, skip_special_tokens=True)

        # ── C) Adaptive v2 (use_cache=True) — Dz + Top-1 gate ──
        torch.cuda.synchronize()
        t0 = time.time()
        adaptive_tokens, n_accept, n_total = generate_adaptive_v2(
            target, enc_out, pca_mean, pca_V, drafter, model_dtype, max_tokens=MAX_TOKENS,
        )
        torch.cuda.synchronize()
        t_adaptive_kv = time.time() - t0
        adaptive_text = processor.decode(adaptive_tokens, skip_special_tokens=True)

        # ── D) Custom Greedy (use_cache=False) — KV benchmark baseline ──
        # Re-implement without KV cache: full prefix each step
        torch.cuda.synchronize()
        t0 = time.time()
        greedy_tokens_no_kv = []
        with torch.no_grad():
            gen_tokens = [target.config.decoder_start_token_id]
            for _ in range(MAX_TOKENS):
                inp = torch.tensor([gen_tokens], device=device)
                out = target.model.decoder(
                    input_ids=inp,
                    encoder_hidden_states=enc_out,
                    output_hidden_states=False,
                    use_cache=False,
                    past_key_values=None,
                )
                next_tok = target.proj_out(out.last_hidden_state)[0, -1].argmax().item()
                gen_tokens.append(next_tok)
                if next_tok == target.config.eos_token_id:
                    break
        torch.cuda.synchronize()
        t_greedy_no_kv = time.time() - t0
        greedy_no_kv_text = processor.decode(gen_tokens, skip_special_tokens=True)

        # ── E) Parity check ──
        adaptive_matches_greedy = int(adaptive_text == greedy_text)
        # Full token-level parity (ignoring special tokens)
        adaptive_tokens_clean = [t for t in adaptive_tokens if t < target.config.eos_token_id]
        greedy_tokens_clean = [t for t in greedy_tokens if t < target.config.eos_token_id]
        token_level_match = int(adaptive_tokens_clean == greedy_tokens_clean)

        res = {
            "idx": i,
            "t_hf_greedy_s": t_hf,
            "t_greedy_kv_s": t_greedy_kv,
            "t_adaptive_kv_s": t_adaptive_kv,
            "t_greedy_no_kv_s": t_greedy_no_kv,
            "speedup_vs_hf": t_hf / max(t_greedy_kv, 1e-9),
            "speedup_adaptive_vs_greedy": t_greedy_kv / max(t_adaptive_kv, 1e-9),
            "speedup_kv_vs_no_kv": t_greedy_no_kv / max(t_greedy_kv, 1e-9),
            "n_accept": n_accept,
            "n_total": n_total,
            "accept_rate": n_accept / max(n_total, 1),
            "parity_text_match": adaptive_matches_greedy,
            "parity_token_match": token_level_match,
            "hf_text": hf_text,
            "greedy_text": greedy_text,
            "adaptive_text": adaptive_text,
        }
        results.append(res)

        print(f"    [{i}] HF={t_hf:.3f}s  greedy(kv)={t_greedy_kv:.3f}s  "
              f"adaptive={t_adaptive_kv:.3f}s  greedy(no_kv)={t_greedy_no_kv:.3f}s", flush=True)
        print(f"         speedup vs HF: {res['speedup_vs_hf']:.2f}x  "
              f"adaptive vs greedy: {res['speedup_adaptive_vs_greedy']:.2f}x  "
              f"KV: {res['speedup_kv_vs_no_kv']:.2f}x  "
              f"parity={res['parity_text_match']}", flush=True)

    # ── Aggregate ──
    n = len(results)
    avg_greedy_kv = np.mean([r["t_greedy_kv_s"] for r in results])
    avg_adaptive_kv = np.mean([r["t_adaptive_kv_s"] for r in results])

    # WERs against matched greedy custom baseline
    greedy_wer_list = []
    adaptive_wer_list = []
    for r in results:
        greedy_wer_list.append(jiwer.wer(r["greedy_text"], r["greedy_text"]) if n > 0 else 0.0)  # identity (0)
        adaptive_wer_list.append(jiwer.wer(r["greedy_text"], r["adaptive_text"]))
    mean_adaptive_vs_greedy_wer = np.mean(adaptive_wer_list) if n > 0 else 0.0

    # Parity stats
    text_parity_rate = np.mean([r["parity_text_match"] for r in results])
    token_parity_rate = np.mean([r["parity_token_match"] for r in results])

    summary = {
        "n_samples": n,
        "hf_greedy_mean_s": float(np.mean([r["t_hf_greedy_s"] for r in results])),
        "greedy_kv_mean_s": float(avg_greedy_kv),
        "adaptive_kv_mean_s": float(avg_adaptive_kv),
        "greedy_no_kv_mean_s": float(np.mean([r["t_greedy_no_kv_s"] for r in results])),
        "avg_speedup_vs_hf": float(np.mean([r["speedup_vs_hf"] for r in results])),
        "avg_speedup_adaptive_vs_greedy": float(np.mean([r["speedup_adaptive_vs_greedy"] for r in results])),
        "avg_speedup_kv_vs_no_kv": float(np.mean([r["speedup_kv_vs_no_kv"] for r in results])),
        "adaptive_vs_greedy_wer": float(mean_adaptive_vs_greedy_wer),
        "text_parity_rate": float(text_parity_rate),
        "token_parity_rate": float(token_parity_rate),
    }
    print(f"    ─────────────────────────────────────", flush=True)
    print(f"    Avg speedup vs HF generate:  {summary['avg_speedup_vs_hf']:.2f}x", flush=True)
    print(f"    Adaptive vs greedy (matched): {summary['avg_speedup_adaptive_vs_greedy']:.2f}x", flush=True)
    print(f"    KV speedup (no_KV vs KV):     {summary['avg_speedup_kv_vs_no_kv']:.2f}x", flush=True)
    print(f"    Adaptive-vs-greedy WER:       {summary['adaptive_vs_greedy_wer']:.4f}", flush=True)
    print(f"    Text parity rate:             {text_parity_rate:.1%}", flush=True)
    print(f"    Token parity rate:            {token_parity_rate:.1%}", flush=True)

    return results, summary


# ─── Main ─────────────────────────────────────────────────────────────
def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"Device: {device}  ({gpu_name})", flush=True)

    pca_rank = args.pca_rank or PCA_RANK_MAP.get(args.model, 64)
    d_draft = args.d_draft or (512 if "large" in args.model else 256)

    print(f"\n{'='*70}", flush=True)
    print(f"EVAL: {args.model}", flush=True)
    print(f"  PCA R={pca_rank}, Draft dim={d_draft}")
    print(f"  Train={args.train_samples}, Eval={args.eval_samples}")
    print(f"{'='*70}", flush=True)

    # ── Load model ──
    t0 = time.time()
    print(f"\nLoading {args.model}...", end=" ", flush=True)
    target = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    target.eval()
    processor = WhisperProcessor.from_pretrained(args.model)
    d_model = target.config.d_model
    print(f"done ({time.time()-t0:.1f}s)  d_model={d_model}", flush=True)

    # ── Load datasets (separate train/eval splits, streaming to avoid full download) ──
    print(f"Loading train: {args.train_dataset} ({args.train_config}/{args.train_split})...", end=" ", flush=True)
    train_ds = load_dataset(args.train_dataset, args.train_config, split=args.train_split, streaming=True)
    print(f"done (streaming, {args.train_samples} samples)", flush=True)

    print(f"Loading eval: {args.eval_dataset} ({args.eval_config}/{args.eval_split})...", end=" ", flush=True)
    eval_ds = load_dataset(args.eval_dataset, args.eval_config, split=args.eval_split, streaming=True)
    print(f"done (streaming, {args.eval_samples} samples)", flush=True)

    text_field = "text" if "text" in eval_ds.features else \
                 ("transcription" if "transcription" in eval_ds.features else "sentence")
    train_text_field = "text" if "text" in train_ds.features else \
                       ("transcription" if "transcription" in train_ds.features else "sentence")
    print(f"  Train text field: '{train_text_field}',  Eval text field: '{text_field}'", flush=True)

    # ── Extract training data ──
    print(f"\nExtracting training data ({args.train_samples} samples from {args.train_split})...", flush=True)
    train_data = extract_training_data(target, processor, train_ds, args.train_samples, device, text_field=train_text_field)
    print(f"  → {len(train_data)} datapoints", flush=True)

    # ── PCA ──
    print("Computing PCA basis...", end=" ", flush=True)
    pca_mean, pca_V = compute_pca_basis_torch([d["h_t"] for d in train_data], pca_rank, device)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"].to(device), pca_mean, pca_V)
    print("done", flush=True)

    # ── Train correction head ──
    print(f"\nTraining correction head (N={len(train_data)}, E={EPOCHS})...", flush=True)
    drafter = CorrectionDrafter(d_draft, pca_rank, d_model, d_model, num_taps=2)
    drafter.to(device).train()
    opt = torch.optim.Adam(drafter.parameters(), lr=1e-3)
    model_dtype = next(target.parameters()).dtype

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for d in train_data:
            z_t = d["z_t"].to(device)
            ctx = d["ctx"].to(device)
            audio = d["audio"].to(device)
            true_tok = torch.tensor([d["true_token"]], device=device)

            delta_z = drafter(z_t, ctx, audio)
            h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V)
            draft_logits = target.proj_out(h_corrected.to(model_dtype))
            ce = F.cross_entropy(draft_logits.view(1, -1), true_tok)
            loss = ce + LAMBDA_CE * torch.mean(delta_z ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:02d}/{EPOCHS}  loss={epoch_loss/len(train_data):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    drafter.eval()
    print(f"  Training done ({time.time()-t0:.0f}s)", flush=True)

    # ── Representation Correction Evaluation ──
    print(f"\n{'='*70}", flush=True)
    print(f"REPRESENTATION SPACE EVALUATION ({args.eval_samples} samples from {args.eval_split})", flush=True)
    print(f"{'='*70}", flush=True)

    metrics = evaluate_correction(
        target, processor, eval_ds, drafter, pca_mean, pca_V,
        0, args.eval_samples, text_field, device,
    )

    # ── Aggregate metrics ──
    ce_deltas = [m["ce_delta"] for m in metrics]
    prob_lifts = [m["prob_lift"] for m in metrics]
    rank_deltas = [m["rank_delta"] for m in metrics]
    baseline_correct = [m["baseline_correct"] for m in metrics]
    corrected_correct = [m["corrected_correct"] for m in metrics]
    dz_norms = [m["dz_norm"] for m in metrics]

    baseline_acc = np.mean(baseline_correct)
    corrected_acc = np.mean(corrected_correct)
    mean_ce_delta = np.mean(ce_deltas)
    mean_prob_lift = np.mean(prob_lifts)
    mean_rank_delta = np.mean(rank_deltas)
    mean_dz_norm = np.mean(dz_norms)

    # Fraction of tokens where correction helps
    frac_ce_improved = np.mean([d > 1e-4 for d in ce_deltas])
    frac_ce_harmed = np.mean([d < -1e-4 for d in ce_deltas])
    frac_ce_neutral = np.mean([abs(d) <= 1e-4 for d in ce_deltas])

    print(f"\n  Correction Quality (over {len(metrics)} tokens):")
    print(f"    Baseline CE:        {np.mean([m['baseline_ce'] for m in metrics]):.4f}")
    print(f"    Corrected CE:       {np.mean([m['corrected_ce'] for m in metrics]):.4f}")
    print(f"    Mean CE Δ:          {mean_ce_delta:+.6f}  (positive = correction helps)")
    print(f"    CE improved:        {frac_ce_improved*100:.1f}% of tokens")
    print(f"    CE harmed:          {frac_ce_harmed*100:.1f}% of tokens")
    print(f"    CE neutral:         {frac_ce_neutral*100:.1f}% of tokens")
    print(f"")
    print(f"    Baseline acc:       {baseline_acc*100:.2f}%")
    print(f"    Corrected acc:      {corrected_acc*100:.2f}%")
    print(f"    Mean prob lift:     {mean_prob_lift:+.6f}")
    print(f"    Mean rank Δ:        {mean_rank_delta:+.3f}  (positive = rank improved)")
    print(f"    Mean |Δz| norm:     {mean_dz_norm:.4f}")

    # ── Speed benchmark (matched baselines) ──
    print(f"\n{'='*70}", flush=True)
    print(f"SPEED BENCHMARK (matched baselines)", flush=True)
    print(f"  A) HF model.generate(use_cache=True)    — external reference", flush=True)
    print(f"  B) Custom greedy (use_cache=True)        — matched baseline (no Dz)", flush=True)
    print(f"  C) Adaptive v2 (use_cache=True)          — Dz correction + Top-1 gate", flush=True)
    print(f"  D) Custom greedy (use_cache=False)       — KV benchmark baseline", flush=True)
    print(f"  E) KV: greedy_no_kv vs greedy_kv         — pure KV speedup", flush=True)
    print(f"  F) Parity: adaptive_v2 text == greedy_custom text?", flush=True)
    print(f"{'='*70}", flush=True)

    speed_results, speed_summary = benchmark_speed(
        target, processor, eval_ds, drafter, pca_mean, pca_V,
        0, min(N_SPEED_SAMPLES, args.eval_samples),
        text_field, device,
    )

    # ── Build artifact ──
    artifact = {
        "model": args.model,
        "train_dataset": f"{args.train_dataset}/{args.train_config}/{args.train_split}",
        "eval_dataset": f"{args.eval_dataset}/{args.eval_config}/{args.eval_split}",
        "pca_rank": pca_rank,
        "d_draft": d_draft,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "epochs": EPOCHS,
        "device": str(device),
        "gpu_name": gpu_name,
        "wall_time_s": time.time() - t0,
        "correction_eval": {
            "n_tokens": len(metrics),
            "baseline_ce_mean": float(np.mean([m["baseline_ce"] for m in metrics])),
            "corrected_ce_mean": float(np.mean([m["corrected_ce"] for m in metrics])),
            "ce_delta_mean": float(mean_ce_delta),
            "frac_ce_improved": float(frac_ce_improved),
            "frac_ce_harmed": float(frac_ce_harmed),
            "frac_ce_neutral": float(frac_ce_neutral),
            "baseline_accuracy": float(baseline_acc),
            "corrected_accuracy": float(corrected_acc),
            "prob_lift_mean": float(mean_prob_lift),
            "rank_delta_mean": float(mean_rank_delta),
            "dz_norm_mean": float(mean_dz_norm),
        },
        "speed_benchmark": {
            "description": "Matched-baseline speed benchmark",
            "variants": {
                "A": "HF model.generate(use_cache=True) - external reference",
                "B": "Custom greedy (use_cache=True) - matched baseline, no Dz",
                "C": "Adaptive v2 (use_cache=True) - Dz correction + Top-1 gate",
                "D": "Custom greedy (use_cache=False) - KV benchmark baseline",
                "E": "KV speedup = D / B (pure KV cache effect)",
                "F": "Parity: adaptive_v2 text == greedy_custom text?",
            },
            "n_samples": len(speed_results),
            "summary": speed_summary,
            "samples": speed_results,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n✅ Artifact → {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Δz Correction Quality Evaluation")
    parser.add_argument("--model", default="openai/whisper-tiny")
    # Training dataset (separate from eval for proper held-out evaluation)
    parser.add_argument("--train-dataset", default="openslr/librispeech_asr")
    parser.add_argument("--train-config", default="clean")
    parser.add_argument("--train-split", default="train.100")
    # Evaluation dataset
    parser.add_argument("--eval-dataset", default="openslr/librispeech_asr")
    parser.add_argument("--eval-config", default="clean")
    parser.add_argument("--eval-split", default="test")
    # Sizes
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=10)
    # PCA / draft dim
    parser.add_argument("--pca-rank", type=int, default=None)
    parser.add_argument("--d-draft", type=int, default=None)
    # Output
    parser.add_argument("--out", default="results/correction_eval.json")
    args = parser.parse_args()

    print(f"Python: {sys.version}", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    sys.exit(run_eval(args))
