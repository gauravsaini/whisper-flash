#!/usr/bin/env python3
"""
experiment_spectral_deep.py

Phase-5 Deep Spectral Exploration — Fixing Rank-1 Collapse

CONTEXT (from Experiment 18):
  - Target manifold α = 0.81 (moderate decay, genuinely rank-4)
  - Draft manifold α = 7.12 (catastrophic collapse to rank-1)
  - Spectral Angle = 67° (subspaces nearly orthogonal)
  - PC-1 cosine = 0.77, PC-2 = 0.38, PC-3/4 = noise
  - ROOT CAUSE: MSE's spectral bias learns PC-1, ignores PC-2..k

THIS EXPERIMENT:
  Part A — Target Manifold Spectral Atlas
    Deep analysis of the TARGET hidden state manifold:
    - Intrinsic dimensionality (eigenvalue gap, participation ratio)
    - Spectral gap structure (where do singular values drop off?)
    - Condition number distribution (how ill-conditioned are trajectories?)
    - Position-dependent spectral variation (does rank change along the sequence?)

  Part B — Three Spectral Training Losses (head-to-head comparison)
    1. MSE Baseline (control)
    2. Whitened MSE (ID 38): ZCA-whitened target space, equalizes PC contributions
    3. Anti-Collapse Loss (ID 37): Penalty when S_i < ε for i >= 2
    4. Per-PC Weighted MSE (ID 39): Inverse singular value reweighting

  Part C — Spectral Metrics Battery
    Same metrics as Exp 18 but across all 4 losses.
    PLUS: Gram matrix rank analysis, condition number matching.
"""

import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, DFlashDecoderLayer, ContinuousDraftModel


# ═══════════════════════════════════════════════════════════════════════
# PART A: Spectral Analysis Utilities
# ═══════════════════════════════════════════════════════════════════════

def compute_svd(matrix):
    """SVD of a (T, D) matrix. Returns U, S, Vt."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def spectral_angle(Vt_a, Vt_b, top_k=4):
    """
    Compute principal angles between top-k subspaces of two matrices.
    Returns mean angle (rad), Grassmann distance, per-PC cosines.
    """
    k = min(top_k, Vt_a.shape[0], Vt_b.shape[0])
    V_a, V_b = Vt_a[:k], Vt_b[:k]
    M = V_a @ V_b.T
    _, sigma, _ = np.linalg.svd(M)
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arccos(sigma)
    
    # Per-PC best-match cosine
    pc_cos = [float(np.max(np.abs(V_a[i] @ V_b.T))) for i in range(k)]
    
    return float(np.mean(angles)), float(np.sqrt(np.sum(angles**2))), pc_cos


def spectral_decay_rate(S):
    """Fit power-law exponent α to singular values: S_i ∝ i^(-α)."""
    S = S[S > 1e-12]
    if len(S) < 3:
        return 0.0, 0.0
    n = len(S)
    log_idx = np.log(np.arange(1, n + 1))
    log_S = np.log(S)
    A = np.stack([log_idx, np.ones(n)], axis=1)
    coeffs = np.linalg.lstsq(A, log_S, rcond=None)[0]
    log_S_pred = A @ coeffs
    ss_res = np.sum((log_S - log_S_pred)**2)
    ss_tot = np.sum((log_S - np.mean(log_S))**2) + 1e-12
    return float(-coeffs[0]), float(1 - ss_res / ss_tot)


def participation_ratio(S):
    """
    Participation ratio = (Σ S_i²)² / (Σ S_i⁴)
    Measures effective dimensionality. Range: 1 (rank-1) to n (uniform).
    """
    S2 = S**2
    S4 = S**4
    if np.sum(S4) < 1e-20:
        return 0.0
    return float(np.sum(S2)**2 / np.sum(S4))


def condition_number(S):
    """Condition number = S_max / S_min (of non-zero singular values)."""
    S = S[S > 1e-12]
    if len(S) < 2:
        return float('inf')
    return float(S[0] / S[-1])


def spectral_gap(S):
    """Largest relative gap between consecutive singular values."""
    S = S[S > 1e-12]
    if len(S) < 2:
        return 0.0, 0
    gaps = np.diff(S) / (S[:-1] + 1e-12)  # relative gaps
    idx = int(np.argmax(np.abs(gaps)))
    return float(np.abs(gaps[idx])), idx


def gram_matrix_rank(H, threshold=0.01):
    """Effective rank of the Gram matrix of normalized H."""
    norms = np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9
    H_norm = H / norms
    G = H_norm @ H_norm.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = eigvals[::-1]  # descending
    # Effective rank: number of eigenvalues > threshold * max
    eff_rank = int(np.sum(eigvals > threshold * eigvals[0]))
    return eff_rank, eigvals


# ═══════════════════════════════════════════════════════════════════════
# PART B: Draft Model (same architecture, different losses)
# ═══════════════════════════════════════════════════════════════════════

class SpectralDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        num_taps = len(config.target_layer_ids)
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
        x = self.norm(x)
        return self.continuous_head(x)


# ═══════════════════════════════════════════════════════════════════════
# PART C: Loss Functions
# ═══════════════════════════════════════════════════════════════════════

def mse_loss(model, noise, ctx, audio, pos, true_hidden):
    """Standard MSE — the control."""
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - true_hidden))


def whitened_mse_loss(model, noise, ctx, audio, pos, true_hidden, whiten_matrix=None):
    """
    ID 38: Whitened MSE.
    
    Pre-multiply the error by the whitening matrix W = V_t @ diag(1/S_t) @ V_t^T
    computed from the target's covariance. This equalizes all PC contributions.
    
    Since whiten_matrix is precomputed and frozen, we pass it via closure.
    Fallback: if no whitening matrix, compute a row-wise decorrelation proxy:
    normalize each row of the error by the row's L2 norm.
    """
    pred = model(noise, ctx, audio, pos)
    error = pred - true_hidden  # (1, T, D)
    
    # Row-wise whitening proxy: normalize each time step's error
    # This prevents PC-1 from dominating by making each step's error unit-norm
    # before squaring (geometrically: measures angular error, not magnitude error)
    error_norms = mx.linalg.norm(error, axis=-1, keepdims=True) + 1e-9
    true_norms = mx.linalg.norm(true_hidden, axis=-1, keepdims=True) + 1e-9
    
    # Cosine-distance-like loss: doesn't suffer from spectral bias
    # because it treats all directions equally after normalization
    pred_norm = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true_norm = true_hidden / true_norms
    
    # 1 - cos(θ) for each vector pair
    cosine_loss = 1.0 - mx.mean(mx.sum(pred_norm * true_norm, axis=-1))
    
    # Also add scaled MSE to preserve magnitude matching
    mse = mx.mean(mx.square(pred - true_hidden))
    
    return 0.5 * mse + 0.5 * cosine_loss


def anti_collapse_loss(model, noise, ctx, audio, pos, true_hidden, lambda_ac=2.0):
    """
    ID 37: Anti-Collapse Loss.
    
    MSE + penalty for rank collapse. We detect collapse via the Gram matrix:
    if all predicted vectors are collinear, G ≈ 11^T (all entries ≈ 1).
    We penalize high off-diagonal entries in the normalized Gram matrix.
    
    Specifically: minimize mean(|G_offdiag|) towards the target's mean(|G_offdiag|).
    Target G has off-diagonal < 1 (diverse directions).
    Collapsed G has off-diagonal ≈ 1 (all same direction).
    
    Also: penalize the condition number proxy — ratio of max to min
    eigenvalue of the Gram matrix (via trace and Frobenius norm).
    """
    pred = model(noise, ctx, audio, pos)
    mse = mx.mean(mx.square(pred - true_hidden))
    
    # Normalized Gram matrices
    pred_n = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true_n = true_hidden / (mx.linalg.norm(true_hidden, axis=-1, keepdims=True) + 1e-9)
    
    G_pred = mx.matmul(pred_n, mx.transpose(pred_n, [0, 2, 1]))  # (1, T, T)
    G_true = mx.matmul(true_n, mx.transpose(true_n, [0, 2, 1]))  # (1, T, T)
    
    # Gram matching loss (structural)
    gram_loss = mx.mean(mx.square(G_pred - G_true))
    
    # Anti-collapse: penalize when off-diagonal of G_pred are too close to ±1
    T = pred.shape[1]
    # Create a mask that zeros the diagonal
    # For T=4: [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
    eye = mx.eye(T)
    offdiag_mask = 1.0 - eye
    
    # Mean absolute off-diagonal (if rank-1, this → 1.0; if diverse, < 1.0)
    offdiag_pred = mx.sum(mx.abs(G_pred[0]) * offdiag_mask) / mx.sum(offdiag_mask)
    
    # Penalty: push off-diagonal cosines AWAY from 1
    # ReLU(|G_offdiag| - 0.8) — only penalize when too collinear
    collapse_penalty = mx.mean(mx.maximum(mx.abs(G_pred[0]) * offdiag_mask - 0.8, 0.0))
    
    # Diversity reward: maximize the variance of predicted hidden states
    # across the T timesteps (higher variance = more diverse directions)
    pred_mean = mx.mean(pred, axis=1, keepdims=True)
    diversity = mx.mean(mx.square(pred - pred_mean))
    
    return mse + lambda_ac * gram_loss + lambda_ac * collapse_penalty - 0.1 * diversity


def per_pc_weighted_mse_loss(model, noise, ctx, audio, pos, true_hidden,
                              inv_sv_weights=None):
    """
    ID 39: Per-PC Weighted MSE.
    
    Project the error onto the target's principal components and weight
    inversely by singular value magnitude. This gives equal gradient to all PCs.
    
    Since target SVD is frozen, we precompute Vt and weights.
    But since we can't easily pass extra args through MLX value_and_grad,
    we use a differentiable proxy: the per-dimension variance-normalized MSE.
    
    For each dimension d, compute the variance of true_hidden[:, :, d]
    and weight the MSE for that dimension by 1/sqrt(var).
    """
    pred = model(noise, ctx, audio, pos)
    error = pred - true_hidden
    
    # Per-dimension variance of the target (across the T timesteps)
    # This is a proxy for SVD weighting: high-variance dimensions ≈ PC-1,
    # low-variance dimensions ≈ PC-k
    target_var = mx.var(true_hidden, axis=1, keepdims=True) + 1e-6  # (1, 1, D)
    
    # Inverse-variance weighting: amplify error in low-variance (neglected) dimensions
    weights = 1.0 / mx.sqrt(target_var)
    weights = weights / mx.mean(weights)  # normalize to mean=1 (keeps loss scale)
    
    weighted_mse = mx.mean(mx.square(error) * weights)
    
    return weighted_mse


# ═══════════════════════════════════════════════════════════════════════
# PART D: Main Experiment
# ═══════════════════════════════════════════════════════════════════════

def run():
    print("=" * 75)
    print("EXPERIMENT: Deep Spectral Exploration — Fixing Rank-1 Collapse")
    print("=" * 75)
    
    # --- Load ---
    print("\n[1/7] Loading Target Model...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    print("[2/7] Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # --- Pre-extract Data ---
    print("[3/7] Pre-extracting training data...")
    train_data = []
    
    for i in range(10):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        token_ids = mx.array([text_tokens], dtype=mx.int32)
        sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
        labels = mx.concatenate([sot, token_ids], axis=1)
        encoder_hidden = encoder_forward(target, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
        
        for t in range(1, labels.shape[1] - config.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = mx.concatenate([hidden_all[lid] for lid in config.target_layer_ids], axis=-1)
            
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            
            train_data.append({
                "noise": noise, "ctx": ctx_feats, "audio": audio_summary,
                "pos": pos_ids, "true_hidden": true_hidden
            })
    
    print(f"   Datapoints: {len(train_data)}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PART A: TARGET MANIFOLD SPECTRAL ATLAS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[4/7] ═══ TARGET MANIFOLD SPECTRAL ATLAS ═══")
    
    all_target_alphas = []
    all_target_pr = []
    all_target_cond = []
    all_target_gaps = []
    all_target_gram_ranks = []
    all_target_sv_spectra = []
    all_target_gram_eigvals = []
    position_alphas = {}  # position → [α values]
    
    for data in train_data:
        true_np = np.array(data["true_hidden"][0])  # (T, D)
        _, S, Vt = compute_svd(true_np)
        
        alpha, r2 = spectral_decay_rate(S)
        pr = participation_ratio(S)
        cn = condition_number(S)
        gap_val, gap_idx = spectral_gap(S)
        gram_rank, gram_eigs = gram_matrix_rank(true_np)
        
        all_target_alphas.append(alpha)
        all_target_pr.append(pr)
        all_target_cond.append(cn)
        all_target_gaps.append((gap_val, gap_idx))
        all_target_gram_ranks.append(gram_rank)
        all_target_sv_spectra.append(S[:config.block_size])
        all_target_gram_eigvals.append(gram_eigs[:config.block_size])
        
        # Track position-dependent spectral structure
        pos = int(np.array(data["pos"][0, 0]))
        if pos not in position_alphas:
            position_alphas[pos] = []
        position_alphas[pos].append(alpha)
    
    print(f"\n   Spectral Decay Rate (α):")
    print(f"     Mean:   {np.mean(all_target_alphas):.4f} ± {np.std(all_target_alphas):.4f}")
    print(f"     Min:    {np.min(all_target_alphas):.4f}")
    print(f"     Max:    {np.max(all_target_alphas):.4f}")
    print(f"     Median: {np.median(all_target_alphas):.4f}")
    
    print(f"\n   Participation Ratio (intrinsic dimensionality):")
    print(f"     Mean:   {np.mean(all_target_pr):.4f} (out of {config.block_size})")
    print(f"     Min:    {np.min(all_target_pr):.4f}")
    print(f"     Max:    {np.max(all_target_pr):.4f}")
    print(f"     → A PR of ~{np.mean(all_target_pr):.1f} means the manifold is genuinely ~{np.mean(all_target_pr):.0f}-dimensional")
    
    print(f"\n   Condition Number:")
    print(f"     Mean:   {np.mean(all_target_cond):.2f}")
    print(f"     Median: {np.median(all_target_cond):.2f}")
    
    print(f"\n   Gram Matrix Effective Rank:")
    print(f"     Mean:   {np.mean(all_target_gram_ranks):.2f} / {config.block_size}")
    gram_rank_hist = {}
    for r in all_target_gram_ranks:
        gram_rank_hist[r] = gram_rank_hist.get(r, 0) + 1
    for r in sorted(gram_rank_hist.keys()):
        pct = gram_rank_hist[r] / len(all_target_gram_ranks) * 100
        print(f"     Rank {r}: {gram_rank_hist[r]} samples ({pct:.1f}%)")
    
    # Singular value spectrum statistics
    sv_matrix = np.array(all_target_sv_spectra)  # (N, block_size)
    print(f"\n   Singular Value Spectrum (mean across all samples):")
    for k in range(min(config.block_size, sv_matrix.shape[1])):
        print(f"     S_{k+1}: {np.mean(sv_matrix[:, k]):.4f} ± {np.std(sv_matrix[:, k]):.4f}")
    
    # Gram eigenvalue spectrum
    ge_matrix = np.array(all_target_gram_eigvals)
    print(f"\n   Gram Matrix Eigenvalue Spectrum (normalized):")
    for k in range(min(config.block_size, ge_matrix.shape[1])):
        mean_eig = np.mean(ge_matrix[:, k])
        total_eig = np.sum(np.mean(ge_matrix, axis=0))
        pct = mean_eig / total_eig * 100 if total_eig > 0 else 0
        print(f"     λ_{k+1}: {mean_eig:.4f} ({pct:.1f}% of total)")
    
    # Position-dependent analysis
    print(f"\n   Position-Dependent Spectral Decay:")
    positions = sorted(position_alphas.keys())
    if len(positions) > 5:
        # Show 5 representative positions
        indices = np.linspace(0, len(positions)-1, 5, dtype=int)
        for idx in indices:
            pos = positions[idx]
            alphas = position_alphas[pos]
            print(f"     Position {pos:3d}: α = {np.mean(alphas):.4f} ± {np.std(alphas):.4f} (n={len(alphas)})")
    
    # ═══════════════════════════════════════════════════════════════════
    # PART B: TRAIN 4 MODELS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[5/7] ═══ TRAINING 4 MODELS (25 epochs) ═══")
    
    models = {
        "MSE": SpectralDraftModel(config),
        "Whitened": SpectralDraftModel(config),
        "AntiCollapse": SpectralDraftModel(config),
        "PerPC": SpectralDraftModel(config),
    }
    
    loss_fns = {
        "MSE": mse_loss,
        "Whitened": whitened_mse_loss,
        "AntiCollapse": anti_collapse_loss,
        "PerPC": per_pc_weighted_mse_loss,
    }
    
    optimizers = {name: optim.Adam(learning_rate=1e-3) for name in models}
    grad_fns = {name: nn.value_and_grad(models[name], loss_fns[name]) for name in models}
    
    epochs = 25
    for epoch in range(epochs):
        losses = {name: 0.0 for name in models}
        
        for data in train_data:
            for name in models:
                l, g = grad_fns[name](
                    models[name], data["noise"], data["ctx"], data["audio"],
                    data["pos"], data["true_hidden"]
                )
                optimizers[name].update(models[name], g)
                mx.eval(models[name].parameters(), optimizers[name].state)
                losses[name] += l.item()
        
        if (epoch + 1) % 5 == 0:
            loss_str = "  ".join([f"{n}: {losses[n]/len(train_data):.5f}" for n in models])
            print(f"   Epoch {epoch+1:02d}/{epochs}  {loss_str}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PART C: EVALUATION — FULL SPECTRAL METRICS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[6/7] ═══ SPECTRAL EVALUATION ON HELD-OUT SAMPLES ═══")
    
    metrics = {}
    for name in models:
        metrics[name] = {
            "cosine": [[] for _ in range(config.block_size)],
            "spectral_angle": [], "grassmann": [],
            "alpha": [], "pr": [],
            "gram_rank": [],
            "pc_cosines": [],
            "alpha_mismatch": [],
            "pr_mismatch": [],
            "top5": 0,
        }
    metrics["_total_top5"] = 0
    
    for i in range(10, 20):
        if i >= len(ds):
            break
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        token_ids = mx.array([text_tokens], dtype=mx.int32)
        sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
        labels = mx.concatenate([sot, token_ids], axis=1)
        encoder_hidden = encoder_forward(target, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
        
        for t in range(1, labels.shape[1] - config.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = mx.concatenate([hidden_all[lid] for lid in config.target_layer_ids], axis=-1)
            
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            
            true_np = np.array(true_hidden[0])
            _, S_true, Vt_true = compute_svd(true_np)
            alpha_true, _ = spectral_decay_rate(S_true)
            pr_true = participation_ratio(S_true)
            
            # True top-5 tokens
            true_logits = target.decoder.token_embedding.as_linear(true_hidden)
            true_top5 = mx.argsort(true_logits, axis=-1)[:, :, -5:]
            mx.eval(true_top5)
            true_top5_np = np.array(true_top5[0])
            
            for name in models:
                pred = models[name](noise, ctx_feats, audio_summary, pos_ids)
                mx.eval(pred)
                pred_np = np.array(pred[0])
                
                # Per-step cosine
                for k in range(config.block_size):
                    cos = float(np.dot(pred_np[k], true_np[k]) / 
                               (np.linalg.norm(pred_np[k]) * np.linalg.norm(true_np[k]) + 1e-9))
                    metrics[name]["cosine"][k].append(cos)
                
                # SVD analysis
                _, S_pred, Vt_pred = compute_svd(pred_np)
                
                # Spectral angle
                angle, grass, pc_cos = spectral_angle(Vt_pred, Vt_true, top_k=config.block_size)
                metrics[name]["spectral_angle"].append(angle)
                metrics[name]["grassmann"].append(grass)
                metrics[name]["pc_cosines"].append(pc_cos)
                
                # Decay rate & participation ratio
                alpha_pred, _ = spectral_decay_rate(S_pred)
                pr_pred = participation_ratio(S_pred)
                metrics[name]["alpha"].append(alpha_pred)
                metrics[name]["pr"].append(pr_pred)
                metrics[name]["alpha_mismatch"].append(abs(alpha_pred - alpha_true))
                metrics[name]["pr_mismatch"].append(abs(pr_pred - pr_true))
                
                # Gram rank
                gram_r, _ = gram_matrix_rank(pred_np)
                metrics[name]["gram_rank"].append(gram_r)
                
                # Top-5
                pred_logits = target.decoder.token_embedding.as_linear(pred)
                pred_top5 = mx.argsort(pred_logits, axis=-1)[:, :, -5:]
                mx.eval(pred_top5)
                for k in range(config.block_size):
                    true_set = set(true_top5_np[k].tolist())
                    pred_set = set(np.array(pred_top5[0, k]).tolist())
                    metrics[name]["top5"] += len(true_set & pred_set)
            
            metrics["_total_top5"] += 5 * config.block_size
    
    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("═══ RESULTS: DEEP SPECTRAL EXPLORATION ═══")
    print("=" * 75)
    
    # --- Per-Step Cosine ---
    print("\n┌─── Per-Step Cosine Similarity ───┐")
    header = f"  {'Step':>6}"
    for name in models:
        header += f"  {name:>12}"
    print(header)
    for k in range(config.block_size):
        row = f"  +{k+1:>4}:"
        for name in models:
            mean_cos = np.mean(metrics[name]["cosine"][k])
            row += f"  {mean_cos:>12.4f}"
        print(row)
    
    # --- Spectral Angle ---
    print("\n┌─── Spectral Angle (↓ better) ───┐")
    for name in models:
        sa = np.mean(metrics[name]["spectral_angle"])
        print(f"  {name:>14}: {sa:.4f} rad ({np.degrees(sa):.2f}°)")
    
    # --- Grassmann Distance ---
    print("\n┌─── Grassmann Distance (↓ better) ───┐")
    for name in models:
        gd = np.mean(metrics[name]["grassmann"])
        print(f"  {name:>14}: {gd:.4f}")
    
    # --- Spectral Decay α ---
    print(f"\n┌─── Spectral Decay Rate α (target: {np.mean(all_target_alphas):.4f}) ───┐")
    for name in models:
        a = np.mean(metrics[name]["alpha"])
        am = np.mean(metrics[name]["alpha_mismatch"])
        print(f"  {name:>14}: α = {a:.4f}  (|Δ| from target: {am:.4f})")
    
    # --- Participation Ratio ---
    target_pr_mean = np.mean(all_target_pr)
    print(f"\n┌─── Participation Ratio (target: {target_pr_mean:.4f}) ───┐")
    for name in models:
        pr = np.mean(metrics[name]["pr"])
        pm = np.mean(metrics[name]["pr_mismatch"])
        print(f"  {name:>14}: PR = {pr:.4f}  (|Δ| from target: {pm:.4f})")
    
    # --- Gram Matrix Rank ---
    target_gr = np.mean(all_target_gram_ranks)
    print(f"\n┌─── Gram Matrix Effective Rank (target: {target_gr:.2f}/{config.block_size}) ───┐")
    for name in models:
        gr = np.mean(metrics[name]["gram_rank"])
        print(f"  {name:>14}: {gr:.2f}/{config.block_size}")
    
    # --- Per-PC Alignment ---
    print("\n┌─── Per-PC Cosine Alignment (best-match) ───┐")
    header = f"  {'PC':>4}"
    for name in models:
        header += f"  {name:>12}"
    print(header)
    n_pcs = len(metrics[list(models.keys())[0]]["pc_cosines"][0]) if metrics[list(models.keys())[0]]["pc_cosines"] else 0
    for pc in range(n_pcs):
        row = f"  {pc+1:>4}:"
        for name in models:
            cos = np.mean([pcs[pc] for pcs in metrics[name]["pc_cosines"]])
            row += f"  {cos:>12.4f}"
        print(row)
    
    # --- Top-5 Token Match ---
    total = metrics["_total_top5"]
    print(f"\n┌─── Top-5 Token Match Rate ───┐")
    for name in models:
        rate = metrics[name]["top5"] / total * 100 if total > 0 else 0
        print(f"  {name:>14}: {rate:.2f}%")
    
    # --- Summary Verdict ---
    print("\n" + "=" * 75)
    print("═══ VERDICT ═══")
    
    # Find the best model on each metric
    best_sa = min(models, key=lambda n: np.mean(metrics[n]["spectral_angle"]))
    best_am = min(models, key=lambda n: np.mean(metrics[n]["alpha_mismatch"]))
    best_gr = max(models, key=lambda n: np.mean(metrics[n]["gram_rank"]))
    best_t5 = max(models, key=lambda n: metrics[n]["top5"])
    best_pr = min(models, key=lambda n: np.mean(metrics[n]["pr_mismatch"]))
    
    print(f"  Best Spectral Angle:      {best_sa}")
    print(f"  Best α Matching:          {best_am}")
    print(f"  Best Gram Rank:           {best_gr}")
    print(f"  Best Top-5 Match:         {best_t5}")
    print(f"  Best PR Matching:         {best_pr}")
    
    # Check if any model broke rank-1 collapse
    for name in models:
        gr = np.mean(metrics[name]["gram_rank"])
        alpha = np.mean(metrics[name]["alpha"])
        if gr > 1.5 and alpha < 5.0:
            print(f"\n  ✅ {name} BROKE RANK-1 COLLAPSE! (Gram rank={gr:.2f}, α={alpha:.2f})")
        else:
            print(f"\n  ❌ {name} still collapsed (Gram rank={gr:.2f}, α={alpha:.2f})")
    
    print("=" * 75)


if __name__ == "__main__":
    run()
