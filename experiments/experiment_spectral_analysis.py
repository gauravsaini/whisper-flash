#!/usr/bin/env python3
"""
experiment_spectral_analysis.py

Phase-4 Spectral Exploration: Spectral Angle & Spectral Decay Diagnostics

MOTIVATION:
All prior experiments measure cosine similarity between individual hidden state vectors.
But cosine similarity conflates magnitude and direction in D=384 dimensional space.
When the draft model achieves 0.55 cosine similarity but 100% verification rejection,
something else is wrong: the *spectral structure* of the trajectory is misaligned.

HYPOTHESES:
1. SPECTRAL ANGLE: The principal components (from SVD) of draft trajectory matrices
   are rotationally misaligned with target trajectory PCs. Even if per-vector cosine
   is decent, the overall subspace that the draft trajectory spans may be rotated away
   from the target's subspace. We measure this via the "spectral angle" between
   the top-k principal subspaces.

2. SPECTRAL DECAY: The singular value decay rate (power-law exponent α) of the draft
   trajectory matrix differs from the target's. If the target trajectory has a steep
   decay (concentrated information in few PCs), but the draft distributes energy
   uniformly (flat spectrum), the draft is wasting capacity on noise dimensions.

3. SPECTRAL ANGLE LOSS: Training with a loss that explicitly penalizes principal
   subspace rotation (via Grassmann distance on singular vectors) may force the draft
   model to generate trajectories that are spectrally aligned, not just point-wise close.

METRICS:
- Spectral Angle (SA): arccos of the product of top-k singular vectors' cosines
- Spectral Decay Rate (α): fitted power-law exponent on singular value spectrum  
- Grassmann Distance: geodesic distance between k-dim subspaces
- Per-step cosine (baseline comparison)
- Top-5 token match rate (functional metric)
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


# ---------------------------------------------------------------------------
# 1. Spectral Analysis Utilities (NumPy for SVD — MLX SVD is limited)
# ---------------------------------------------------------------------------

def compute_svd_spectrum(trajectory_matrix):
    """
    Compute the singular value spectrum of a trajectory matrix.
    
    Args:
        trajectory_matrix: (T, D) numpy array — T timesteps, D dimensions
    Returns:
        U: (T, k) left singular vectors
        S: (k,) singular values (sorted descending)
        Vt: (k, D) right singular vectors
    """
    U, S, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)
    return U, S, Vt


def spectral_angle(S_draft, S_target, Vt_draft, Vt_target, top_k=8):
    """
    Compute the spectral angle between top-k principal subspaces.
    
    The spectral angle measures how well the draft trajectory's principal
    directions align with the target's. This is the angle between subspaces
    on the Grassmann manifold.
    
    Args:
        S_draft, S_target: singular values (descending)
        Vt_draft, Vt_target: right singular vectors (top-k rows = top-k PCs)
        top_k: number of principal components to compare
    Returns:
        angle: spectral angle in radians (0 = perfectly aligned)
        cos_angles: per-PC cosine similarities
    """
    k = min(top_k, len(S_draft), len(S_target), Vt_draft.shape[0], Vt_target.shape[0])
    
    # Top-k right singular vectors (each row is a PC direction in D-space)
    V_d = Vt_draft[:k, :]   # (k, D)
    V_t = Vt_target[:k, :]  # (k, D)
    
    # Compute the cosine of the principal angles between the two subspaces
    # This uses the SVD of the cross-product matrix V_d @ V_t^T
    M = V_d @ V_t.T  # (k, k)
    _, sigma_cross, _ = np.linalg.svd(M)
    
    # The principal angles are arccos(sigma_cross), clamped to [-1, 1]
    sigma_cross = np.clip(sigma_cross, -1.0, 1.0)
    principal_angles = np.arccos(sigma_cross)
    
    # Per-PC cosine: how well each draft PC aligns with *its best match* in target
    cos_angles = []
    for i in range(k):
        v_d = V_d[i]
        # Best alignment with any target PC
        dots = np.abs(V_d[i] @ V_t.T)  # (k,) absolute cosines with each target PC
        cos_angles.append(float(np.max(dots)))
    
    # Grassmann distance: sqrt(sum of squared principal angles)
    grassmann_dist = float(np.sqrt(np.sum(principal_angles**2)))
    
    # Mean spectral angle
    mean_angle = float(np.mean(principal_angles))
    
    return mean_angle, grassmann_dist, cos_angles, principal_angles


def spectral_decay_rate(S, min_components=3):
    """
    Fit a power-law decay exponent to the singular value spectrum.
    
    S_i ∝ i^(-α)  →  log(S_i) ∝ -α log(i)
    
    Higher α = steeper decay = information concentrated in fewer PCs.
    Lower α = flatter spectrum = energy spread across many dimensions.
    
    A well-trained draft model should match the target's α.
    
    Args:
        S: singular values (descending, positive)
        min_components: minimum number of singular values to fit
    Returns:
        alpha: power-law exponent (positive = decaying)
        r_squared: goodness of fit
    """
    S = np.array(S)
    S = S[S > 1e-12]  # Drop near-zero singular values
    
    if len(S) < min_components:
        return 0.0, 0.0
    
    n = len(S)
    log_idx = np.log(np.arange(1, n + 1))
    log_S = np.log(S)
    
    # Linear regression: log(S) = -alpha * log(i) + c
    A = np.stack([log_idx, np.ones(n)], axis=1)
    result = np.linalg.lstsq(A, log_S, rcond=None)
    coeffs = result[0]
    
    alpha = -coeffs[0]  # Negate because we fit -alpha
    
    # R-squared
    log_S_pred = A @ coeffs
    ss_res = np.sum((log_S - log_S_pred)**2)
    ss_tot = np.sum((log_S - np.mean(log_S))**2) + 1e-12
    r_squared = 1 - ss_res / ss_tot
    
    return float(alpha), float(r_squared)


def spectral_energy_ratio(S, top_k=4):
    """
    Fraction of total spectral energy captured by top-k singular values.
    
    Measures how "low-rank" the trajectory is.
    Higher = more concentrated, easier to approximate.
    """
    S = np.array(S)
    total = np.sum(S**2)
    if total < 1e-12:
        return 0.0
    top_energy = np.sum(S[:top_k]**2)
    return float(top_energy / total)


# ---------------------------------------------------------------------------
# 2. Spectral-Angle-Aware Draft Model
# ---------------------------------------------------------------------------

class SpectralDraftModel(nn.Module):
    """
    Standard ContinuousDraftModel architecture (with cross-attention),
    but designed to be trained with spectral angle loss in addition to MSE.
    """
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


# ---------------------------------------------------------------------------
# 3. Loss Functions
# ---------------------------------------------------------------------------

def mse_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred - true_hidden))


def spectral_angle_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden,
                            lambda_spectral=0.3):
    """
    Combined MSE + Spectral Angle Loss.
    
    The spectral component penalizes misalignment of the trajectory's
    principal structure by computing a differentiable approximation of
    subspace alignment: the Frobenius norm of the cross-covariance matrix
    between draft and target trajectories (which is maximized when subspaces
    are aligned).
    
    Note: Full SVD is not differentiable in MLX, so we use a proxy:
    the nuclear norm of the cross-covariance matrix C = pred^T @ true.
    This is sum of singular values of C, which is maximized when pred and true
    share the same column space.
    """
    pred = model(noise, target_hidden, audio_summary, position_ids)
    
    # 1. Standard MSE
    mse = mx.mean(mx.square(pred - true_hidden))
    
    # 2. Spectral alignment proxy: Gram matrix alignment
    # Compute normalized Gram matrices for both trajectories
    # G_pred[i,j] = pred[i] · pred[j] / (||pred[i]|| ||pred[j]||)
    # G_true[i,j] = true[i] · true[j] / (||true[i]|| ||true[j]||)
    # Penalize L2 distance between Gram matrices
    
    # Normalize along feature dimension
    pred_norm = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    true_norm = true_hidden / (mx.linalg.norm(true_hidden, axis=-1, keepdims=True) + 1e-9)
    
    # Gram matrices: (batch, T, T) 
    G_pred = mx.matmul(pred_norm, mx.transpose(pred_norm, [0, 2, 1]))
    G_true = mx.matmul(true_norm, mx.transpose(true_norm, [0, 2, 1]))
    
    # Spectral structure loss: Frobenius norm of Gram matrix difference
    gram_loss = mx.mean(mx.square(G_pred - G_true))
    
    # 3. Cross-covariance trace (directional alignment reward)
    # Maximize trace(pred_norm^T @ true_norm) — aligns principal directions
    cross_cov = mx.matmul(mx.transpose(pred_norm, [0, 2, 1]), true_norm)  # (batch, D, D)
    # Frobenius norm of cross-cov (proxy for sum of singular values)
    alignment_reward = mx.mean(mx.sum(cross_cov * cross_cov, axis=(-2, -1)))
    
    return mse + lambda_spectral * gram_loss - 0.1 * lambda_spectral * alignment_reward


# ---------------------------------------------------------------------------
# 4. Main Experiment
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("EXPERIMENT: Spectral Angle & Spectral Decay Analysis")
    print("=" * 70)
    
    # --- Load Target Model ---
    print("\n[1/6] Loading Target Model...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Two models: MSE baseline vs Spectral Angle Loss
    model_mse = SpectralDraftModel(config)
    model_spectral = SpectralDraftModel(config)
    
    # --- Load Dataset ---
    print("[2/6] Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # --- Pre-extract Training Data ---
    print("[3/6] Pre-extracting training data (10 samples)...")
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
    
    print(f"   Training datapoints: {len(train_data)}")
    
    # --- Train Both Models ---
    print("[4/6] Training (25 epochs)...")
    opt_mse = optim.Adam(learning_rate=1e-3)
    opt_spectral = optim.Adam(learning_rate=1e-3)
    
    grad_mse = nn.value_and_grad(model_mse, mse_loss_fn)
    grad_spectral = nn.value_and_grad(model_spectral, spectral_angle_loss_fn)
    
    epochs = 25
    for epoch in range(epochs):
        loss_mse_total = 0.0
        loss_spectral_total = 0.0
        
        for data in train_data:
            # MSE model
            l_mse, g_mse = grad_mse(
                model_mse, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_mse.update(model_mse, g_mse)
            mx.eval(model_mse.parameters(), opt_mse.state)
            loss_mse_total += l_mse.item()
            
            # Spectral model
            l_sp, g_sp = grad_spectral(
                model_spectral, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_spectral.update(model_spectral, g_sp)
            mx.eval(model_spectral.parameters(), opt_spectral.state)
            loss_spectral_total += l_sp.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:02d}/{epochs}  MSE Loss: {loss_mse_total/len(train_data):.5f}  "
                  f"Spectral Loss: {loss_spectral_total/len(train_data):.5f}")
    
    # --- Spectral Analysis on Held-out Data ---
    print("\n[5/6] Spectral Analysis on Held-out Samples (10-19)...")
    
    # Per-step metrics
    step_metrics = {
        "mse": {"cosine": [[] for _ in range(config.block_size)],
                "spectral_angle": [], "grassmann": [],
                "alpha_draft": [], "alpha_target": [],
                "energy_ratio_draft": [], "energy_ratio_target": [],
                "pc_cosines": []},
        "spectral": {"cosine": [[] for _ in range(config.block_size)],
                     "spectral_angle": [], "grassmann": [],
                     "alpha_draft": [], "alpha_target": [],
                     "energy_ratio_draft": [], "energy_ratio_target": [],
                     "pc_cosines": []}
    }
    
    top5_match = {"mse": 0, "spectral": 0, "total": 0}
    
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
            
            # Get predictions
            pred_mse = model_mse(noise, ctx_feats, audio_summary, pos_ids)
            pred_spectral = model_spectral(noise, ctx_feats, audio_summary, pos_ids)
            mx.eval(pred_mse, pred_spectral)
            
            # Convert to numpy for spectral analysis
            true_np = np.array(true_hidden[0])   # (block_size, D)
            pred_mse_np = np.array(pred_mse[0])
            pred_sp_np = np.array(pred_spectral[0])
            
            # Per-step cosine similarity
            for k in range(config.block_size):
                h_true = true_np[k]
                h_mse = pred_mse_np[k]
                h_sp = pred_sp_np[k]
                
                cos_mse = float(np.dot(h_mse, h_true) / (np.linalg.norm(h_mse) * np.linalg.norm(h_true) + 1e-9))
                cos_sp = float(np.dot(h_sp, h_true) / (np.linalg.norm(h_sp) * np.linalg.norm(h_true) + 1e-9))
                
                step_metrics["mse"]["cosine"][k].append(cos_mse)
                step_metrics["spectral"]["cosine"][k].append(cos_sp)
            
            # Trajectory-level spectral analysis (SVD on block_size x D matrix)
            _, S_true, Vt_true = compute_svd_spectrum(true_np)
            _, S_mse, Vt_mse = compute_svd_spectrum(pred_mse_np)
            _, S_sp, Vt_sp = compute_svd_spectrum(pred_sp_np)
            
            # Spectral angle
            angle_mse, grass_mse, pc_cos_mse, _ = spectral_angle(S_mse, S_true, Vt_mse, Vt_true, top_k=min(4, config.block_size))
            angle_sp, grass_sp, pc_cos_sp, _ = spectral_angle(S_sp, S_true, Vt_sp, Vt_true, top_k=min(4, config.block_size))
            
            step_metrics["mse"]["spectral_angle"].append(angle_mse)
            step_metrics["mse"]["grassmann"].append(grass_mse)
            step_metrics["mse"]["pc_cosines"].append(pc_cos_mse)
            
            step_metrics["spectral"]["spectral_angle"].append(angle_sp)
            step_metrics["spectral"]["grassmann"].append(grass_sp)
            step_metrics["spectral"]["pc_cosines"].append(pc_cos_sp)
            
            # Spectral decay rate
            alpha_true, r2_true = spectral_decay_rate(S_true)
            alpha_mse, r2_mse = spectral_decay_rate(S_mse)
            alpha_sp, r2_sp = spectral_decay_rate(S_sp)
            
            step_metrics["mse"]["alpha_draft"].append(alpha_mse)
            step_metrics["mse"]["alpha_target"].append(alpha_true)
            step_metrics["spectral"]["alpha_draft"].append(alpha_sp)
            step_metrics["spectral"]["alpha_target"].append(alpha_true)
            
            # Energy ratio (top-2 PCs for block_size=4)
            er_true = spectral_energy_ratio(S_true, top_k=2)
            er_mse = spectral_energy_ratio(S_mse, top_k=2)
            er_sp = spectral_energy_ratio(S_sp, top_k=2)
            
            step_metrics["mse"]["energy_ratio_draft"].append(er_mse)
            step_metrics["mse"]["energy_ratio_target"].append(er_true)
            step_metrics["spectral"]["energy_ratio_draft"].append(er_sp)
            step_metrics["spectral"]["energy_ratio_target"].append(er_true)
            
            # Top-5 token match
            draft_logits_mse = target.decoder.token_embedding.as_linear(pred_mse[:, :, :])
            draft_logits_sp = target.decoder.token_embedding.as_linear(pred_spectral[:, :, :])
            true_logits = target.decoder.token_embedding.as_linear(true_hidden[:, :, :])
            
            true_top5 = mx.argsort(true_logits, axis=-1)[:, :, -5:]
            draft_top5_mse = mx.argsort(draft_logits_mse, axis=-1)[:, :, -5:]
            draft_top5_sp = mx.argsort(draft_logits_sp, axis=-1)[:, :, -5:]
            mx.eval(true_top5, draft_top5_mse, draft_top5_sp)
            
            true_top5_np = np.array(true_top5[0])
            for k in range(config.block_size):
                true_set = set(true_top5_np[k].tolist())
                mse_set = set(np.array(draft_top5_mse[0, k]).tolist())
                sp_set = set(np.array(draft_top5_sp[0, k]).tolist())
                
                top5_match["mse"] += len(true_set & mse_set)
                top5_match["spectral"] += len(true_set & sp_set)
                top5_match["total"] += 5
    
    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS: SPECTRAL ANGLE & SPECTRAL DECAY ANALYSIS")
    print("=" * 70)
    
    print("\n--- Per-Step Cosine Similarity ---")
    for k in range(config.block_size):
        mean_mse = np.mean(step_metrics["mse"]["cosine"][k])
        mean_sp = np.mean(step_metrics["spectral"]["cosine"][k])
        delta = mean_sp - mean_mse
        print(f"  Step +{k+1}: MSE={mean_mse:.4f}  Spectral={mean_sp:.4f}  Delta={delta:+.4f}")
    
    print("\n--- Spectral Angle (lower = better subspace alignment) ---")
    mean_sa_mse = np.mean(step_metrics["mse"]["spectral_angle"])
    mean_sa_sp = np.mean(step_metrics["spectral"]["spectral_angle"])
    print(f"  MSE Model:      {mean_sa_mse:.4f} rad ({np.degrees(mean_sa_mse):.2f}°)")
    print(f"  Spectral Model: {mean_sa_sp:.4f} rad ({np.degrees(mean_sa_sp):.2f}°)")
    
    print("\n--- Grassmann Distance (lower = closer subspaces) ---")
    mean_gd_mse = np.mean(step_metrics["mse"]["grassmann"])
    mean_gd_sp = np.mean(step_metrics["spectral"]["grassmann"])
    print(f"  MSE Model:      {mean_gd_mse:.4f}")
    print(f"  Spectral Model: {mean_gd_sp:.4f}")
    
    print("\n--- Spectral Decay Rate α (power-law exponent) ---")
    mean_alpha_target = np.mean(step_metrics["mse"]["alpha_target"])
    mean_alpha_mse = np.mean(step_metrics["mse"]["alpha_draft"])
    mean_alpha_sp = np.mean(step_metrics["spectral"]["alpha_draft"])
    print(f"  Target:         α = {mean_alpha_target:.4f}")
    print(f"  MSE Draft:      α = {mean_alpha_mse:.4f}  (Δ from target: {abs(mean_alpha_mse - mean_alpha_target):.4f})")
    print(f"  Spectral Draft: α = {mean_alpha_sp:.4f}  (Δ from target: {abs(mean_alpha_sp - mean_alpha_target):.4f})")
    
    print("\n--- Spectral Energy Ratio (top-2 PCs, higher = more concentrated) ---")
    mean_er_target = np.mean(step_metrics["mse"]["energy_ratio_target"])
    mean_er_mse = np.mean(step_metrics["mse"]["energy_ratio_draft"])
    mean_er_sp = np.mean(step_metrics["spectral"]["energy_ratio_draft"])
    print(f"  Target:         {mean_er_target:.4f}")
    print(f"  MSE Draft:      {mean_er_mse:.4f}")
    print(f"  Spectral Draft: {mean_er_sp:.4f}")
    
    print("\n--- Principal Component Alignment (per-PC cosine with best match) ---")
    if step_metrics["mse"]["pc_cosines"]:
        n_pcs = len(step_metrics["mse"]["pc_cosines"][0])
        for pc_idx in range(n_pcs):
            cos_mse = np.mean([pcs[pc_idx] for pcs in step_metrics["mse"]["pc_cosines"]])
            cos_sp = np.mean([pcs[pc_idx] for pcs in step_metrics["spectral"]["pc_cosines"]])
            print(f"  PC-{pc_idx+1}: MSE={cos_mse:.4f}  Spectral={cos_sp:.4f}  Delta={cos_sp - cos_mse:+.4f}")
    
    print("\n--- Top-5 Token Match Rate ---")
    if top5_match["total"] > 0:
        rate_mse = top5_match["mse"] / top5_match["total"] * 100
        rate_sp = top5_match["spectral"] / top5_match["total"] * 100
        print(f"  MSE Model:      {rate_mse:.2f}%")
        print(f"  Spectral Model: {rate_sp:.2f}%")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("  - If Spectral Angle is large (>0.3 rad / >17°): subspaces are rotated")
    print("  - If α mismatch is large (>0.5): draft has wrong rank structure")
    print("  - If Energy Ratio mismatch: draft distributes energy differently")
    print("  - If PC-1 cosine is low (<0.5): even the dominant direction is wrong")
    print("=" * 70)


if __name__ == "__main__":
    run()
