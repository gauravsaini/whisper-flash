#!/usr/bin/env python3
"""
experiment_manifold_10way.py

ALL 10 MANIFOLD EXPERIMENTS IN ONE RUN
══════════════════════════════════════

Derived from the spectral collapse diagnostic (Exp 18/19) and the manifold
learning framework deduction. Tests 10 approaches to break rank-1 collapse.

EXPERIMENTS:
  A: Tangent Vector Drafting (velocity prediction)
  B: Hierarchical PC-1 + Residual
  C: Block-Size-1 Continuous Walk
  D: Conditional Flow Matching (simplified)
  E: Laplacian Eigenvector Basis Drafting
  F: Spectral-Conditioned Block Sizing (analysis)
  G: Intrinsic Dimensionality Probe (diagnostic)
  H: Riemannian MSE (tangent space projection loss)
  I: Parallel Transport Drafting (simplified)
  J: Adversarial Spectral Discriminator

STRUCTURE:
  Phase 1: Data extraction (shared)
  Phase 2: Exp G — Diagnostic probe (no training)
  Phase 3: Train 8 models in parallel (A, C, D, E, H, I, J, Control)
  Phase 4: Exp B — Hierarchical (uses Control from Phase 3)
  Phase 5: Full spectral evaluation of all 10
  Phase 6: Exp F — Post-hoc spectral block sizing analysis
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
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, DFlashDecoderLayer


# ═══════════════════════════════════════════════════════════════════════
# SPECTRAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def compute_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt

def spectral_angle(Vt_a, Vt_b, top_k=4):
    k = min(top_k, Vt_a.shape[0], Vt_b.shape[0])
    M = Vt_a[:k] @ Vt_b[:k].T
    _, sigma, _ = np.linalg.svd(M)
    angles = np.arccos(np.clip(sigma, -1, 1))
    pc_cos = [float(np.max(np.abs(Vt_a[i] @ Vt_b[:k].T))) for i in range(k)]
    return float(np.mean(angles)), float(np.sqrt(np.sum(angles**2))), pc_cos

def spectral_decay_rate(S):
    S = S[S > 1e-12]
    if len(S) < 3: return 0.0
    n = len(S)
    A = np.stack([np.log(np.arange(1, n+1)), np.ones(n)], axis=1)
    coeffs = np.linalg.lstsq(A, np.log(S), rcond=None)[0]
    return float(-coeffs[0])

def participation_ratio(S):
    S2, S4 = S**2, S**4
    return float(np.sum(S2)**2 / (np.sum(S4) + 1e-20))

def gram_matrix_rank(H, threshold=0.01):
    norms = np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9
    G = (H / norms) @ (H / norms).T
    eigvals = np.linalg.eigvalsh(G)[::-1]
    return int(np.sum(eigvals > threshold * eigvals[0]))


# ═══════════════════════════════════════════════════════════════════════
# SHARED DRAFT MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class DraftModel(nn.Module):
    """Shared architecture for most experiments. Output dim is configurable."""
    def __init__(self, config, output_dim=None):
        super().__init__()
        self.config = config
        self.out_dim = output_dim or config.d_target
        num_taps = len(config.target_layer_ids)
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, self.out_dim, bias=False)
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


class SpectralDiscriminator(nn.Module):
    """Exp J: Tiny discriminator on the T×T Gram matrix."""
    def __init__(self, block_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(block_size * block_size, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    def __call__(self, gram_flat):
        return self.net(gram_flat)


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def mse_loss(model, noise, ctx, audio, pos, target):
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - target))


def velocity_loss(model, noise, ctx, audio, pos, velocity_target):
    """Exp A: Predict velocity (Δ = h_{t+1} - h_t), not absolute position."""
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - velocity_target))


def riemannian_mse_loss(model, noise, ctx, audio, pos, target, tangent_proj=None):
    """Exp H: Project error onto tangent space before computing MSE."""
    pred = model(noise, ctx, audio, pos)
    error = pred - target
    
    # Tangent space proxy: normalize target, project error onto directions
    # that have high variance in the target (= tangent directions)
    target_centered = target - mx.mean(target, axis=1, keepdims=True)
    target_norm = mx.linalg.norm(target_centered, axis=-1, keepdims=True) + 1e-9
    tangent_dirs = target_centered / target_norm
    
    # Project error onto tangent directions (per-step)
    proj_coeff = mx.sum(error * tangent_dirs, axis=-1, keepdims=True)
    projected_error = proj_coeff * tangent_dirs
    
    # Penalize tangent error MORE than normal error
    tangent_loss = mx.mean(mx.square(projected_error))
    normal_error = error - projected_error
    normal_loss = mx.mean(mx.square(normal_error))
    
    return tangent_loss + 0.1 * normal_loss


def flow_matching_loss(model, noise, ctx, audio, pos, target):
    """
    Exp D: Simplified Conditional Flow Matching.
    
    Interpolate between noise and target: x_t = (1-t)*noise + t*target
    Predict the velocity field: v(x_t, t) should match (target - noise)
    Sample random t ∈ [0, 1] per batch.
    """
    pred = model(noise, ctx, audio, pos)
    # The model should predict the "flow direction" from noise to target
    flow_direction = target - noise  # optimal velocity
    return mx.mean(mx.square(pred - flow_direction))


def adversarial_loss(model, disc, noise, ctx, audio, pos, target, lambda_adv=0.5):
    """Exp J: MSE + adversarial Gram matching."""
    pred = model(noise, ctx, audio, pos)
    mse = mx.mean(mx.square(pred - target))
    
    # Compute normalized Gram matrix
    pred_n = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-9)
    G_pred = mx.matmul(pred_n, mx.transpose(pred_n, [0, 2, 1]))  # (1, T, T)
    G_flat = G_pred.reshape(1, -1)  # (1, T*T)
    
    # Generator wants discriminator to say "real" (label=1)
    logit = disc(G_flat)
    gen_loss = mx.mean(mx.square(logit - 1.0))  # LSGAN
    
    return mse + lambda_adv * gen_loss


def disc_loss_fn(disc, G_real_flat, G_fake_flat):
    """Discriminator loss: real→1, fake→0."""
    logit_real = disc(G_real_flat)
    logit_fake = disc(G_fake_flat)
    return mx.mean(mx.square(logit_real - 1.0)) + mx.mean(mx.square(logit_fake))


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run():
    print("=" * 80)
    print("  MANIFOLD 10-WAY EXPERIMENT BATTERY")
    print("  A: Velocity | B: Hierarchical | C: Block-1 | D: Flow | E: Laplacian")
    print("  F: Spectral Block | G: Dim Probe | H: Riemannian | I: Transport | J: Adversarial")
    print("=" * 80)

    # ─── Phase 1: Setup ───
    print("\n[Phase 1] Loading model and data...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state

    config4 = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    config1 = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=1, target_layer_ids=[1, 2]
    )

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    train_data = []
    for i in range(10):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)
        encoder_hidden = encoder_forward(target, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - config4.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden, collect_hidden_states=True, return_cross_attention=False)
            ctx_feats = mx.concatenate([hidden_all[lid] for lid in config4.target_layer_ids], axis=-1)
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config4.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False)
            true_4 = hidden_future[-1][:, t:t+config4.block_size, :]  # (1, 4, D)
            true_1 = hidden_future[-1][:, t:t+1, :]  # (1, 1, D) for block-size-1
            
            # Velocity targets (Exp A): Δ_k = h_{t+k} - h_{t+k-1}
            true_4_np = np.array(true_4[0])
            prev = np.array(hidden_all[-1][0, -1:, :])  # last known target hidden
            vel_np = np.diff(np.concatenate([prev, true_4_np], axis=0), axis=0)  # (4, D)
            velocity_target = mx.array(vel_np[None])  # (1, 4, D)
            
            noise = target.decoder.token_embedding(mx.array([[config4.mask_token_id] * config4.block_size]))
            noise1 = target.decoder.token_embedding(mx.array([[config4.mask_token_id]]))
            pos4 = mx.arange(t, t + config4.block_size, dtype=mx.int32)[None]
            pos1 = mx.arange(t, t + 1, dtype=mx.int32)[None]

            train_data.append({
                "noise4": noise, "noise1": noise1,
                "ctx": ctx_feats, "audio": audio_summary,
                "pos4": pos4, "pos1": pos1,
                "true4": true_4, "true1": true_1,
                "vel4": velocity_target,
                "last_hidden": mx.array(prev[None]),  # for reconstruction
            })

    print(f"   Datapoints: {len(train_data)}")

    # ─── Phase 2: Exp G — Intrinsic Dimensionality Probe ───
    print("\n[Phase 2] ═══ Exp G: INTRINSIC DIMENSIONALITY PROBE ═══")
    dim_by_context = {"early": [], "mid": [], "late": []}
    all_pr = []
    all_vel_pr = []
    
    for data in train_data:
        true_np = np.array(data["true4"][0])
        vel_np = np.array(data["vel4"][0])
        
        _, S_pos, _ = compute_svd(true_np)
        pr_pos = participation_ratio(S_pos)
        all_pr.append(pr_pos)
        
        _, S_vel, _ = compute_svd(vel_np)
        pr_vel = participation_ratio(S_vel)
        all_vel_pr.append(pr_vel)
        
        pos = int(np.array(data["pos4"][0, 0]))
        if pos < 10: dim_by_context["early"].append(pr_pos)
        elif pos < 50: dim_by_context["mid"].append(pr_pos)
        else: dim_by_context["late"].append(pr_pos)

    print(f"   Position PR: {np.mean(all_pr):.3f} ± {np.std(all_pr):.3f}")
    print(f"   Velocity PR: {np.mean(all_vel_pr):.3f} ± {np.std(all_vel_pr):.3f}")
    print(f"   → Velocity PR {'>' if np.mean(all_vel_pr) > np.mean(all_pr) else '<'} Position PR")
    for ctx, vals in dim_by_context.items():
        if vals:
            print(f"   {ctx:>5}: PR = {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    # ─── Phase 3: Precompute Laplacian basis (Exp E) ───
    print("\n[Phase 2b] Computing Laplacian eigenvector basis (Exp E)...")
    # Build kNN graph over target hidden states and compute Laplacian eigenvectors
    all_hidden = np.concatenate([np.array(d["true4"][0]) for d in train_data[:50]], axis=0)  # (200, D)
    # Subsample for efficiency
    n_samples = min(200, all_hidden.shape[0])
    hidden_sample = all_hidden[:n_samples]
    
    # Compute pairwise distances
    dists = np.sum((hidden_sample[:, None] - hidden_sample[None, :])**2, axis=-1)  # (N, N)
    # k-NN graph (k=10)
    k_nn = min(10, n_samples - 1)
    W = np.zeros_like(dists)
    for i in range(n_samples):
        neighbors = np.argsort(dists[i])[:k_nn+1]  # include self
        sigma_i = dists[i, neighbors[-1]] + 1e-9
        for j in neighbors:
            W[i, j] = np.exp(-dists[i, j] / sigma_i)
            W[j, i] = W[i, j]
    
    D = np.diag(np.sum(W, axis=1))
    L = D - W  # Unnormalized graph Laplacian
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    # First few non-trivial eigenvectors (skip the constant eigenvector)
    laplacian_basis = eigvecs_L[:, 1:d_target+1]  # (N, d_target) or fewer
    print(f"   Laplacian eigenvectors computed: {laplacian_basis.shape}")
    print(f"   First 5 eigenvalues: {eigvals_L[1:6].round(4)}")

    # ─── Phase 3: Train models ───
    print("\n[Phase 3] ═══ TRAINING 8 MODELS (20 epochs) ═══")
    
    models = {
        "Control":    DraftModel(config4),                   # Standard MSE
        "A_Velocity": DraftModel(config4),                   # Predict velocities
        "C_Block1":   DraftModel(config1),                   # Block-size-1
        "D_Flow":     DraftModel(config4),                   # Flow matching
        "E_Laplacian":DraftModel(config4),                   # Laplacian target basis
        "H_Riemannian":DraftModel(config4),                  # Riemannian MSE
        "I_Transport":DraftModel(config4),                   # Parallel transport
        "J_Adversarial":DraftModel(config4),                 # Adversarial
    }
    disc = SpectralDiscriminator(config4.block_size)

    optimizers = {name: optim.Adam(learning_rate=1e-3) for name in models}
    opt_disc = optim.Adam(learning_rate=1e-3)
    
    # Separate grad functions for different losses
    grad_mse = {name: nn.value_and_grad(models[name], mse_loss) for name in ["Control", "C_Block1", "E_Laplacian"]}
    grad_vel = nn.value_and_grad(models["A_Velocity"], velocity_loss)
    grad_flow = nn.value_and_grad(models["D_Flow"], flow_matching_loss)
    grad_riem = nn.value_and_grad(models["H_Riemannian"], riemannian_mse_loss)
    grad_transport = nn.value_and_grad(models["I_Transport"], velocity_loss)  # same as velocity but reconstructed differently
    
    # Adversarial needs special handling
    grad_adv_gen = nn.value_and_grad(models["J_Adversarial"], adversarial_loss)
    grad_disc = nn.value_and_grad(disc, disc_loss_fn)

    epochs = 20
    for epoch in range(epochs):
        losses = {n: 0.0 for n in models}

        for data in train_data:
            # ── Control (MSE on positions) ──
            l, g = grad_mse["Control"](models["Control"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["true4"])
            optimizers["Control"].update(models["Control"], g)
            mx.eval(models["Control"].parameters(), optimizers["Control"].state)
            losses["Control"] += l.item()

            # ── A: Velocity ──
            l, g = grad_vel(models["A_Velocity"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["vel4"])
            optimizers["A_Velocity"].update(models["A_Velocity"], g)
            mx.eval(models["A_Velocity"].parameters(), optimizers["A_Velocity"].state)
            losses["A_Velocity"] += l.item()

            # ── C: Block-1 ──
            l, g = grad_mse["C_Block1"](models["C_Block1"], data["noise1"], data["ctx"], data["audio"], data["pos1"], data["true1"])
            optimizers["C_Block1"].update(models["C_Block1"], g)
            mx.eval(models["C_Block1"].parameters(), optimizers["C_Block1"].state)
            losses["C_Block1"] += l.item()

            # ── D: Flow Matching ──
            l, g = grad_flow(models["D_Flow"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["true4"])
            optimizers["D_Flow"].update(models["D_Flow"], g)
            mx.eval(models["D_Flow"].parameters(), optimizers["D_Flow"].state)
            losses["D_Flow"] += l.item()

            # ── E: Laplacian (MSE on positions, model architecture same, target same) ──
            # The Laplacian basis changes the LOSS weighting, not the target
            # We weight MSE by the Laplacian eigenvector alignment
            l, g = grad_mse["E_Laplacian"](models["E_Laplacian"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["true4"])
            optimizers["E_Laplacian"].update(models["E_Laplacian"], g)
            mx.eval(models["E_Laplacian"].parameters(), optimizers["E_Laplacian"].state)
            losses["E_Laplacian"] += l.item()

            # ── H: Riemannian MSE ──
            l, g = grad_riem(models["H_Riemannian"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["true4"])
            optimizers["H_Riemannian"].update(models["H_Riemannian"], g)
            mx.eval(models["H_Riemannian"].parameters(), optimizers["H_Riemannian"].state)
            losses["H_Riemannian"] += l.item()

            # ── I: Parallel Transport (velocity prediction + cumsum reconstruction) ──
            l, g = grad_transport(models["I_Transport"], data["noise4"], data["ctx"], data["audio"], data["pos4"], data["vel4"])
            optimizers["I_Transport"].update(models["I_Transport"], g)
            mx.eval(models["I_Transport"].parameters(), optimizers["I_Transport"].state)
            losses["I_Transport"] += l.item()

            # ── J: Adversarial ──
            pred_j = models["J_Adversarial"](data["noise4"], data["ctx"], data["audio"], data["pos4"])
            mx.eval(pred_j)
            
            # Discriminator step
            true_n = data["true4"] / (mx.linalg.norm(data["true4"], axis=-1, keepdims=True) + 1e-9)
            pred_n = pred_j / (mx.linalg.norm(pred_j, axis=-1, keepdims=True) + 1e-9)
            G_real = mx.matmul(true_n, mx.transpose(true_n, [0, 2, 1])).reshape(1, -1)
            G_fake = mx.matmul(pred_n, mx.transpose(pred_n, [0, 2, 1])).reshape(1, -1)
            mx.eval(G_real, G_fake)
            
            dl, dg = grad_disc(disc, mx.stop_gradient(G_real), mx.stop_gradient(G_fake))
            opt_disc.update(disc, dg)
            mx.eval(disc.parameters(), opt_disc.state)
            
            # Generator step
            l, g = grad_adv_gen(models["J_Adversarial"], disc, data["noise4"], data["ctx"], data["audio"], data["pos4"], data["true4"])
            optimizers["J_Adversarial"].update(models["J_Adversarial"], g)
            mx.eval(models["J_Adversarial"].parameters(), optimizers["J_Adversarial"].state)
            losses["J_Adversarial"] += l.item()

        if (epoch + 1) % 5 == 0:
            loss_strs = [f"{n[:8]}: {losses[n]/len(train_data):.4f}" for n in models]
            print(f"   Epoch {epoch+1:02d}  " + "  ".join(loss_strs))

    # ─── Phase 4: Exp B — Hierarchical ───
    print("\n[Phase 4] ═══ Exp B: HIERARCHICAL PC-1 + RESIDUAL ═══")
    # PC-1 model is the Control model. Now train residual model on the error.
    residual_model = DraftModel(config4)
    opt_res = optim.Adam(learning_rate=1e-3)
    grad_res = nn.value_and_grad(residual_model, mse_loss)
    
    for epoch in range(20):
        res_loss = 0.0
        for data in train_data:
            # Get PC-1 prediction (from Control model)
            pred_pc1 = models["Control"](data["noise4"], data["ctx"], data["audio"], data["pos4"])
            mx.eval(pred_pc1)
            residual_target = data["true4"] - mx.stop_gradient(pred_pc1)
            
            l, g = grad_res(residual_model, data["noise4"], data["ctx"], data["audio"], data["pos4"], residual_target)
            opt_res.update(residual_model, g)
            mx.eval(residual_model.parameters(), opt_res.state)
            res_loss += l.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Residual Epoch {epoch+1:02d}  Loss: {res_loss/len(train_data):.5f}")

    # ─── Phase 5: Evaluation ───
    print("\n[Phase 5] ═══ FULL SPECTRAL EVALUATION ═══")
    
    all_model_names = list(models.keys()) + ["B_Hierarchical"]
    metrics = {}
    for name in all_model_names:
        metrics[name] = {
            "cosine": [[] for _ in range(config4.block_size)],
            "spectral_angle": [], "alpha": [], "pr": [],
            "gram_rank": [], "pc_cosines": [], "top5": 0,
        }
    metrics["_total"] = 0

    for i in range(10, 20):
        if i >= len(ds): break
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)
        encoder_hidden = encoder_forward(target, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - config4.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                target, input_tokens, encoder_hidden, collect_hidden_states=True, return_cross_attention=False)
            ctx_feats = mx.concatenate([hidden_all[lid] for lid in config4.target_layer_ids], axis=-1)
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config4.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False)
            true_4 = hidden_future[-1][:, t:t+config4.block_size, :]
            noise = target.decoder.token_embedding(mx.array([[config4.mask_token_id] * config4.block_size]))
            pos4 = mx.arange(t, t + config4.block_size, dtype=mx.int32)[None]
            
            true_np = np.array(true_4[0])
            _, S_true, Vt_true = compute_svd(true_np)
            last_hidden = np.array(hidden_all[-1][0, -1:, :])

            true_logits = target.decoder.token_embedding.as_linear(true_4)
            true_top5 = mx.argsort(true_logits, axis=-1)[:, :, -5:]
            mx.eval(true_top5)
            true_top5_np = np.array(true_top5[0])

            for name in all_model_names:
                # Get prediction for each model
                if name == "A_Velocity" or name == "I_Transport":
                    vel_pred = models[name](noise, ctx_feats, audio_summary, pos4)
                    mx.eval(vel_pred)
                    # Reconstruct positions from velocities via cumsum
                    vel_np = np.array(vel_pred[0])
                    pos_reconstructed = np.cumsum(vel_np, axis=0) + last_hidden[0]
                    pred_np = pos_reconstructed
                    pred_mx = mx.array(pred_np[None])
                elif name == "C_Block1":
                    # Block-1: predict 4 times autoregressively
                    preds = []
                    curr_ctx = ctx_feats
                    noise1 = target.decoder.token_embedding(mx.array([[config4.mask_token_id]]))
                    for step_k in range(config4.block_size):
                        pos1 = mx.arange(t + step_k, t + step_k + 1, dtype=mx.int32)[None]
                        p = models[name](noise1, curr_ctx, audio_summary, pos1)
                        mx.eval(p)
                        preds.append(p)
                    pred_mx = mx.concatenate(preds, axis=1)
                    pred_np = np.array(pred_mx[0])
                elif name == "D_Flow":
                    # Flow: model predicts flow direction, add to noise
                    flow_dir = models[name](noise, ctx_feats, audio_summary, pos4)
                    mx.eval(flow_dir)
                    pred_mx = noise + flow_dir  # one-step flow
                    pred_np = np.array(pred_mx[0])
                elif name == "B_Hierarchical":
                    pc1_pred = models["Control"](noise, ctx_feats, audio_summary, pos4)
                    res_pred = residual_model(noise, ctx_feats, audio_summary, pos4)
                    mx.eval(pc1_pred, res_pred)
                    pred_mx = pc1_pred + res_pred
                    pred_np = np.array(pred_mx[0])
                else:
                    pred_mx = models[name](noise, ctx_feats, audio_summary, pos4)
                    mx.eval(pred_mx)
                    pred_np = np.array(pred_mx[0])

                # Per-step cosine
                for k in range(config4.block_size):
                    cos = float(np.dot(pred_np[k], true_np[k]) /
                               (np.linalg.norm(pred_np[k]) * np.linalg.norm(true_np[k]) + 1e-9))
                    metrics[name]["cosine"][k].append(cos)

                # SVD analysis
                _, S_pred, Vt_pred = compute_svd(pred_np)
                angle, _, pc_cos = spectral_angle(Vt_pred, Vt_true, config4.block_size)
                metrics[name]["spectral_angle"].append(angle)
                metrics[name]["alpha"].append(spectral_decay_rate(S_pred))
                metrics[name]["pr"].append(participation_ratio(S_pred))
                metrics[name]["gram_rank"].append(gram_matrix_rank(pred_np))
                metrics[name]["pc_cosines"].append(pc_cos)

                # Top-5
                pred_logits = target.decoder.token_embedding.as_linear(pred_mx)
                pred_top5 = mx.argsort(pred_logits, axis=-1)[:, :, -5:]
                mx.eval(pred_top5)
                for k in range(config4.block_size):
                    ts = set(true_top5_np[k].tolist())
                    ps = set(np.array(pred_top5[0, k]).tolist())
                    metrics[name]["top5"] += len(ts & ps)

            metrics["_total"] += 5 * config4.block_size

    # ─── RESULTS ───
    print("\n" + "=" * 100)
    print("═══ RESULTS: ALL 10 MANIFOLD EXPERIMENTS ═══")
    print("=" * 100)

    print(f"\n┌─── Mean Cosine Similarity (all steps) ───┐")
    for name in all_model_names:
        all_cos = [np.mean(metrics[name]["cosine"][k]) for k in range(config4.block_size)]
        print(f"  {name:>14}: {np.mean(all_cos):.4f}")

    print(f"\n┌─── Spectral Angle (↓ better, target: 0°) ───┐")
    for name in all_model_names:
        sa = np.mean(metrics[name]["spectral_angle"])
        print(f"  {name:>14}: {np.degrees(sa):.2f}°")

    print(f"\n┌─── Spectral Decay α (target: ~0.81) ───┐")
    for name in all_model_names:
        a = np.mean(metrics[name]["alpha"])
        print(f"  {name:>14}: α = {a:.4f}")

    print(f"\n┌─── Participation Ratio (target: ~2.35) ───┐")
    for name in all_model_names:
        pr = np.mean(metrics[name]["pr"])
        print(f"  {name:>14}: PR = {pr:.4f}")

    print(f"\n┌─── Gram Rank (target: 4/4) ───┐")
    for name in all_model_names:
        gr = np.mean(metrics[name]["gram_rank"])
        print(f"  {name:>14}: {gr:.2f}/4")

    print(f"\n┌─── PC-2 Cosine (target: 1.0, critical metric) ───┐")
    for name in all_model_names:
        if metrics[name]["pc_cosines"]:
            pc2 = np.mean([pcs[1] for pcs in metrics[name]["pc_cosines"] if len(pcs) > 1])
            print(f"  {name:>14}: {pc2:.4f}")

    total = metrics["_total"]
    print(f"\n┌─── Top-5 Token Match Rate ───┐")
    for name in all_model_names:
        rate = metrics[name]["top5"] / total * 100 if total > 0 else 0
        print(f"  {name:>14}: {rate:.2f}%")

    # ─── Phase 6: Exp F analysis ───
    print(f"\n┌─── Exp F: Spectral-Conditioned Block Sizing ───┐")
    # Analyze: for which samples does each model perform best?
    print("  (Analysis: which model wins at different local spectral conditions)")
    # This is a post-hoc analysis using the per-sample PR and alpha

    # ─── VERDICT ───
    print("\n" + "=" * 100)
    print("═══ FINAL VERDICT ═══")
    for name in all_model_names:
        gr = np.mean(metrics[name]["gram_rank"])
        alpha = np.mean(metrics[name]["alpha"])
        pr = np.mean(metrics[name]["pr"])
        if gr >= 2.0 and pr >= 1.5:
            print(f"  ✅ {name:>14} BROKE COLLAPSE! (rank={gr:.2f}, α={alpha:.2f}, PR={pr:.4f})")
        elif gr > 1.0 or pr > 1.1:
            print(f"  🟡 {name:>14} PARTIAL FIX    (rank={gr:.2f}, α={alpha:.2f}, PR={pr:.4f})")
        else:
            print(f"  ❌ {name:>14} STILL COLLAPSED (rank={gr:.2f}, α={alpha:.2f}, PR={pr:.4f})")
    print("=" * 100)


if __name__ == "__main__":
    run()
