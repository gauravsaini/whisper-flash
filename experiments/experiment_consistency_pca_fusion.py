#!/usr/bin/env python3
"""
experiment_consistency_pca_fusion.py

Phase-2 Item 24: Consistency-Model Continuous Drafting + PCA-Subspace Fusion

HYPOTHESIS: Training a consistency-model draft network to predict in the PCA
subspace (R=64) rather than full D=384 will combine:
  - Consistency model's drift-immune trajectory generation (0.72 cosine from Exp 9)
  - PCA's 83% parameter savings and structural regularization (0.5474 cosine from Exp 6)
  - Bottleneck-free MLP architecture (no cross-attention, 0.78 cosine from Timeline K)

The PCA projection restricts consistency denoising to the principal semantic
directions of the target manifold, inherently filtering high-frequency noise
and acting as a structural regularizer ON TOP OF the temporal regularizer
provided by consistency training.

METRICS TO BEAT:
  - Consistency (2-shot) in full space: cosine 0.7225, greedy 15.31%, top-5 25.51%
  - PCA subspace (MSE): cosine 0.5474, top-5 5.27%
  - Bottleneck-free: cosine 0.7837
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
# 1. Consistency Draft Model operating in PCA Subspace
# ---------------------------------------------------------------------------

class ConsistencyPCADraftModel(nn.Module):
    """
    Consistency model that denoises in the PCA subspace (R dimensions)
    instead of the full D_target dimensions.

    Key architectural choice: the model's continuous_head outputs R-dimensional
    subspace coordinates. The PCA up-projection (fixed, frozen) reconstructs
    back to D_target only at evaluation time.

    This means:
      - Training operates in R-dimensional space (cheaper, regularized)
      - Consistency denoising is on the PCA subspace manifold
      - The fixed PCA basis acts as a structural prior
    """
    def __init__(self, config: WhisperDFlashConfig, pca_rank: int = 64):
        super().__init__()
        self.config = config
        self.pca_rank = pca_rank

        # Input projection: noisy PCA coordinates → d_draft
        self.input_proj = nn.Linear(pca_rank, config.d_draft, bias=False)

        # Context projections (tapped layers)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)

        # Noise level (sigma) embedding
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, config.d_draft)
        )

        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)

        # KEY: Output head predicts PCA subspace coordinates (R dims, not D_target)
        self.continuous_head = nn.Linear(config.d_draft, pca_rank, bias=False)
        self.target_layer_ids = config.target_layer_ids

    def __call__(self, noisy_z, target_hidden, audio_summary, position_ids, sigma):
        """
        Args:
            noisy_z: (batch, block_size, pca_rank) — noisy PCA subspace coords
            target_hidden: (batch, seq_len, num_taps * d_target)
            audio_summary: (batch, 1, d_target)
            position_ids: (batch, block_size)
            sigma: (batch, 1) — noise level
        Returns:
            F_out: (batch, block_size, pca_rank) — predicted clean PCA coords
        """
        x = self.input_proj(noisy_z) + self.pos_embed(position_ids)

        # Embed noise level
        ln_sigma = mx.log(mx.clip(sigma, 1e-9, 1e9))
        if len(ln_sigma.shape) == 1:
            ln_sigma = ln_sigma[:, None]
        sigma_emb = self.sigma_mlp(ln_sigma)

        x = x + sigma_emb[:, None, :]

        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)

        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)

        x = self.norm(x)
        F_out = self.continuous_head(x)
        return F_out


# ---------------------------------------------------------------------------
# 2. Bottleneck-Free Consistency-PCA Model (MLP, no cross-attention)
# ---------------------------------------------------------------------------

class BottleneckFreeConsistencyPCAModel(nn.Module):
    """
    Combines all three mechanisms:
      1. Consistency model (temporal regularizer, drift-immune)
      2. PCA subspace (structural regularizer, 83% param savings)
      3. Bottleneck-free MLP (no cross-attention, fastest inference)

    Takes only the last target hidden state + audio summary and predicts
    block_size PCA subspace coordinates via a simple MLP with sigma conditioning.
    """
    def __init__(self, config: WhisperDFlashConfig, pca_rank: int = 64):
        super().__init__()
        self.config = config
        self.pca_rank = pca_rank

        num_taps = len(config.target_layer_ids)
        # Context: last target hidden (taps * d_target) + audio summary (d_target)
        ctx_dim = num_taps * config.d_target + config.d_target
        self.ctx_proj = nn.Linear(ctx_dim, config.d_draft)

        # Sigma embedding
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, config.d_draft)
        )

        # MLP predicting block_size PCA coordinates
        self.mlp = nn.Sequential(
            nn.Linear(config.d_draft, config.d_draft * 2),
            nn.GELU(),
            nn.Linear(config.d_draft * 2, config.block_size * pca_rank)
        )

        self.norm = nn.LayerNorm(pca_rank)
        self.target_layer_ids = config.target_layer_ids

    def __call__(self, noisy_z, target_hidden, audio_summary, position_ids, sigma):
        """
        Args:
            noisy_z: unused for bottleneck-free (we generate from context)
            target_hidden: (bsz, seq_len, taps * d_target)
            audio_summary: (bsz, 1, d_target)
            position_ids: unused
            sigma: (bsz, 1) noise level
        Returns:
            (bsz, block_size, pca_rank) — predicted clean PCA coordinates
        """
        last_target = target_hidden[:, -1:, :]
        ctx_input = mx.concatenate([last_target, audio_summary], axis=-1)

        hidden = self.ctx_proj(ctx_input)  # (bsz, 1, d_draft)

        # Add sigma conditioning
        ln_sigma = mx.log(mx.clip(sigma, 1e-9, 1e9))
        if len(ln_sigma.shape) == 1:
            ln_sigma = ln_sigma[:, None]
        sigma_emb = self.sigma_mlp(ln_sigma)
        hidden = hidden + sigma_emb[:, None, :]

        predicted = self.mlp(hidden)  # (bsz, 1, block_size * pca_rank)

        bsz = predicted.shape[0]
        predicted = predicted.reshape(bsz, self.config.block_size, self.pca_rank)
        predicted = self.norm(predicted)
        return predicted


# ---------------------------------------------------------------------------
# 3. Consistency Prediction Wrapper (operates in PCA subspace)
# ---------------------------------------------------------------------------

def get_consistency_prediction_pca(model, z, target_hidden, audio_summary, position_ids, sigma):
    """Consistency model boundary condition in PCA subspace."""
    sigma_min = 0.002

    if not isinstance(sigma, mx.array):
        sigma = mx.array(sigma)
    if len(sigma.shape) == 1:
        sigma = sigma[:, None]
    elif len(sigma.shape) == 0:
        sigma = sigma[None, None]

    c_skip = (sigma_min ** 2) / ((sigma - sigma_min) ** 2 + sigma_min ** 2)
    c_out = (sigma - sigma_min) / mx.sqrt((sigma - sigma_min) ** 2 + sigma_min ** 2)

    c_skip = c_skip[:, :, None]  # (batch, 1, 1) for broadcasting
    c_out = c_out[:, :, None]

    F_out = model(z, target_hidden, audio_summary, position_ids, sigma)

    return c_skip * z + c_out * F_out


# ---------------------------------------------------------------------------
# 4. Consistency Training Loss (PCA subspace)
# ---------------------------------------------------------------------------

def consistency_loss_pca(
    online_model, target_model, clean_z, ctx, audio, pos, sigma_n, sigma_np1
):
    """Consistency training loss operating entirely in PCA subspace."""
    z_noise = mx.random.normal(clean_z.shape)

    z_n = clean_z + sigma_n * z_noise
    z_np1 = clean_z + sigma_np1 * z_noise

    batch_size = clean_z.shape[0]
    sigma_n_arr = mx.full((batch_size, 1), sigma_n)
    sigma_np1_arr = mx.full((batch_size, 1), sigma_np1)

    pred_online = get_consistency_prediction_pca(
        online_model, z_np1, ctx, audio, pos, sigma_np1_arr
    )
    pred_target = get_consistency_prediction_pca(
        target_model, z_n, ctx, audio, pos, sigma_n_arr
    )

    return mx.mean(mx.square(pred_online - pred_target))


# ---------------------------------------------------------------------------
# 5. Parameter Utilities
# ---------------------------------------------------------------------------

def copy_parameters(model):
    from mlx.utils import tree_map
    return tree_map(lambda x: mx.array(x), model.parameters())

def update_target_parameters(online_model, target_model, ema_mu=0.95):
    from mlx.utils import tree_map
    def ema_update(target_param, online_param):
        return ema_mu * target_param + (1 - ema_mu) * online_param
    new_params = tree_map(ema_update, target_model.parameters(), online_model.parameters())
    target_model.update(new_params)

def get_sigma_schedule(num_steps=10, sigma_min=0.002, sigma_max=80.0):
    return [sigma_min * ((sigma_max / sigma_min) ** (i / (num_steps - 1))) for i in range(num_steps)]


# ---------------------------------------------------------------------------
# 6. Main Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT: Consistency-Model + PCA-Subspace Fusion (Phase-2 Item 24)")
    print("=" * 70)

    print("\nLoading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    pca_rank = 64

    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )

    # --- Instantiate all 4 models ---
    # A. MSE Baseline (full-rank, standard ContinuousDraftModel)
    baseline_mse = ContinuousDraftModel(config)

    # B. Consistency in full D space (reproduction of Exp 9 baseline)
    from experiment_consistency_drafting import ConsistencyDraftModel
    consistency_full_online = ConsistencyDraftModel(config)
    consistency_full_target = ConsistencyDraftModel(config)

    # C. Consistency in PCA subspace (NEW)
    consistency_pca_online = ConsistencyPCADraftModel(config, pca_rank=pca_rank)
    consistency_pca_target = ConsistencyPCADraftModel(config, pca_rank=pca_rank)

    # D. Bottleneck-Free Consistency PCA (NEW — the triple fusion)
    bf_pca_online = BottleneckFreeConsistencyPCAModel(config, pca_rank=pca_rank)
    bf_pca_target = BottleneckFreeConsistencyPCAModel(config, pca_rank=pca_rank)

    # --- Force initialize all models ---
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    sigma_init = mx.ones((1, 1))
    pca_noise_init = mx.zeros((1, config.block_size, pca_rank))

    _ = baseline_mse(noise_init, ctx_init, audio_init, pos_init)
    _ = consistency_full_online(noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = consistency_full_target(noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = consistency_pca_online(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = consistency_pca_target(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = bf_pca_online(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = bf_pca_target(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)

    # Sync target network params
    consistency_full_target.update(copy_parameters(consistency_full_online))
    consistency_pca_target.update(copy_parameters(consistency_pca_online))
    bf_pca_target.update(copy_parameters(bf_pca_online))

    # --- Count parameters ---
    from mlx.utils import tree_flatten
    def count_params(model):
        return sum(x.size for _, x in tree_flatten(model.parameters()))

    params_baseline = count_params(baseline_mse)
    params_full = count_params(consistency_full_online)
    params_pca = count_params(consistency_pca_online)
    params_bf_pca = count_params(bf_pca_online)

    print(f"\n--- Parameter Counts ---")
    print(f"MSE Baseline (full D):          {params_baseline:,}")
    print(f"Consistency (full D):           {params_full:,}")
    print(f"Consistency (PCA R={pca_rank}):       {params_pca:,}")
    print(f"BF-Consistency-PCA (R={pca_rank}):    {params_bf_pca:,}")
    print(f"PCA Param Savings vs Full:      {(1 - params_pca/params_full)*100:.1f}%")
    print(f"BF-PCA Savings vs Full:         {(1 - params_bf_pca/params_full)*100:.1f}%")

    # --- Load dataset ---
    print("\nLoading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    print("Pre-extracting dataset context features (10 training samples)...")
    data_tensors = []
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

        for t in range(1, labels.shape[1] - config.block_size, 3):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )

            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)

            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]

            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            data_tensors.append({
                "noise": noise,
                "ctx": ctx_feats,
                "audio": audio_summary,
                "pos": pos_ids,
                "true_hidden": true_hidden
            })

    print(f"Pre-extraction complete. Extracted {len(data_tensors)} train points.")

    # --- Compute PCA/SVD basis from training hidden states ---
    print("\nComputing PCA/SVD subspace components...")
    all_true_h = np.concatenate([np.array(d["true_hidden"]) for d in data_tensors], axis=0)
    M_samples, B_block, D_dim = all_true_h.shape
    X = all_true_h.reshape(-1, D_dim)

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    V = Vt[:pca_rank, :].T  # (D_dim, pca_rank)
    mean_mx = mx.array(mean)  # (1, D_dim)
    V_mx = mx.array(V)       # (D_dim, pca_rank)

    # Compute explained variance
    total_var = np.sum(S ** 2)
    explained_var = np.sum(S[:pca_rank] ** 2) / total_var * 100
    print(f"PCA Rank: {pca_rank}, Explained Variance: {explained_var:.1f}%")
    print(f"Subspace dimension: {pca_rank} (compressed from {d_target})")

    # Project true hidden states to PCA subspace coordinates
    for d in data_tensors:
        true_h = d["true_hidden"]
        d["true_z"] = (true_h - mean_mx) @ V_mx  # (1, B, pca_rank)

    # --- Training ---
    epochs = 15
    sigmas = get_sigma_schedule(num_steps=10, sigma_min=0.002, sigma_max=80.0)

    # A. Train MSE Baseline
    print("\n--- Training Model A: MSE Baseline (full D) ---")
    opt_base = optim.Adam(learning_rate=1e-3)
    def baseline_loss_fn(model, noise, ctx, audio, pos, true_hidden):
        pred = model(noise, ctx, audio, pos)
        return mx.mean(mx.square(pred - true_hidden))
    grad_base = nn.value_and_grad(baseline_mse, baseline_loss_fn)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            loss, grads = grad_base(
                baseline_mse, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_base.update(baseline_mse, grads)
            mx.eval(baseline_mse.parameters(), opt_base.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"  Trained in {time.time() - t0:.1f}s")

    # B. Train Consistency (full D space)
    print("\n--- Training Model B: Consistency (full D) ---")
    from experiment_consistency_drafting import consistency_loss_fn as full_consistency_loss
    opt_full = optim.Adam(learning_rate=1e-3)
    grad_full = nn.value_and_grad(consistency_full_online, full_consistency_loss)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            n = np.random.randint(0, len(sigmas) - 1)
            loss, grads = grad_full(
                consistency_full_online, consistency_full_target,
                data["true_hidden"], data["ctx"], data["audio"], data["pos"],
                sigmas[n], sigmas[n+1]
            )
            opt_full.update(consistency_full_online, grads)
            update_target_parameters(consistency_full_online, consistency_full_target, ema_mu=0.95)
            mx.eval(consistency_full_online.parameters(), consistency_full_target.parameters(), opt_full.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"  Trained in {time.time() - t0:.1f}s")

    # C. Train Consistency-PCA (NEW)
    print("\n--- Training Model C: Consistency-PCA (R=64) [NEW] ---")
    opt_pca = optim.Adam(learning_rate=1e-3)
    grad_pca = nn.value_and_grad(consistency_pca_online, consistency_loss_pca)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            n = np.random.randint(0, len(sigmas) - 1)
            loss, grads = grad_pca(
                consistency_pca_online, consistency_pca_target,
                data["true_z"], data["ctx"], data["audio"], data["pos"],
                sigmas[n], sigmas[n+1]
            )
            opt_pca.update(consistency_pca_online, grads)
            update_target_parameters(consistency_pca_online, consistency_pca_target, ema_mu=0.95)
            mx.eval(consistency_pca_online.parameters(), consistency_pca_target.parameters(), opt_pca.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"  Trained in {time.time() - t0:.1f}s")

    # D. Train Bottleneck-Free Consistency-PCA (NEW — triple fusion)
    print("\n--- Training Model D: BF-Consistency-PCA (R=64, no cross-attn) [NEW] ---")
    opt_bf = optim.Adam(learning_rate=1e-3)
    grad_bf = nn.value_and_grad(bf_pca_online, consistency_loss_pca)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            n = np.random.randint(0, len(sigmas) - 1)
            loss, grads = grad_bf(
                bf_pca_online, bf_pca_target,
                data["true_z"], data["ctx"], data["audio"], data["pos"],
                sigmas[n], sigmas[n+1]
            )
            opt_bf.update(bf_pca_online, grads)
            update_target_parameters(bf_pca_online, bf_pca_target, ema_mu=0.95)
            mx.eval(bf_pca_online.parameters(), bf_pca_target.parameters(), opt_bf.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"  Trained in {time.time() - t0:.1f}s")

    # --- Evaluation ---
    print("\n" + "=" * 70)
    print("EVALUATING on held-out validation samples (samples 10 to 19)...")
    print("=" * 70)

    sigma_max = sigmas[-1]
    sigma_mid = 10.0

    models_list = [
        ("MSE Baseline", "base"),
        ("Consistency Full-D (2-shot)", "full"),
        ("Consistency-PCA (2-shot)", "pca"),
        ("BF-Consistency-PCA (2-shot)", "bf_pca"),
    ]

    metrics = {}
    for _, key in models_list:
        metrics[key] = {
            "sim": [], "acc": [], "top5_acc": [],
            "sim_per_step": [[] for _ in range(config.block_size)]
        }

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

        for t in range(1, labels.shape[1] - config.block_size):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)

            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]

            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            # A. Baseline MSE prediction
            pred_base = baseline_mse(noise, ctx_feats, audio_summary, pos_ids)

            # B. Consistency Full-D (2-shot refinement)
            z1 = mx.random.normal(true_hidden.shape)
            y1 = sigma_max * z1
            sigma_max_arr = mx.full((1, 1), sigma_max)
            from experiment_consistency_drafting import get_consistency_prediction
            pred1_full = get_consistency_prediction(
                consistency_full_online, y1, ctx_feats, audio_summary, pos_ids, sigma_max_arr
            )
            z2_full = mx.random.normal(true_hidden.shape)
            y2_full = pred1_full + math.sqrt(max(sigma_mid**2 - 0.002**2, 1e-9)) * z2_full
            sigma_mid_arr = mx.full((1, 1), sigma_mid)
            pred_full = get_consistency_prediction(
                consistency_full_online, y2_full, ctx_feats, audio_summary, pos_ids, sigma_mid_arr
            )

            # C. Consistency-PCA (2-shot in subspace, then reconstruct)
            true_z = (true_hidden - mean_mx) @ V_mx  # reference only
            z1_pca = mx.random.normal((1, config.block_size, pca_rank))
            y1_pca = sigma_max * z1_pca
            pred1_pca = get_consistency_prediction_pca(
                consistency_pca_online, y1_pca, ctx_feats, audio_summary, pos_ids, sigma_max_arr
            )
            z2_pca = mx.random.normal((1, config.block_size, pca_rank))
            y2_pca = pred1_pca + math.sqrt(max(sigma_mid**2 - 0.002**2, 1e-9)) * z2_pca
            pred_z_pca = get_consistency_prediction_pca(
                consistency_pca_online, y2_pca, ctx_feats, audio_summary, pos_ids, sigma_mid_arr
            )
            # Reconstruct to full D_target space
            pred_pca = pred_z_pca @ V_mx.T + mean_mx

            # D. BF-Consistency-PCA (2-shot in subspace, then reconstruct)
            y1_bf = sigma_max * z1_pca  # reuse noise
            pred1_bf = get_consistency_prediction_pca(
                bf_pca_online, y1_bf, ctx_feats, audio_summary, pos_ids, sigma_max_arr
            )
            z2_bf = mx.random.normal((1, config.block_size, pca_rank))
            y2_bf = pred1_bf + math.sqrt(max(sigma_mid**2 - 0.002**2, 1e-9)) * z2_bf
            pred_z_bf = get_consistency_prediction_pca(
                bf_pca_online, y2_bf, ctx_feats, audio_summary, pos_ids, sigma_mid_arr
            )
            pred_bf = pred_z_bf @ V_mx.T + mean_mx

            # --- Compute similarities and accuracies ---
            all_preds = {
                "base": pred_base,
                "full": pred_full,
                "pca": pred_pca,
                "bf_pca": pred_bf
            }

            for key, pred in all_preds.items():
                for k in range(config.block_size):
                    h_true = true_hidden[0, k]
                    h_pred = pred[0, k]
                    sim = (mx.sum(h_pred * h_true) / (mx.linalg.norm(h_pred) * mx.linalg.norm(h_true) + 1e-9)).item()
                    metrics[key]["sim"].append(sim)
                    metrics[key]["sim_per_step"][k].append(sim)

            # Token projections
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]

            for key, pred in all_preds.items():
                pred_logits = target.decoder.token_embedding.as_linear(pred.reshape(-1, d_target))
                pred_tokens = mx.argmax(pred_logits, axis=-1).tolist()
                for idx in range(len(true_tokens)):
                    metrics[key]["acc"].append(1.0 if pred_tokens[idx] == true_tokens[idx] else 0.0)
                    metrics[key]["top5_acc"].append(1.0 if pred_tokens[idx] in top5_indices[idx].tolist() else 0.0)

    # --- Print Results ---
    print("\n" + "=" * 80)
    print("RESULTS: CONSISTENCY-MODEL + PCA-SUBSPACE FUSION (Phase-2 Item 24)")
    print("=" * 80)

    print("\n--- 1. Hidden Representation Cosine Similarity ---")
    base_sim = np.mean(metrics["base"]["sim"])
    for name, key in models_list:
        mean_sim = np.mean(metrics[key]["sim"])
        delta = mean_sim - base_sim
        print(f"  {name:40s}: {mean_sim:.4f}  (Delta vs MSE: {delta:+.4f})")

    print("\n--- 2. Greedy Token Accuracy (Projection Match) ---")
    base_acc = np.mean(metrics["base"]["acc"])
    for name, key in models_list:
        mean_acc = np.mean(metrics[key]["acc"]) * 100
        delta = mean_acc - base_acc * 100
        print(f"  {name:40s}: {mean_acc:.2f}%  (Delta vs MSE: {delta:+.2f}%)")

    print("\n--- 3. Top-5 Expected Token Acceptance Rate ---")
    base_top5 = np.mean(metrics["base"]["top5_acc"])
    for name, key in models_list:
        mean_top5 = np.mean(metrics[key]["top5_acc"]) * 100
        delta = mean_top5 - base_top5 * 100
        print(f"  {name:40s}: {mean_top5:.2f}%  (Delta vs MSE: {delta:+.2f}%)")

    print("\n--- 4. Parameter Efficiency ---")
    print(f"  {'MSE Baseline (full D)':40s}: {params_baseline:>8,} params")
    print(f"  {'Consistency (full D)':40s}: {params_full:>8,} params")
    print(f"  {'Consistency-PCA (R=64)':40s}: {params_pca:>8,} params  ({(1-params_pca/params_full)*100:.1f}% savings)")
    print(f"  {'BF-Consistency-PCA (R=64)':40s}: {params_bf_pca:>8,} params  ({(1-params_bf_pca/params_full)*100:.1f}% savings)")

    print("\n--- 5. Cosine Similarity Drift Per Step ---")
    for k in range(config.block_size):
        parts = [f"Step +{k+1}:"]
        for name, key in models_list:
            mean_k = np.mean(metrics[key]["sim_per_step"][k])
            label = key[:8]
            parts.append(f"{label}={mean_k:.4f}")
        print("  " + " | ".join(parts))

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
