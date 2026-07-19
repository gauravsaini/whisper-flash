#!/usr/bin/env python3
"""
experiment_spectral_id42.py

Phase-6 ID 42: Per-Position Stochastic Noise Injection

ROOT CAUSE (from Experiment 18 + 19):
  The draft model receives IDENTICAL mask token embeddings for all T positions.
  The only differentiator is position embedding (256-dim), overwhelmed by shared
  context (768-dim). This structurally forces rank-1 outputs. 5 loss fixes failed.

FIX:
  Replace identical mask tokens with UNIQUE per-position noise vectors:
    ε_k ~ N(0, σ²I) for each position k ∈ {1..T}
  
  This breaks input symmetry. The model must map DIVERSE inputs → DIVERSE outputs.
  
  We test 3 noise strategies:
    A) Gaussian noise (ε ~ N(0, 1))
    B) Scaled Gaussian noise (ε ~ N(0, σ²I) with σ = ||target_hidden||)
    C) Orthogonal noise (QR decomposition of random matrix → guaranteed orthogonal)

  Each is trained with standard MSE loss. If the architecture fix works,
  even vanilla MSE should produce diverse outputs.

SPECTRAL METRICS (same battery as Exp 19):
  - Spectral Angle, Grassmann Distance
  - Decay Rate α, Participation Ratio
  - Gram Matrix Effective Rank
  - Per-PC Cosine Alignment
  - Top-5 Token Match
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
# Spectral Utilities (from Exp 18/19)
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
    if len(S) < 3: return 0.0, 0.0
    n = len(S)
    A = np.stack([np.log(np.arange(1, n+1)), np.ones(n)], axis=1)
    coeffs = np.linalg.lstsq(A, np.log(S), rcond=None)[0]
    return float(-coeffs[0]), 0.0

def participation_ratio(S):
    S2, S4 = S**2, S**4
    return float(np.sum(S2)**2 / (np.sum(S4) + 1e-20))

def gram_matrix_rank(H, threshold=0.01):
    norms = np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9
    G = (H / norms) @ (H / norms).T
    eigvals = np.linalg.eigvalsh(G)[::-1]
    return int(np.sum(eigvals > threshold * eigvals[0])), eigvals


# ═══════════════════════════════════════════════════════════════════════
# Draft Models with Different Noise Strategies
# ═══════════════════════════════════════════════════════════════════════

class IdenticalInputDraftModel(nn.Module):
    """CONTROL: Original architecture — identical mask tokens for all positions."""
    def __init__(self, config):
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


class StochasticNoiseDraftModel(nn.Module):
    """
    ID 42: Per-position stochastic noise injection.
    Instead of identical mask tokens, each position gets unique noise.
    The noise is generated fresh every forward pass (stochastic).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_taps = len(config.target_layer_ids)
        # Input proj takes RANDOM noise (d_target-dim) per position
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
        # noise_embedding is unique per position (generated by caller)
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
        x = self.norm(x)
        return self.continuous_head(x)


class MultiHeadDraftModel(nn.Module):
    """
    ID 44: Position-specialized multi-head output.
    Shared backbone but T separate output heads, one per position.
    """
    def __init__(self, config):
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
        # T SEPARATE output heads
        self.output_heads = [nn.Linear(config.d_draft, config.d_target, bias=False)
                            for _ in range(config.block_size)]
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
        x = self.norm(x)
        
        # Apply position-specific output heads
        outputs = []
        for k in range(self.config.block_size):
            out_k = self.output_heads[k](x[:, k:k+1, :])  # (1, 1, d_target)
            outputs.append(out_k)
        return mx.concatenate(outputs, axis=1)  # (1, T, d_target)


class StochasticMultiHeadDraftModel(nn.Module):
    """
    ID 42 + 44 COMBINED: Stochastic noise + multi-head output.
    Maximum architectural diversity: unique noise input AND specialized output.
    """
    def __init__(self, config):
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
        self.output_heads = [nn.Linear(config.d_draft, config.d_target, bias=False)
                            for _ in range(config.block_size)]
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
        x = self.norm(x)
        outputs = []
        for k in range(self.config.block_size):
            outputs.append(self.output_heads[k](x[:, k:k+1, :]))
        return mx.concatenate(outputs, axis=1)


# ═══════════════════════════════════════════════════════════════════════
# Loss and Noise Generation
# ═══════════════════════════════════════════════════════════════════════

def mse_loss(model, noise, ctx, audio, pos, true_hidden):
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - true_hidden))


def generate_stochastic_noise(block_size, d_target):
    """Generate unique Gaussian noise for each position."""
    return mx.random.normal(shape=(1, block_size, d_target)) * 0.1


def generate_orthogonal_noise(block_size, d_target):
    """Generate orthogonal noise vectors via QR decomposition."""
    # Random matrix → QR → orthogonal columns
    rand_matrix = np.random.randn(d_target, block_size)
    Q, _ = np.linalg.qr(rand_matrix)
    # Q is (d_target, block_size) with orthonormal columns
    # Transpose to (block_size, d_target) and scale
    ortho_noise = Q[:, :block_size].T * 0.1
    return mx.array(ortho_noise[None], dtype=mx.float32)


# ═══════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════

def run():
    print("=" * 75)
    print("EXPERIMENT: Per-Position Noise Injection (ID 42 + 44)")
    print("  Testing if architectural diversity breaks rank-1 collapse")
    print("=" * 75)

    print("\n[1/6] Loading Target Model...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state

    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )

    print("[2/6] Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # --- Pre-extract Data ---
    print("[3/6] Pre-extracting training data...")
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
                target, input_tokens, encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = mx.concatenate([hidden_all[lid] for lid in config.target_layer_ids], axis=-1)
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
            
            # Identical mask token noise (for control model)
            mask_noise = target.decoder.token_embedding(
                mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            train_data.append({
                "mask_noise": mask_noise,
                "ctx": ctx_feats, "audio": audio_summary,
                "pos": pos_ids, "true_hidden": true_hidden
            })

    print(f"   Datapoints: {len(train_data)}")

    # --- Models ---
    models = {
        "Control":      IdenticalInputDraftModel(config),
        "Gaussian":     StochasticNoiseDraftModel(config),
        "Orthogonal":   StochasticNoiseDraftModel(config),
        "MultiHead":    MultiHeadDraftModel(config),
        "Gauss+MH":     StochasticMultiHeadDraftModel(config),
    }

    from mlx.utils import tree_flatten
    for name, model in models.items():
        params = sum(x.size for _, x in tree_flatten(model.parameters()))
        print(f"   {name:>12}: {params:>10,} params")

    optimizers = {name: optim.Adam(learning_rate=1e-3) for name in models}
    grad_fns = {name: nn.value_and_grad(models[name], mse_loss) for name in models}

    # --- Train ---
    print("\n[4/6] Training (25 epochs)...")
    epochs = 25
    for epoch in range(epochs):
        losses = {n: 0.0 for n in models}
        for data in train_data:
            # Generate fresh noise for stochastic models
            gauss_noise = generate_stochastic_noise(config.block_size, d_target)
            ortho_noise = generate_orthogonal_noise(config.block_size, d_target)

            noise_map = {
                "Control":    data["mask_noise"],
                "Gaussian":   gauss_noise,
                "Orthogonal": ortho_noise,
                "MultiHead":  data["mask_noise"],
                "Gauss+MH":   gauss_noise,
            }

            for name in models:
                l, g = grad_fns[name](
                    models[name], noise_map[name],
                    data["ctx"], data["audio"], data["pos"], data["true_hidden"]
                )
                optimizers[name].update(models[name], g)
                mx.eval(models[name].parameters(), optimizers[name].state)
                losses[name] += l.item()

        if (epoch + 1) % 5 == 0:
            loss_str = "  ".join([f"{n}: {losses[n]/len(train_data):.5f}" for n in models])
            print(f"   Epoch {epoch+1:02d}/{epochs}  {loss_str}")

    # --- Evaluate ---
    print("\n[5/6] Spectral Evaluation on Held-out Samples...")

    metrics = {}
    for name in models:
        metrics[name] = {
            "cosine": [[] for _ in range(config.block_size)],
            "spectral_angle": [], "grassmann": [],
            "alpha": [], "pr": [],
            "gram_rank": [], "pc_cosines": [],
            "top5": 0,
        }
    metrics["_total"] = 0

    for i in range(10, 20):
        if i >= len(ds): break
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
            mask_noise = target.decoder.token_embedding(
                mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            true_np = np.array(true_hidden[0])
            _, S_true, Vt_true = compute_svd(true_np)

            true_logits = target.decoder.token_embedding.as_linear(true_hidden)
            true_top5 = mx.argsort(true_logits, axis=-1)[:, :, -5:]
            mx.eval(true_top5)
            true_top5_np = np.array(true_top5[0])

            gauss_noise = generate_stochastic_noise(config.block_size, d_target)
            ortho_noise = generate_orthogonal_noise(config.block_size, d_target)
            noise_map = {
                "Control": mask_noise, "Gaussian": gauss_noise,
                "Orthogonal": ortho_noise, "MultiHead": mask_noise,
                "Gauss+MH": gauss_noise,
            }

            for name in models:
                pred = models[name](noise_map[name], ctx_feats, audio_summary, pos_ids)
                mx.eval(pred)
                pred_np = np.array(pred[0])

                for k in range(config.block_size):
                    cos = float(np.dot(pred_np[k], true_np[k]) /
                               (np.linalg.norm(pred_np[k]) * np.linalg.norm(true_np[k]) + 1e-9))
                    metrics[name]["cosine"][k].append(cos)

                _, S_pred, Vt_pred = compute_svd(pred_np)
                angle, grass, pc_cos = spectral_angle(Vt_pred, Vt_true, config.block_size)
                alpha, _ = spectral_decay_rate(S_pred)
                pr = participation_ratio(S_pred)
                gr, _ = gram_matrix_rank(pred_np)

                metrics[name]["spectral_angle"].append(angle)
                metrics[name]["grassmann"].append(grass)
                metrics[name]["pc_cosines"].append(pc_cos)
                metrics[name]["alpha"].append(alpha)
                metrics[name]["pr"].append(pr)
                metrics[name]["gram_rank"].append(gr)

                pred_logits = target.decoder.token_embedding.as_linear(pred)
                pred_top5 = mx.argsort(pred_logits, axis=-1)[:, :, -5:]
                mx.eval(pred_top5)
                for k in range(config.block_size):
                    true_set = set(true_top5_np[k].tolist())
                    pred_set = set(np.array(pred_top5[0, k]).tolist())
                    metrics[name]["top5"] += len(true_set & pred_set)

            metrics["_total"] += 5 * config.block_size

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("═══ RESULTS: ARCHITECTURAL DIVERSITY EXPERIMENT ═══")
    print("=" * 80)

    model_names = list(models.keys())

    print("\n┌─── Per-Step Cosine Similarity ───┐")
    header = f"  {'Step':>6}" + "".join(f"  {n:>12}" for n in model_names)
    print(header)
    for k in range(config.block_size):
        row = f"  +{k+1:>4}:"
        for name in model_names:
            row += f"  {np.mean(metrics[name]['cosine'][k]):>12.4f}"
        print(row)

    print("\n┌─── Spectral Angle (↓ better, target: 0°) ───┐")
    for name in model_names:
        sa = np.mean(metrics[name]["spectral_angle"])
        print(f"  {name:>12}: {sa:.4f} rad ({np.degrees(sa):.2f}°)")

    print("\n┌─── Grassmann Distance (↓ better) ───┐")
    for name in model_names:
        print(f"  {name:>12}: {np.mean(metrics[name]['grassmann']):.4f}")

    print("\n┌─── Spectral Decay α (target: ~0.81) ───┐")
    for name in model_names:
        a = np.mean(metrics[name]["alpha"])
        print(f"  {name:>12}: α = {a:.4f}")

    print("\n┌─── Participation Ratio (target: ~2.35) ───┐")
    for name in model_names:
        pr = np.mean(metrics[name]["pr"])
        print(f"  {name:>12}: PR = {pr:.4f}")

    print("\n┌─── Gram Matrix Effective Rank (target: 4/4) ───┐")
    for name in model_names:
        gr = np.mean(metrics[name]["gram_rank"])
        print(f"  {name:>12}: {gr:.2f}/4")

    print("\n┌─── Per-PC Cosine Alignment ───┐")
    n_pcs = len(metrics[model_names[0]]["pc_cosines"][0]) if metrics[model_names[0]]["pc_cosines"] else 0
    header = f"  {'PC':>4}" + "".join(f"  {n:>12}" for n in model_names)
    print(header)
    for pc in range(n_pcs):
        row = f"  {pc+1:>4}:"
        for name in model_names:
            cos = np.mean([pcs[pc] for pcs in metrics[name]["pc_cosines"]])
            row += f"  {cos:>12.4f}"
        print(row)

    total = metrics["_total"]
    print(f"\n┌─── Top-5 Token Match Rate ───┐")
    for name in model_names:
        rate = metrics[name]["top5"] / total * 100 if total > 0 else 0
        print(f"  {name:>12}: {rate:.2f}%")

    # Verdict
    print("\n" + "=" * 80)
    print("═══ VERDICT ═══")
    for name in model_names:
        gr = np.mean(metrics[name]["gram_rank"])
        alpha = np.mean(metrics[name]["alpha"])
        pr = np.mean(metrics[name]["pr"])
        if gr > 1.5 and pr > 1.3:
            print(f"  ✅ {name} BROKE RANK-1 COLLAPSE! (Gram rank={gr:.2f}, α={alpha:.2f}, PR={pr:.4f})")
        else:
            print(f"  ❌ {name} still collapsed (Gram rank={gr:.2f}, α={alpha:.2f}, PR={pr:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    run()
