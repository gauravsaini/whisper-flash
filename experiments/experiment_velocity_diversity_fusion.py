#!/usr/bin/env python3
"""
experiment_velocity_diversity_fusion.py

VELOCITY-DIVERSITY FUSION (ID 52)
══════════════════════════════════════════

Hypothesis: Combining tangent-space velocity prediction (predicting Δ = h_{t+k} - h_{t+k-1})
with architectural diversity (per-position orthogonal noise + multi-head outputs) will
simultaneously achieve high token match rates (25%+) AND high spectral diversity (gram rank > 2).

TIMELINE Deduction (line 1738-1743):
  A_Velocity → 24.86% top-5 match, spectral angle 59.95°
  Orthogonal Noise → PC-2 cosine 0.466, gram rank 1.63
  IF combined → top-5 > 25% AND gram rank > 2.0

EXPERIMENTS:
  A: Control — standard MSE, identical mask inputs
  B: Orthogonal Noise — standard MSE, per-position orthogonal noise (ID 42 fix)
  C: Velocity — tangent-space prediction (Δ), identical mask inputs
  D: Velocity + Orthogonal Noise — tangent prediction + orthogonal noise
  E: Velocity + MultiHead — tangent prediction + T separate output heads
  F: Velocity + Orthogonal Noise + MultiHead (FULL FUSION)
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
# SPECTRAL UTILITIES (unchanged from experiment_manifold_10way.py)
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
# NOISE GENERATORS (from experiment_spectral_id42.py)
# ═══════════════════════════════════════════════════════════════════════

def generate_orthogonal_noise(block_size, d_target, scale=0.1):
    rand_matrix = np.random.randn(d_target, block_size)
    Q, _ = np.linalg.qr(rand_matrix)
    ortho_noise = Q[:, :block_size].T * scale
    return mx.array(ortho_noise[None], dtype=mx.float32)


# ═══════════════════════════════════════════════════════════════════════
# DRAFT MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class BaseDraftModel(nn.Module):
    """Shared backbone. output_dim and velocity_mode configure behavior."""

    def __init__(self, config, output_dim=None, velocity_mode=False):
        super().__init__()
        self.config = config
        self.out_dim = output_dim or config.d_target
        self.velocity_mode = velocity_mode
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


class MultiHeadDraftModel(nn.Module):
    """Shared backbone with T separate output heads. velocity_mode configurable."""

    def __init__(self, config, output_dim=None, velocity_mode=False):
        super().__init__()
        self.config = config
        self.out_dim = output_dim or config.d_target
        self.velocity_mode = velocity_mode
        num_taps = len(config.target_layer_ids)
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.output_heads = [nn.Linear(config.d_draft, self.out_dim, bias=False)
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
            out_k = self.output_heads[k](x[:, k:k+1, :])
            outputs.append(out_k)
        return mx.concatenate(outputs, axis=1)


# ═══════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def mse_loss(model, noise, ctx, audio, pos, target):
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - target))


def velocity_loss(model, noise, ctx, audio, pos, target):
    """Predict velocity (Δ = h_{t+k} - h_{t+k-1}) instead of absolute positions."""
    pred = model(noise, ctx, audio, pos)
    return mx.mean(mx.square(pred - target))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    print("═══ VELOCITY-DIVERSITY FUSION (ID 52) ═══")
    print()

    t_start = time.time()

    # ─── Load target model ───
    print("[1/6] Loading target model...")
    whisper = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(whisper.is_multilingual, num_languages=whisper.num_languages)
    d_target = whisper.dims.n_text_state
    print(f"  d_target = {d_target}")

    config = WhisperDFlashConfig(
        d_target=d_target,
        d_draft=256,
        num_layers=2,
        vocab_size=whisper.dims.n_vocab,
        block_size=4,
        max_target_positions=getattr(whisper.dims, 'n_text_max', 448),
        target_layer_ids=[1, 2],
        mask_token_id=tokenizer.no_timestamps,
    )

    # ─── Load dataset ───
    print("[2/6] Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print(f"  Total samples: {len(ds)}")

    # ─── Extract training data ───
    print("[3/6] Extracting train data (samples 0-9)...")
    train_data = []
    for i in range(10):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        n_mels = whisper.dims.n_mels
        padding = 16000 * 30 - len(audio)
        mel = log_mel_spectrogram(audio, n_mels=n_mels, padding=padding if padding > 0 else 0)
        mel_mx = mx.array(mel[None], dtype=mx.float32)

        sot_token = tokenizer.sot
        labels = mx.concatenate([
            mx.array([[sot_token]], dtype=mx.int32),
            mx.array([text_tokens], dtype=mx.int32),
        ], axis=1)

        encoder_hidden = encoder_forward(whisper, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - config.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                whisper, input_tokens, encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False,
            )
            ctx_feats = mx.concatenate(
                [hidden_all[lid] for lid in config.target_layer_ids], axis=-1)

            _, _, hidden_future = decoder_forward_with_hidden_states(
                whisper, labels[:, :t+1+config.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False,
            )
            true_hidden = mx.stop_gradient(hidden_future[-1][:, t:t+config.block_size, :])

            # Velocity target: Δ for each of the 4 positions relative to previous
            # Include last hidden state as anchor: [h_t, h_{t+1}, h_{t+2}, h_{t+3}, h_{t+4}]
            # → 4 deltas: Δ_k = h_{t+k} - h_{t+k-1}
            prev_np = np.array(hidden_future[-1][0, t-1:t, :])  # (1, 384)
            true_np = np.array(true_hidden[0])                     # (4, 384)
            stacked = np.concatenate([prev_np, true_np], axis=0)   # (5, 384)
            vel_np = np.diff(stacked, axis=0)                     # (4, 384)
            vel_target = mx.array(vel_np[None])

            mask_tokens = mx.array([[config.mask_token_id] * config.block_size], dtype=mx.int32)
            noise = whisper.decoder.token_embedding(mask_tokens)
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            train_data.append({
                "noise": noise,
                "ctx": ctx_feats,
                "audio": audio_summary,
                "pos": pos_ids,
                "true_hidden": true_hidden,
                "vel_target": vel_target,
            })

    print(f"  Extracted {len(train_data)} training datapoints")

    # ─── Instantiate models ───
    print("[4/6] Instantiating 6 model variants...")

    models = {}
    optimizers = {}

    # A: Control
    models["Control"] = BaseDraftModel(config, velocity_mode=False)
    optimizers["Control"] = optim.Adam(learning_rate=1e-3)

    # B: Orthogonal Noise
    models["OrthoNoise"] = BaseDraftModel(config, velocity_mode=False)
    optimizers["OrthoNoise"] = optim.Adam(learning_rate=1e-3)

    # C: Velocity (identical mask inputs, predicts Δ)
    models["Velocity"] = BaseDraftModel(config, velocity_mode=True)
    optimizers["Velocity"] = optim.Adam(learning_rate=1e-3)

    # D: Velocity + Orthogonal Noise
    models["Vel+Ortho"] = BaseDraftModel(config, velocity_mode=True)
    optimizers["Vel+Ortho"] = optim.Adam(learning_rate=1e-3)

    # E: Velocity + MultiHead
    models["Vel+MH"] = MultiHeadDraftModel(config, velocity_mode=True)
    optimizers["Vel+MH"] = optim.Adam(learning_rate=1e-3)

    # F: Velocity + Orthogonal Noise + MultiHead (FULL FUSION)
    models["Vel+Ortho+MH"] = MultiHeadDraftModel(config, velocity_mode=True)
    optimizers["Vel+Ortho+MH"] = optim.Adam(learning_rate=1e-3)

    model_names = list(models.keys())

    # ─── Force initialization ───
    for name, model in models.items():
        data = train_data[0]
        _ = model(data["noise"], data["ctx"], data["audio"], data["pos"])

    # ─── Training ───
    print("[5/6] Training all models...")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Epochs: 25, Samples: {len(train_data)}, Block size: {config.block_size}")

    grad_fns = {}
    for name in model_names:
        if "Vel" in name:
            grad_fns[name] = nn.value_and_grad(models[name], velocity_loss)
        else:
            grad_fns[name] = nn.value_and_grad(models[name], mse_loss)

    epochs = 25
    for epoch in range(epochs):
        losses = {n: 0.0 for n in model_names}
        for data in train_data:
            ortho_noise = generate_orthogonal_noise(config.block_size, d_target)
            noise_map = {
                "Control":     data["noise"],
                "OrthoNoise":  ortho_noise,
                "Velocity":    data["noise"],
                "Vel+Ortho":   ortho_noise,
                "Vel+MH":      data["noise"],
                "Vel+Ortho+MH": ortho_noise,
            }
            for name in model_names:
                target = data["vel_target"] if "Vel" in name else data["true_hidden"]
                l, g = grad_fns[name](models[name], noise_map[name],
                                      data["ctx"], data["audio"], data["pos"], target)
                optimizers[name].update(models[name], g)
                mx.eval(models[name].parameters(), optimizers[name].state)
                losses[name] += l.item()
        if (epoch + 1) % 5 == 0:
            loss_str = " | ".join([f"{n}: {losses[n]/len(train_data):.5f}" for n in model_names])
            print(f"  Epoch {epoch+1:02d}/{epochs}  {loss_str}")

    # ─── Evaluation ───
    print()
    print("[6/6] Evaluating all models on held-out samples 10-19...")

    metrics = {}
    for name in model_names:
        metrics[name] = {
            "cosine": [[] for _ in range(config.block_size)],
            "spectral_angle": [],
            "alpha": [],
            "pr": [],
            "gram_rank": [],
            "pc_cosines": [],
            "top5_tokens": 0,
            "top5_total": 0,
        }

    eval_count = 0
    for i in range(10, 20):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        n_mels = whisper.dims.n_mels
        padding = 16000 * 30 - len(audio)
        mel = log_mel_spectrogram(audio, n_mels=n_mels, padding=padding if padding > 0 else 0)
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        sot_token = tokenizer.sot
        labels = mx.concatenate([
            mx.array([[sot_token]], dtype=mx.int32),
            mx.array([text_tokens], dtype=mx.int32),
        ], axis=1)

        encoder_hidden = encoder_forward(whisper, mel_mx)
        audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - config.block_size, 2):
            input_tokens = labels[:, :t+1]
            _, _, hidden_all = decoder_forward_with_hidden_states(
                whisper, input_tokens, encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False,
            )
            ctx_feats = mx.concatenate(
                [hidden_all[lid] for lid in config.target_layer_ids], axis=-1)
            last_hidden = hidden_all[-1][:, -1:, :]

            _, _, hidden_future = decoder_forward_with_hidden_states(
                whisper, labels[:, :t+1+config.block_size], encoder_hidden,
                collect_hidden_states=True, return_cross_attention=False,
            )
            true_hidden = np.array(hidden_future[-1][0, t:t+config.block_size, :])

            mask_tokens = mx.array([[config.mask_token_id] * config.block_size], dtype=mx.int32)
            noise_input = whisper.decoder.token_embedding(mask_tokens)
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]

            for name in model_names:
                ortho_noise = generate_orthogonal_noise(config.block_size, d_target)
                noise_use = ortho_noise if ("Ortho" in name) else noise_input

                pred = models[name](noise_use, ctx_feats, audio_summary, pos_ids)
                pred_np = np.array(pred[0])

                if "Vel" in name:
                    pred_np = np.cumsum(pred_np, axis=0) + np.array(last_hidden[0])

                topk = min(4, true_hidden.shape[0])
                for k in range(topk):
                    p_norm = np.linalg.norm(pred_np[k]) + 1e-9
                    t_norm = np.linalg.norm(true_hidden[k]) + 1e-9
                    cs = float(np.dot(pred_np[k], true_hidden[k]) / (p_norm * t_norm))
                    metrics[name]["cosine"][k].append(cs)

                # Spectral analysis
                _, S_pred, Vt_pred = compute_svd(pred_np)
                _, S_true, Vt_true = compute_svd(true_hidden)
                angle, _, pc_cos = spectral_angle(Vt_pred, Vt_true, config.block_size)
                metrics[name]["spectral_angle"].append(angle)
                metrics[name]["alpha"].append(spectral_decay_rate(S_pred))
                metrics[name]["pr"].append(participation_ratio(S_pred))
                metrics[name]["gram_rank"].append(gram_matrix_rank(pred_np))
                metrics[name]["pc_cosines"].append(pc_cos)

                # Top-5 token match
                pred_mx = mx.array(pred_np[None], dtype=mx.float32)
                pred_logits = whisper.decoder.token_embedding.as_linear(pred_mx)
                pred_top5 = mx.argsort(pred_logits, axis=-1)
                pred_top5_np = np.array(pred_top5[0, :, -5:])

                true_tokens = labels[:, t:t+config.block_size]
                true_tokens_np = np.array(true_tokens[0])
                for k in range(config.block_size):
                    metrics[name]["top5_total"] += 1
                    if true_tokens_np[k] in pred_top5_np[k]:
                        metrics[name]["top5_tokens"] += 1

                if eval_count == 0 and name == model_names[0]:
                    print(f"    [DEBUG] True tokens[{t}]: {true_tokens_np}")
                    print(f"    [DEBUG] Pred top-5[{t}]: {pred_top5_np}")
                    cos_k = []
                    for k in range(config.block_size):
                        c = float(np.dot(pred_np[k], true_hidden[k]) / (np.linalg.norm(pred_np[k])*np.linalg.norm(true_hidden[k])+1e-9))
                        cos_k.append(f"{c:.4f}")
                    print(f"    [DEBUG] Cos per step: {cos_k}")

                eval_count += 1

    # ─── Print results ───
    print()
    print("=" * 80)
    print("  RESULTS: VELOCITY-DIVERSITY FUSION (ID 52)")
    print("=" * 80)
    print()

    header = f"{'Model':>20} | {'Cos':>7} | {'SpAng':>6} | {'Decay':>6} | {'PR':>6} | {'GramR':>5} | {'PC-2':>6} | {'Top-5':>7}"
    print(header)
    print("-" * len(header))

    for name in model_names:
        cosines = [np.mean(metrics[name]["cosine"][k]) for k in range(config.block_size)]
        mean_cos = np.mean(cosines)
        mean_angle = np.degrees(np.mean(metrics[name]["spectral_angle"]))
        mean_alpha = np.mean(metrics[name]["alpha"])
        mean_pr = np.mean(metrics[name]["pr"])
        mean_gr = np.mean(metrics[name]["gram_rank"])
        pc2 = np.mean([c[1] if len(c) > 1 else 0.0 for c in metrics[name]["pc_cosines"]])
        top5_pct = (metrics[name]["top5_tokens"] / max(metrics[name]["top5_total"], 1)) * 100
        print(f"{name:>20} | {mean_cos:>7.4f} | {mean_angle:>6.1f}° | {mean_alpha:>6.2f} | {mean_pr:>6.4f} | {mean_gr:>4.1f}/4 | {pc2:>6.4f} | {top5_pct:>6.2f}%")

    print()

    # ─── Verdicts ───
    print("═══ VERDICT ═══")
    for name in model_names:
        gr = np.mean(metrics[name]["gram_rank"])
        pr = np.mean(metrics[name]["pr"])
        top5_pct = (metrics[name]["top5_tokens"] / max(metrics[name]["top5_total"], 1)) * 100
        if gr >= 2.0 and pr >= 1.5 and top5_pct >= 10.0:
            print(f"  ✅ {name:>20}: BROKE COLLAPSE! GramR={gr:.2f}, PR={pr:.4f}, Top-5={top5_pct:.2f}% — THE FUSION WORKS!")
        elif gr > 1.5 or pr > 1.1 or top5_pct >= 15.0:
            print(f"  🟡 {name:>20}: PARTIAL — GramR={gr:.2f}, PR={pr:.4f}, Top-5={top5_pct:.2f}%")
        else:
            print(f"  ❌ {name:>20}: COLLAPSED — GramR={gr:.2f}, PR={pr:.4f}, Top-5={top5_pct:.2f}%")

    print()
    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    run()
