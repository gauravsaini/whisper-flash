#!/usr/bin/env python3
"""
experiment_delta.py

Moonshot #8: Continuous State Delta-Prediction (Residual Drafting)
Compares three speculative drafting architectures to stabilize manifold trajectories:
1. Absolute Baseline (Predicts absolute future hidden states directly)
2. Anchor Residual (Predicts the delta relative to the prefix anchor state)
3. Cumulative Step Residual (Predicts cumulative step-wise deltas along the block)
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel

# --- 1. Define Model Variants ---

class AbsoluteDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids, h_anchor=None):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        return self.continuous_head(x)

class AnchorResidualDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids, h_anchor):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        delta = self.continuous_head(x)
        # h_anchor is (batch, 1, d_target), delta is (batch, block_size, d_target)
        return h_anchor + delta

class CumulativeResidualDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids, h_anchor):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        step_deltas = self.continuous_head(x)
        # Cumulative sum of step deltas along the block_size axis (axis=1)
        return h_anchor + mx.cumsum(step_deltas, axis=1)

# --- 2. Loss Function ---

def loss_fn(model, noise, target_hidden, audio_summary, position_ids, h_anchor, true_hidden):
    pred = model(noise, target_hidden, audio_summary, position_ids, h_anchor)
    return mx.mean(mx.square(pred - true_hidden))

def copy_parameters(model):
    from mlx.utils import tree_map
    return tree_map(lambda x: mx.array(x), model.parameters())

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Initialize models
    model_absolute = AbsoluteDrafter(config)
    model_anchor = AnchorResidualDrafter(config)
    model_cumulative = CumulativeResidualDrafter(config)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    h_anchor_init = mx.zeros((1, 1, d_target))
    
    _ = model_absolute(noise_init, ctx_init, audio_init, pos_init, h_anchor_init)
    _ = model_anchor(noise_init, ctx_init, audio_init, pos_init, h_anchor_init)
    _ = model_cumulative(noise_init, ctx_init, audio_init, pos_init, h_anchor_init)
    
    # Align starting weights
    initial_params = copy_parameters(model_absolute)
    model_anchor.update(initial_params)
    model_cumulative.update(initial_params)

    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
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
        
        for t in range(1, labels.shape[1] - config.block_size - 1, 3):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            
            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)
            
            # Anchor state h_t is the top layer's hidden state at the last token of the prefix
            h_anchor = hidden_target[-1][:, -1:, :] # (1, 1, d_target)
            
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            # Future true states are indices t+1 to t+block_size
            true_hidden = hidden_future[-1][:, t+1:t+1+config.block_size, :]
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t + 1, t + 1 + config.block_size, dtype=mx.int32)[None]
            
            data_tensors.append({
                "noise": noise,
                "ctx": ctx_feats,
                "audio": audio_summary,
                "pos": pos_ids,
                "h_anchor": h_anchor,
                "true_hidden": true_hidden
            })
            
    print(f"Extracted {len(data_tensors)} training samples.")
    
    # -----------------------------------------------------------------------
    # Train Models
    # -----------------------------------------------------------------------
    
    # 1. Train Absolute
    print("\n--- Training Model 1: Absolute Baseline ---")
    opt_abs = optim.Adam(learning_rate=1e-3)
    grad_abs = nn.value_and_grad(model_absolute, loss_fn)
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_abs(
                model_absolute, data["noise"], data["ctx"], data["audio"], data["pos"], data["h_anchor"], data["true_hidden"]
            )
            opt_abs.update(model_absolute, grads)
            mx.eval(model_absolute.parameters(), opt_abs.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Absolute Baseline trained in {time.time() - t0:.1f}s.")
    
    # 2. Train Anchor Residual
    print("\n--- Training Model 2: Anchor Residual ---")
    opt_anch = optim.Adam(learning_rate=1e-3)
    grad_anch = nn.value_and_grad(model_anchor, loss_fn)
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_anch(
                model_anchor, data["noise"], data["ctx"], data["audio"], data["pos"], data["h_anchor"], data["true_hidden"]
            )
            opt_anch.update(model_anchor, grads)
            mx.eval(model_anchor.parameters(), opt_anch.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Anchor Residual trained in {time.time() - t0:.1f}s.")
    
    # 3. Train Cumulative Residual
    print("\n--- Training Model 3: Cumulative Step Residual ---")
    opt_cum = optim.Adam(learning_rate=1e-3)
    grad_cum = nn.value_and_grad(model_cumulative, loss_fn)
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_cum(
                model_cumulative, data["noise"], data["ctx"], data["audio"], data["pos"], data["h_anchor"], data["true_hidden"]
            )
            opt_cum.update(model_cumulative, grads)
            mx.eval(model_cumulative.parameters(), opt_cum.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Cumulative Residual trained in {time.time() - t0:.1f}s.")
    
    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    
    metrics = {
        "abs": {"sim": [], "acc": [], "top5_acc": [], "mse": []},
        "anch": {"sim": [], "acc": [], "top5_acc": [], "mse": []},
        "cum": {"sim": [], "acc": [], "top5_acc": [], "mse": []}
    }
    
    for i in range(10, 15):
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
        
        for t in range(1, labels.shape[1] - config.block_size - 1):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)
            h_anchor = hidden_target[-1][:, -1:, :]
            
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t+1:t+1+config.block_size, :]
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t + 1, t + 1 + config.block_size, dtype=mx.int32)[None]
            
            # Predictions
            pred_abs = model_absolute(noise, ctx_feats, audio_summary, pos_ids, h_anchor)
            pred_anch = model_anchor(noise, ctx_feats, audio_summary, pos_ids, h_anchor)
            pred_cum = model_cumulative(noise, ctx_feats, audio_summary, pos_ids, h_anchor)
            
            # Reconstruction MSE
            metrics["abs"]["mse"].append(mx.mean(mx.square(pred_abs - true_hidden)).item())
            metrics["anch"]["mse"].append(mx.mean(mx.square(pred_anch - true_hidden)).item())
            metrics["cum"]["mse"].append(mx.mean(mx.square(pred_cum - true_hidden)).item())
            
            # Cosine similarity evaluation
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                
                h_a = pred_abs[0, k]
                sim_a = (mx.sum(h_a * h_true) / (mx.linalg.norm(h_a) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["abs"]["sim"].append(sim_a)
                
                h_an = pred_anch[0, k]
                sim_an = (mx.sum(h_an * h_true) / (mx.linalg.norm(h_an) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["anch"]["sim"].append(sim_an)
                
                h_c = pred_cum[0, k]
                sim_c = (mx.sum(h_c * h_true) / (mx.linalg.norm(h_c) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["cum"]["sim"].append(sim_c)
                
            # Logit projections
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            abs_logits = target.decoder.token_embedding.as_linear(pred_abs.reshape(-1, d_target))
            anch_logits = target.decoder.token_embedding.as_linear(pred_anch.reshape(-1, d_target))
            cum_logits = target.decoder.token_embedding.as_linear(pred_cum.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            abs_tokens = mx.argmax(abs_logits, axis=-1).tolist()
            anch_tokens = mx.argmax(anch_logits, axis=-1).tolist()
            cum_tokens = mx.argmax(cum_logits, axis=-1).tolist()
            
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                metrics["abs"]["acc"].append(1.0 if abs_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["anch"]["acc"].append(1.0 if  anch_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["cum"]["acc"].append(1.0 if cum_tokens[idx] == true_tokens[idx] else 0.0)
                
                abs_top5 = abs_tokens[idx] in top5_indices[idx].tolist()
                anch_top5 = anch_tokens[idx] in top5_indices[idx].tolist()
                cum_top5 = cum_tokens[idx] in top5_indices[idx].tolist()
                
                metrics["abs"]["top5_acc"].append(1.0 if abs_top5 else 0.0)
                metrics["anch"]["top5_acc"].append(1.0 if anch_top5 else 0.0)
                metrics["cum"]["top5_acc"].append(1.0 if cum_top5 else 0.0)

    print("\n" + "="*70)
    print("RESULTS: CONTINUOUS STATE RESIDUAL DRAFTING")
    print("="*70)
    print(f"{'Metric':<30} | {'Absolute':<12} | {'Anchor Resid':<12} | {'Cumulative':<12}")
    print("-" * 70)
    print(f"{'Mean Reconstruction MSE':<30} | {np.mean(metrics['abs']['mse']):<12.5f} | {np.mean(metrics['anch']['mse']):<12.5f} | {np.mean(metrics['cum']['mse']):<12.5f}")
    print(f"{'Mean Cosine Similarity':<30} | {np.mean(metrics['abs']['sim']):<12.4f} | {np.mean(metrics['anch']['sim']):<12.4f} | {np.mean(metrics['cum']['sim']):<12.4f}")
    print(f"{'Greedy Token Accuracy':<30} | {np.mean(metrics['abs']['acc'])*100:<11.2f}% | {np.mean(metrics['anch']['acc'])*100:<11.2f}% | {np.mean(metrics['cum']['acc'])*100:<11.2f}%")
    print(f"{'Top-5 Expected Token Acc':<30} | {np.mean(metrics['abs']['top5_acc'])*100:<11.2f}% | {np.mean(metrics['anch']['top5_acc'])*100:<11.2f}% | {np.mean(metrics['cum']['top5_acc'])*100:<11.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_experiment()
