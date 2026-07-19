#!/usr/bin/env python3
"""
experiment_moe_drafting.py

Moonshot #16: Continuous MoE Routing (Representation Space)
Tests whether dynamically routing continuous representations through a Mixture-of-Experts (MoE) 
layer allows the drafter to allocate capacity based on trajectory complexity, improving 
overall cosine similarity and parameter efficiency.
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

class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts (simple 2-layer MLPs)
        self.experts = []
        for _ in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            )
            self.experts.append(expert)
            
    def __call__(self, x):
        # x: (B, seq_len, d_model)
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # Router logits
        logits = self.router(x_flat) # (B*L, num_experts)
        probs = mx.softmax(logits, axis=-1) # (B*L, num_experts)
        
        # Top-1 routing for simplicity (with soft assignment during training)
        # Using soft routing so all experts get gradients weighted by prob
        out = mx.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)
            out = out + probs[:, i:i+1] * expert_out
            
        return out.reshape(B, L, D)

class MoEDraftModel(nn.Module):
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
        
        # Insert MoE before final projection
        self.moe = MoELayer(config.d_draft, num_experts=4)
        
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        # Route through MoE
        x = self.moe(x)
        
        x = self.norm(x)
        return self.continuous_head(x)

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred - true_hidden))

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    model_baseline = ContinuousDraftModel(config)
    model_moe = MoEDraftModel(config)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    # Train on 10 samples for speed
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
            
            # Force evaluation to prevent MLX from building a massive computation graph
            mx.eval(ctx_feats, true_hidden, noise, pos_ids, audio_summary)
            
            data_tensors.append({
                "noise": noise,
                "ctx": ctx_feats,
                "audio": audio_summary,
                "pos": pos_ids,
                "true_hidden": true_hidden
            })
            
    print(f"Extracted {len(data_tensors)} training samples.")
    
    epochs = 4
    
    # 1. Train Baseline
    print("\n--- Training Model 1: Dense Baseline ---")
    opt_base = optim.Adam(learning_rate=1e-3)
    val_and_grad_base = nn.value_and_grad(model_baseline, mse_loss)
    
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = val_and_grad_base(
                model_baseline, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_base.update(model_baseline, grads)
            mx.eval(model_baseline.parameters(), opt_base.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Baseline trained in {time.time() - t0:.1f}s.")
    
    # 2. Train MoE
    print("\n--- Training Model 2: Continuous MoE Drafter ---")
    opt_moe = optim.Adam(learning_rate=1e-3)
    val_and_grad_moe = nn.value_and_grad(model_moe, mse_loss)
    
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = val_and_grad_moe(
                model_moe, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_moe.update(model_moe, grads)
            mx.eval(model_moe.parameters(), opt_moe.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"MoE trained in {time.time() - t0:.1f}s.")
    
    print("\nEvaluating on validation samples...")
    
    sims = {"dense": [], "moe": []}
    accs = {"dense": [], "moe": []}
    
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
            
            p_dense = model_baseline(noise, ctx_feats, audio_summary, pos_ids)
            p_moe = model_moe(noise, ctx_feats, audio_summary, pos_ids)
            
            for k in range(config.block_size):
                ht = true_hidden[0, k]
                hd = p_dense[0, k]
                hm = p_moe[0, k]
                
                sims["dense"].append((mx.sum(hd * ht) / (mx.linalg.norm(hd) * mx.linalg.norm(ht) + 1e-9)).item())
                sims["moe"].append((mx.sum(hm * ht) / (mx.linalg.norm(hm) * mx.linalg.norm(ht) + 1e-9)).item())
                
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            t_tokens = mx.argmax(true_logits, axis=-1).tolist()
            d_tokens = mx.argmax(target.decoder.token_embedding.as_linear(p_dense.reshape(-1, d_target)), axis=-1).tolist()
            m_tokens = mx.argmax(target.decoder.token_embedding.as_linear(p_moe.reshape(-1, d_target)), axis=-1).tolist()
            
            accs["dense"].extend([1.0 if d == t else 0.0 for d, t in zip(d_tokens, t_tokens)])
            accs["moe"].extend([1.0 if m == t else 0.0 for m, t in zip(m_tokens, t_tokens)])

    print("\n" + "="*70)
    print("RESULTS: CONTINUOUS MoE ROUTING")
    print("="*70)
    print(f"{'Metric':<25} | {'Dense Baseline':<20} | {'MoE Drafter (4 Experts)':<20}")
    print("-" * 70)
    print(f"{'Mean Cosine Similarity':<25} | {np.mean(sims['dense']):<20.4f} | {np.mean(sims['moe']):<20.4f}")
    print(f"{'Greedy Token Accuracy':<25} | {np.mean(accs['dense'])*100:<19.2f}% | {np.mean(accs['moe'])*100:<19.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_experiment()
