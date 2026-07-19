#!/usr/bin/env python3
"""
experiment_low_rank.py

Moonshot #6: Low-Rank Continuous Subspace Drafting
Compares three speculative drafting architectures to optimize memory and compute constraints:
1. Full-Rank Baseline (Direct D_draft -> D_target linear mapping)
2. Factorized Low-Rank (Learnable bottleneck layer of rank R)
3. Subspace-Projected (Fixed SVD/PCA-based project-down and project-up transformation)
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

class FullRankDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        # We manually import DFlashDecoderLayer to avoid import issues
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        return self.continuous_head(x)

class FactorizedDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig, rank: int = 64):
        super().__init__()
        self.config = config
        self.rank = rank
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # Learnable low-rank bottleneck
        self.proj_down = nn.Linear(config.d_draft, rank, bias=False)
        self.proj_up = nn.Linear(rank, config.d_target, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        # Factorized low-rank map
        bottleneck = self.proj_down(x)
        return self.proj_up(bottleneck)

class SubspaceDrafter(nn.Module):
    def __init__(self, config: WhisperDFlashConfig, rank: int = 64):
        super().__init__()
        self.config = config
        self.rank = rank
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        from whisper_flash_mlx.draft_model import DFlashDecoderLayer
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # Predicts low-rank subspace coordinates directly
        self.continuous_head = nn.Linear(config.d_draft, rank, bias=False)

    def __call__(self, noise, target_hidden, audio_summary, position_ids):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        return self.continuous_head(x)

# --- 2. Loss Functions ---

def baseline_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred - true_hidden))

def subspace_loss(model, noise, target_hidden, audio_summary, position_ids, true_z):
    pred_z = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_z - true_z))

def copy_parameters(src, dst):
    if isinstance(src, dict) and isinstance(dst, dict):
        for k, v in src.items():
            if k in dst:
                if isinstance(v, dict) and isinstance(dst[k], dict):
                    copy_parameters(v, dst[k])
                elif isinstance(v, list) and isinstance(dst[k], list):
                    copy_parameters(v, dst[k])
                elif not isinstance(v, (dict, list)) and not isinstance(dst[k], (dict, list)):
                    if hasattr(v, "shape") and hasattr(dst[k], "shape"):
                        if v.shape == dst[k].shape:
                            dst[k] = mx.array(v)
    elif isinstance(src, list) and isinstance(dst, list):
        for i in range(min(len(src), len(dst))):
            v = src[i]
            d = dst[i]
            if isinstance(v, dict) and isinstance(d, dict):
                copy_parameters(v, d)
            elif isinstance(v, list) and isinstance(d, list):
                copy_parameters(v, d)
            elif not isinstance(v, (dict, list)) and not isinstance(d, (dict, list)):
                if hasattr(v, "shape") and hasattr(d, "shape"):
                    if v.shape == d.shape:
                        dst[i] = mx.array(v)

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    rank = 64
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Initialize models
    model_baseline = FullRankDrafter(config)
    model_factorized = FactorizedDrafter(config, rank=rank)
    model_subspace = SubspaceDrafter(config, rank=rank)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = model_baseline(noise_init, ctx_init, audio_init, pos_init)
    _ = model_factorized(noise_init, ctx_init, audio_init, pos_init)
    _ = model_subspace(noise_init, ctx_init, audio_init, pos_init)
    
    # Align starting weights for non-head parameters to ensure fair comparison
    base_params = model_baseline.parameters()
    copy_parameters(base_params, model_factorized.parameters())
    copy_parameters(base_params, model_subspace.parameters())
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    for i in range(5):
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
            
    print(f"Extracted {len(data_tensors)} training samples.")
    
    # -----------------------------------------------------------------------
    # Compute SVD / PCA subspace components from true hidden training states
    # -----------------------------------------------------------------------
    print("Computing PCA/SVD subspace components...")
    all_true_h = np.concatenate([np.array(d["true_hidden"]) for d in data_tensors], axis=0) # (M, B, D)
    M_samples, B_block, D_dim = all_true_h.shape
    X = all_true_h.reshape(-1, D_dim) # (M_samples * B_block, D_dim)
    
    mean = np.mean(X, axis=0, keepdims=True) # (1, D_dim)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Top R principal components
    V = Vt[:rank, :].T # (D_dim, rank)
    
    # Convert PCA parameters to MLX arrays
    mean_mx = mx.array(mean) # (1, D_dim)
    V_mx = mx.array(V) # (D_dim, rank)
    
    # Inject PCA projections into data_tensors
    for d in data_tensors:
        true_h = d["true_hidden"]
        # Project true hidden to subspace coordinates z
        true_z = (true_h - mean_mx) @ V_mx # (1, B, rank)
        d["true_z"] = true_z
        
    print(f"Subspace dimension configured: {rank} (compressed from {d_target}).")
    
    # -----------------------------------------------------------------------
    # Train Models
    # -----------------------------------------------------------------------
    
    # 1. Train Baseline
    print("\n--- Training Model 1: Full-Rank Baseline ---")
    opt_base = optim.Adam(learning_rate=1e-3)
    grad_base = nn.value_and_grad(model_baseline, baseline_loss)
    t0 = time.time()
    for epoch in range(3):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_base(
                model_baseline, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_base.update(model_baseline, grads)
            mx.eval(model_baseline.parameters(), opt_base.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/3 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_base = time.time() - t0
    print(f"Full-Rank Baseline trained in {t_base:.1f}s.")
    
    # 2. Train Factorized
    print("\n--- Training Model 2: Factorized Low-Rank ---")
    opt_fact = optim.Adam(learning_rate=1e-3)
    grad_fact = nn.value_and_grad(model_factorized, baseline_loss)
    t0 = time.time()
    for epoch in range(3):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_fact(
                model_factorized, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_fact.update(model_factorized, grads)
            mx.eval(model_factorized.parameters(), opt_fact.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/3 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_fact = time.time() - t0
    print(f"Factorized Low-Rank trained in {t_fact:.1f}s.")
    
    # 3. Train Subspace
    print("\n--- Training Model 3: PCA/SVD Subspace-Projected ---")
    opt_sub = optim.Adam(learning_rate=1e-3)
    grad_sub = nn.value_and_grad(model_subspace, subspace_loss)
    t0 = time.time()
    for epoch in range(3):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = grad_sub(
                model_subspace, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_z"]
            )
            opt_sub.update(model_subspace, grads)
            mx.eval(model_subspace.parameters(), opt_sub.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/3 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_sub = time.time() - t0
    print(f"Subspace-Projected trained in {t_sub:.1f}s.")
    
    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    
    metrics = {
        "base": {"sim": [], "acc": [], "top5_acc": [], "param_count": 256 * d_target},
        "fact": {"sim": [], "acc": [], "top5_acc": [], "param_count": 256 * rank + rank * d_target},
        "sub": {"sim": [], "acc": [], "top5_acc": [], "param_count": 256 * rank}
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
            
            # 1. Baseline prediction
            pred_base = model_baseline(noise, ctx_feats, audio_summary, pos_ids)
            
            # 2. Factorized prediction
            pred_fact = model_factorized(noise, ctx_feats, audio_summary, pos_ids)
            
            # 3. Subspace prediction and SVD reconstruction
            pred_z = model_subspace(noise, ctx_feats, audio_summary, pos_ids)
            pred_sub = pred_z @ V_mx.T + mean_mx
            
            # Cosine similarity evaluation in original D_target space
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                
                h_base = pred_base[0, k]
                sim_base = (mx.sum(h_base * h_true) / (mx.linalg.norm(h_base) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["base"]["sim"].append(sim_base)
                
                h_fact = pred_fact[0, k]
                sim_fact = (mx.sum(h_fact * h_true) / (mx.linalg.norm(h_fact) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["fact"]["sim"].append(sim_fact)
                
                h_sub = pred_sub[0, k]
                sim_sub = (mx.sum(h_sub * h_true) / (mx.linalg.norm(h_sub) * mx.linalg.norm(h_true) + 1e-9)).item()
                metrics["sub"]["sim"].append(sim_sub)
                
            # Logit projections
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            base_logits = target.decoder.token_embedding.as_linear(pred_base.reshape(-1, d_target))
            fact_logits = target.decoder.token_embedding.as_linear(pred_fact.reshape(-1, d_target))
            sub_logits = target.decoder.token_embedding.as_linear(pred_sub.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            base_tokens = mx.argmax(base_logits, axis=-1).tolist()
            fact_tokens = mx.argmax(fact_logits, axis=-1).tolist()
            sub_tokens = mx.argmax(sub_logits, axis=-1).tolist()
            
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                # Greedy match
                metrics["base"]["acc"].append(1.0 if base_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["fact"]["acc"].append(1.0 if fact_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["sub"]["acc"].append(1.0 if sub_tokens[idx] == true_tokens[idx] else 0.0)
                
                # Top 5 match
                base_top5 = base_tokens[idx] in top5_indices[idx].tolist()
                fact_top5 = fact_tokens[idx] in top5_indices[idx].tolist()
                sub_top5 = sub_tokens[idx] in top5_indices[idx].tolist()
                
                metrics["base"]["top5_acc"].append(1.0 if base_top5 else 0.0)
                metrics["fact"]["top5_acc"].append(1.0 if fact_top5 else 0.0)
                metrics["sub"]["top5_acc"].append(1.0 if sub_top5 else 0.0)

    print("\n" + "="*70)
    print("RESULTS: LOW-RANK SUBSPACE SPECULATIVE DRAFTING")
    print("="*70)
    print(f"{'Metric':<30} | {'Baseline':<10} | {'Factorized':<10} | {'Subspace (PCA)':<10}")
    print("-" * 70)
    print(f"{'Output Head Parameters':<30} | {metrics['base']['param_count']:<10d} | {metrics['fact']['param_count']:<10d} | {metrics['sub']['param_count']:<10d}")
    print(f"{'Param Savings vs Baseline':<30} | {'0.0%':<10} | {((1 - metrics['fact']['param_count']/metrics['base']['param_count'])*100):-7.2f}% | {((1 - metrics['sub']['param_count']/metrics['base']['param_count'])*100):-7.2f}%")
    print(f"{'Mean Cosine Similarity':<30} | {np.mean(metrics['base']['sim']):<10.4f} | {np.mean(metrics['fact']['sim']):<10.4f} | {np.mean(metrics['sub']['sim']):<10.4f}")
    print(f"{'Greedy Token Accuracy':<30} | {np.mean(metrics['base']['acc'])*100:<9.2f}% | {np.mean(metrics['fact']['acc'])*100:<9.2f}% | {np.mean(metrics['sub']['acc'])*100:<9.2f}%")
    print(f"{'Top-5 Expected Token Acc':<30} | {np.mean(metrics['base']['top5_acc'])*100:<9.2f}% | {np.mean(metrics['fact']['top5_acc'])*100:<9.2f}% | {np.mean(metrics['sub']['top5_acc'])*100:<9.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_experiment()
