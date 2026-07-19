#!/usr/bin/env python3
"""
experiment_subspace_drafting.py

Moonshot #6: Low-Rank Continuous Subspace Drafting
Trains continuous draft models constrained to a low-rank subspace (rank=64)
using:
1. Static SVD/PCA projection of target hidden states.
2. A learnable low-rank bottleneck (factorized output head).
Compares them against the standard full-dimensional MSE training baseline (d_target=384).
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
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel, DFlashDecoderLayer

# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

class StaticSubspaceDraftModel(nn.Module):
    """Draft model predicting in a static low-rank subspace."""
    def __init__(self, config: WhisperDFlashConfig, rank: int):
        super().__init__()
        self.config = config
        self.rank = rank
        
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # Predicts directly in rank-dimensional subspace
        self.continuous_head = nn.Linear(config.d_draft, rank, bias=False)
        
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
            
        x = self.norm(x)
        predicted_subspace = self.continuous_head(x)
        return predicted_subspace

    def count_params(self) -> int:
        from mlx.utils import tree_flatten
        return sum(x.size for k, x in tree_flatten(self.parameters()))


class LearnableSubspaceDraftModel(nn.Module):
    """Draft model with a learnable low-rank bottleneck head."""
    def __init__(self, config: WhisperDFlashConfig, rank: int):
        super().__init__()
        self.config = config
        self.rank = rank
        
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # Factorized low-rank head (d_draft -> rank -> d_target)
        self.continuous_head_down = nn.Linear(config.d_draft, rank, bias=False)
        self.continuous_head_up = nn.Linear(rank, config.d_target, bias=False)
        
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = self.input_proj(noise_embedding) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
            
        x = self.norm(x)
        z = self.continuous_head_down(x)
        predicted_hidden = self.continuous_head_up(z)
        return predicted_hidden

    def count_params(self) -> int:
        from mlx.utils import tree_flatten
        return sum(x.size for k, x in tree_flatten(self.parameters()))


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def static_subspace_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden, V_r):
    pred_z = model(noise, target_hidden, audio_summary, position_ids) # (1, block_size, rank)
    true_z = true_hidden @ V_r # (1, block_size, rank)
    return mx.mean(mx.square(pred_z - true_z))


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment():
    rank = 64
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    d_target = target.dims.n_text_state
    print(f"Target hidden dimension: {d_target}, Subspace Rank: {rank}")
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Initialize the three models
    draft_baseline = ContinuousDraftModel(config)
    draft_static = StaticSubspaceDraftModel(config, rank=rank)
    draft_learnable = LearnableSubspaceDraftModel(config, rank=rank)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = draft_baseline(noise_init, ctx_init, audio_init, pos_init)
    _ = draft_static(noise_init, ctx_init, audio_init, pos_init)
    _ = draft_learnable(noise_init, ctx_init, audio_init, pos_init)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # 1. Collect dataset tensors
    print("Pre-extracting dataset context features...")
    data_tensors = []
    all_true_hiddens = []
    
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
            # Convert to numpy for SVD computation
            all_true_hiddens.append(np.array(true_hidden.reshape(-1, d_target)))
            
    print(f"Extracted {len(data_tensors)} training samples.")
    
    # 2. Compute SVD Projection Matrix
    print("Computing SVD on target hidden states...")
    H = np.concatenate(all_true_hiddens, axis=0) # (N, d_target)
    # H has shape (Num_samples * block_size, d_target)
    # Perform SVD to get principal components
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    # Take the top `rank` principal components (columns of V, or rows of Vt)
    V_r_np = Vt[:rank, :].T # (d_target, rank)
    V_r = mx.array(V_r_np)
    
    # Measure explained variance
    total_var = np.sum(S**2)
    explained_var = np.sum(S[:rank]**2)
    print(f"Explained Variance Ratio by top-{rank} components: {explained_var / total_var * 100:.2f}%")
    
    # 3. Train Model 1 (Baseline MSE)
    print("\n--- Training Model 1: Baseline MSE (Full-Space) ---")
    opt_base = optim.Adam(learning_rate=1e-3)
    loss_and_grad_base = nn.value_and_grad(draft_baseline, mse_loss)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_base(
                draft_baseline, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_base.update(draft_baseline, grads)
            mx.eval(draft_baseline.parameters(), opt_base.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_base = time.time() - t0
    print(f"Baseline trained in {t_base:.1f}s.")
    
    # 4. Train Model 2 (Static Subspace SVD)
    print("\n--- Training Model 2: Static Subspace (SVD) ---")
    opt_static = optim.Adam(learning_rate=1e-3)
    
    def static_loss_wrapper(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
        return static_subspace_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden, V_r)
        
    loss_and_grad_static = nn.value_and_grad(draft_static, static_loss_wrapper)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_static(
                draft_static, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_static.update(draft_static, grads)
            mx.eval(draft_static.parameters(), opt_static.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_static = time.time() - t0
    print(f"Static Subspace model trained in {t_static:.1f}s.")
    
    # 5. Train Model 3 (Learnable Subspace Bottleneck)
    print("\n--- Training Model 3: Learnable Subspace (Bottleneck) ---")
    opt_learnable = optim.Adam(learning_rate=1e-3)
    loss_and_grad_learnable = nn.value_and_grad(draft_learnable, mse_loss)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_learnable(
                draft_learnable, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_learnable.update(draft_learnable, grads)
            mx.eval(draft_learnable.parameters(), opt_learnable.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    t_learnable = time.time() - t0
    print(f"Learnable Subspace model trained in {t_learnable:.1f}s.")
    
    # 6. Evaluation
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    
    metrics = {
        "baseline": {"sim": [], "acc": [], "top5_acc": []},
        "static": {"sim": [], "acc": [], "top5_acc": []},
        "learnable": {"sim": [], "acc": [], "top5_acc": []}
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
            
            # Predict
            pred_base = draft_baseline(noise, ctx_feats, audio_summary, pos_ids)
            pred_static_z = draft_static(noise, ctx_feats, audio_summary, pos_ids)
            # Reconstruct static predictions: z @ V_r^T
            pred_static = pred_static_z @ V_r.T
            pred_learnable = draft_learnable(noise, ctx_feats, audio_summary, pos_ids)
            
            # Compute similarity
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                h_base = pred_base[0, k]
                h_static = pred_static[0, k]
                h_learn = pred_learnable[0, k]
                
                sim_base = (mx.sum(h_base * h_true) / (mx.linalg.norm(h_base) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_static = (mx.sum(h_static * h_true) / (mx.linalg.norm(h_static) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_learn = (mx.sum(h_learn * h_true) / (mx.linalg.norm(h_learn) * mx.linalg.norm(h_true) + 1e-9)).item()
                
                metrics["baseline"]["sim"].append(sim_base)
                metrics["static"]["sim"].append(sim_static)
                metrics["learnable"]["sim"].append(sim_learn)
                
            # Logit projections and accuracy
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            base_logits = target.decoder.token_embedding.as_linear(pred_base.reshape(-1, d_target))
            static_logits = target.decoder.token_embedding.as_linear(pred_static.reshape(-1, d_target))
            learnable_logits = target.decoder.token_embedding.as_linear(pred_learnable.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            base_tokens = mx.argmax(base_logits, axis=-1).tolist()
            static_tokens = mx.argmax(static_logits, axis=-1).tolist()
            learnable_tokens = mx.argmax(learnable_logits, axis=-1).tolist()
            
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                # Greedy accuracy
                metrics["baseline"]["acc"].append(1.0 if base_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["static"]["acc"].append(1.0 if static_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["learnable"]["acc"].append(1.0 if learnable_tokens[idx] == true_tokens[idx] else 0.0)
                
                # Top-5 accuracy
                base_top5 = base_tokens[idx] in top5_indices[idx].tolist()
                static_top5 = static_tokens[idx] in top5_indices[idx].tolist()
                learnable_top5 = learnable_tokens[idx] in top5_indices[idx].tolist()
                
                metrics["baseline"]["top5_acc"].append(1.0 if base_top5 else 0.0)
                metrics["static"]["top5_acc"].append(1.0 if static_top5 else 0.0)
                metrics["learnable"]["top5_acc"].append(1.0 if learnable_top5 else 0.0)

    # Param counts
    try:
        from mlx.utils import tree_flatten
        params_base = sum(x.size for k, x in tree_flatten(draft_baseline.parameters()))
        params_static = sum(x.size for k, x in tree_flatten(draft_static.parameters()))
        params_learnable = sum(x.size for k, x in tree_flatten(draft_learnable.parameters()))
    except Exception:
        params_base = draft_baseline.count_params() if hasattr(draft_baseline, "count_params") else 0
        params_static = draft_static.count_params() if hasattr(draft_static, "count_params") else 0
        params_learnable = draft_learnable.count_params() if hasattr(draft_learnable, "count_params") else 0


    print("\n" + "="*70)
    print("RESULTS: LOW-RANK SUBSPACE DRAFTING (RANK=64) VS BASELINE (D=384)")
    print("="*70)
    
    print("--- 1. Hidden Representation Cosine Similarity ---")
    print(f"Baseline (Full)     : {np.mean(metrics['baseline']['sim']):.4f}")
    print(f"Static Subspace     : {np.mean(metrics['static']['sim']):.4f}  (Delta: {np.mean(metrics['static']['sim']) - np.mean(metrics['baseline']['sim']):+.4f})")
    print(f"Learnable Subspace  : {np.mean(metrics['learnable']['sim']):.4f}  (Delta: {np.mean(metrics['learnable']['sim']) - np.mean(metrics['baseline']['sim']):+.4f})")
    
    print("\n--- 2. Greedy Token Accuracy (Projection Match) ---")
    print(f"Baseline (Full)     : {np.mean(metrics['baseline']['acc'])*100:.2f}%")
    print(f"Static Subspace     : {np.mean(metrics['static']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['static']['acc']) - np.mean(metrics['baseline']['acc']))*100:+.2f}%)")
    print(f"Learnable Subspace  : {np.mean(metrics['learnable']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['learnable']['acc']) - np.mean(metrics['baseline']['acc']))*100:+.2f}%)")
    
    print("\n--- 3. Top-5 Expected Token Acceptance Rate ---")
    print(f"Baseline (Full)     : {np.mean(metrics['baseline']['top5_acc'])*100:.2f}%")
    print(f"Static Subspace     : {np.mean(metrics['static']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['static']['top5_acc']) - np.mean(metrics['baseline']['top5_acc']))*100:+.2f}%)")
    print(f"Learnable Subspace  : {np.mean(metrics['learnable']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['learnable']['top5_acc']) - np.mean(metrics['baseline']['top5_acc']))*100:+.2f}%)")
    
    print("\n--- 4. Model Parameter Counts ---")
    print(f"Baseline (Full)     : {params_base:,} params")
    print(f"Static Subspace     : {params_static:,} params (Saved: {params_base - params_static:,} params)")
    print(f"Learnable Subspace  : {params_learnable:,} params (Saved: {params_base - params_learnable:,} params)")
    print("="*70)
    
if __name__ == "__main__":
    run_experiment()
