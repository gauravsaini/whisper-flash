#!/usr/bin/env python3
"""
experiment_consistency_drafting.py

Moonshot: Consistency-Model Drafting (One-shot jump)
Translates the concept of Consistency Models (Song et al., 2023) to continuous state drafting.
Instead of predicting the future hidden states sequentially or in a single raw step,
we train a Consistency Model that maps any noisy future state block back to the clean target sequence.
This allows high-fidelity multi-step prediction in a single one-shot jump, or few-shot iterative refinement.
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

# --- 1. Consistency Draft Model Architecture ---
class ConsistencyDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        
        # Project the noisy/clean draft input sequence
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        
        # Context projections (tapped layers context)
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        # Noise level (time) embedding: maps ln(sigma) -> d_draft
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, config.d_draft)
        )
        
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)
        self.target_layer_ids = config.target_layer_ids

    def __call__(self, noisy_y, target_hidden, audio_summary, position_ids, sigma):
        # noisy_y: (batch, block_size, d_target)
        # target_hidden: (batch, block_size * tapped_layers, d_target)
        # audio_summary: (batch, 1, d_target)
        # position_ids: (batch, block_size)
        # sigma: (batch, 1) or scalar
        
        # Project inputs and add positional embedding
        x = self.input_proj(noisy_y) + self.pos_embed(position_ids)
        
        # Embed noise level
        ln_sigma = mx.log(mx.clip(sigma, 1e-9, 1e9))
        if len(ln_sigma.shape) == 1:
            ln_sigma = ln_sigma[:, None]
        sigma_emb = self.sigma_mlp(ln_sigma) # (batch, d_draft)
        
        # Add noise embedding to each draft token representation (broadcasting along block_size)
        x = x + sigma_emb[:, None, :]
        
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        F_out = self.continuous_head(x)
        return F_out

# --- 2. Consistency Prediction Wrapper ---
def get_consistency_prediction(model, y, target_hidden, audio_summary, position_ids, sigma):
    sigma_min = 0.002
    
    # Ensure sigma is an array and has shape (batch, 1)
    if not isinstance(sigma, mx.array):
        sigma = mx.array(sigma)
    if len(sigma.shape) == 1:
        sigma = sigma[:, None]
    elif len(sigma.shape) == 0:
        sigma = sigma[None, None]
    
    # Boundary condition scaling coefficients
    c_skip = (sigma_min ** 2) / ((sigma - sigma_min) ** 2 + sigma_min ** 2)
    c_out = (sigma - sigma_min) / mx.sqrt((sigma - sigma_min) ** 2 + sigma_min ** 2)
    
    # Reshape to (batch, 1, 1) for broadcasting with Y (batch, block_size, d_target)
    c_skip = c_skip[:, :, None]
    c_out = c_out[:, :, None]
    
    F_out = model(y, target_hidden, audio_summary, position_ids, sigma)
    
    return c_skip * y + c_out * F_out


# --- 3. Consistency Training Loss ---
def consistency_loss_fn(
    online_model, target_model, clean_x, ctx, audio, pos, sigma_n, sigma_np1
):
    z = mx.random.normal(clean_x.shape)
    
    # Noisy targets at consecutive noise levels
    x_n = clean_x + sigma_n * z
    x_np1 = clean_x + sigma_np1 * z
    
    batch_size = clean_x.shape[0]
    sigma_n_arr = mx.full((batch_size, 1), sigma_n)
    sigma_np1_arr = mx.full((batch_size, 1), sigma_np1)
    
    # Online network maps x_np1 (higher noise) to clean prediction
    pred_online = get_consistency_prediction(
        online_model, x_np1, ctx, audio, pos, sigma_np1_arr
    )
    
    # Target network maps x_n (lower noise) to clean prediction
    pred_target = get_consistency_prediction(
        target_model, x_n, ctx, audio, pos, sigma_n_arr
    )
    
    # Consistency loss is the distance between predictions
    return mx.mean(mx.square(pred_online - pred_target))

# --- 4. Parameter Utilities ---
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

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Instantiate models
    draft_baseline = ContinuousDraftModel(config)
    consistency_online = ConsistencyDraftModel(config)
    consistency_target = ConsistencyDraftModel(config)
    
    # Force initializations
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    sigma_init = mx.ones((1, 1))
    
    _ = draft_baseline(noise_init, ctx_init, audio_init, pos_init)
    _ = consistency_online(noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = consistency_target(noise_init, ctx_init, audio_init, pos_init, sigma_init)
    
    # Match consistency online and target params
    initial_params = copy_parameters(consistency_online)
    consistency_target.update(initial_params)
    
    print("Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    for i in range(5):  # 5 training samples
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
    
    # Optimizers
    optimizer_baseline = optim.Adam(learning_rate=1e-3)
    optimizer_consistency = optim.Adam(learning_rate=1e-3)
    
    def baseline_loss_fn(model, noise, ctx, audio, pos, true_hidden):
        pred = model(noise, ctx, audio, pos)
        return mx.mean(mx.square(pred - true_hidden))
        
    loss_and_grad_baseline = nn.value_and_grad(draft_baseline, baseline_loss_fn)
    loss_and_grad_consistency = nn.value_and_grad(consistency_online, consistency_loss_fn)
    
    # --- Training loops ---
    epochs = 15
    print("\nTraining Baseline MSE model...")
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            loss, grads = loss_and_grad_baseline(
                draft_baseline, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            optimizer_baseline.update(draft_baseline, grads)
            mx.eval(draft_baseline.parameters(), optimizer_baseline.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Baseline Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Baseline MSE model trained in {time.time() - t0:.1f}s.")
    
    print("\nTraining Consistency Model (Consistency Training)...")
    sigmas = get_sigma_schedule(num_steps=10, sigma_min=0.002, sigma_max=80.0)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            # Sample noise levels n uniformly
            n = np.random.randint(0, len(sigmas) - 1)
            sigma_n = sigmas[n]
            sigma_np1 = sigmas[n+1]
            
            loss, grads = loss_and_grad_consistency(
                consistency_online, consistency_target, data["true_hidden"], data["ctx"], data["audio"], data["pos"], sigma_n, sigma_np1
            )
            optimizer_consistency.update(consistency_online, grads)
            
            # EMA Update target parameters
            update_target_parameters(consistency_online, consistency_target, ema_mu=0.95)
            
            mx.eval(consistency_online.parameters(), consistency_target.parameters(), optimizer_consistency.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Consistency Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Consistency model trained in {time.time() - t0:.1f}s.")
    
    # --- 5. Evaluation ---
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    metrics = {
        "baseline": {"sim": [], "acc": [], "top5_acc": [], "sim_per_step": [[] for _ in range(config.block_size)]},
        "oneshot": {"sim": [], "acc": [], "top5_acc": [], "sim_per_step": [[] for _ in range(config.block_size)]},
        "twoshot": {"sim": [], "acc": [], "top5_acc": [], "sim_per_step": [[] for _ in range(config.block_size)]}
    }
    
    sigma_max = sigmas[-1]
    sigma_min = sigmas[0]
    sigma_mid = 10.0 # Refinement noise level
    
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
            pred_baseline = draft_baseline(noise, ctx_feats, audio_summary, pos_ids)
            
            # 2. Consistency model - One-shot Jump
            z1 = mx.random.normal(true_hidden.shape)
            y1 = sigma_max * z1
            sigma_max_arr = mx.full((1, 1), sigma_max)
            pred_oneshot = get_consistency_prediction(
                consistency_online, y1, ctx_feats, audio_summary, pos_ids, sigma_max_arr
            )
            
            # 3. Consistency model - Two-shot Refinement
            # Step 1 prediction
            pred1 = get_consistency_prediction(
                consistency_online, y1, ctx_feats, audio_summary, pos_ids, sigma_max_arr
            )
            # Denoise step 2: add noise back to sigma_mid, then predict again
            z2 = mx.random.normal(true_hidden.shape)
            y2 = pred1 + math.sqrt(max(sigma_mid**2 - sigma_min**2, 1e-9)) * z2
            sigma_mid_arr = mx.full((1, 1), sigma_mid)
            pred_twoshot = get_consistency_prediction(
                consistency_online, y2, ctx_feats, audio_summary, pos_ids, sigma_mid_arr
            )
            
            # Compute step-wise similarities and evaluate
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                h_base = pred_baseline[0, k]
                h_one = pred_oneshot[0, k]
                h_two = pred_twoshot[0, k]
                
                sim_base = (mx.sum(h_base * h_true) / (mx.linalg.norm(h_base) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_one = (mx.sum(h_one * h_true) / (mx.linalg.norm(h_one) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_two = (mx.sum(h_two * h_true) / (mx.linalg.norm(h_two) * mx.linalg.norm(h_true) + 1e-9)).item()
                
                metrics["baseline"]["sim"].append(sim_base)
                metrics["oneshot"]["sim"].append(sim_one)
                metrics["twoshot"]["sim"].append(sim_two)
                
                metrics["baseline"]["sim_per_step"][k].append(sim_base)
                metrics["oneshot"]["sim_per_step"][k].append(sim_one)
                metrics["twoshot"]["sim_per_step"][k].append(sim_two)
                
            # Compute token accuracies
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            base_logits = target.decoder.token_embedding.as_linear(pred_baseline.reshape(-1, d_target))
            one_logits = target.decoder.token_embedding.as_linear(pred_oneshot.reshape(-1, d_target))
            two_logits = target.decoder.token_embedding.as_linear(pred_twoshot.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            base_tokens = mx.argmax(base_logits, axis=-1).tolist()
            one_tokens = mx.argmax(one_logits, axis=-1).tolist()
            two_tokens = mx.argmax(two_logits, axis=-1).tolist()
            
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                metrics["baseline"]["acc"].append(1.0 if base_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["oneshot"]["acc"].append(1.0 if one_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["twoshot"]["acc"].append(1.0 if two_tokens[idx] == true_tokens[idx] else 0.0)
                
                metrics["baseline"]["top5_acc"].append(1.0 if base_tokens[idx] in top5_indices[idx].tolist() else 0.0)
                metrics["oneshot"]["top5_acc"].append(1.0 if one_tokens[idx] in top5_indices[idx].tolist() else 0.0)
                metrics["twoshot"]["top5_acc"].append(1.0 if two_tokens[idx] in top5_indices[idx].tolist() else 0.0)

    # Calculate average scores
    print("\n" + "="*60)
    print("RESULTS: CONSISTENCY-MODEL DRAFTING (ONE-SHOT JUMP & REFINEMENT)")
    print("="*60)
    
    print("\n--- 1. Hidden Representation Cosine Similarity ---")
    print(f"MSE Baseline       : {np.mean(metrics['baseline']['sim']):.4f}")
    print(f"Consistency (1-shot): {np.mean(metrics['oneshot']['sim']):.4f}  (Delta: {np.mean(metrics['oneshot']['sim']) - np.mean(metrics['baseline']['sim']):+.4f})")
    print(f"Consistency (2-shot): {np.mean(metrics['twoshot']['sim']):.4f}  (Delta: {np.mean(metrics['twoshot']['sim']) - np.mean(metrics['baseline']['sim']):+.4f})")
    
    print("\n--- 2. Greedy Token Accuracy (Projection Match) ---")
    print(f"MSE Baseline       : {np.mean(metrics['baseline']['acc'])*100:.2f}%")
    print(f"Consistency (1-shot): {np.mean(metrics['oneshot']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['oneshot']['acc']) - np.mean(metrics['baseline']['acc']))*100:+.2f}%)")
    print(f"Consistency (2-shot): {np.mean(metrics['twoshot']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['twoshot']['acc']) - np.mean(metrics['baseline']['acc']))*100:+.2f}%)")
    
    print("\n--- 3. Top-5 Expected Token Acceptance Rate ---")
    print(f"MSE Baseline       : {np.mean(metrics['baseline']['top5_acc'])*100:.2f}%")
    print(f"Consistency (1-shot): {np.mean(metrics['oneshot']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['oneshot']['top5_acc']) - np.mean(metrics['baseline']['top5_acc']))*100:+.2f}%)")
    print(f"Consistency (2-shot): {np.mean(metrics['twoshot']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['twoshot']['top5_acc']) - np.mean(metrics['baseline']['top5_acc']))*100:+.2f}%)")
    
    print("\n--- 4. Cosine Similarity Drift Per Step ---")
    for k in range(config.block_size):
        mean_base = np.mean(metrics['baseline']['sim_per_step'][k])
        mean_one = np.mean(metrics['oneshot']['sim_per_step'][k])
        mean_two = np.mean(metrics['twoshot']['sim_per_step'][k])
        print(f"Step +{k+1}: Baseline={mean_base:.4f} | CM(1-shot)={mean_one:.4f} | CM(2-shot)={mean_two:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
