#!/usr/bin/env python3
"""
experiment_dynamic_contrastive.py

Moonshot #7: Dynamic Temperature-Weighted Contrastive Loss for Drafter
Trains the continuous draft model using a dynamic temperature scaled by target entropy,
and compares it to the fixed-temperature logit contrastive baseline.
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

def fixed_contrastive_loss(
    model, target_model, noise, target_hidden, audio_summary, position_ids, true_hidden,
    tau_pred=1.0, tau_target=1.0, alpha=1.0, mse_weight=0.1
):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    N, B, D = pred_hidden.shape
    pred_h_flat = pred_hidden.reshape(-1, D)
    true_h_flat = true_hidden.reshape(-1, D)
    
    pred_logits = target_model.decoder.token_embedding.as_linear(pred_h_flat)
    true_logits = target_model.decoder.token_embedding.as_linear(true_h_flat)
    
    P_target = mx.softmax(true_logits / tau_target, axis=-1)
    y_target = mx.argmax(true_logits, axis=-1)
    
    logits_scaled = pred_logits / tau_pred
    max_logits = mx.max(logits_scaled, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_scaled - max_logits)
    
    w_neg = mx.power(1.0 - P_target, alpha)
    weighted_exp = w_neg * exp_logits
    sum_weighted_exp = mx.sum(weighted_exp, axis=-1)
    
    indices = mx.arange(y_target.shape[0])
    pos_P_target = P_target[indices, y_target]
    pos_exp_logits = exp_logits[indices, y_target]
    
    correction = (1.0 - mx.power(1.0 - pos_P_target, alpha)) * pos_exp_logits
    sum_weighted_exp = sum_weighted_exp + correction
    
    log_denom = max_logits.squeeze(-1) + mx.log(sum_weighted_exp + 1e-9)
    pos_logits = logits_scaled[indices, y_target]
    
    contrastive = mx.mean(log_denom - pos_logits)
    mse = mx.mean(mx.square(pred_h_flat - true_h_flat))
    return contrastive + mse_weight * mse

def dynamic_contrastive_loss(
    model, target_model, noise, target_hidden, audio_summary, position_ids, true_hidden,
    tau_pred=1.0, tau_base=0.5, gamma=0.5, alpha=1.0, mse_weight=0.1
):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    N, B, D = pred_hidden.shape
    pred_h_flat = pred_hidden.reshape(-1, D)
    true_h_flat = true_hidden.reshape(-1, D)
    
    pred_logits = target_model.decoder.token_embedding.as_linear(pred_h_flat)
    true_logits = target_model.decoder.token_embedding.as_linear(true_h_flat)
    
    # Shannon entropy of true logits
    P_base = mx.softmax(true_logits, axis=-1)
    H = -mx.sum(P_base * mx.log(P_base + 1e-9), axis=-1, keepdims=True) # (N*B, 1)
    
    # Dynamic target temperature
    tau_target = tau_base + gamma * H # (N*B, 1)
    
    P_target = mx.softmax(true_logits / tau_target, axis=-1)
    y_target = mx.argmax(true_logits, axis=-1)
    
    logits_scaled = pred_logits / tau_pred
    max_logits = mx.max(logits_scaled, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_scaled - max_logits)
    
    w_neg = mx.power(1.0 - P_target, alpha)
    weighted_exp = w_neg * exp_logits
    sum_weighted_exp = mx.sum(weighted_exp, axis=-1)
    
    indices = mx.arange(y_target.shape[0])
    pos_P_target = P_target[indices, y_target]
    pos_exp_logits = exp_logits[indices, y_target]
    
    correction = (1.0 - mx.power(1.0 - pos_P_target, alpha)) * pos_exp_logits
    sum_weighted_exp = sum_weighted_exp + correction
    
    log_denom = max_logits.squeeze(-1) + mx.log(sum_weighted_exp + 1e-9)
    pos_logits = logits_scaled[indices, y_target]
    
    contrastive = mx.mean(log_denom - pos_logits)
    mse = mx.mean(mx.square(pred_h_flat - true_h_flat))
    return contrastive + mse_weight * mse

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
    
    draft_fixed = ContinuousDraftModel(config)
    draft_dynamic = ContinuousDraftModel(config)
    
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = draft_fixed(noise_init, ctx_init, audio_init, pos_init)
    _ = draft_dynamic(noise_init, ctx_init, audio_init, pos_init)
    
    initial_params = copy_parameters(draft_fixed)
    draft_dynamic.update(initial_params)
    
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
    
    # Train Model 1 (Fixed Logit-Aware Loss)
    print("\n--- Training Model 1: Fixed Temperature Logit Contrastive Loss ---")
    opt_fixed = optim.Adam(learning_rate=1e-3)
    
    def fixed_loss_wrapper(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
        return fixed_contrastive_loss(
            model, target, noise, target_hidden, audio_summary, position_ids, true_hidden,
            tau_pred=1.0, tau_target=1.0, alpha=1.0, mse_weight=0.1
        )
        
    loss_and_grad_fixed = nn.value_and_grad(draft_fixed, fixed_loss_wrapper)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_fixed(
                draft_fixed, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_fixed.update(draft_fixed, grads)
            mx.eval(draft_fixed.parameters(), opt_fixed.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Fixed baseline trained in {time.time() - t0:.1f}s.")
    
    # Train Model 2 (Dynamic Temperature Loss)
    print("\n--- Training Model 2: Dynamic Temperature-Weighted Contrastive Loss ---")
    opt_dynamic = optim.Adam(learning_rate=1e-3)
    
    def dynamic_loss_wrapper(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
        return dynamic_contrastive_loss(
            model, target, noise, target_hidden, audio_summary, position_ids, true_hidden,
            tau_pred=1.0, tau_base=0.5, gamma=0.5, alpha=1.0, mse_weight=0.1
        )
        
    loss_and_grad_dynamic = nn.value_and_grad(draft_dynamic, dynamic_loss_wrapper)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_dynamic(
                draft_dynamic, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_dynamic.update(draft_dynamic, grads)
            mx.eval(draft_dynamic.parameters(), opt_dynamic.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Dynamic model trained in {time.time() - t0:.1f}s.")
    
    # Evaluation
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    metrics = {
        "fixed": {"sim": [], "acc": [], "top5_acc": []},
        "dynamic": {"sim": [], "acc": [], "top5_acc": []}
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
            
            pred_fixed = draft_fixed(noise, ctx_feats, audio_summary, pos_ids)
            pred_dynamic = draft_dynamic(noise, ctx_feats, audio_summary, pos_ids)
            
            # Cosine similarity
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                h_fixed = pred_fixed[0, k]
                h_dynamic = pred_dynamic[0, k]
                
                sim_fixed = (mx.sum(h_fixed * h_true) / (mx.linalg.norm(h_fixed) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_dynamic = (mx.sum(h_dynamic * h_true) / (mx.linalg.norm(h_dynamic) * mx.linalg.norm(h_true) + 1e-9)).item()
                
                metrics["fixed"]["sim"].append(sim_fixed)
                metrics["dynamic"]["sim"].append(sim_dynamic)
                
            # Logit projection accuracy
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            fixed_logits = target.decoder.token_embedding.as_linear(pred_fixed.reshape(-1, d_target))
            dynamic_logits = target.decoder.token_embedding.as_linear(pred_dynamic.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            fixed_tokens = mx.argmax(fixed_logits, axis=-1).tolist()
            dynamic_tokens = mx.argmax(dynamic_logits, axis=-1).tolist()
            
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                metrics["fixed"]["acc"].append(1.0 if fixed_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["dynamic"]["acc"].append(1.0 if dynamic_tokens[idx] == true_tokens[idx] else 0.0)
                
                fixed_top5 = fixed_tokens[idx] in top5_indices[idx].tolist()
                dynamic_top5 = dynamic_tokens[idx] in top5_indices[idx].tolist()
                
                metrics["fixed"]["top5_acc"].append(1.0 if fixed_top5 else 0.0)
                metrics["dynamic"]["top5_acc"].append(1.0 if dynamic_top5 else 0.0)

    print("\n" + "="*50)
    print("RESULTS: FIXED VS DYNAMIC TEMPERATURE CONTRASTIVE LOSS")
    print("="*50)
    print("--- 1. Hidden Representation Cosine Similarity ---")
    print(f"Fixed Baseline     : {np.mean(metrics['fixed']['sim']):.4f}")
    print(f"Dynamic Contrast   : {np.mean(metrics['dynamic']['sim']):.4f}  (Delta: {np.mean(metrics['dynamic']['sim']) - np.mean(metrics['fixed']['sim']):+.4f})")
    
    print("\n--- 2. Greedy Token Accuracy (Projection Match) ---")
    print(f"Fixed Baseline     : {np.mean(metrics['fixed']['acc'])*100:.2f}%")
    print(f"Dynamic Contrast   : {np.mean(metrics['dynamic']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['dynamic']['acc']) - np.mean(metrics['fixed']['acc']))*100:+.2f}%)")
    
    print("\n--- 3. Top-5 Expected Token Acceptance Rate ---")
    print(f"Fixed Baseline     : {np.mean(metrics['fixed']['top5_acc'])*100:.2f}%")
    print(f"Dynamic Contrast   : {np.mean(metrics['dynamic']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['dynamic']['top5_acc']) - np.mean(metrics['fixed']['top5_acc']))*100:+.2f}%)")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
