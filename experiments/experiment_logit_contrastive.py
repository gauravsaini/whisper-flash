#!/usr/bin/env python3
"""
experiment_logit_contrastive.py

Moonshot #4: Logit-Aware Contrastive Loss for Drafter
Trains the continuous draft model using Logit-Aware Contrastive Loss
and compares it to the standard MSE training baseline.
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

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def logit_aware_contrastive_loss(
    model, target_model, noise, target_hidden, audio_summary, position_ids, true_hidden,
    tau_pred=1.0, tau_target=1.0, alpha=1.0, mse_weight=0.1
):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    
    # Shapes: (batch, block_size, d_target)
    N, B, D = pred_hidden.shape
    pred_h_flat = pred_hidden.reshape(-1, D)
    true_h_flat = true_hidden.reshape(-1, D)
    
    # Project hidden states to logit space via target lm_head (token_embedding.as_linear)
    pred_logits = target_model.decoder.token_embedding.as_linear(pred_h_flat) # (N*B, V)
    true_logits = target_model.decoder.token_embedding.as_linear(true_h_flat) # (N*B, V)
    
    # Target probabilities & greedy target tokens
    P_target = mx.softmax(true_logits / tau_target, axis=-1) # (N*B, V)
    y_target = mx.argmax(true_logits, axis=-1) # (N*B,)
    
    # Scaled predicted logits
    logits_scaled = pred_logits / tau_pred
    max_logits = mx.max(logits_scaled, axis=-1, keepdims=True) # (N*B, 1)
    exp_logits = mx.exp(logits_scaled - max_logits) # (N*B, V)
    
    # Negative weights: (1 - P_target)^alpha
    w_neg = mx.power(1.0 - P_target, alpha)
    
    # Raw sum of weighted exponentials
    weighted_exp = w_neg * exp_logits # (N*B, V)
    sum_weighted_exp = mx.sum(weighted_exp, axis=-1) # (N*B,)
    
    # Add correction to set weight of positive class to 1.0 (instead of (1.0 - P_target)^alpha)
    indices = mx.arange(y_target.shape[0])
    pos_P_target = P_target[indices, y_target]
    pos_exp_logits = exp_logits[indices, y_target]
    
    correction = (1.0 - mx.power(1.0 - pos_P_target, alpha)) * pos_exp_logits
    sum_weighted_exp = sum_weighted_exp + correction
    
    # Log-denominator
    log_denom = max_logits.squeeze(-1) + mx.log(sum_weighted_exp + 1e-9) # (N*B,)
    
    # Positive term
    pos_logits = logits_scaled[indices, y_target] # (N*B,)
    
    # Contrastive Loss
    contrastive = mx.mean(log_denom - pos_logits)
    
    # MSE loss to preserve representation layout
    mse = mx.mean(mx.square(pred_h_flat - true_h_flat))
    
    # Total loss
    total_loss = contrastive + mse_weight * mse
    return total_loss

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
    
    # We will initialize two models identically to ensure comparison is fair
    draft_mse = ContinuousDraftModel(config)
    draft_contrastive = ContinuousDraftModel(config)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = draft_mse(noise_init, ctx_init, audio_init, pos_init)
    _ = draft_contrastive(noise_init, ctx_init, audio_init, pos_init)
    
    # Copy parameters from draft_mse to draft_contrastive
    initial_params = copy_parameters(draft_mse)
    draft_contrastive.update(initial_params)
    
    print("Calling load_dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print("load_dataset finished.")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    for i in range(5):  # Reduced to 5 samples
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
    
    # 2. Train Model 1 (MSE baseline)
    print("\n--- Training Model 1: MSE Baseline ---")
    opt_mse = optim.Adam(learning_rate=1e-3)
    loss_and_grad_mse = nn.value_and_grad(draft_mse, mse_loss)
    
    t0 = time.time()
    for epoch in range(5):  # Reduced to 5 epochs
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_mse(
                draft_mse, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_mse.update(draft_mse, grads)
            mx.eval(draft_mse.parameters(), opt_mse.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/5 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"MSE baseline trained in {time.time() - t0:.1f}s.")
    
    # 3. Train Model 2 (Logit-Aware Contrastive Loss)
    print("\n--- Training Model 2: Logit-Aware Contrastive Loss ---")
    opt_contrastive = optim.Adam(learning_rate=1e-3)
    
    def contrastive_loss_wrapper(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
        return logit_aware_contrastive_loss(
            model, target, noise, target_hidden, audio_summary, position_ids, true_hidden,
            tau_pred=1.0, tau_target=1.0, alpha=1.0, mse_weight=0.1
        )
        
    loss_and_grad_contrastive = nn.value_and_grad(draft_contrastive, contrastive_loss_wrapper)
    
    t0 = time.time()
    for epoch in range(5):  # Reduced to 5 epochs
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_contrastive(
                draft_contrastive, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt_contrastive.update(draft_contrastive, grads)
            mx.eval(draft_contrastive.parameters(), opt_contrastive.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/5 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Logit-Aware Contrastive model trained in {time.time() - t0:.1f}s.")
    
    # 4. Evaluation
    print("\nEvaluating on held-out validation samples (samples 10 to 14)...")
    
    metrics = {
        "mse": {"sim": [], "acc": [], "top5_acc": []},
        "contrastive": {"sim": [], "acc": [], "top5_acc": []}
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
            pred_mse = draft_mse(noise, ctx_feats, audio_summary, pos_ids)
            pred_contrastive = draft_contrastive(noise, ctx_feats, audio_summary, pos_ids)
            
            # 1. Cosine similarity
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                h_mse = pred_mse[0, k]
                h_contr = pred_contrastive[0, k]
                
                sim_mse = (mx.sum(h_mse * h_true) / (mx.linalg.norm(h_mse) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_contr = (mx.sum(h_contr * h_true) / (mx.linalg.norm(h_contr) * mx.linalg.norm(h_true) + 1e-9)).item()
                
                metrics["mse"]["sim"].append(sim_mse)
                metrics["contrastive"]["sim"].append(sim_contr)
                
            # 2. Logit projections and accuracy
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            mse_logits = target.decoder.token_embedding.as_linear(pred_mse.reshape(-1, d_target))
            contr_logits = target.decoder.token_embedding.as_linear(pred_contrastive.reshape(-1, d_target))
            
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            mse_tokens = mx.argmax(mse_logits, axis=-1).tolist()
            contr_tokens = mx.argmax(contr_logits, axis=-1).tolist()
            
            # Top 5 target tokens for semantic alignment check
            # We want to see if the draft model predicted token is in the top-5 expected tokens of the target model
            top5_indices = mx.argpartition(-true_logits, 5, axis=-1)[:, :5]
            
            for idx in range(len(true_tokens)):
                # Greedy accuracy
                metrics["mse"]["acc"].append(1.0 if mse_tokens[idx] == true_tokens[idx] else 0.0)
                metrics["contrastive"]["acc"].append(1.0 if contr_tokens[idx] == true_tokens[idx] else 0.0)
                
                # Top-5 accuracy (does prediction fall in target's top-5?)
                mse_top5 = mse_tokens[idx] in top5_indices[idx].tolist()
                contr_top5 = contr_tokens[idx] in top5_indices[idx].tolist()
                
                metrics["mse"]["top5_acc"].append(1.0 if mse_top5 else 0.0)
                metrics["contrastive"]["top5_acc"].append(1.0 if contr_top5 else 0.0)

    print("\n" + "="*50)
    print("RESULTS: LOGIT-AWARE CONTRASTIVE LOSS VS MSE BASELINE")
    print("="*50)
    
    print("--- 1. Hidden Representation Cosine Similarity ---")
    print(f"MSE Baseline       : {np.mean(metrics['mse']['sim']):.4f}")
    print(f"Logit Contrastive  : {np.mean(metrics['contrastive']['sim']):.4f}  (Delta: {np.mean(metrics['contrastive']['sim']) - np.mean(metrics['mse']['sim']):+.4f})")
    
    print("\n--- 2. Greedy Token Accuracy (Projection Match) ---")
    print(f"MSE Baseline       : {np.mean(metrics['mse']['acc'])*100:.2f}%")
    print(f"Logit Contrastive  : {np.mean(metrics['contrastive']['acc'])*100:.2f}%  (Delta: {(np.mean(metrics['contrastive']['acc']) - np.mean(metrics['mse']['acc']))*100:+.2f}%)")
    
    print("\n--- 3. Top-5 Expected Token Acceptance Rate ---")
    print(f"MSE Baseline       : {np.mean(metrics['mse']['top5_acc'])*100:.2f}%")
    print(f"Logit Contrastive  : {np.mean(metrics['contrastive']['top5_acc'])*100:.2f}%  (Delta: {(np.mean(metrics['contrastive']['top5_acc']) - np.mean(metrics['mse']['top5_acc']))*100:+.2f}%)")
    print("="*50)
    
if __name__ == "__main__":
    run_experiment()

