#!/usr/bin/env python3
"""
experiment_adaptive_threshold.py

Moonshot #9: Adaptive Attention-Entropy Cosine Verification Threshold
Compares static vs dynamic (cross-attention entropy-based) verification thresholds
for speculative drafting to minimize False Acceptance Rates (FAR) while maximizing throughput.
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

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=1, target_layer_ids=[1, 2]
    )
    
    draft = ContinuousDraftModel(config)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # Pre-extract data for training
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
    
    # Train the model
    print("\n--- Training Continuous Draft Model ---")
    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    t0 = time.time()
    for epoch in range(15):
        loss_sum = 0.0
        for data in data_tensors:
            loss, grads = loss_and_grad_fn(
                draft, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            opt.update(draft, grads)
            mx.eval(draft.parameters(), opt.state)
            loss_sum += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:02d}/15 - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Model trained in {time.time() - t0:.1f}s.")
    
    # Evaluation
    print("\nEvaluating static vs adaptive threshold on held-out samples...")
    
    static_accepts = 0
    static_false_accepts = 0
    
    adaptive_accepts = 0
    adaptive_false_accepts = 0
    
    total_steps = 0
    
    dynamic_thresholds = []
    align_entropies = []
    
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
            
            # Forward pass with return_cross_attention=True
            logits_target, _, hidden_target, cross_attns = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=True
            )
            
            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)
            
            # Extract true future state
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
            
            # Draft prediction
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            pred_hidden = draft(noise, ctx_feats, audio_summary, pos_ids)
            
            # Cosine similarity
            h_true = true_hidden[0, 0]
            h_pred = pred_hidden[0, 0]
            sim = (mx.sum(h_pred * h_true) / (mx.linalg.norm(h_pred) * mx.linalg.norm(h_true) + 1e-9)).item()
            
            # Project predicted hidden state to token
            pred_logits = target.decoder.token_embedding.as_linear(h_pred)
            pred_token = mx.argmax(pred_logits, axis=-1).item()
            
            # Target greedy token
            true_logits = target.decoder.token_embedding.as_linear(h_true)
            true_token = mx.argmax(true_logits, axis=-1).item()
            
            # Calculate alignment entropy of the cross-attention weights for the last token in the prefix
            # cross_attns is a list of arrays (one per layer). Each layer's attention shape is (1, heads, seq_len, 1500)
            # We take the last decoder layer's cross attention weights at the last index
            # Convert raw attention logits to probability distribution via softmax
            last_attn = cross_attns[-1][0, :, -1, :] # shape (heads, 1500)
            last_attn_probs = mx.softmax(last_attn, axis=-1)
            mean_attn = mx.mean(last_attn_probs, axis=0) # shape (1500,)
            mean_attn = mean_attn / (mx.sum(mean_attn) + 1e-9)
            
            entropy = -mx.sum(mean_attn * mx.log(mean_attn + 1e-9)).item()
            align_entropies.append(entropy)
            
            # Adaptive threshold definition: scale from 0.90 to 0.98 based on entropy
            # Typical entropy values for 1500 frames are around 2.0 to 5.0
            tau_dynamic = 0.85 + 0.025 * entropy
            tau_dynamic = min(max(tau_dynamic, 0.90), 0.98)
            dynamic_thresholds.append(tau_dynamic)
            
            # 1. Static Threshold (tau = 0.95)
            is_static_accepted = sim >= 0.95
            if is_static_accepted:
                static_accepts += 1
                if pred_token != true_token:
                    static_false_accepts += 1
                    
            # 2. Adaptive Threshold (tau_dynamic)
            is_adaptive_accepted = sim >= tau_dynamic
            if is_adaptive_accepted:
                adaptive_accepts += 1
                if pred_token != true_token:
                    adaptive_false_accepts += 1
                    
            total_steps += 1

    static_ar = (static_accepts / total_steps) * 100
    static_far = (static_false_accepts / max(static_accepts, 1)) * 100
    
    adaptive_ar = (adaptive_accepts / total_steps) * 100
    adaptive_far = (adaptive_false_accepts / max(adaptive_accepts, 1)) * 100
    
    print("\n" + "="*70)
    print("RESULTS: ADAPTIVE ATTENTION-ENTROPY COSINE VERIFICATION THRESHOLD")
    print("="*70)
    print(f"Total Steps Evaluated      : {total_steps}")
    print(f"Mean Alignment Entropy     : {np.mean(align_entropies):.4f}")
    print(f"Mean Adaptive Threshold    : {np.mean(dynamic_thresholds):.4f}")
    print("-" * 70)
    print(f"Static Threshold (tau=0.95)  - Acceptance Rate: {static_ar:.2f}% | False Acceptance Rate (FAR): {static_far:.2f}%")
    print(f"Adaptive Threshold (tau_dyn) - Acceptance Rate: {adaptive_ar:.2f}% | False Acceptance Rate (FAR): {adaptive_far:.2f}%")
    print(f"Delta (Adaptive - Static)    - Acceptance Rate: {adaptive_ar - static_ar:+.2f}% | FAR: {adaptive_far - static_far:+.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_experiment()
