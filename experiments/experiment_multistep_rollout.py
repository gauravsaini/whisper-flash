#!/usr/bin/env python3
"""
experiment_multistep_rollout.py

Moonshot 2.2: Multi-step Rollout Stability.
Measures how the continuous hidden state prediction degrades (drifts) 
as we predict further into the future (k=1, 2, 3, 4 steps ahead) 
from a single anchor state, using the ContinuousDraftModel.
"""

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

def run():
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    d_target = target.dims.n_text_state
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    draft = ContinuousDraftModel(config)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-training Continuous Drafter (15 epochs on 10 samples) to ensure stable vectors...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    # Train
    for epoch in range(15):
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
                
                # Context features
                ctx_feats = [hidden_target[layer_id] for layer_id in draft.target_layer_ids]
                ctx_feats = mx.concatenate(ctx_feats, axis=-1)
                
                # True future hidden states (k=1 to block_size)
                _, _, hidden_future = decoder_forward_with_hidden_states(
                    target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
                )
                true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
                
                noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
                pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
                
                loss, grads = loss_and_grad_fn(draft, noise, ctx_feats, audio_summary, pos_ids, true_hidden)
                optimizer.update(draft, grads)
                mx.eval(draft.parameters(), optimizer.state)

    print("Training complete. Evaluating drift over k steps.")
    
    # Evaluate drift
    drift_stats = {k: [] for k in range(1, config.block_size + 1)}
    
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
        
        # Get all true hidden states
        _, _, hidden_all = decoder_forward_with_hidden_states(
            target, labels, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
        )
        true_hiddens = hidden_all[-1][0] # (Seq_len, D)
        
        for t in range(1, labels.shape[1] - config.block_size):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            ctx_feats = [hidden_target[layer_id] for layer_id in draft.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            
            pred_hidden = draft(noise, ctx_feats, audio_summary, pos_ids)[0] # (4, D)
            
            for k in range(config.block_size):
                h_draft = pred_hidden[k]
                h_true = true_hiddens[t + k]
                
                sim = mx.sum(h_draft * h_true) / (mx.linalg.norm(h_draft) * mx.linalg.norm(h_true) + 1e-9)
                drift_stats[k+1].append(sim.item())
                
    print("\n" + "="*40)
    print("CONTINUOUS HIDDEN STATE DRIFT (Cosine Sim)")
    print("="*40)
    for k in range(1, config.block_size + 1):
        mean_sim = np.mean(drift_stats[k])
        std_sim = np.std(drift_stats[k])
        print(f"Step +{k}: Cosine Sim = {mean_sim:.4f} (std: {std_sim:.4f})")
    print("="*40)
    
if __name__ == "__main__":
    run()
