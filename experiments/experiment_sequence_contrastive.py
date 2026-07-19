#!/usr/bin/env python3
"""
experiment_sequence_contrastive.py

Moonshot #10: Contrastive Sequence-Level Trajectory Calibration
Investigating if applying InfoNCE contrastive loss over the entire drafted sequence trajectory 
prevents cumulative manifold drift better than independent step-wise MSE.
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

def mse_loss(model, batch):
    losses = []
    for d in batch:
        pred = model(d["noise"], d["ctx"], d["audio"], d["pos"])
        losses.append(mx.mean(mx.square(pred - d["true_hidden"])))
    return mx.mean(mx.stack(losses))

def sequence_contrastive_loss(model, batch, tau=0.1):
    preds = []
    trues = []
    for d in batch:
        pred = model(d["noise"], d["ctx"], d["audio"], d["pos"])
        preds.append(pred)
        trues.append(d["true_hidden"])
        
    preds = mx.concatenate(preds, axis=0) # (N, B, D)
    trues = mx.concatenate(trues, axis=0) # (N, B, D)
    
    N, B, D = preds.shape
    pred_seq = preds.reshape(N, -1)
    true_seq = trues.reshape(N, -1)
    
    pred_norm = pred_seq / (mx.linalg.norm(pred_seq, axis=-1, keepdims=True) + 1e-9)
    true_norm = true_seq / (mx.linalg.norm(true_seq, axis=-1, keepdims=True) + 1e-9)
    
    sim_matrix = mx.matmul(pred_norm, true_norm.T) / tau
    labels = mx.arange(N)
    return mx.mean(nn.losses.cross_entropy(sim_matrix, labels))

def hybrid_loss(model, batch, alpha=0.5, tau=0.1):
    preds = []
    trues = []
    mse_losses = []
    for d in batch:
        pred = model(d["noise"], d["ctx"], d["audio"], d["pos"])
        preds.append(pred)
        trues.append(d["true_hidden"])
        mse_losses.append(mx.mean(mx.square(pred - d["true_hidden"])))
        
    mse = mx.mean(mx.stack(mse_losses))
    
    preds = mx.concatenate(preds, axis=0)
    trues = mx.concatenate(trues, axis=0)
    
    N, B, D = preds.shape
    pred_seq = preds.reshape(N, -1)
    true_seq = trues.reshape(N, -1)
    
    pred_norm = pred_seq / (mx.linalg.norm(pred_seq, axis=-1, keepdims=True) + 1e-9)
    true_norm = true_seq / (mx.linalg.norm(true_seq, axis=-1, keepdims=True) + 1e-9)
    
    sim_matrix = mx.matmul(pred_norm, true_norm.T) / tau
    labels = mx.arange(N)
    contrastive = mx.mean(nn.losses.cross_entropy(sim_matrix, labels))
    
    return alpha * mse + (1 - alpha) * contrastive

def copy_parameters(src_params, dest_model):
    from mlx.utils import tree_map
    params_copy = tree_map(lambda x: mx.array(x), src_params)
    dest_model.update(params_copy)

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=8, target_layer_ids=[1, 2] # Use 8 steps to see if trajectory holds up
    )
    
    model_mse = ContinuousDraftModel(config)
    model_contrastive = ContinuousDraftModel(config)
    model_hybrid = ContinuousDraftModel(config)
    
    # Initialize and sync weights
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = model_mse(noise_init, ctx_init, audio_init, pos_init)
    _ = model_contrastive(noise_init, ctx_init, audio_init, pos_init)
    _ = model_hybrid(noise_init, ctx_init, audio_init, pos_init)
    
    base_params = model_mse.parameters()
    copy_parameters(base_params, model_contrastive)
    copy_parameters(base_params, model_hybrid)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    # Use 15 samples, but we must batch them to compute InfoNCE properly (N > 1)
    for i in range(15):
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
        
        for t in range(1, labels.shape[1] - config.block_size, 5):
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
    
    # We must batch data for InfoNCE. Batch size = 16
    batch_size = 16
    batched_data = []
    for i in range(0, len(data_tensors), batch_size):
        batch = data_tensors[i:i+batch_size]
        if len(batch) < 2: continue # Need > 1 for contrastive
        # Keep as list of dicts!
        batched_data.append(batch)
    print(f"Formed {len(batched_data)} batches (Batch Size: {batch_size}).")
    
    # --- Training loops ---
    epochs = 15
    
    print("\n--- Training Model 1: MSE Baseline ---")
    opt_mse = optim.Adam(learning_rate=1e-3)
    val_and_grad_mse = nn.value_and_grad(model_mse, mse_loss)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        for batch in batched_data:
            loss, grads = val_and_grad_mse(model_mse, batch)
            opt_mse.update(model_mse, grads)
            mx.eval(model_mse.parameters(), opt_mse.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(batched_data):.5f}")
    print(f"MSE trained in {time.time() - t0:.1f}s.")
    
    print("\n--- Training Model 2: Sequence Contrastive InfoNCE ---")
    opt_contrastive = optim.Adam(learning_rate=1e-3)
    val_and_grad_contrastive = nn.value_and_grad(model_contrastive, sequence_contrastive_loss)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        for batch in batched_data:
            loss, grads = val_and_grad_contrastive(model_contrastive, batch)
            opt_contrastive.update(model_contrastive, grads)
            mx.eval(model_contrastive.parameters(), opt_contrastive.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(batched_data):.5f}")
    print(f"Contrastive trained in {time.time() - t0:.1f}s.")
    
    print("\n--- Training Model 3: Hybrid (MSE + InfoNCE) ---")
    opt_hybrid = optim.Adam(learning_rate=1e-3)
    val_and_grad_hybrid = nn.value_and_grad(model_hybrid, hybrid_loss)
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0.0
        for batch in batched_data:
            loss, grads = val_and_grad_hybrid(model_hybrid, batch)
            opt_hybrid.update(model_hybrid, grads)
            mx.eval(model_hybrid.parameters(), opt_hybrid.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(batched_data):.5f}")
    print(f"Hybrid trained in {time.time() - t0:.1f}s.")
    
    print("\nEvaluating Trajectory Stability on validation...")
    
    sims = {"mse": [[] for _ in range(8)], "contrastive": [[] for _ in range(8)], "hybrid": [[] for _ in range(8)]}
    accs = {"mse": [], "contrastive": [], "hybrid": []}
    
    for i in range(15, 20):
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
        
        for t in range(1, labels.shape[1] - config.block_size, 5):
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
            
            p_mse = model_mse(noise, ctx_feats, audio_summary, pos_ids)
            p_contr = model_contrastive(noise, ctx_feats, audio_summary, pos_ids)
            p_hyb = model_hybrid(noise, ctx_feats, audio_summary, pos_ids)
            
            for k in range(config.block_size):
                ht = true_hidden[0, k]
                hm = p_mse[0, k]
                hc = p_contr[0, k]
                hh = p_hyb[0, k]
                
                sims["mse"][k].append((mx.sum(hm * ht) / (mx.linalg.norm(hm) * mx.linalg.norm(ht) + 1e-9)).item())
                sims["contrastive"][k].append((mx.sum(hc * ht) / (mx.linalg.norm(hc) * mx.linalg.norm(ht) + 1e-9)).item())
                sims["hybrid"][k].append((mx.sum(hh * ht) / (mx.linalg.norm(hh) * mx.linalg.norm(ht) + 1e-9)).item())
                
            # Logit accuracy for whole block
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            t_tokens = mx.argmax(true_logits, axis=-1).tolist()
            
            m_tokens = mx.argmax(target.decoder.token_embedding.as_linear(p_mse.reshape(-1, d_target)), axis=-1).tolist()
            c_tokens = mx.argmax(target.decoder.token_embedding.as_linear(p_contr.reshape(-1, d_target)), axis=-1).tolist()
            h_tokens = mx.argmax(target.decoder.token_embedding.as_linear(p_hyb.reshape(-1, d_target)), axis=-1).tolist()
            
            accs["mse"].extend([1.0 if m == t else 0.0 for m, t in zip(m_tokens, t_tokens)])
            accs["contrastive"].extend([1.0 if c == t else 0.0 for c, t in zip(c_tokens, t_tokens)])
            accs["hybrid"].extend([1.0 if h == t else 0.0 for h, t in zip(h_tokens, t_tokens)])

    print("\n" + "="*80)
    print("RESULTS: SEQUENCE-LEVEL TRAJECTORY CALIBRATION (B=8)")
    print("="*80)
    print(f"{'Metric':<25} | {'Step-wise MSE':<15} | {'Pure InfoNCE':<15} | {'Hybrid (MSE+NCE)':<15}")
    print("-" * 80)
    print(f"{'Mean CosSim (Step 1)':<25} | {np.mean(sims['mse'][0]):<15.4f} | {np.mean(sims['contrastive'][0]):<15.4f} | {np.mean(sims['hybrid'][0]):<15.4f}")
    print(f"{'Mean CosSim (Step 8)':<25} | {np.mean(sims['mse'][7]):<15.4f} | {np.mean(sims['contrastive'][7]):<15.4f} | {np.mean(sims['hybrid'][7]):<15.4f}")
    print(f"{'Overall Mean CosSim':<25} | {np.mean([np.mean(s) for s in sims['mse']]):<15.4f} | {np.mean([np.mean(s) for s in sims['contrastive']]):<15.4f} | {np.mean([np.mean(s) for s in sims['hybrid']]):<15.4f}")
    print(f"{'Greedy Token Acc':<25} | {np.mean(accs['mse'])*100:<14.2f}% | {np.mean(accs['contrastive'])*100:<14.2f}% | {np.mean(accs['hybrid'])*100:<14.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_experiment()
