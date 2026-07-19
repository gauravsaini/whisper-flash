#!/usr/bin/env python3
"""
experiment_span_verification.py

Moonshot #14: Span-level Semantic Graph Verification
Compares strict lexical verification, strict cosine verification, and
Span-level Semantic Graph Verification (dynamic programming alignment)
to maximize Mean Acceptance Tokens (MAT) while maintaining safety.
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

def verify_span_graph(pred_hidden, true_hidden, threshold=0.52, window=1):
    # pred_hidden shape: (B, D)
    # true_hidden shape: (B, D)
    B, D = pred_hidden.shape
    
    # Compute similarity matrix S
    norm_pred = pred_hidden / (mx.linalg.norm(pred_hidden, axis=-1, keepdims=True) + 1e-9)
    norm_true = true_hidden / (mx.linalg.norm(true_hidden, axis=-1, keepdims=True) + 1e-9)
    S = norm_pred @ norm_true.T # shape (B, B)
    S_np = np.array(S)
    
    # DP table: DP[i][j] stores if there is a valid path from start to (i, j)
    # i is draft index (0 to B-1), j is target index (0 to B-1)
    DP = np.zeros((B, B), dtype=bool)
    
    # Backpointer to trace the alignment path: parent[i][j] = k (target index)
    parent = np.full((B, B), -1, dtype=int)
    
    # Initialize first row (i=0)
    for j in range(min(window + 1, B)):
        if S_np[0, j] >= threshold:
            DP[0, j] = True
            
    # Fill DP table
    for i in range(1, B):
        for j in range(max(0, i - window), min(B, i + window + 1)):
            if S_np[i, j] >= threshold:
                # We can transition from any k <= j in the previous row
                for k in range(max(0, i - 1 - window), min(j + 1, B)):
                    if DP[i-1, k]:
                        DP[i, j] = True
                        parent[i][j] = k
                        break
                        
    # Find the maximum accepted draft length and the matched path
    accepted_len = 0
    best_j = -1
    for i in range(B):
        if np.any(DP[i]):
            accepted_len = i + 1
            # Find the best target index j for this length
            for j in range(B):
                if DP[i, j]:
                    best_j = j
        else:
            break
            
    # Reconstruct the alignment path (mapping of draft index i -> target index)
    path = {}
    if accepted_len > 0 and best_j != -1:
        curr_j = best_j
        for i in range(accepted_len - 1, -1, -1):
            path[i] = curr_j
            curr_j = parent[i][curr_j]
            
    return accepted_len, path

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    draft = ContinuousDraftModel(config)
    
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
    print("\nEvaluating verification methods on held-out samples...")
    
    mat = {"lexical": [], "cosine": [], "graph": []}
    far = {"lexical": {"accepts": 0, "false": 0}, "cosine": {"accepts": 0, "false": 0}, "graph": {"accepts": 0, "false": 0}}
    
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
            
            # Predict
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            pred_hidden = draft(noise, ctx_feats, audio_summary, pos_ids)
            
            # Project to tokens
            pred_flat = pred_hidden.reshape(-1, d_target)
            true_flat = true_hidden.reshape(-1, d_target)
            
            pred_logits = target.decoder.token_embedding.as_linear(pred_flat)
            true_logits = target.decoder.token_embedding.as_linear(true_flat)
            
            pred_tokens = mx.argmax(pred_logits, axis=-1).tolist()
            true_tokens = mx.argmax(true_logits, axis=-1).tolist()
            
            # Cosine similarity matrix
            norm_pred = pred_flat / (mx.linalg.norm(pred_flat, axis=-1, keepdims=True) + 1e-9)
            norm_true = true_flat / (mx.linalg.norm(true_flat, axis=-1, keepdims=True) + 1e-9)
            cos_sims = mx.sum(norm_pred * norm_true, axis=-1).tolist()
            
            # 1. Strict Lexical Verification
            lex_len = 0
            for k in range(config.block_size):
                if pred_tokens[k] == true_tokens[k]:
                    lex_len += 1
                else:
                    break
            mat["lexical"].append(lex_len)
            far["lexical"]["accepts"] += lex_len
            # In strict lexical, false acceptances are strictly 0 since tokens must match exactly
            
            # 2. Strict Cosine Verification (threshold = 0.52)
            cos_len = 0
            for k in range(config.block_size):
                if cos_sims[k] >= 0.52:
                    cos_len += 1
                else:
                    break
            mat["cosine"].append(cos_len)
            far["cosine"]["accepts"] += cos_len
            for k in range(cos_len):
                if pred_tokens[k] != true_tokens[k]:
                    far["cosine"]["false"] += 1
                    
            # 3. Span-level Semantic Graph Verification
            graph_len, path = verify_span_graph(pred_flat, true_flat, threshold=0.52, window=1)
            mat["graph"].append(graph_len)
            far["graph"]["accepts"] += graph_len
            for draft_idx, target_idx in path.items():
                if pred_tokens[draft_idx] != true_tokens[target_idx]:
                    far["graph"]["false"] += 1

    print("\n" + "="*80)
    print("RESULTS: SPAN-LEVEL SEMANTIC GRAPH VERIFICATION")
    print("="*80)
    print(f"{'Verification Method':<30} | {'Mean Acceptance (MAT)':<22} | {'False Acceptance Rate (FAR)':<25}")
    print("-" * 80)
    
    lex_mat = np.mean(mat["lexical"])
    lex_far = (far["lexical"]["false"] / max(far["lexical"]["accepts"], 1)) * 100
    print(f"{'Strict Lexical (Baseline)':<30} | {lex_mat:<22.4f} | {lex_far:<24.2f}%")
    
    cos_mat = np.mean(mat["cosine"])
    cos_far = (far["cosine"]["false"] / max(far["cosine"]["accepts"], 1)) * 100
    print(f"{'Strict Cosine (tau=0.52)':<30} | {cos_mat:<22.4f} | {cos_far:<24.2f}%")
    
    graph_mat = np.mean(mat["graph"])
    graph_far = (far["graph"]["false"] / max(far["graph"]["accepts"], 1)) * 100
    print(f"{'Span Graph (tau=0.52, W=1)':<30} | {graph_mat:<22.4f} | {graph_far:<24.2f}%")
    
    print("\nDelta (Span Graph vs Strict Cosine):")
    print(f"Mean Acceptance Length (MAT) Gain: {graph_mat - cos_mat:+.4f} ({((graph_mat - cos_mat)/max(cos_mat, 1e-9))*100:+.2f}%)")
    print(f"False Acceptance Rate (FAR) Change: {graph_far - cos_far:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_experiment()
