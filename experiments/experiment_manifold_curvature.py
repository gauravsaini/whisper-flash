#!/usr/bin/env python3
"""
experiment_manifold_curvature.py

Moonshot 3.9: Manifold Curvature Penalty / Trajectory Curvature Alignment
- Penalizes the difference in second-order differences (acceleration/curvature)
  between the predicted draft trajectory and the true target trajectory.
- Training loss: L_total = L_MSE + lambda * MSE(d^2/dt^2 H_pred, d^2/dt^2 H_true).
- Forces the draft model to generate smooth continuous hidden state trajectories 
  aligned with the target manifold dynamics, preventing compounding spatial drift.
- Benchmarks: Cosine Similarity, Greedy Token Accuracy, Speculative Acceptance Rate, and WER.
"""

import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
import evaluate
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel
from whisper_flash_mlx.generate import crop_self_attention_cache

# --- Loss functions ---
def mse_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def curvature_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden, lambda_curv=0.5):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    
    # 1. Standard reconstruction MSE
    mse = mx.mean(mx.square(pred_hidden - true_hidden))
    
    # 2. Trajectory Curvature Alignment Penalty (second derivative matching)
    if pred_hidden.shape[1] >= 3:
        pred_accel = pred_hidden[:, 2:, :] - 2 * pred_hidden[:, 1:-1, :] + pred_hidden[:, :-2, :]
        true_accel = true_hidden[:, 2:, :] - 2 * true_hidden[:, 1:-1, :] + true_hidden[:, :-2, :]
        curv_penalty = mx.mean(mx.square(pred_accel - true_accel))
    else:
        curv_penalty = mx.array(0.0)
        
    return mse + lambda_curv * curv_penalty

# --- Speculative Generation Loop with Span-Level Graph Verification ---
def generate_speculative(
    draft_model, target, mel, max_length=448, temperature=0.0
):
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    block_size = draft_model.config.block_size
    mask_token_id = draft_model.mask_token_id
    
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    output_list = [mask_token_id] * (max_length + block_size)
    output_list[0] = 50258  # SOT
    
    # Initialize target_hidden by running prompt prefill
    logits_init, kv_cache, all_hidden_init = decoder_forward_with_hidden_states(
        target, mx.array([[50258]], dtype=mx.int32), encoder_hidden, collect_hidden_states=True, return_cross_attention=False
    )
    first_token = mx.argmax(logits_init[:, -1:, :], axis=-1).item()
    output_list[1] = first_token
    target_hidden = all_hidden_init[draft_model.target_layer_ids[0]]
    for layer_id in draft_model.target_layer_ids[1:]:
        target_hidden = mx.concatenate([target_hidden, all_hidden_init[layer_id]], axis=-1)
        
    start = 1
    current_block_size = block_size
    acceptance_lengths = []
    block_sizes = []
    
    while start < max_length:
        block_ids_list = output_list[start: start + current_block_size]
        while len(block_ids_list) < current_block_size:
            block_ids_list.append(mask_token_id)
        block_ids = mx.array([block_ids_list], dtype=mx.int32)
        block_positions = mx.arange(start, start + current_block_size)[None]
        
        # Draft step
        if current_block_size > 1:
            noise_embedding = target.decoder.token_embedding(block_ids)
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                audio_summary=audio_summary,
                position_ids=block_positions,
            )
            
            # Project using target lm_head
            draft_logits = target.decoder.token_embedding.as_linear(draft_hidden[:, :-1, :])
            draft_tokens = mx.argmax(draft_logits, axis=-1)
            mx.eval(draft_tokens)
            
            draft_tokens_list = draft_tokens.tolist()[0]
            for i, t in enumerate(draft_tokens_list):
                block_ids_list[i + 1] = t
            block_ids = mx.array([block_ids_list], dtype=mx.int32)
            
        # Verify step
        logits, kv_cache, all_hidden_verify = decoder_forward_with_hidden_states(
            target, block_ids, encoder_hidden, kv_cache=kv_cache, collect_hidden_states=True, return_cross_attention=False
        )
        
        posterior = mx.argmax(logits, axis=-1)
        mx.eval(posterior)
        
        posterior_list = posterior.tolist()[0]
        acceptance_length = 0
        true_hidden = all_hidden_verify[-1][0]
        
        K = current_block_size - 1
        
        if current_block_size > 1 and K > 0:
            hat_H = draft_hidden[0, :K]
            H = true_hidden[:K]
            norm_hat_H = hat_H / (mx.linalg.norm(hat_H, axis=-1, keepdims=True) + 1e-9)
            norm_H = H / (mx.linalg.norm(H, axis=-1, keepdims=True) + 1e-9)
            node_sims = mx.sum(norm_hat_H * norm_H, axis=-1).tolist()
            
            # Span-level semantic graph verification
            G_hat = mx.matmul(norm_hat_H, norm_hat_H.T)
            G = mx.matmul(norm_H, norm_H.T)
            g_hat_flat = G_hat.reshape(-1)
            g_flat = G.reshape(-1)
            topo_sim = mx.sum(g_hat_flat * g_flat) / (mx.linalg.norm(g_hat_flat) * mx.linalg.norm(g_flat) + 1e-9)
            M_graph = 0.5 * np.mean(node_sims) + 0.5 * topo_sim.item()
            
            if M_graph >= 0.95:
                acceptance_length = K
            else:
                for i in range(1, current_block_size):
                    lexical_match = block_ids_list[i] == posterior_list[i - 1]
                    sim_val = node_sims[i - 1]
                    semantic_match = sim_val >= 0.97
                    if lexical_match or semantic_match:
                        acceptance_length += 1
                    else:
                        break
        else:
            acceptance_length = 0
            
        # Accept tokens
        for i in range(acceptance_length + 1):
            output_list[start + i] = block_ids_list[i]
            
        # Fallback token correction
        output_list[start + acceptance_length + 1] = posterior_list[acceptance_length]
        
        start += acceptance_length + 1

        kv_cache = crop_self_attention_cache(kv_cache, start)
        
        acceptance_lengths.append(acceptance_length + 1)
        block_sizes.append(current_block_size)
        
        ctx_layer = all_hidden_verify[draft_model.target_layer_ids[0]][:, : acceptance_length + 1, :]
        for layer_id in draft_model.target_layer_ids[1:]:
            ctx_layer = mx.concatenate([ctx_layer, all_hidden_verify[layer_id][:, : acceptance_length + 1, :]], axis=-1)
        target_hidden = mx.concatenate([target_hidden, ctx_layer], axis=1)
        
        if 50257 in output_list[:start]:
            break
            
    final_ids = output_list[:start]
    decoded_text = tokenizer.decode(final_ids)
    accept_rate = sum(acceptance_lengths) / sum(block_sizes)
    return decoded_text, accept_rate

def copy_parameters(model):
    from mlx.utils import tree_map
    return tree_map(lambda x: mx.array(x), model.parameters())

def run():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    d_target = target.dims.n_text_state
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    
    # Initialize baseline and curvature draft models identically
    draft_baseline = ContinuousDraftModel(config)
    draft_curvature = ContinuousDraftModel(config)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, d_target))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    
    _ = draft_baseline(noise_init, ctx_init, audio_init, pos_init)
    _ = draft_curvature(noise_init, ctx_init, audio_init, pos_init)
    
    initial_params = copy_parameters(draft_baseline)
    draft_curvature.update(initial_params)
    
    print("Loading Dataset...")
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
            
    print(f"Pre-extraction complete. Extracted {len(data_tensors)} train points.")
    
    # Optimizers
    optimizer_baseline = optim.Adam(learning_rate=1e-3)
    optimizer_curvature = optim.Adam(learning_rate=1e-3)
    
    loss_and_grad_baseline = nn.value_and_grad(draft_baseline, mse_loss_fn)
    loss_and_grad_curvature = nn.value_and_grad(draft_curvature, curvature_loss_fn)
    
    # Training
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
    
    print("\nTraining Curvature-Aligned model (lambda=0.5)...")
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            loss, grads = loss_and_grad_curvature(
                draft_curvature, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            optimizer_curvature.update(draft_curvature, grads)
            mx.eval(draft_curvature.parameters(), optimizer_curvature.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs} - Curvature Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Curvature-Aligned model trained in {time.time() - t0:.1f}s.")
    
    # Evaluation
    eval_samples = list(range(10, min(20, len(ds))))
    print(f"\nEvaluating speculative decoding on {len(eval_samples)} held-out samples...")
    
    results = {
        "baseline": {"wers": [], "accept_rates": [], "similarities": [], "accs": [], "texts": []},
        "curvature": {"wers": [], "accept_rates": [], "similarities": [], "accs": [], "texts": []}
    }
    
    references = [ds[idx]["text"] for idx in eval_samples]
    
    # Run evaluation
    for mode, model in [("baseline", draft_baseline), ("curvature", draft_curvature)]:
        print(f"Evaluating model: {mode}...")
        for idx in eval_samples:
            sample = ds[idx]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel)[None]
            
            text, accept_rate = generate_speculative(
                draft_model=model,
                target=target,
                mel=mel_mx,
                temperature=0.0
            )
            
            results[mode]["texts"].append(text)
            results[mode]["accept_rates"].append(accept_rate)
            
        wer = wer_metric.compute(predictions=results[mode]["texts"], references=references)
        results[mode]["wer"] = wer
        results[mode]["mean_accept_rate"] = np.mean(results[mode]["accept_rates"])
        
    # Evaluate raw cosine similarity and token accuracy on test points
    print("\nMeasuring raw representation stats on held-out test points...")
    for idx in eval_samples:
        sample = ds[idx]
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
            
            pred_base = draft_baseline(noise, ctx_feats, audio_summary, pos_ids)
            pred_curv = draft_curvature(noise, ctx_feats, audio_summary, pos_ids)
            
            for k in range(config.block_size):
                h_true = true_hidden[0, k]
                h_base = pred_base[0, k]
                h_curv = pred_curv[0, k]
                
                sim_base = (mx.sum(h_base * h_true) / (mx.linalg.norm(h_base) * mx.linalg.norm(h_true) + 1e-9)).item()
                sim_curv = (mx.sum(h_curv * h_true) / (mx.linalg.norm(h_curv) * mx.linalg.norm(h_true) + 1e-9)).item()
                
                results["baseline"]["similarities"].append(sim_base)
                results["curvature"]["similarities"].append(sim_curv)
                
            # Token matches
            true_logits = target.decoder.token_embedding.as_linear(true_hidden.reshape(-1, d_target))
            base_logits = target.decoder.token_embedding.as_linear(pred_base.reshape(-1, d_target))
            curv_logits = target.decoder.token_embedding.as_linear(pred_curv.reshape(-1, d_target))
            
            true_tok = mx.argmax(true_logits, axis=-1).tolist()
            base_tok = mx.argmax(base_logits, axis=-1).tolist()
            curv_tok = mx.argmax(curv_logits, axis=-1).tolist()
            
            for index in range(len(true_tok)):
                results["baseline"]["accs"].append(1.0 if base_tok[index] == true_tok[index] else 0.0)
                results["curvature"]["accs"].append(1.0 if curv_tok[index] == true_tok[index] else 0.0)

    print("\n" + "="*50)
    print("RESULTS: MANIFOLD CURVATURE / TRAJECTORY ALIGNMENT")
    print("="*50)
    
    print("\n--- 1. Word Error Rate (WER) ---")
    print(f"Baseline MSE Model                 : {results['baseline']['wer']:.4f}")
    print(f"Curvature-Aligned Model            : {results['curvature']['wer']:.4f}  (Delta: {results['curvature']['wer'] - results['baseline']['wer']:+.4f})")
    
    print("\n--- 2. Mean Speculative Acceptance Rate ---")
    print(f"Baseline MSE Model                 : {results['baseline']['mean_accept_rate']*100:.2f}%")
    print(f"Curvature-Aligned Model            : {results['curvature']['mean_accept_rate']*100:.2f}%  (Delta: {(results['curvature']['mean_accept_rate'] - results['baseline']['mean_accept_rate'])*100:+.2f}%)")
    
    print("\n--- 3. Raw Latent Cosine Similarity ---")
    print(f"Baseline MSE Model                 : {np.mean(results['baseline']['similarities']):.4f}")
    print(f"Curvature-Aligned Model            : {np.mean(results['curvature']['similarities']):.4f}  (Delta: {np.mean(results['curvature']['similarities']) - np.mean(results['baseline']['similarities']):+.4f})")
    
    print("\n--- 4. Greedy Token Accuracy (Projection Match) ---")
    print(f"Baseline MSE Model                 : {np.mean(results['baseline']['accs'])*100:.2f}%")
    print(f"Curvature-Aligned Model            : {np.mean(results['curvature']['accs'])*100:.2f}%  (Delta: {(np.mean(results['curvature']['accs']) - np.mean(results['baseline']['accs']))*100:+.2f}%)")
    print("="*50)

if __name__ == "__main__":
    run()
