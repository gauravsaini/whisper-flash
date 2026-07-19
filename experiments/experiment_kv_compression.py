#!/usr/bin/env python3
"""
experiment_kv_compression.py

Moonshot 3.8: Difficulty-Aware Continuous KV Compression
- Implements dynamic compression of the target model's self-attention KV cache.
- During speculative verification, if the fallback token's confidence is high (max_prob > 0.70), 
  the region is classified as "easy", and we compress the newly accepted KV cache entries along the sequence length dimension (average pooling every 2 entries).
- Compares:
  1. Baseline speculative decoding (No KV compression)
  2. Difficulty-Aware KV compression (compression only in easy regions)
  3. Aggressive KV compression (compress new entries at every step)
- Benchmarks final KV cache length reduction, acceptance rate, and Word Error Rate (WER).
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

# --- Loss function ---
def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

# --- KV Cache Compression Helper ---
def compress_kv_cache(kv_cache, A):
    if A < 2:
        return kv_cache
    new_cache = []
    for self_kv, cross_kv in kv_cache:
        if self_kv is not None:
            k, v = self_kv
            batch, L, D = k.shape
            
            # Context up to L-A
            k_prev = k[:, :L-A, :]
            v_prev = v[:, :L-A, :]
            
            # Newly added KV entries
            k_new = k[:, L-A:, :]
            v_new = v[:, L-A:, :]
            
            # Compress new entries by average pooling every 2 elements
            k_comp_list = []
            v_comp_list = []
            for idx in range(0, A, 2):
                if idx + 1 < A:
                    k_comp_list.append(mx.mean(k_new[:, idx:idx+2, :], axis=1, keepdims=True))
                    v_comp_list.append(mx.mean(v_new[:, idx:idx+2, :], axis=1, keepdims=True))
                else:
                    k_comp_list.append(k_new[:, idx:idx+1, :])
                    v_comp_list.append(v_new[:, idx:idx+1, :])
            
            k_comp = mx.concatenate([k_prev] + k_comp_list, axis=1)
            v_comp = mx.concatenate([v_prev] + v_comp_list, axis=1)
            
            self_kv = (k_comp, v_comp)
        new_cache.append((self_kv, cross_kv))
    return new_cache

# --- Speculative Generation Loop with KV Cache Compression ---
def generate_speculative(
    draft_model, target, mel, compression_mode='none', max_length=448, temperature=0.0
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
            
            # Span-level semantic graph verification (using our validated 0.95 threshold)
            G_hat = mx.matmul(norm_hat_H, norm_hat_H.T)
            G = mx.matmul(norm_H, norm_H.T)
            g_hat_flat = G_hat.reshape(-1)
            g_flat = G.reshape(-1)
            topo_sim = mx.sum(g_hat_flat * g_flat) / (mx.linalg.norm(g_hat_flat) * mx.linalg.norm(g_flat) + 1e-9)
            M_graph = 0.5 * np.mean(node_sims) + 0.5 * topo_sim.item()
            
            if M_graph >= 0.95:
                acceptance_length = K
            else:
                # Fallback to token-level verification at tau=0.97
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

        
        # Cache crop (before compression)
        kv_cache = crop_self_attention_cache(kv_cache, start)
        
        # Apply KV Compression
        target_token_logits = logits[0, acceptance_length]
        max_prob = mx.max(mx.softmax(target_token_logits, axis=-1)).item()
        
        is_easy = max_prob >= 0.70
        total_accepted = acceptance_length + 1
        
        if compression_mode == 'difficulty' and is_easy:
            kv_cache = compress_kv_cache(kv_cache, total_accepted)
        elif compression_mode == 'aggressive':
            kv_cache = compress_kv_cache(kv_cache, total_accepted)
            
        acceptance_lengths.append(total_accepted)
        block_sizes.append(current_block_size)
        
        # Context extraction for draft model
        # Note: If cache is compressed, start/length coordinates are offset.
        # However, target_hidden represents draft model tap history which is appended.
        ctx_layer = all_hidden_verify[draft_model.target_layer_ids[0]][:, : acceptance_length + 1, :]
        for layer_id in draft_model.target_layer_ids[1:]:
            ctx_layer = mx.concatenate([ctx_layer, all_hidden_verify[layer_id][:, : acceptance_length + 1, :]], axis=-1)
        target_hidden = mx.concatenate([target_hidden, ctx_layer], axis=1)
        
        # Stop condition
        if 50257 in output_list[:start]:
            break
            
    final_ids = output_list[:start]
    decoded_text = tokenizer.decode(final_ids)
    accept_rate = sum(acceptance_lengths) / sum(block_sizes)
    
    # Record the final KV cache sequence length
    final_kv_len = kv_cache[0][0][0].shape[1] if kv_cache and kv_cache[0][0] is not None else start
    
    return decoded_text, accept_rate, final_kv_len

def run():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    draft = ContinuousDraftModel(config)
    
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
    
    print("Pre-training Continuous Drafter (15 epochs)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    start_train = time.time()
    for epoch in range(15):
        epoch_loss = 0
        for data in data_tensors:
            loss, grads = loss_and_grad_fn(
                draft, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/15 - Loss: {epoch_loss/len(data_tensors):.5f}")
    print(f"Draft model trained in {time.time() - start_train:.1f}s.")
    
    # Evaluation on 10 held-out samples
    eval_samples = list(range(10, min(20, len(ds))))
    print(f"\nEvaluating speculative decoding on {len(eval_samples)} held-out samples...")
    
    results = {
        "none": {"wers": [], "accept_rates": [], "kv_lens": [], "texts": []},
        "difficulty": {"wers": [], "accept_rates": [], "kv_lens": [], "texts": []},
        "aggressive": {"wers": [], "accept_rates": [], "kv_lens": [], "texts": []}
    }
    
    references = [ds[idx]["text"] for idx in eval_samples]
        
    for mode in ["none", "difficulty", "aggressive"]:
        print(f"Running mode: {mode}...")
        for idx in eval_samples:
            sample = ds[idx]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel)[None]
            
            text, accept_rate, kv_len = generate_speculative(
                draft_model=draft,
                target=target,
                mel=mel_mx,
                compression_mode=mode,
                temperature=0.0
            )
            
            results[mode]["texts"].append(text)
            results[mode]["accept_rates"].append(accept_rate)
            results[mode]["kv_lens"].append(kv_len)
            
        wer = wer_metric.compute(predictions=results[mode]["texts"], references=references)
        results[mode]["wer"] = wer
        results[mode]["mean_accept_rate"] = np.mean(results[mode]["accept_rates"])
        results[mode]["mean_kv_len"] = np.mean(results[mode]["kv_lens"])
        
    print("\n" + "="*50)
    print("RESULTS: DIFFICULTY-AWARE CONTINUOUS KV COMPRESSION")
    print("="*50)
    
    print("\n--- 1. Word Error Rate (WER) ---")
    print(f"No Compression (Baseline)          : {results['none']['wer']:.4f}")
    print(f"Difficulty-Aware KV Compression    : {results['difficulty']['wer']:.4f}  (Delta: {results['difficulty']['wer'] - results['none']['wer']:+.4f})")
    print(f"Aggressive KV Compression          : {results['aggressive']['wer']:.4f}  (Delta: {results['aggressive']['wer'] - results['none']['wer']:+.4f})")
    
    print("\n--- 2. Mean KV Cache Sequence Length ---")
    print(f"No Compression (Baseline)          : {results['none']['mean_kv_len']:.1f} tokens")
    print(f"Difficulty-Aware KV Compression    : {results['difficulty']['mean_kv_len']:.1f} tokens  (Savings: {100 * (1 - results['difficulty']['mean_kv_len'] / results['none']['mean_kv_len']):.1f}%)")
    print(f"Aggressive KV Compression          : {results['aggressive']['mean_kv_len']:.1f} tokens  (Savings: {100 * (1 - results['aggressive']['mean_kv_len'] / results['none']['mean_kv_len']):.1f}%)")
    
    print("\n--- 3. Mean Speculative Acceptance Rate ---")
    print(f"No Compression (Baseline)          : {results['none']['mean_accept_rate']*100:.2f}%")
    print(f"Difficulty-Aware KV Compression    : {results['difficulty']['mean_accept_rate']*100:.2f}%  (Delta: {(results['difficulty']['mean_accept_rate'] - results['none']['mean_accept_rate'])*100:+.2f}%)")
    print(f"Aggressive KV Compression          : {results['aggressive']['mean_accept_rate']*100:.2f}%  (Delta: {(results['aggressive']['mean_accept_rate'] - results['none']['mean_accept_rate'])*100:+.2f}%)")
    print("="*50)
    
if __name__ == "__main__":
    run()
