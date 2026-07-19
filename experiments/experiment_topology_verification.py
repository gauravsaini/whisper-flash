#!/usr/bin/env python3
"""
experiment_semantic_graph.py

Moonshot 3.7: Span-level Semantic Graph Verification
- Implements a verification topology over a sequence of draft continuous states.
- Instead of checking token similarity sequentially and rolling back at the first deviation,
  we build a Pairwise Cosine Similarity Graph (Gram Matrix) for the drafted block and the target block.
- We measure node-to-node similarity (local alignment) and Gram matrix cosine similarity (topological consistency).
- We compute a global Graph Metric M_graph. If M_graph >= 0.95, we accept the entire span of B tokens.
  Otherwise, we fall back to sequential token-level verification at a strict threshold of tau = 0.97.
- Evaluates WER, Acceptance Rate, and Speedup compared to strict lexical verification and token-level semantic verification.
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

# --- Speculative Generation Loop with Span-Level Graph Verification ---
def generate_speculative(
    draft_model, target, mel, verification_mode='lexical', max_length=448, temperature=0.0
):
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    block_size = draft_model.config.block_size
    mask_token_id = draft_model.mask_token_id
    
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    sot_seq = tokenizer.sot_sequence_including_notimestamps
    output_list = [mask_token_id] * (max_length + block_size + len(sot_seq))
    output_list[:len(sot_seq)] = sot_seq
    
    logits_init, kv_cache, all_hidden_init = decoder_forward_with_hidden_states(
        target, mx.array([sot_seq], dtype=mx.int32), encoder_hidden, collect_hidden_states=True, return_cross_attention=False
    )
    first_token = mx.argmax(logits_init[:, -1:, :], axis=-1).item()
    output_list[len(sot_seq)] = first_token
    
    target_hidden = all_hidden_init[draft_model.target_layer_ids[0]]
    for layer_id in draft_model.target_layer_ids[1:]:
        target_hidden = mx.concatenate([target_hidden, all_hidden_init[layer_id]], axis=-1)
        
    start = len(sot_seq)
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
            # Extract drafted and target states
            hat_H = draft_hidden[0, :K] # (K, D)
            H = true_hidden[:K] # (K, D)
            
            # Compute cosine similarities for matched nodes
            norm_hat_H = hat_H / (mx.linalg.norm(hat_H, axis=-1, keepdims=True) + 1e-9)
            norm_H = H / (mx.linalg.norm(H, axis=-1, keepdims=True) + 1e-9)
            node_sims = mx.sum(norm_hat_H * norm_H, axis=-1) # (K,)
            node_sims_list = node_sims.tolist()
            
            if verification_mode == 'lexical':
                # Strict lexical match
                for i in range(1, current_block_size):
                    if block_ids_list[i] == posterior_list[i - 1]:
                        acceptance_length += 1
                    else:
                        break
            elif verification_mode == 'token_semantic':
                # Token-level semantic verification with safe threshold 0.97
                for i in range(1, current_block_size):
                    lexical_match = block_ids_list[i] == posterior_list[i - 1]
                    sim_val = node_sims_list[i - 1]
                    semantic_match = sim_val >= 0.97
                    
                    if lexical_match or semantic_match:
                        acceptance_length += 1
                    else:
                        break
            elif verification_mode == 'graph_semantic':
                # Span-level semantic graph verification
                # 1. Compute Gram Matrices
                G_hat = mx.matmul(norm_hat_H, norm_hat_H.T) # (K, K)
                G = mx.matmul(norm_H, norm_H.T) # (K, K)
                
                # 2. Compute Topological Similarity
                g_hat_flat = G_hat.reshape(-1)
                g_flat = G.reshape(-1)
                topo_sim = mx.sum(g_hat_flat * g_flat) / (mx.linalg.norm(g_hat_flat) * mx.linalg.norm(g_flat) + 1e-9)
                topo_sim_val = topo_sim.item()
                
                # 3. Compute overall Graph Metric
                mean_node_sim = mx.mean(node_sims).item()
                alpha = 0.5
                M_graph = alpha * mean_node_sim + (1 - alpha) * topo_sim_val
                
                if M_graph >= 0.95:
                    # Accept the entire block!
                    acceptance_length = K
                else:
                    # Fall back to token-level verification at tau=0.97
                    for i in range(1, current_block_size):
                        lexical_match = block_ids_list[i] == posterior_list[i - 1]
                        sim_val = node_sims_list[i - 1]
                        semantic_match = sim_val >= 0.97
                        
                        if lexical_match or semantic_match:
                            acceptance_length += 1
                        else:
                            break
            elif verification_mode == 'graph_curvature':
                G_hat = mx.matmul(norm_hat_H, norm_hat_H.T)
                G = mx.matmul(norm_H, norm_H.T)
                g_hat_flat = G_hat.reshape(-1)
                g_flat = G.reshape(-1)
                topo_sim = mx.sum(g_hat_flat * g_flat) / (mx.linalg.norm(g_hat_flat) * mx.linalg.norm(g_flat) + 1e-9)
                topo_sim_val = topo_sim.item()
                mean_node_sim = mx.mean(node_sims).item()
                M_graph = 0.5 * mean_node_sim + 0.5 * topo_sim_val
                
                if K > 1:
                    delta_hat = hat_H[1:] - hat_H[:-1]
                    delta_H = H[1:] - H[:-1]
                    norm_delta_hat = delta_hat / (mx.linalg.norm(delta_hat, axis=-1, keepdims=True) + 1e-9)
                    norm_delta_H = delta_H / (mx.linalg.norm(delta_H, axis=-1, keepdims=True) + 1e-9)
                    curve_sims = mx.sum(norm_delta_hat * norm_delta_H, axis=-1)
                    M_curve = mx.mean(curve_sims).item()
                else:
                    M_curve = 1.0
                    
                if M_graph >= 0.95 and M_curve >= 0.95:
                    acceptance_length = K
                else:
                    for i in range(1, current_block_size):
                        lexical_match = block_ids_list[i] == posterior_list[i - 1]
                        sim_val = node_sims_list[i - 1]
                        semantic_match = sim_val >= 0.97
                        if lexical_match or semantic_match:
                            acceptance_length += 1
                        else:
                            break
        else:
            # Block size is 1 (fallback), greedy acceptance
            acceptance_length = 0
            
        # Accept tokens
        for i in range(acceptance_length + 1):
            output_list[start + i] = block_ids_list[i]
            
        # Fallback token correction
        output_list[start + acceptance_length + 1] = posterior_list[acceptance_length]
        
        start += acceptance_length + 1

        
        # Cache updates
        kv_cache = crop_self_attention_cache(kv_cache, start)
        
        acceptance_lengths.append(acceptance_length + 1)
        block_sizes.append(current_block_size)
        
        # Context extraction for draft model
        ctx_layer = all_hidden_verify[draft_model.target_layer_ids[0]][:, : acceptance_length + 1, :]
        for layer_id in draft_model.target_layer_ids[1:]:
            ctx_layer = mx.concatenate([ctx_layer, all_hidden_verify[layer_id][:, : acceptance_length + 1, :]], axis=-1)
        target_hidden = mx.concatenate([target_hidden, ctx_layer], axis=1)
        
        # Stop condition
        if 50257 in output_list[:start]:
            break
            
    final_ids = output_list[:start]
    clean_ids = [t for t in final_ids if t < 50257]
    decoded_text = tokenizer.decode(clean_ids).strip()
    import re
    def normalize(t):
        return re.sub(r'[^\w\s]', '', t).lower().strip()
    decoded_text = normalize(decoded_text)
    accept_rate = sum(acceptance_lengths) / sum(block_sizes)
    return decoded_text, accept_rate

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
        sot = mx.array([tokenizer.sot_sequence_including_notimestamps], dtype=mx.int32)
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
    
    # Evaluation on held-out samples 10 to 19 (larger validation pool of 10 samples)
    eval_samples = list(range(10, min(20, len(ds))))
    print(f"\nEvaluating speculative decoding on {len(eval_samples)} held-out samples...")
    
    results = {
        "lexical": {"wer": 0, "accept_rates": [], "mean_accept_rate": 0, "texts": [], "refs": []},
        "token_semantic": {"wer": 0, "accept_rates": [], "mean_accept_rate": 0, "texts": [], "refs": []},
        "graph_semantic": {"wer": 0, "accept_rates": [], "mean_accept_rate": 0, "texts": [], "refs": []},
        "graph_curvature": {"wer": 0, "accept_rates": [], "mean_accept_rate": 0, "texts": [], "refs": []},
    }
    
    # Ground truth references
    references = []
    for idx in eval_samples:
        references.append(ds[idx]["text"])
        
    for mode in ["lexical", "token_semantic", "graph_semantic", "graph_curvature"]:
        print(f"Running mode: {mode}...")
        for idx in eval_samples:
            sample = ds[idx]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel)[None]
            
            text, accept_rate = generate_speculative(
                draft_model=draft,
                target=target,
                mel=mel_mx,
                verification_mode=mode,
                temperature=0.0
            )
            
            # Clean text formatting for WER computation
            # Store refs and normalize for evaluation
            import re
            def normalize(t):
                return re.sub(r'[^\w\s]', '', t).lower().strip()
                
            results[mode]["texts"].append(text)
            results[mode]["refs"].append(normalize(sample["text"]))
            results[mode]["accept_rates"].append(accept_rate)
            
        # Calculate WER over the batch
        wer = wer_metric.compute(predictions=results[mode]["texts"], references=results[mode]["refs"])
        results[mode]["wer"] = wer
        results[mode]["mean_accept_rate"] = np.mean(results[mode]["accept_rates"])
        
    print("\n" + "="*50)
    print("RESULTS: SPAN-LEVEL SEMANTIC GRAPH VERIFICATION")
    print("="*50)
    
    print("\n--- 1. Word Error Rate (WER) ---")
    print(f"Strict Lexical Verification        : {results['lexical']['wer']:.4f}")
    print(f"Token-Level Semantic (tau=0.97)    : {results['token_semantic']['wer']:.4f}  (Delta: {results['token_semantic']['wer'] - results['lexical']['wer']:+.4f})")
    print(f"Span-Level Graph Verification      : {results['graph_semantic']['wer']:.4f}  (Delta: {results['graph_semantic']['wer'] - results['lexical']['wer']:+.4f})")
    print(f"Span-Level Graph+Curvature         : {results['graph_curvature']['wer']:.4f}  (Delta: {results['graph_curvature']['wer'] - results['lexical']['wer']:+.4f})")
    
    print("\n--- 2. Mean Speculative Acceptance Rate ---")
    print(f"Strict Lexical Verification        : {results['lexical']['mean_accept_rate']*100:.2f}%")
    print(f"Token-Level Semantic (tau=0.97)    : {results['token_semantic']['mean_accept_rate']*100:.2f}%  (Delta: {(results['token_semantic']['mean_accept_rate'] - results['lexical']['mean_accept_rate'])*100:+.2f}%)")
    print(f"Span-Level Graph Verification      : {results['graph_semantic']['mean_accept_rate']*100:.2f}%  (Delta: {(results['graph_semantic']['mean_accept_rate'] - results['lexical']['mean_accept_rate'])*100:+.2f}%)")
    print(f"Span-Level Graph+Curvature         : {results['graph_curvature']['mean_accept_rate']*100:.2f}%  (Delta: {(results['graph_curvature']['mean_accept_rate'] - results['lexical']['mean_accept_rate'])*100:+.2f}%)")
    
    # Expected speedup proxy relative to lexical
    speedup_token = results['token_semantic']['mean_accept_rate'] / (results['lexical']['mean_accept_rate'] + 1e-9)
    speedup_graph = results['graph_semantic']['mean_accept_rate'] / (results['lexical']['mean_accept_rate'] + 1e-9)
    speedup_curve = results['graph_curvature']['mean_accept_rate'] / (results['lexical']['mean_accept_rate'] + 1e-9)
    
    print("\n--- 3. Relative Speedup Factor (vs Lexical) ---")
    print(f"Token-Level Semantic (tau=0.97)    : {speedup_token:.3f}x")
    print(f"Span-Level Graph Verification      : {speedup_graph:.3f}x")
    print(f"Span-Level Graph+Curvature         : {speedup_curve:.3f}x  (Speedup Gain vs Graph: {speedup_curve - speedup_graph:+.3f}x)")
    print("="*50)
    
if __name__ == "__main__":
    run()
