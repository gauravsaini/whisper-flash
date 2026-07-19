#!/usr/bin/env python3
"""
experiment_id23.py

Unified Conditional ASR v2 — Fully Continuous Pipeline
Integrates:
- Consistency-Model Drafting in PCA Subspace (from ID 28)
- Bottleneck-Free Architecture (from ID 28)
- Curvature-Aligned Training (from Exp 3.9)
- Span-Level Graph Verification (from Exp 3.9)
- 20-step rollout chunks
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
from whisper_flash_mlx.draft_model import WhisperDFlashConfig
from whisper_flash_mlx.generate import crop_self_attention_cache

# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------

class BottleneckFreeConsistencyPCAModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig, pca_rank: int = 64):
        super().__init__()
        self.config = config
        self.pca_rank = pca_rank

        num_taps = len(config.target_layer_ids)
        ctx_dim = num_taps * config.d_target + config.d_target
        self.ctx_proj = nn.Linear(ctx_dim, config.d_draft)

        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, config.d_draft)
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.d_draft, config.d_draft * 2),
            nn.GELU(),
            nn.Linear(config.d_draft * 2, config.block_size * pca_rank)
        )

        self.norm = nn.LayerNorm(pca_rank)
        self.target_layer_ids = config.target_layer_ids

    def __call__(self, noisy_z, target_hidden, audio_summary, position_ids, sigma):
        last_target = target_hidden[:, -1:, :]
        ctx_input = mx.concatenate([last_target, audio_summary], axis=-1)
        hidden = self.ctx_proj(ctx_input)
        
        ln_sigma = mx.log(mx.clip(sigma, 1e-9, 1e9))
        if len(ln_sigma.shape) == 1:
            ln_sigma = ln_sigma[:, None]
        sigma_emb = self.sigma_mlp(ln_sigma)
        hidden = hidden + sigma_emb[:, None, :]
        
        predicted = self.mlp(hidden)
        bsz = predicted.shape[0]
        predicted = predicted.reshape(bsz, self.config.block_size, self.pca_rank)
        predicted = self.norm(predicted)
        return predicted

def get_consistency_prediction_pca(model, z, target_hidden, audio_summary, position_ids, sigma):
    sigma_min = 0.002
    if not isinstance(sigma, mx.array):
        sigma = mx.array(sigma)
    if len(sigma.shape) == 1:
        sigma = sigma[:, None]
    elif len(sigma.shape) == 0:
        sigma = sigma[None, None]

    c_skip = (sigma_min ** 2) / ((sigma - sigma_min) ** 2 + sigma_min ** 2)
    c_out = (sigma - sigma_min) / mx.sqrt((sigma - sigma_min) ** 2 + sigma_min ** 2)

    c_skip = c_skip[:, :, None]
    c_out = c_out[:, :, None]

    F_out = model(z, target_hidden, audio_summary, position_ids, sigma)
    return c_skip * z + c_out * F_out

def curvature_consistency_loss_pca(online_model, target_model, clean_z, ctx, audio, pos, sigma_n, sigma_np1, lambda_curv=0.5):
    z_noise = mx.random.normal(clean_z.shape)
    z_n = clean_z + sigma_n * z_noise
    z_np1 = clean_z + sigma_np1 * z_noise

    batch_size = clean_z.shape[0]
    sigma_n_arr = mx.full((batch_size, 1), sigma_n)
    sigma_np1_arr = mx.full((batch_size, 1), sigma_np1)

    pred_online = get_consistency_prediction_pca(online_model, z_np1, ctx, audio, pos, sigma_np1_arr)
    pred_target = get_consistency_prediction_pca(target_model, z_n, ctx, audio, pos, sigma_n_arr)

    mse = mx.mean(mx.square(pred_online - pred_target))
    
    if pred_online.shape[1] >= 3:
        pred_accel = pred_online[:, 2:, :] - 2 * pred_online[:, 1:-1, :] + pred_online[:, :-2, :]
        target_accel = clean_z[:, 2:, :] - 2 * clean_z[:, 1:-1, :] + clean_z[:, :-2, :]
        curv_penalty = mx.mean(mx.square(pred_accel - target_accel))
    else:
        curv_penalty = mx.array(0.0)

    return mse + lambda_curv * curv_penalty

# ---------------------------------------------------------------------------
# 2. Adapter for Generation
# ---------------------------------------------------------------------------

class PipelineAdapterModel(nn.Module):
    def __init__(self, bf_pca_model, V_mx, mean_mx):
        super().__init__()
        self.bf = bf_pca_model
        self.V = V_mx
        self.mean = mean_mx
        self.config = bf_pca_model.config
        self.mask_token_id = bf_pca_model.config.mask_token_id
        self.target_layer_ids = bf_pca_model.config.target_layer_ids

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids):
        bsz = target_hidden.shape[0]
        pca_rank = self.bf.pca_rank
        block_size = self.config.block_size
        
        sigma_max = 80.0
        sigma_mid = 10.0
        
        z1_pca = mx.random.normal((bsz, block_size, pca_rank))
        y1_pca = sigma_max * z1_pca
        sigma_max_arr = mx.full((bsz, 1), sigma_max)
        
        pred1 = get_consistency_prediction_pca(self.bf, y1_pca, target_hidden, audio_summary, position_ids, sigma_max_arr)
        
        z2_pca = mx.random.normal((bsz, block_size, pca_rank))
        y2_pca = pred1 + math.sqrt(max(sigma_mid**2 - 0.002**2, 1e-9)) * z2_pca
        sigma_mid_arr = mx.full((bsz, 1), sigma_mid)
        
        pred2_z = get_consistency_prediction_pca(self.bf, y2_pca, target_hidden, audio_summary, position_ids, sigma_mid_arr)
        
        pred_full = pred2_z @ self.V.T + self.mean
        return pred_full

# ---------------------------------------------------------------------------
# 3. Span-Level Graph Verification Generation Loop
# ---------------------------------------------------------------------------

def generate_speculative(draft_model, target, mel, max_length=448, temperature=0.0):
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    block_size = draft_model.config.block_size
    mask_token_id = draft_model.mask_token_id
    
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    output_list = [mask_token_id] * (max_length + block_size)
    output_list[0] = 50258  # SOT
    
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
            
        for i in range(acceptance_length + 1):
            output_list[start + i] = block_ids_list[i]
            
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

# ---------------------------------------------------------------------------
# 4. Main Experiment
# ---------------------------------------------------------------------------

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

def run():
    print("="*60)
    print("Unified Conditional ASR v2 — Fully Continuous Pipeline")
    print("="*60)
    
    print("Loading Target Model...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    d_target = target.dims.n_text_state
    pca_rank = 64
    
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=20, target_layer_ids=[1, 2]
    )
    
    online_model = BottleneckFreeConsistencyPCAModel(config, pca_rank=pca_rank)
    target_model_draft = BottleneckFreeConsistencyPCAModel(config, pca_rank=pca_rank)
    
    pca_noise_init = mx.zeros((1, config.block_size, pca_rank))
    ctx_init = mx.zeros((1, config.block_size, len(config.target_layer_ids) * d_target))
    audio_init = mx.zeros((1, 1, d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    sigma_init = mx.ones((1, 1))
    
    _ = online_model(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)
    _ = target_model_draft(pca_noise_init, ctx_init, audio_init, pos_init, sigma_init)
    target_model_draft.update(copy_parameters(online_model))
    
    print("Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-extracting dataset context features...")
    data_tensors = []
    
    # We load 5 points just for quick pre-training validation of the integration concept
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
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            
            data_tensors.append({
                "ctx": ctx_feats, "audio": audio_summary, "pos": pos_ids, "true_hidden": true_hidden
            })
            
    print("Computing PCA/SVD subspace components...")
    all_true_h = np.concatenate([np.array(d["true_hidden"]) for d in data_tensors], axis=0)
    M_samples, B_block, D_dim = all_true_h.shape
    X = all_true_h.reshape(-1, D_dim)
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt[:pca_rank, :].T
    mean_mx = mx.array(mean)
    V_mx = mx.array(V)
    
    for d in data_tensors:
        d["true_z"] = (d["true_hidden"] - mean_mx) @ V_mx
        
    print("Training Unified Pipeline Model (15 epochs)...")
    epochs = 15
    sigmas = get_sigma_schedule(num_steps=10, sigma_min=0.002, sigma_max=80.0)
    opt = optim.Adam(learning_rate=1e-3)
    grad_fn = nn.value_and_grad(online_model, curvature_consistency_loss_pca)
    
    t0 = time.time()
    for epoch in range(epochs):
        loss_sum = 0
        for data in data_tensors:
            n = np.random.randint(0, len(sigmas) - 1)
            loss, grads = grad_fn(
                online_model, target_model_draft, data["true_z"], data["ctx"], data["audio"], data["pos"],
                sigmas[n], sigmas[n+1], lambda_curv=0.5
            )
            opt.update(online_model, grads)
            update_target_parameters(online_model, target_model_draft, ema_mu=0.95)
            mx.eval(online_model.parameters(), target_model_draft.parameters(), opt.state)
            loss_sum += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {loss_sum/len(data_tensors):.5f}")
    print(f"Trained in {time.time() - t0:.1f}s")
    
    # Evaluation
    adapter = PipelineAdapterModel(online_model, V_mx, mean_mx)
    eval_samples = list(range(10, min(15, len(ds))))
    print(f"\nEvaluating speculative decoding on {len(eval_samples)} held-out samples...")
    
    results = {"baseline": {"wers": [], "texts": [], "times": []}, "unified": {"wers": [], "texts": [], "times": [], "acc_rates": []}}
    references = [ds[idx]["text"] for idx in eval_samples]
    
    from whisper_flash_mlx.generate import whisper_dflash_generate as generate_discrete
    
    for idx in eval_samples:
        sample = ds[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        start = time.perf_counter()
        res_base = generate_discrete(None, target, mel_mx, return_stats=True)
        t_base = time.perf_counter() - start
        dec_base = tokenizer.decode(res_base.output_ids[0].tolist())
        results["baseline"]["texts"].append(dec_base)
        results["baseline"]["times"].append(t_base)
        
        start = time.perf_counter()
        dec_uni, acc_rate = generate_speculative(adapter, target, mel_mx, temperature=0.0)
        t_uni = time.perf_counter() - start
        results["unified"]["texts"].append(dec_uni)
        results["unified"]["times"].append(t_uni)
        results["unified"]["acc_rates"].append(acc_rate)
        
    w_base = wer_metric.compute(predictions=results["baseline"]["texts"], references=references)
    w_uni = wer_metric.compute(predictions=results["unified"]["texts"], references=references)
    mean_acc = np.mean(results["unified"]["acc_rates"])
    mean_t_base = np.mean(results["baseline"]["times"])
    mean_t_uni = np.mean(results["unified"]["times"])
    speedup = mean_t_base / mean_t_uni
    
    print("\n" + "="*50)
    print("RESULTS: UNIFIED CONDITIONAL ASR v2")
    print("="*50)
    print(f"Baseline WER      : {w_base:.4f}")
    print(f"Unified WER       : {w_uni:.4f}")
    print(f"Acceptance Rate   : {mean_acc*100:.2f}%")
    print(f"Avg Speedup       : {speedup:.2f}x")
    print("="*50)

if __name__ == "__main__":
    run()
