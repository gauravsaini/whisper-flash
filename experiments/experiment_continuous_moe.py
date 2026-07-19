import math
import copy
import mlx.core as mx
import numpy as np
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig

# -------------------------------------------------------------------------
# Continuous MoE Drafter
# -------------------------------------------------------------------------
class ExpertMLP(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        # Simplified bottleneck-free design (no cross attention)
        d_in = config.d_target * len(config.target_layer_ids) + config.d_target
        d_out = config.d_target
        self.net = nn.Sequential(
            nn.Linear(d_in, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, config.d_draft),
            nn.GELU(),
            nn.Linear(config.d_draft, d_out)
        )
        
    def __call__(self, x):
        return self.net(x)

class ContinuousMoEDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig, num_experts: int = 2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        
        # We need an audio projection for bottleneck-free MLP
        self.audio_proj = nn.Linear(config.d_target, config.d_target)
        self.noise_proj = nn.Linear(config.d_target, config.d_target)
        
        # MoE Router: takes context and audio to predict difficulty and routing weights
        d_in = config.d_target * len(config.target_layer_ids) + config.d_target
        self.router = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, num_experts + 1) # First output is difficulty [0, 1], rest are logits
        )
        
        self.experts = [ExpertMLP(config) for _ in range(num_experts)]
        self.layer_norm = nn.LayerNorm(config.d_target)

    def __call__(self, noise_embedding, target_hidden, audio_summary, position_ids):
        # noise_embedding: (B, S, D)
        # target_hidden: (B, 1, D * num_layers)
        # audio_summary: (B, 1, D)
        B, S, D = noise_embedding.shape
        
        ctx = mx.repeat(target_hidden, S, axis=1) # (B, S, D*num_layers)
        aud = mx.repeat(audio_summary, S, axis=1) # (B, S, D)
        
        router_in = mx.concatenate([ctx, aud], axis=-1)
        router_out = self.router(router_in) # (B, S, num_experts + 1)
        
        # Difficulty coordinate (0 = easy, 1 = hard)
        difficulty = mx.sigmoid(router_out[..., 0:1]) 
        
        # Routing weights
        routing_logits = router_out[..., 1:]
        routing_weights = mx.softmax(routing_logits, axis=-1) # (B, S, num_experts)
        
        # Input to experts
        x_in = mx.concatenate([ctx, self.audio_proj(aud) + self.noise_proj(noise_embedding)], axis=-1)
        
        # Compute expert outputs
        expert_outs = [expert(x_in) for expert in self.experts]
        
        # Combine
        combined_out = mx.zeros_like(expert_outs[0])
        for i in range(self.num_experts):
            combined_out = combined_out + routing_weights[..., i:i+1] * expert_outs[i]
            
        out = self.layer_norm(combined_out)
        
        return out, difficulty, routing_weights

# -------------------------------------------------------------------------
# Dynamic B Verification
# -------------------------------------------------------------------------
def generate_speculative(draft_model, target, mel, max_length=150, max_B=20):
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    mask_token_id = draft_model.config.mask_token_id
    
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    output_list = [mask_token_id] * (max_length + max_B)
    output_list[0] = 50258
    
    logits_init, kv_cache, all_hidden_init = decoder_forward_with_hidden_states(
        target, mx.array([[50258]], dtype=mx.int32), encoder_hidden, collect_hidden_states=True, return_cross_attention=False
    )
    first_token = mx.argmax(logits_init[:, -1:, :], axis=-1).item()
    output_list[1] = first_token
    
    target_hidden = all_hidden_init[draft_model.config.target_layer_ids[0]]
    for layer_id in draft_model.config.target_layer_ids[1:]:
        target_hidden = mx.concatenate([target_hidden, all_hidden_init[layer_id]], axis=-1)
        
    start = 1
    
    acceptance_lengths = []
    block_sizes = []
    difficulties = []
    
    while start < max_length:
        # Step 1: Predict difficulty using just the first step of the block
        noise_embedding_init = target.decoder.token_embedding(mx.array([[mask_token_id]], dtype=mx.int32))
        pos_init = mx.arange(start, start + 1)[None]
        
        _, diff_val, _ = draft_model(noise_embedding_init, target_hidden[:, -1:, :], audio_summary, pos_init)
        d_val = diff_val[0, 0, 0].item()
        
        # Dynamic block size B in [1, max_B]
        current_block_size = max(1, int(max_B * (1.0 - d_val)))
        
        block_ids_list = output_list[start: start + current_block_size]
        while len(block_ids_list) < current_block_size:
            block_ids_list.append(mask_token_id)
        block_ids = mx.array([block_ids_list], dtype=mx.int32)
        block_positions = mx.arange(start, start + current_block_size)[None]
        
        # Draft step
        if current_block_size > 1:
            noise_embedding = target.decoder.token_embedding(block_ids)
            draft_hidden, _, _ = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden[:, -1:, :],
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
            
        for i in range(acceptance_length + 1):
            output_list[start + i] = block_ids_list[i]
        output_list[start + acceptance_length + 1] = posterior_list[acceptance_length]
        
        start += acceptance_length + 1
        
        from whisper_flash_mlx.generate import crop_self_attention_cache
        kv_cache = crop_self_attention_cache(kv_cache, start)
        
        acceptance_lengths.append(acceptance_length + 1)
        block_sizes.append(current_block_size)
        difficulties.append(d_val)
        
        ctx_layer = all_hidden_verify[draft_model.config.target_layer_ids[0]][:, : acceptance_length + 1, :]
        for layer_id in draft_model.config.target_layer_ids[1:]:
            ctx_layer = mx.concatenate([ctx_layer, all_hidden_verify[layer_id][:, : acceptance_length + 1, :]], axis=-1)
        target_hidden = mx.concatenate([target_hidden, ctx_layer], axis=1)
        
        if 50257 in output_list[:start]:
            break
            
    final_ids = output_list[:start]
    decoded_text = tokenizer.decode(final_ids)
    accept_rate = sum(acceptance_lengths) / sum(block_sizes)
    return decoded_text, accept_rate, np.mean(difficulties), sum(block_sizes)/len(block_sizes)

# -------------------------------------------------------------------------
# Loss Function
# -------------------------------------------------------------------------
def moe_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden, diff, weights = model(noise, target_hidden, audio_summary, position_ids)
    
    # 1. Prediction MSE
    mse = mx.mean(mx.square(pred_hidden - true_hidden))
    
    # 2. Difficulty label: measure target_hidden local gradient norm as proxy for acoustic difficulty
    # Approximation: if true_hidden varies a lot, it's hard.
    # For now, we'll use distance to target_hidden as a proxy for difficulty.
    B, S, D = true_hidden.shape
    dist = mx.mean(mx.square(true_hidden - target_hidden[..., -D:]))
    # normalize to 0-1 via sigmoid
    target_diff = mx.sigmoid(dist - 0.5)
    diff_loss = mx.mean(mx.square(diff - target_diff))
    
    # 3. Load balancing loss for MoE
    mean_routing = mx.mean(weights, axis=(0, 1))
    load_loss = mx.sum(mean_routing * mx.log(mean_routing + 1e-9))
    
    return mse + 0.5 * diff_loss + 0.1 * load_loss

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def run():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    import evaluate
    wer_metric = evaluate.load("wer")
    
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=20, target_layer_ids=[1, 2] # Large block size!
    )
    
    draft_moe = ContinuousMoEDraftModel(config, num_experts=2)
    
    # Force initialization
    noise_init = mx.zeros((1, config.block_size, config.d_target))
    ctx_init = mx.zeros((1, 1, len(config.target_layer_ids) * config.d_target))
    audio_init = mx.zeros((1, 1, config.d_target))
    pos_init = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = draft_moe(noise_init, ctx_init, audio_init, pos_init)
    
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
        
        for t in range(1, labels.shape[1] - config.block_size, 5):
            input_token = labels[:, :t+1]
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            
            ctx_feats = [hidden_target[layer_id] for layer_id in config.target_layer_ids]
            ctx_feats = mx.concatenate(ctx_feats, axis=-1)
            ctx_last = ctx_feats[:, -1:, :]
            
            _, _, hidden_future = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
            )
            true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
            
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
            pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
            
            data_tensors.append({
                "noise": noise,
                "ctx": ctx_last,
                "audio": audio_summary,
                "pos": pos_ids,
                "true_hidden": true_hidden
            })
            
    print(f"Pre-extraction complete. Extracted {len(data_tensors)} train points.")
    
    print("Training MoE Drafter (25 epochs)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(draft_moe, moe_loss_fn)
    
    for epoch in range(25):
        loss_sum = 0
        for data in data_tensors:
            loss, grads = loss_and_grad(
                draft_moe, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            optimizer.update(draft_moe, grads)
            mx.eval(draft_moe.parameters(), optimizer.state)
            loss_sum += loss.item()
        print(f"Epoch {epoch+1:02d}/25 - Loss: {loss_sum/len(data_tensors):.5f}")
        
    print("\nEvaluating MoE Dynamic Block Decoding on 10 held-out samples...")
    eval_samples = list(range(10, min(20, len(ds))))
    
    texts = []
    accept_rates = []
    diffs = []
    avg_bs = []
    references = [ds[idx]["text"] for idx in eval_samples]
    
    for idx in eval_samples:
        sample = ds[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        text, accept_rate, avg_diff, avg_b = generate_speculative(
            draft_model=draft_moe,
            target=target,
            mel=mel_mx,
            max_B=20
        )
        
        texts.append(text)
        accept_rates.append(accept_rate)
        diffs.append(avg_diff)
        avg_bs.append(avg_b)
        
    wer = wer_metric.compute(predictions=texts, references=references)
    
    print("\n" + "="*50)
    print("RESULTS: CONTINUOUS MOE EXPERTS & DYNAMIC BLOCK SIZE")
    print("="*50)
    print(f"--- 1. Word Error Rate (WER) ---")
    print(f"Continuous MoE Drafter             : {wer:.4f}")
    print(f"--- 2. Mean Speculative Acceptance Rate ---")
    print(f"Continuous MoE Drafter             : {np.mean(accept_rates)*100:.2f}%")
    print(f"--- 3. Mean Difficulty Coordinate ---")
    print(f"Average Route Difficulty [0, 1]    : {np.mean(diffs):.4f}")
    print(f"--- 4. Mean Block Size ---")
    print(f"Average Dynamic Block Size (B)     : {np.mean(avg_bs):.2f} tokens / step")
    print("="*50)

if __name__ == "__main__":
    run()
