#!/usr/bin/env python3
"""
experiment_hybrid_verification.py

Moonshot 2.4: Hybrid Verification
- Easy regions (confidence > 0.6): Use semantic verification (cosine sim > 0.8)
- Hard regions (confidence <= 0.6): Fall back to strict discrete lexical verification
"""

import time
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
from whisper_flash_mlx.generate import whisper_dflash_generate as generate_discrete, crop_self_attention_cache

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

# Custom hybrid generation function based on generate_continuous.py
def generate_hybrid(
    draft_model, target, mel, max_length=448, temperature=0.0
):
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    block_size = draft_model.config.block_size
    mask_token_id = draft_model.mask_token_id
    
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    output_list = [mask_token_id] * (max_length + block_size)
    output_list[0] = 50258  # SOT
    
    # Initialize target_hidden by running a forward pass on the prompt
    _, kv_cache, all_hidden_init = decoder_forward_with_hidden_states(
        target, mx.array([[50258]], dtype=mx.int32), encoder_hidden, collect_hidden_states=True, return_cross_attention=False
    )
    target_hidden = all_hidden_init[draft_model.target_layer_ids[0]]
    for layer_id in draft_model.target_layer_ids[1:]:
        target_hidden = mx.concatenate([target_hidden, all_hidden_init[layer_id]], axis=-1)
    
    start = 1
    current_block_size = block_size
    acceptance_lengths = []
    block_sizes = []
    
    decode_start = time.perf_counter()
    
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
        
        # Difficulty routing
        target_token_logits = logits[0, 0]
        max_prob = mx.max(mx.softmax(target_token_logits, axis=-1)).item()
        is_easy = max_prob > 0.6
        
        posterior_list = posterior.tolist()[0]
        acceptance_length = 0
        true_hidden = all_hidden_verify[-1][0]
        
        for i in range(1, current_block_size):
            if i < len(block_ids_list):
                h_draft = draft_hidden[0, i-1]
                h_target = true_hidden[i-1]
                sim = mx.sum(h_draft * h_target) / (mx.linalg.norm(h_draft) * mx.linalg.norm(h_target) + 1e-9)
                sim_val = sim.item()
                
                lexical_match = block_ids_list[i] == posterior_list[i - 1]
                semantic_match = sim_val > 0.80
                
                if lexical_match:
                    acceptance_length += 1
                elif is_easy and semantic_match:
                    acceptance_length += 1
                else:
                    break
            else:
                break
                
        for i in range(acceptance_length + 1):
            idx = start + i
            output_list[idx] = block_ids_list[i] if i < acceptance_length else posterior_list[i]
            
        start += acceptance_length + 1
        
        # Cache updates
        kv_cache = crop_self_attention_cache(kv_cache, start)
        
        acceptance_lengths.append(acceptance_length + 1)
        block_sizes.append(current_block_size)
        
        # Context extraction
        ctx_layer = all_hidden_verify[draft_model.target_layer_ids[0]][:, : acceptance_length + 1, :]
        for layer_id in draft_model.target_layer_ids[1:]:
            ctx_layer = mx.concatenate([ctx_layer, all_hidden_verify[layer_id][:, : acceptance_length + 1, :]], axis=-1)
        target_hidden = mx.concatenate([target_hidden, ctx_layer], axis=1)
        
        # Stop condition
        if 50257 in output_list[:start]:
            break
            
    final_ids = output_list[:start]
    return tokenizer.decode(final_ids), sum(acceptance_lengths) / sum(block_sizes)


def run():
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    draft = ContinuousDraftModel(config)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-training Continuous Drafter (15 epochs on 10 samples)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
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
                
                ctx_feats = [hidden_target[layer_id] for layer_id in draft.target_layer_ids]
                ctx_feats = mx.concatenate(ctx_feats, axis=-1)
                
                _, _, hidden_future = decoder_forward_with_hidden_states(
                    target, labels[:, :t+1+config.block_size], encoder_hidden, collect_hidden_states=True, return_cross_attention=False
                )
                true_hidden = hidden_future[-1][:, t:t+config.block_size, :]
                
                noise = target.decoder.token_embedding(mx.array([[config.mask_token_id] * config.block_size]))
                pos_ids = mx.arange(t, t + config.block_size, dtype=mx.int32)[None]
                
                loss, grads = loss_and_grad_fn(draft, noise, ctx_feats, audio_summary, pos_ids, true_hidden)
                optimizer.update(draft, grads)
                mx.eval(draft.parameters(), optimizer.state)

    print("Evaluating Hybrid Verification (Easy=Semantic, Hard=Discrete)...")
    
    results = []
    
    for i in range(10, 15):
        if i >= len(ds): break
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        true_text = sample["text"]
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        # 1. Baseline
        res_baseline = generate_discrete(None, target, mel_mx, return_stats=True)
        dec_baseline = tokenizer.decode(res_baseline.output_ids[0].tolist())
        
        # 2. Hybrid Continuous + Discrete
        dec_hybrid, acc_rate = generate_hybrid(draft, target, mel_mx)
        
        results.append({
            "wer_base": wer_metric.compute(predictions=[dec_baseline.lower()], references=[true_text.lower()]),
            "wer_hybrid": wer_metric.compute(predictions=[dec_hybrid.lower()], references=[true_text.lower()]),
            "acc_rate": acc_rate
        })
        print(f"Sample {i}: Acc={acc_rate*100:.1f}% | WER Base: {results[-1]['wer_base']:.3f} | WER Hybrid: {results[-1]['wer_hybrid']:.3f}")
        
    avg_acc = np.mean([r["acc_rate"] for r in results])
    avg_wer_base = np.mean([r["wer_base"] for r in results])
    avg_wer_hybrid = np.mean([r["wer_hybrid"] for r in results])
    
    print("\n" + "="*40)
    print("HYBRID CONTINUOUS + DISCRETE DRAFTING")
    print("="*40)
    print(f"Avg Acceptance Rate:  {avg_acc*100:.1f}%")
    print(f"Avg WER (Baseline):   {avg_wer_base:.3f}")
    print(f"Avg WER (Hybrid):     {avg_wer_hybrid:.3f}")
    print("="*40)

if __name__ == "__main__":
    run()
