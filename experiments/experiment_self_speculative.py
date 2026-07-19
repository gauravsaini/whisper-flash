#!/usr/bin/env python3
"""
experiment_self_speculative.py

Moonshot 4: Self-Speculative Continuous Drafting (Layer Skip)
- Train the LayerSkipDraftModel on 10 samples for 15 epochs.
- Evaluate on 5 samples using continuous speculative decoding.
- Compare speed and WER against baseline.
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
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, LayerSkipDraftModel
from whisper_flash_mlx.generate_continuous import whisper_dflash_generate as generate_continuous

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def run():
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    # Tap only Layer 1. Whisper-tiny has 4 layers. We skip layers 2,3,4.
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1]
    )
    draft = LayerSkipDraftModel(config)
    
    print("Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-training Layer-Skip Drafter (15 epochs on 10 samples)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    start_train = time.time()
    for epoch in range(15):
        epoch_loss = 0
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
                epoch_loss += loss.item()
    print(f"Training took {time.time() - start_train:.2f}s")
    
    print("\nEvaluating Layer-Skip Drafting...")
    results = []
    
    for i in range(10, 15):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        true_text = sample["text"]
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        # Baseline decoding
        start_base = time.time()
        res_base = generate_continuous(None, target, mel_mx, return_stats=True)
        t_base = time.time() - start_base
        dec_base = tokenizer.decode(res_base.output_ids[0].tolist())
        
        # Layer-Skip drafting
        start_ls = time.time()
        res_ls = generate_continuous(draft, target, mel_mx, return_stats=True)
        t_ls = time.time() - start_ls
        dec_ls = tokenizer.decode(res_ls.output_ids[0].tolist())
        
        wer_base = wer_metric.compute(predictions=[dec_base.lower()], references=[true_text.lower()])
        wer_ls = wer_metric.compute(predictions=[dec_ls.lower()], references=[true_text.lower()])
        acc_rate = sum(res_ls.acceptance_lengths) / sum(res_ls.block_sizes)
        
        results.append({
            "wer_base": wer_base,
            "wer_ls": wer_ls,
            "acc_rate": acc_rate,
            "speedup": t_base / t_ls
        })
        print(f"Sample {i}: Speedup={t_base/t_ls:.2f}x | Acc={acc_rate*100:.1f}% | WER Base: {wer_base:.3f} | WER LS: {wer_ls:.3f}")
        
    avg_speedup = np.mean([r["speedup"] for r in results])
    avg_acc = np.mean([r["acc_rate"] for r in results])
    avg_wer_base = np.mean([r["wer_base"] for r in results])
    avg_wer_ls = np.mean([r["wer_ls"] for r in results])
    
    print("\n" + "="*40)
    print("LAYER-SKIP CONTINUOUS DRAFTING")
    print("="*40)
    print(f"Avg Speedup:          {avg_speedup:.2f}x")
    print(f"Avg Acceptance Rate:  {avg_acc*100:.1f}%")
    print(f"Avg WER (Baseline):   {avg_wer_base:.3f}")
    print(f"Avg WER (Layer-Skip): {avg_wer_ls:.3f}")
    print("="*40)

if __name__ == "__main__":
    run()
