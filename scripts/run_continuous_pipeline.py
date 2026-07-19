#!/usr/bin/env python3
"""
run_continuous_pipeline.py

Step 1 of the Continuous Drafter Roadmap:
1. Quickly trains a Continuous Drafter on a few samples to give it basic competence.
2. Plugs it into `generate_continuous.py`.
3. Decodes a held-out test sample using standard discrete verify vs continuous verify.
4. Measures WER, Acceptance Rate, Rollbacks, and Speedup.
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
from whisper_flash_mlx.generate_continuous import whisper_dflash_generate as generate_continuous
from whisper_flash_mlx.generate import whisper_dflash_generate as generate_discrete

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def run():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    d_target = target.dims.n_text_state
    config = WhisperDFlashConfig(
        d_target=d_target,
        d_draft=256,
        num_layers=2,
        vocab_size=target.dims.n_vocab,
        block_size=4,  # Block size for speculative decoding
        target_layer_ids=[1, 2]
    )
    
    draft = ContinuousDraftModel(config)
    
    print("Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Pre-training Continuous Drafter (10 epochs on 10 samples)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    for epoch in range(10):
        for i in range(10):
            sample = ds[i]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            text = sample["text"]
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel)[None]
            text_tokens = tokenizer.encode(text)
            token_ids = mx.array([text_tokens], dtype=mx.int32)
            sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
            labels = mx.concatenate([sot, token_ids], axis=1)
            encoder_hidden = encoder_forward(target, mel_mx)
            audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
            
            for t in range(1, labels.shape[1] - 1, 3):
                input_token = labels[:, :t+1]
                _, _, hidden_target = decoder_forward_with_hidden_states(
                    target, input_token, encoder_hidden, 
                    collect_hidden_states=True, return_cross_attention=False
                )
                true_hidden = hidden_target[-1][:, -1:, :]
                ctx_feats = [hidden_target[layer_id] for layer_id in draft.target_layer_ids]
                ctx_feats = mx.concatenate(ctx_feats, axis=-1)
                
                noise = target.decoder.token_embedding(mx.array([[config.mask_token_id]]))
                pos_ids = mx.array([[input_token.shape[1]]], dtype=mx.int32)
                
                loss, grads = loss_and_grad_fn(draft, noise, ctx_feats, audio_summary, pos_ids, true_hidden)
                optimizer.update(draft, grads)
                mx.eval(draft.parameters(), optimizer.state)

    print("Training complete. Starting Inference Evaluation.")
    
    # Evaluate on next 5 samples
    results = []
    
    for i in range(10, 15):
        if i >= len(ds): break
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        true_text = sample["text"]
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        # 1. Baseline Target-Only Autoregressive
        start = time.perf_counter()
        res_baseline = generate_discrete(None, target, mel_mx, return_stats=True)
        t_base = time.perf_counter() - start
        dec_baseline = tokenizer.decode(res_baseline.output_ids[0].tolist())
        
        # 2. Continuous Speculative Decoding
        start = time.perf_counter()
        res_cont = generate_continuous(draft, target, mel_mx, return_stats=True)
        t_cont = time.perf_counter() - start
        dec_cont = tokenizer.decode(res_cont.output_ids[0].tolist())
        
        acc_rate = sum(res_cont.acceptance_lengths) / sum(res_cont.block_sizes) if sum(res_cont.block_sizes) > 0 else 0.0
        
        results.append({
            "t_base": t_base,
            "t_cont": t_cont,
            "wer_base": wer_metric.compute(predictions=[dec_baseline.lower()], references=[true_text.lower()]),
            "wer_cont": wer_metric.compute(predictions=[dec_cont.lower()], references=[true_text.lower()]),
            "acc_rate": acc_rate,
            "speedup": t_base / t_cont
        })
        
        print(f"Sample {i}: Speedup={t_base/t_cont:.2f}x, Acc={acc_rate*100:.1f}%")
        print(f"  WER Base: {results[-1]['wer_base']:.3f} | WER Cont: {results[-1]['wer_cont']:.3f}")
        
    avg_speedup = np.mean([r["speedup"] for r in results])
    avg_acc = np.mean([r["acc_rate"] for r in results])
    avg_wer_base = np.mean([r["wer_base"] for r in results])
    avg_wer_cont = np.mean([r["wer_cont"] for r in results])
    
    print("\n" + "="*50)
    print("CONTINUOUS SPECULATIVE DECODING (E2E RESULTS)")
    print("="*50)
    print(f"Avg Speedup:          {avg_speedup:.3f}x")
    print(f"Avg Acceptance Rate:  {avg_acc*100:.1f}%")
    print(f"Avg WER (Baseline):   {avg_wer_base:.3f}")
    print(f"Avg WER (Continuous): {avg_wer_cont:.3f}")
    print("="*50)

if __name__ == "__main__":
    run()
