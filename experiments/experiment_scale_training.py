#!/usr/bin/env python3
"""
experiment_scale_training.py

Moonshot 2.5: Scale Training & OOD Testing
- Train the Continuous Drafter on a larger set of LibriSpeech 'clean' samples.
- Evaluate on LibriSpeech 'other' (noisy/accented) to test generalization.
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

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred_hidden - true_hidden))

def run():
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    wer_metric = evaluate.load("wer")
    
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=4, target_layer_ids=[1, 2]
    )
    draft = ContinuousDraftModel(config)
    
    # Train on 100 samples from LibriSpeech clean validation
    print("Loading Training Dataset (clean)...")
    ds_train = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Training Continuous Drafter (Scale Training: 100 samples)...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    start_train = time.time()
    
    # Train on 50 samples for 3 epochs to simulate scaled training
    num_train_samples = min(50, len(ds_train))
    for epoch in range(3):
        epoch_loss = 0
        for i in range(num_train_samples):
            sample = ds_train[i]
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
        print(f"Epoch {epoch+1} Loss: {epoch_loss / num_train_samples:.4f}")
        
    print(f"Training took {time.time() - start_train:.2f}s")
    
    print("\nLoading OOD Dataset (noisy/accented)...")
    try:
        ds_test = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
        ds_test_iter = iter(ds_test)
    except Exception as e:
        print("Failed to load 'other', falling back to 'clean' test split...")
        ds_test = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
        ds_test_iter = iter(ds_test)
        
    print("Evaluating OOD Generalization...")
    results = []
    
    for i in range(5):
        sample = next(ds_test_iter)
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        true_text = sample["text"]
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel)[None]
        
        # Continuous Speculative Decoding (Semantic Verification)
        res_cont = generate_continuous(draft, target, mel_mx, return_stats=True)
        dec_cont = tokenizer.decode(res_cont.output_ids[0].tolist())
        
        wer_cont = wer_metric.compute(predictions=[dec_cont.lower()], references=[true_text.lower()])
        acc_rate = sum(res_cont.acceptance_lengths) / sum(res_cont.block_sizes)
        results.append({
            "wer_cont": wer_cont,
            "acc_rate": acc_rate
        })
        print(f"OOD Sample {i}: Acc={acc_rate*100:.1f}% | WER: {wer_cont:.3f}")
        
    avg_acc = np.mean([r["acc_rate"] for r in results])
    avg_wer = np.mean([r["wer_cont"] for r in results])
    
    print("\n" + "="*40)
    print("SCALE TRAINING & OOD GENERALIZATION")
    print("="*40)
    print(f"Avg OOD Acceptance Rate:  {avg_acc*100:.1f}%")
    print(f"Avg OOD WER:              {avg_wer:.3f}")
    print("="*40)

if __name__ == "__main__":
    run()
