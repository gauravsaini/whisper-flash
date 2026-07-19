#!/usr/bin/env python3
"""
test_semantic_equivalence.py

Step 3 of the Continuous Drafter Roadmap: Semantic Equivalence Mapping.
We evaluate the continuous drafter by checking cases where the discrete tokens 
differ, but the continuous hidden states are highly similar (cosine >= 0.8).
We decode the tokens to see if they are semantically equivalent (synonyms, 
sub-word boundaries, etc.) to validate "semantic verification".
"""

import numpy as np
import mlx.core as mx
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, decoder_forward_with_hidden_states, encoder_forward
from whisper_flash_mlx.draft_model import WhisperDFlashConfig
from experiment_continuous_drafting import ContinuousDraftModel, mse_loss
import mlx.optimizers as optim
import mlx.nn as nn

def run_semantic_mapping():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    d_target = target.dims.n_text_state
    config = WhisperDFlashConfig(
        d_target=d_target,
        d_draft=256,
        num_layers=2,
        vocab_size=target.dims.n_vocab,
        block_size=1,
        target_layer_ids=[1, 2]
    )
    
    draft = ContinuousDraftModel(config)
    
    # We will quickly train it on 10 samples so it's a bit more capable than last time
    print("Loading Dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    print("Quickly training Continuous Drafter...")
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    for epoch in range(15):
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
            
            # Use random time steps to train quickly
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

    print("Training done. Evaluating Semantic Equivalence on held-out samples...")
    
    equivalence_pairs = []
    
    # Evaluate on next 5 samples
    for i in range(10, 15):
        if i >= len(ds): break
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
        
        for t in range(1, labels.shape[1] - 1):
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
            
            # Predict continuous state
            pred_hidden = draft(noise, ctx_feats, audio_summary, pos_ids)
            
            # Compute cosine similarity
            h_true = true_hidden[0, 0, :]
            h_pred = pred_hidden[0, 0, :]
            sim = mx.sum(h_true * h_pred) / (mx.linalg.norm(h_true) * mx.linalg.norm(h_pred) + 1e-9)
            sim_val = sim.item()
            
            # Project to vocabulary to see the discrete tokens
            target_logits = target.decoder.token_embedding.as_linear(true_hidden)[0, -1, :]
            draft_logits = target.decoder.token_embedding.as_linear(pred_hidden)[0, -1, :]
            
            t_tok = mx.argmax(target_logits).item()
            d_tok = mx.argmax(draft_logits).item()
            
            if t_tok != d_tok:
                # Target and Draft disagree on the discrete token
                # But what if their semantic similarity is high?
                if sim_val > 0.6:  # using 0.6 since model is under-trained
                    t_str = tokenizer.decode([t_tok])
                    d_str = tokenizer.decode([d_tok])
                    
                    # Context for printing
                    ctx_str = tokenizer.decode(labels[0, max(0, t-3):t+1].tolist())
                    
                    equivalence_pairs.append({
                        "context": ctx_str,
                        "target_token": t_str,
                        "draft_token": d_str,
                        "similarity": sim_val
                    })

    print("\n" + "="*60)
    print("SEMANTIC EQUIVALENCE MAPPING (Draft != Target, but high Cosine Sim)")
    print("="*60)
    
    # Sort by similarity descending
    equivalence_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    for pair in equivalence_pairs[:15]:
        print(f"Context: '...{pair['context']}'")
        print(f"  Target: '{pair['target_token']}'")
        print(f"  Draft : '{pair['draft_token']}'")
        print(f"  Cosine Similarity: {pair['similarity']:.4f}")
        print("-" * 40)
        
    print(f"\nTotal semantic mismatch pairs found: {len(equivalence_pairs)}")

if __name__ == "__main__":
    run_semantic_mapping()
