#!/usr/bin/env python3
"""
validate_idea_2.py

Validates Idea #2: Multi-granular Verifier.
Hypothesis: Strict token-by-token verification is too rigid. Different tokens can 
represent the same semantic meaning (e.g., synonyms or different BPE chunks).
Validation: We measure the cosine similarity of the target model's hidden states 
when forced to diverge. If the top-2 predicted tokens result in highly similar 
hidden states at the next step, it proves we can use hidden state similarity 
for verification instead of strict token ID matching.
"""

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, decoder_forward_with_hidden_states, encoder_forward

def validate_multi_granular():
    model_name = "mlx-community/whisper-tiny"
    dataset_name = "hf-internal-testing/librispeech_asr_dummy"
    config = "clean"
    split = "validation"
    num_samples = 3

    print("Loading Target Model...")
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    print(f"Loading Dataset {dataset_name}...")
    ds = load_dataset(dataset_name, config, split=split)
    
    similarities = []
    
    for i in range(min(num_samples, len(ds))):
        print(f"\nProcessing sample {i+1}/{num_samples}...")
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
        
        seq_len = labels.shape[1] - 1
        
        for t in tqdm(range(seq_len)):
            input_token = labels[:, :t+1]
            
            # Forward pass to get logits for step t
            logits_target, kv_cache, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, 
                collect_hidden_states=True, return_cross_attention=False
            )
            
            current_target_logits = logits_target[0, -1, :] # (vocab_size)
            
            # Get top 2 tokens
            top_k_indices = mx.argpartition(-current_target_logits, kth=2)[:2]
            top_k_logits = current_target_logits[top_k_indices]
            
            # Sort them so top1 is first
            sorted_idx = mx.argsort(-top_k_logits)
            top1_tok = top_k_indices[sorted_idx[0]].item()
            top2_tok = top_k_indices[sorted_idx[1]].item()
            
            # Forward pass branch 1 (Top 1)
            tok1_input = mx.concatenate([input_token, mx.array([[top1_tok]])], axis=1)
            _, _, hidden1 = decoder_forward_with_hidden_states(
                target, tok1_input, encoder_hidden, 
                collect_hidden_states=True, return_cross_attention=False
            )
            
            # Forward pass branch 2 (Top 2)
            tok2_input = mx.concatenate([input_token, mx.array([[top2_tok]])], axis=1)
            _, _, hidden2 = decoder_forward_with_hidden_states(
                target, tok2_input, encoder_hidden, 
                collect_hidden_states=True, return_cross_attention=False
            )
            
            # Compare final hidden state of the newly added token
            h1 = hidden1[-1][0, -1, :]
            h2 = hidden2[-1][0, -1, :]
            
            sim = mx.sum(h1 * h2) / (mx.linalg.norm(h1) * mx.linalg.norm(h2) + 1e-9)
            similarities.append(sim.item())
                
    # --- Reporting ---
    print("\n" + "="*50)
    print("VALIDATION RESULTS FOR IDEA #2 (Multi-granular Verifier)")
    print("="*50)
    
    mean_sim = np.mean(similarities)
    print(f"Mean Cosine Similarity between Top-1 and Top-2 semantic branches: {mean_sim:.4f}")
    percent_highly_similar = sum(1 for s in similarities if s > 0.95) / len(similarities) * 100
    print(f"Percentage of steps where Top-1 and Top-2 are highly similar (>0.95): {percent_highly_similar:.2f}%")
    print("-> If similarity is high, we can safely use embedding distance to verify drafts instead of exact token ID matching!")

if __name__ == "__main__":
    validate_multi_granular()
