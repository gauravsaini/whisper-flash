#!/usr/bin/env python3
"""
experiment_continuous_drafting.py

Moonshot #2: Continuous Hidden-State Speculation
Instead of training the drafter to predict discrete tokens via Cross-Entropy, 
we train it to directly predict the continuous hidden state of the target model 
using Mean Squared Error (MSE). We then verify predictions using Cosine Similarity.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer

from whisper_flash_mlx.target_model import load_target_model, decoder_forward_with_hidden_states, encoder_forward
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel, DFlashDecoderLayer
from whisper_flash_mlx.utils import extract_context_feature

# --- 1. Define Continuous Drafter ---
class ContinuousDraftModel(nn.Module):
    def __init__(self, config: WhisperDFlashConfig):
        super().__init__()
        self.config = config
        
        # We don't need input embedding projection because we input continuous noise/states
        self.input_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        
        num_taps = len(config.target_layer_ids)
        self.fc = nn.Linear(num_taps * config.d_target, config.d_draft, bias=False)
        self.hidden_norm = nn.LayerNorm(config.d_draft)
        self.audio_proj = nn.Linear(config.d_target, config.d_draft, bias=False)
        self.pos_embed = nn.Embedding(config.max_target_positions, config.d_draft)
        
        self.layers = [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        self.norm = nn.LayerNorm(config.d_draft)
        
        # KEY DIFFERENCE: Instead of outputting to vocab, we output to target hidden dimension
        self.continuous_head = nn.Linear(config.d_draft, config.d_target, bias=False)
        
        self.target_layer_ids = config.target_layer_ids

    def __call__(self, noise, target_hidden, audio_summary, position_ids):
        x = self.input_proj(noise) + self.pos_embed(position_ids)
        ctx = self.hidden_norm(self.fc(target_hidden))
        audio_ctx = self.audio_proj(audio_summary)
        
        for layer in self.layers:
            x = layer(x, ctx, audio_ctx, mask=None)
            
        x = self.norm(x)
        # Directly predict the target hidden state
        predicted_hidden = self.continuous_head(x) 
        return predicted_hidden

def mse_loss(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred_hidden = model(noise, target_hidden, audio_summary, position_ids)
    # Mean Squared Error between predicted and true continuous state
    return mx.mean(mx.square(pred_hidden - true_hidden))

def run_experiment():
    print("Loading Target Model (mlx-community/whisper-tiny)...")
    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    
    # Tiny model target hidden dim is 384
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
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = nn.value_and_grad(draft, mse_loss)
    
    print("Loading Dataset (hf-internal-testing/librispeech_asr_dummy)...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # We will collect training data from the first 5 samples, and test on the 6th
    num_train = 5
    epochs = 10
    
    train_data = []
    
    print("Extracting continuous hidden states for training...")
    for i in range(num_train):
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
        
        seq_len = labels.shape[1] - 1
        for t in range(seq_len):
            input_token = labels[:, :t+1]
            
            # Target forward pass
            _, _, hidden_target = decoder_forward_with_hidden_states(
                target, input_token, encoder_hidden, 
                collect_hidden_states=True, return_cross_attention=False
            )
            
            # True next hidden state (from top layer of target model at step t)
            true_hidden = hidden_target[-1][:, -1:, :] # (1, 1, d_target)
            
            # Context features from tapped layers
            ctx_feats = extract_context_feature(hidden_target, draft.target_layer_ids)
            
            # Noise (mask token embedding)
            noise = target.decoder.token_embedding(mx.array([[config.mask_token_id]]))
            pos_ids = mx.array([[input_token.shape[1]]], dtype=mx.int32)
            
            train_data.append({
                "noise": noise,
                "ctx": ctx_feats,
                "audio": audio_summary,
                "pos": pos_ids,
                "true_hidden": true_hidden
            })

    print(f"Collected {len(train_data)} training examples. Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for data in train_data:
            loss, grads = loss_and_grad_fn(
                draft, data["noise"], data["ctx"], data["audio"], data["pos"], data["true_hidden"]
            )
            optimizer.update(draft, grads)
            mx.eval(draft.parameters(), optimizer.state)
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {epoch_loss/len(train_data):.4f}")

    print("\nEvaluating on Test Sample...")
    test_sample = ds[5]
    audio = np.array(test_sample["audio"]["array"], dtype=np.float32)
    text = test_sample["text"]
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
    mel_mx = mx.array(mel)[None]
    text_tokens = tokenizer.encode(text)
    token_ids = mx.array([text_tokens], dtype=mx.int32)
    sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
    labels = mx.concatenate([sot, token_ids], axis=1)
    
    encoder_hidden = encoder_forward(target, mel_mx)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)
    
    similarities = []
    
    seq_len = labels.shape[1] - 1
    for t in range(seq_len):
        input_token = labels[:, :t+1]
        _, _, hidden_target = decoder_forward_with_hidden_states(
            target, input_token, encoder_hidden, 
            collect_hidden_states=True, return_cross_attention=False
        )
        
        true_hidden = hidden_target[-1][:, -1:, :]
        ctx_feats = extract_context_feature(hidden_target, draft.target_layer_ids)
        noise = target.decoder.token_embedding(mx.array([[config.mask_token_id]]))
        pos_ids = mx.array([[input_token.shape[1]]], dtype=mx.int32)
        
        # Predict continuous state
        pred_hidden = draft(noise, ctx_feats, audio_summary, pos_ids)
        
        # Cosine Similarity
        h_true = true_hidden[0, 0, :]
        h_pred = pred_hidden[0, 0, :]
        sim = mx.sum(h_true * h_pred) / (mx.linalg.norm(h_true) * mx.linalg.norm(h_pred) + 1e-9)
        similarities.append(sim.item())
        
    mean_sim = np.mean(similarities)
    print("\n" + "="*50)
    print("RESULTS: CONTINUOUS HIDDEN-STATE SPECULATION")
    print("="*50)
    print(f"Mean Cosine Similarity (Test Set): {mean_sim:.4f}")
    print(f"Tokens with > 0.90 similarity: {sum(1 for s in similarities if s > 0.90) / len(similarities) * 100:.1f}%")
    
if __name__ == "__main__":
    run_experiment()
