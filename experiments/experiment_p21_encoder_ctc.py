"""P21: Encoder-Only CTC Decoding (Approximation)

Goal: Test if the causal decoder can be completely bypassed by training a linear
classifier directly on the 1500 encoder frames. Since MLX doesn't have CTC loss,
we extract a hard alignment using Whisper's cross-attention weights, and train
a frame-wise cross-entropy classifier.

If the encoder frames are linearly separable into the target tokens, it proves
an encoder-only non-autoregressive architecture (like Wav2Vec2) is viable for Whisper.
"""

import math
import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
BLANK_ID = 51864  # Just use the last token in vocab as blank, or 0. We'll use vocab_size as blank.

def extract_alignment_and_targets(model, audio_arr):
    """Run Whisper to get tokens and cross-attention alignment."""
    from mlx_whisper.audio import log_mel_spectrogram
    
    mel = log_mel_spectrogram(audio_arr, n_mels=80)
    if mel.shape[0] < 3000:
        mel = np.pad(mel, [(0, 3000 - mel.shape[0]), (0, 0)])
    else:
        mel = mel[:3000, :]
    mel = mx.array(mel)[None]  # (1, 3000, 80)
    
    enc = encoder_forward(model, mel)  # (1, 1500, D)
    
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    output_ids = []
    cross_attns = []
    
    kv_cache = None
    for i in range(50):
        # We need return_cross_attention=True
        logits, kv_cache, _, cross_qk = decoder_forward_with_hidden_states(
            model, mx.array([[SOT_ID if i==0 else output_ids[-1]]]), enc, kv_cache=kv_cache,
            return_cross_attention=True
        )
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        
        # cross_qk is list of layers: each is (B, n_heads, 1, n_frames)
        # Average across layers and heads for this step
        step_attn = mx.mean(mx.concatenate([mx.mean(layer_qk, axis=1, keepdims=True) for layer_qk in cross_qk], axis=1), axis=1) # (1, 1, 1500)
        cross_attns.append(step_attn)
        
        if tok == EOS_ID:
            break
            
    # cross_attns: (1, n_tokens, 1500)
    attns = mx.concatenate(cross_attns, axis=1)[0]  # (n_tokens, 1500)
    
    # Argmax over frames for each token
    # To ensure monotonicity and avoid collisions, we could do DP, but argmax is a fine approximation for a quick test.
    frame_indices = mx.argmax(attns, axis=1).tolist()
    
    # Build 1500-length target sequence
    vocab_size = model.decoder.token_embedding.weight.shape[0]
    blank = vocab_size # We'll add 1 to vocab size for blank
    targets = np.full(1500, blank, dtype=np.int32)
    
    for tok, frame_idx in zip(output_ids, frame_indices):
        targets[frame_idx] = tok
        
    return enc[0], mx.array(targets), output_ids, vocab_size + 1

def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    print("Extracting encoder frames and cross-attention alignment...")
    enc_frames, targets, target_tokens, num_classes = extract_alignment_and_targets(whisper_model, audio)
    
    print(f"Encoder shape: {enc_frames.shape}")
    print(f"Total non-blank targets: {sum(targets.tolist() != num_classes - 1 for targets in targets)}")
    
    # Train linear classifier: D -> num_classes
    D = enc_frames.shape[1]
    linear_head = nn.Linear(D, num_classes)
    mx.eval(linear_head.parameters())
    
    optimizer = optim.Adam(learning_rate=1e-2)
    
    def loss_fn(model_params, x, y):
        linear_head.update(model_params)
        logits = linear_head(x)  # (1500, num_classes)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        return loss
        
    loss_and_grad_fn = nn.value_and_grad(linear_head, loss_fn)
    
    print("\nTraining linear CTC approximation head...")
    epochs = 200
    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(linear_head.parameters(), enc_frames, targets)
        optimizer.update(linear_head, grads)
        mx.eval(linear_head.parameters(), optimizer.state)
        
        if (epoch + 1) % 40 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    # Evaluate
    logits = linear_head(enc_frames)
    preds = mx.argmax(logits, axis=-1).tolist()
    
    # CTC decode (collapse repeats, remove blanks)
    blank = num_classes - 1
    decoded = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
        
    tokenizer = get_tokenizer(multilingual=False)
    
    gt_text = tokenizer.decode([t for t in target_tokens if t < tokenizer.eot])
    pred_text = tokenizer.decode([t for t in decoded if t < tokenizer.eot])
    
    print(f"\n--- Ground Truth (Autoregressive Decoder) ---")
    print(f"Tokens: {target_tokens}")
    print(f"Text:   {gt_text}")
    print(f"\n--- Encoder Linear Head (CTC Approx) ---")
    print(f"Tokens: {decoded}")
    print(f"Text:   {pred_text}")
    
    matches = sum(1 for a, b in zip(target_tokens, decoded) if a == b)
    match_rate = matches / max(1, len(decoded), len(target_tokens))
    print(f"\nToken Match Rate: {match_rate*100:.1f}%")
    
    out_path = Path("results/p21_encoder_ctc.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P21: Encoder CTC Approx",
            "loss_final": float(loss.item()),
            "match_rate": match_rate,
            "gt_text": gt_text,
            "pred_text": pred_text
        }, f, indent=2)

if __name__ == "__main__":
    main()
