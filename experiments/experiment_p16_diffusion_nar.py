"""P16: Diffusion on the Continuous Hidden Manifold (NAR ASR Prototype)

Goal: Completely bypass the autoregressive branching ceiling by generating the
entire sequence of hidden states (z) in one shot using a diffusion process.

This script implements a tiny prototype:
1. Extract ground-truth hidden states from Whisper teacher-forcing.
2. Train a tiny conditional diffusion model (DDPM) to overfit and denoise these states.
3. Generate z from pure noise, project via lm_head, and check if it recovers the text.
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

EOS_ID, SOT_ID = 50257, 50258


# ════════════════════════════════════════════════════════════════
# DDPM Utilities
# ════════════════════════════════════════════════════════════════

def get_timestep_embedding(timesteps: mx.array, embedding_dim: int) -> mx.array:
    """Build sinusoidal embeddings for timesteps."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = mx.pad(emb, ((0, 0), (0, 1)))
    return emb


class DenoiseNet(nn.Module):
    """A tiny transformer-based denoiser.
    Predicts noise given x_t, t, and encoder context.
    """
    def __init__(self, d_model: int = 384, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Self attention for the sequence
        self.layers = []
        for _ in range(num_layers):
            layer = {
                "attn": nn.MultiHeadAttention(d_model, n_heads),
                "ln1": nn.LayerNorm(d_model),
                "cross": nn.MultiHeadAttention(d_model, n_heads),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                "ln3": nn.LayerNorm(d_model)
            }
            self.layers.append(layer)
            
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x_t: mx.array, t: mx.array, enc: mx.array) -> mx.array:
        # t is (B,)
        t_emb = get_timestep_embedding(t, x_t.shape[-1])
        t_emb = self.time_mlp(t_emb)[:, None, :]  # (B, 1, D)
        
        h = x_t + t_emb
        
        for layer in self.layers:
            # Self attn
            h_sa = layer["attn"](layer["ln1"](h), layer["ln1"](h), layer["ln1"](h))
            h = h + h_sa
            # Cross attn to encoder
            h_ca = layer["cross"](layer["ln2"](h), enc, enc)
            h = h + h_ca
            # FFN
            h_ffn = layer["mlp"](layer["ln3"](h))
            h = h + h_ffn
            
        return self.out_proj(h)


class DDPM:
    def __init__(self, num_timesteps: int = 100):
        self.num_timesteps = num_timesteps
        # Linear schedule
        self.beta = mx.linspace(1e-4, 0.02, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = mx.cumprod(self.alpha, axis=0)
        
    def q_sample(self, x_0: mx.array, t: mx.array, noise: mx.array) -> mx.array:
        """Forward diffusion process: add noise to x_0 at timestep t."""
        alpha_bar_t = self.alpha_bar[t][:, None, None]
        return mx.sqrt(alpha_bar_t) * x_0 + mx.sqrt(1 - alpha_bar_t) * noise

    def compute_loss(self, model: DenoiseNet, x_0: mx.array, enc: mx.array) -> mx.array:
        B = x_0.shape[0]
        # Sample random t for each item in the batch
        t = mx.random.randint(0, self.num_timesteps, shape=(B,))
        noise = mx.random.normal(x_0.shape)
        
        x_t = self.q_sample(x_0, t, noise)
        pred_noise = model(x_t, t, enc)
        
        loss = mx.mean(mx.square(pred_noise - noise))
        return loss

    def p_sample_loop(self, model: DenoiseNet, shape: tuple, enc: mx.array) -> mx.array:
        """Reverse diffusion process: denoise from pure noise to x_0."""
        x = mx.random.normal(shape)
        B = shape[0]
        
        for i in reversed(range(self.num_timesteps)):
            t = mx.full((B,), i, dtype=mx.int32)
            pred_noise = model(x, t, enc)
            
            alpha_t = self.alpha[t][:, None, None]
            alpha_bar_t = self.alpha_bar[t][:, None, None]
            beta_t = self.beta[t][:, None, None]
            
            # Predict x_{t-1}
            x = (1 / mx.sqrt(alpha_t)) * (x - ((1 - alpha_t) / mx.sqrt(1 - alpha_bar_t)) * pred_noise)
            
            if i > 0:
                noise = mx.random.normal(shape)
                x = x + mx.sqrt(beta_t) * noise
                
        return x


# ════════════════════════════════════════════════════════════════
# Experiment Loop
# ════════════════════════════════════════════════════════════════

def extract_ground_truth_states(model, audio_arr, max_len=30):
    """Run Whisper with teacher forcing to get target hidden states."""
    from mlx_whisper.audio import log_mel_spectrogram
    import librosa
    
    # Process audio
    if len(audio_arr) > 16000 * 30:
        audio_arr = audio_arr[:16000 * 30]
    mel = log_mel_spectrogram(audio_arr, n_mels=80)
    if mel.shape[0] < 3000:
        mel = np.pad(mel, [(0, 3000 - mel.shape[0]), (0, 0)])
    else:
        mel = mel[:3000, :]
    mel = mx.array(mel)[None]  # (1, 3000, 80)
    
    enc = encoder_forward(model, mel)
    
    # Standard decoding to get tokens
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    output_ids = [SOT_ID]
    hidden_states = []
    
    kv_cache = None
    for _ in range(max_len):
        logits, kv_cache, h, _ = decoder_forward_with_hidden_states(
            model, mx.array([[output_ids[-1]]]), enc, kv_cache=kv_cache,
            collect_hidden_states=True, return_cross_attention=True)
        
        # h is a list of hidden states for each layer. The last one is the input to lm_head.
        last_h = h[-1]  # shape (1, 1, D)
        hidden_states.append(last_h)
        
        tok = mx.argmax(logits[:, -1:, :], axis=-1).item()
        output_ids.append(tok)
        if tok == EOS_ID:
            break
            
    # Stack hidden states: (1, T, D)
    gt_z = mx.concatenate(hidden_states, axis=1)
    return enc, gt_z, output_ids


def main():
    model_name = "mlx-community/whisper-tiny-mlx"
    print(f"Loading {model_name}...")
    whisper_model = load_target_model(model_name, dtype=mx.float32)
    
    # 1. Get dummy audio
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    
    print("Extracting ground-truth hidden states (teacher forcing)...")
    enc, gt_z, gt_tokens = extract_ground_truth_states(whisper_model, audio, max_len=50)
    
    T = gt_z.shape[1]
    D = gt_z.shape[2]
    print(f"Target sequence length: {T}, Dim: {D}")
    
    # 2. Setup Diffusion Denoiser
    denoiser = DenoiseNet(d_model=D, n_heads=4, num_layers=2)
    mx.eval(denoiser.parameters())
    ddpm = DDPM(num_timesteps=100)
    
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    def loss_fn(model_params, x_0, enc):
        denoiser.update(model_params)
        return ddpm.compute_loss(denoiser, x_0, enc)
        
    loss_and_grad_fn = nn.value_and_grad(denoiser, loss_fn)
    
    # 3. Overfit on this single sample
    print("\nTraining tiny diffusion model to overfit this sample...")
    epochs = 100
    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(denoiser.parameters(), gt_z, enc)
        optimizer.update(denoiser, grads)
        mx.eval(denoiser.parameters(), optimizer.state)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    # 4. Generate via reverse diffusion
    print("\nGenerating hidden states from pure noise...")
    t0 = time.perf_counter()
    pred_z = ddpm.p_sample_loop(denoiser, shape=(1, T, D), enc=enc)
    mx.eval(pred_z)
    gen_time = time.perf_counter() - t0
    
    # 5. Project to text via Whisper's lm_head
    print("Projecting generated states to text...")
    logits = whisper_model.decoder.token_embedding.as_linear(pred_z)  # (1, T, vocab)
    pred_tokens = mx.argmax(logits, axis=-1)[0].tolist()
    
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=False)
    
    gt_text = tokenizer.decode([t for t in gt_tokens if t < tokenizer.eot])
    pred_text = tokenizer.decode([t for t in pred_tokens if t < tokenizer.eot])
    
    print(f"\n--- Ground Truth (Autoregressive) ---")
    print(f"Tokens: {gt_tokens}")
    print(f"Text:   {gt_text}")
    print(f"\n--- Diffusion Generated (Non-Autoregressive) ---")
    print(f"Tokens: {pred_tokens}")
    print(f"Text:   {pred_text}")
    print(f"Generation Time: {gen_time:.3f}s for {T} tokens")
    
    # Compute token match percentage
    matches = sum(1 for a, b in zip(gt_tokens[1:], pred_tokens) if a == b)
    match_rate = matches / max(1, len(pred_tokens))
    print(f"\nToken Match Rate: {match_rate*100:.1f}%")
    
    out_path = Path("results/p16_diffusion_nar.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P16: Diffusion NAR",
            "loss_final": float(loss.item()),
            "match_rate": match_rate,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "gen_time_s": gen_time
        }, f, indent=2)

if __name__ == "__main__":
    main()
