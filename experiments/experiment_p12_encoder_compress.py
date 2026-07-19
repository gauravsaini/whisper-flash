"""P12: Learnable Encoder Compression — Stride-4 with Learned Projection.

E2 proved stride-2 avg_pool is lossless but stride-4 avg_pool degrades WER
by +0.012 because simple pooling destroys fine-grained temporal boundaries.

This experiment tests whether a tiny learnable projection layer
(1D conv or linear) can compress 1500→375 frames while preserving
the temporal boundaries that avg_pool destroys.

Architecture:
  1. Encode audio → 1500 frames (standard Whisper encoder)
  2. Apply 1D convolution with stride-4 → 375 frames
  3. Train the conv layer with frozen encoder+decoder to minimize WER
  4. At inference: 4× fewer encoder frames → 4× fewer cross-attention keys

This is composable with Q8 + KV cache.
"""

from __future__ import annotations

import json
import time
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
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ══════════════════════════════════════════════════════════════════
# Learned Encoder Compressor
# ══════════════════════════════════════════════════════════════════

class LearnedEncoderCompressor(nn.Module):
    """Learnable 1D convolution to compress encoder frames.
    
    Takes (B, T, D) encoder output and produces (B, T//stride, D)
    via a learned convolutional layer that preserves temporal boundaries.
    """
    
    def __init__(self, d_model: int, stride: int = 4, kernel_size: int = 7):
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        # Depthwise 1D convolution: each feature dimension is convolved independently
        # This learns a weighted combination of adjacent frames
        self.proj = nn.Linear(d_model * kernel_size, d_model)
        self.kernel_size = kernel_size
        # Initialize close to avg_pool (uniform weights)
        # This ensures the model starts from the lossless stride-2 baseline
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, D) encoder output
        Returns:
            (B, T//stride, D) compressed output
        """
        B, T, D = x.shape
        stride = self.stride
        k = self.kernel_size
        pad = k // 2
        
        # Pad the time dimension
        if pad > 0:
            x_padded = mx.pad(x, [(0, 0), (pad, pad), (0, 0)])
        else:
            x_padded = x
        
        # Extract patches with stride
        T_out = T // stride
        # For each output position, gather kernel_size frames centered at stride*i
        indices = []
        for i in range(T_out):
            center = i * stride + pad  # center position in padded tensor
            start = center - pad
            indices.append(start)
        
        # Gather patches: (B, T_out, kernel_size, D)
        patches = []
        for idx in indices:
            patch = x_padded[:, idx:idx + k, :]  # (B, k, D)
            patches.append(patch)
        
        patches = mx.stack(patches, axis=1)  # (B, T_out, k, D)
        patches = patches.reshape(B, T_out, k * D)  # (B, T_out, k*D)
        
        # Project back to D
        out = self.proj(patches)  # (B, T_out, D)
        return out


class SimpleLinearCompressor(nn.Module):
    """Simple linear compression: reshape + project."""
    
    def __init__(self, d_model: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(d_model * stride, d_model)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        stride = self.stride
        T_trim = (T // stride) * stride
        x = x[:, :T_trim, :]
        x = x.reshape(B, T_trim // stride, stride * D)
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════

def train_compressor(
    model, compressor, train_samples: list, *,
    epochs: int = 20,
    lr: float = 1e-3,
) -> list[float]:
    """Train the compressor to match full-encoder decoding quality.
    
    Loss = MSE between full-encoder hidden states and compressed-encoder hidden states.
    """
    optimizer = optim.Adam(learning_rate=lr)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_tokens = 0
        
        for mel, ref, idx in train_samples:
            # Full encoder
            enc_full = encoder_forward(model, mel)
            mx.eval(enc_full)
            
            # Compressed encoder
            enc_compressed = compressor(enc_full)
            mx.eval(enc_compressed)
            
            # Greedy decode with full encoder to get target token sequence
            dec = mx.array([[SOT_ID]], dtype=mx.int32)
            logits_full, kv_full, hs_full = decoder_forward_with_hidden_states(
                model, dec, enc_full, kv_cache=None, collect_hidden_states=True)
            first_full = sample(logits_full[:, -1:, :], 0.0)
            mx.eval(first_full)
            target_ids = [SOT_ID, first_full.item()]
            
            while len(target_ids) < 100:
                inp = mx.array([[target_ids[-1]]], dtype=mx.int32)
                logits_full, kv_full, hs_full = decoder_forward_with_hidden_states(
                    model, inp, enc_full, kv_cache=kv_full, collect_hidden_states=True)
                tok = sample(logits_full[:, -1:, :], 0.0)
                mx.eval(tok)
                tid = tok.item()
                target_ids.append(tid)
                if tid == EOS_ID:
                    break
            
            # Now compute loss: run target tokens through decoder with compressed encoder
            # Compare logits
            target_tokens = mx.array([target_ids], dtype=mx.int32)
            
            def loss_fn(compressor_params):
                # Reconstruct compressed encoder
                enc_comp = compressor(enc_full)
                # Run decoder with compressed encoder
                logits_comp, _, _ = decoder_forward_with_hidden_states(
                    model, target_tokens, enc_comp, kv_cache=None, collect_hidden_states=False)
                # Run decoder with full encoder for target logits
                logits_target, _, _ = decoder_forward_with_hidden_states(
                    model, target_tokens, enc_full, kv_cache=None, collect_hidden_states=False)
                # MSE loss on logits (we want compressed to match full)
                loss = mx.mean((logits_comp - logits_target) ** 2)
                return loss
            
            loss, grads = nn.value_and_grad(compressor, loss_fn)(compressor.parameters())
            optimizer.update(compressor, grads)
            mx.eval(compressor.parameters(), optimizer.state)
            
            epoch_loss += loss.item() * len(target_ids)
            n_tokens += len(target_ids)
        
        avg_loss = epoch_loss / max(n_tokens, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.6f}")
    
    return losses


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def generate_with_compression(
    model, mel: mx.array, compressor,
    use_kv_cache: bool = True,
    max_tokens: int = 448,
) -> tuple[list[int], float]:
    """Greedy decode with compressed encoder."""
    t0 = time.perf_counter()
    
    enc = encoder_forward(model, mel)
    mx.eval(enc)
    
    enc = compressor(enc)
    mx.eval(enc)
    
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]
    
    while len(output_ids) < max_tokens:
        last_tok = output_ids[-1]
        if last_tok == EOS_ID:
            break
        inp = mx.array([[last_tok]], dtype=mx.int32)
        if use_kv_cache:
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        else:
            full_seq = mx.array([output_ids], dtype=mx.int32)
            logits, _, _ = decoder_forward_with_hidden_states(
                model, full_seq, enc, kv_cache=None, collect_hidden_states=False)
        tok = sample(logits[:, -1:, :], 0.0)
        mx.eval(tok)
        output_ids.append(tok.item())
    
    return output_ids, time.perf_counter() - t0


def load_dataset(n_samples: int = 20):
    from datasets import load_dataset as hf_load
    from mlx_whisper.audio import log_mel_spectrogram

    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation")
    samples = []
    for i in range(min(n_samples, len(ds))):
        audio = ds[i]["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        mel = log_mel_spectrogram(arr, n_mels=80, padding=16000 * 30 - len(arr))
        mel = mx.array(mel)[None]
        ref = ds[i].get("text", ds[i].get("transcription", ""))
        samples.append((mel, ref, i))
    return samples


def decode_tokens(model, token_ids: list[int]) -> str:
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)
    text_tokens = [t for t in token_ids[1:] if t < tokenizer.eot]
    return tokenizer.decode(text_tokens).strip()


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import wer
    return wer([r.strip().lower() for r in refs], [h.strip().lower() for h in hyps])


def avg_pool_compress(enc: mx.array, stride: int) -> mx.array:
    """Simple avg_pool baseline for comparison."""
    B, T, D = enc.shape
    T_trim = (T // stride) * stride
    return mx.mean(enc[:, :T_trim, :].reshape(B, T_trim // stride, stride, D), axis=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P12: Learnable Encoder Compression")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--n-train", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  P12: Learnable Encoder Compression")
    print(f"  Model: {args.model}")
    print(f"  Stride: {args.stride}, Epochs: {args.epochs}")
    print(f"{'#'*60}")

    model = load_target_model(args.model, dtype=mx.float16)
    d_model = model.dims.n_audio_state

    total_samples = args.n_train + args.n_eval
    all_samples = load_dataset(total_samples)
    train_samples = all_samples[:args.n_train]
    eval_samples = all_samples[args.n_train:args.n_train + args.n_eval]

    # Phase 1: Baselines (no compression, avg_pool stride-2, avg_pool stride-4)
    configs = [
        ("Baseline (1500)", 1),
        ("AvgPool S2 (750)", 2),
        (f"AvgPool S{args.stride} ({1500//args.stride})", args.stride),
    ]
    
    baseline_results = []
    for name, stride in configs:
        print(f"\n--- {name} ---")
        refs, hyps = [], []
        total_time = 0
        for mel, ref, idx in eval_samples:
            enc = encoder_forward(model, mel)
            mx.eval(enc)
            if stride > 1:
                enc_c = avg_pool_compress(enc, stride)
                mx.eval(enc_c)
            else:
                enc_c = enc
            
            t0 = time.perf_counter()
            dec = mx.array([[SOT_ID]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, dec, enc_c, kv_cache=None, collect_hidden_states=False)
            first = sample(logits[:, -1:, :], 0.0)
            mx.eval(first)
            ids = [SOT_ID, first.item()]
            while len(ids) < 448:
                inp = mx.array([[ids[-1]]], dtype=mx.int32)
                logits, kv_cache, _ = decoder_forward_with_hidden_states(
                    model, inp, enc_c, kv_cache=kv_cache, collect_hidden_states=False)
                tok = sample(logits[:, -1:, :], 0.0)
                mx.eval(tok)
                tid = tok.item()
                ids.append(tid)
                if tid == EOS_ID:
                    break
            wall = time.perf_counter() - t0
            text = decode_tokens(model, ids)
            refs.append(ref)
            hyps.append(text)
            total_time += wall
            print(f"  Sample {idx}: {wall:.3f}s | {text[:60]}")
        
        wer_val = compute_wer(refs, hyps)
        baseline_results.append({
            "name": name, "stride": stride, "wer": round(wer_val, 6),
            "total_time_s": round(total_time, 4),
        })
        print(f"  WER: {wer_val:.4f}, Time: {total_time:.3f}s")

    # Phase 2: Train learned compressor
    print(f"\n--- Training Learned S{args.stride} Compressor ---")
    compressor = SimpleLinearCompressor(d_model, stride=args.stride)
    mx.eval(compressor.parameters())
    
    losses = train_compressor(model, compressor, train_samples, epochs=args.epochs)

    # Phase 3: Evaluate learned compressor
    print(f"\n--- Evaluating Learned S{args.stride} ---")
    refs, hyps = [], []
    total_time = 0
    for mel, ref, idx in eval_samples:
        ids, wall = generate_with_compression(model, mel, compressor)
        text = decode_tokens(model, ids)
        refs.append(ref)
        hyps.append(text)
        total_time += wall
        print(f"  Sample {idx}: {wall:.3f}s | {text[:60]}")
    
    learned_wer = compute_wer(refs, hyps)
    learned_result = {
        "name": f"Learned S{args.stride} ({1500//args.stride})",
        "stride": args.stride,
        "wer": round(learned_wer, 6),
        "total_time_s": round(total_time, 4),
    }
    print(f"  WER: {learned_wer:.4f}, Time: {total_time:.3f}s")

    # Summary
    all_results = baseline_results + [learned_result]
    print(f"\n\n{'='*70}")
    print(f"  RESULTS — P12 Encoder Compression ({args.model})")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Frames':>6} {'WER':>8} {'ΔWER':>8} {'Time(s)':>8}")
    print("-" * 70)
    base_wer = baseline_results[0]["wer"]
    for r in all_results:
        frames = 1500 // r["stride"]
        delta = r["wer"] - base_wer
        print(f"{r['name']:<25} {frames:>6} {r['wer']:>8.4f} {delta:>+8.4f} {r['total_time_s']:>8.3f}")
    print("=" * 70)

    # Save
    out_path = args.output or f"results/p12_encoder_compress_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "P12: Learnable Encoder Compression",
            "model": args.model,
            "stride": args.stride,
            "epochs": args.epochs,
            "training_losses": losses,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
