"""Minimal smoke test for whisper-flash-mlx.

Uses whisper-tiny (~39MB) + synthetic audio + 1 training step.
Total download: ~39MB. Peak RAM: ~300MB. Runtime: ~30s.

Tests:
  1. Target model loading + encoder/decoder with hidden states
  2. Draft model forward pass
  3. One training step (loss + gradient)
  4. Speculative generation loop
"""

import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# ── 0. Setup ──────────────────────────────────────────────────────────
print("=" * 60)
print("whisper-flash-mlx  SMOKE TEST  (whisper-tiny, synthetic audio)")
print("=" * 60)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅  {name}")
    else:
        FAIL += 1
        print(f"  ❌  {name}  {detail}")

# ── 1. Load target model (whisper-tiny, ~39 MB) ──────────────────────
print("\n[1/4] Loading whisper-tiny target model...")
t0 = time.perf_counter()

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
    get_token_embedding,
    project_to_logits,
    crop_self_attention_cache,
)

target = load_target_model("mlx-community/whisper-tiny")
mx.eval(target.parameters())
print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

check("Model loaded", target is not None)
check("Encoder exists", hasattr(target, "encoder"))
check("Decoder exists", hasattr(target, "decoder"))

# Dimensions for whisper-tiny
D_MODEL = target.dims.n_text_state     # 384
N_LAYERS = target.dims.n_text_layer    # 4
N_VOCAB = target.dims.n_vocab          # 51865
print(f"  d_model={D_MODEL}, layers={N_LAYERS}, vocab={N_VOCAB}")

# ── 2. Encoder + decoder with hidden states (synthetic audio) ────────
print("\n[2/4] Encoder + decoder with hidden-state extraction...")

# 1s of audio → mel spectrogram, padded to 30s (3000 frames, 80 mels)
# mlx-whisper log_mel_spectrogram returns (frames, n_mels); pad to 30s
from mlx_whisper.audio import log_mel_spectrogram
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
mel = log_mel_spectrogram(audio, padding=16000 * 30 - len(audio))  # (3000, 80)
mel = mel[None]  # (1, 3000, 80)
check("Mel shape", mel.shape == (1, 3000, 80), f"got {mel.shape}")


# Encoder
enc_hidden = encoder_forward(target, mel)
mx.eval(enc_hidden)
check("Encoder output shape", enc_hidden.shape[0] == 1 and enc_hidden.shape[2] == D_MODEL,
      f"got {enc_hidden.shape}")

# Decoder with hidden states
sot = mx.array([[50258]], dtype=mx.int32)  # SOT token
logits, kv_cache, hidden_states = decoder_forward_with_hidden_states(
    target, sot, enc_hidden, kv_cache=None, collect_hidden_states=True,
)
mx.eval(logits)
check("Logits shape", logits.shape == (1, 1, N_VOCAB), f"got {logits.shape}")
check("Hidden states count", len(hidden_states) == N_LAYERS + 1,
      f"expected {N_LAYERS+1}, got {len(hidden_states)}")
check("Hidden state shape", hidden_states[0].shape == (1, 1, D_MODEL),
      f"got {hidden_states[0].shape}")

# KV cache cropping
cropped = crop_self_attention_cache(kv_cache, 1)
check("Cache crop works", len(cropped) == N_LAYERS)

# ── 3. Draft model forward + 1 training step ─────────────────────────
print("\n[3/4] Draft model forward pass + training step...")

from whisper_flash_mlx.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash_mlx.utils import extract_context_feature

# Tiny draft model matching whisper-tiny dimensions
config = WhisperDFlashConfig(
    d_draft=128,          # small
    num_layers=2,         # 2 layers
    num_heads=4,
    ffn_dim=256,
    d_target=D_MODEL,
    num_target_layers=N_LAYERS,
    vocab_size=N_VOCAB,
    max_target_positions=448,
    block_size=4,         # small block
    num_context_layers=2, # only 2 taps (whisper-tiny has 4 layers)
)
draft = WhisperDFlashDraftModel(config)
mx.eval(draft.parameters())

nparams = draft.count_params()
print(f"  Draft model: {nparams:,} parameters")
check("Draft params reasonable", 50_000 < nparams < 5_000_000, f"got {nparams:,}")

# Forward pass
B = config.block_size
noise_emb = get_token_embedding(target, mx.array([[50258] + [0]*(B-1)], dtype=mx.int32))

# Build context from hidden states
ctx_feature = extract_context_feature(hidden_states, config.target_layer_ids)
mx.eval(ctx_feature)
check("Context feature shape",
      ctx_feature.shape == (1, 1, len(config.target_layer_ids) * D_MODEL),
      f"got {ctx_feature.shape}")

audio_summary = mx.mean(enc_hidden, axis=1, keepdims=True)
positions = mx.arange(0, B)[None]

draft_out = draft(
    noise_embedding=noise_emb,
    target_hidden=ctx_feature,
    audio_summary=audio_summary,
    position_ids=positions,
)
mx.eval(draft_out)
check("Draft output shape", draft_out.shape == (1, B, D_MODEL), f"got {draft_out.shape}")

# Project to logits
draft_logits = project_to_logits(target, draft_out)
mx.eval(draft_logits)
check("Draft logits shape", draft_logits.shape == (1, B, N_VOCAB), f"got {draft_logits.shape}")

# --- One training step ---
from whisper_flash_mlx.train import position_weighted_loss

labels = mx.array([[50258, 318, 257, 1353]], dtype=mx.int32)  # dummy labels, B=4
loss_val = position_weighted_loss(draft_logits, labels, gamma=7.0)
mx.eval(loss_val)
check("Loss is finite", float(loss_val) > 0 and np.isfinite(float(loss_val)),
      f"got {float(loss_val)}")

# Gradient step
import mlx.optimizers as optim
optimizer = optim.AdamW(learning_rate=1e-3)

def loss_fn(model):
    h = model(
        noise_embedding=noise_emb,
        target_hidden=ctx_feature,
        audio_summary=audio_summary,
        position_ids=positions,
    )
    logits = project_to_logits(target, h)
    return position_weighted_loss(logits, labels, gamma=7.0)

loss_before = float(loss_val)
loss, grads = nn.value_and_grad(draft, loss_fn)(draft)
optimizer.update(draft, grads)
mx.eval(draft.parameters(), optimizer.state)

loss_after_val = loss_fn(draft)
mx.eval(loss_after_val)
loss_after = float(loss_after_val)
check("Gradient step ran", True)
check("Loss changed after step", abs(loss_after - loss_before) > 1e-6,
      f"before={loss_before:.4f}, after={loss_after:.4f}")

# ── 4. Speculative generation loop ───────────────────────────────────
print("\n[4/4] Speculative generation loop...")

from whisper_flash_mlx.generate import whisper_dflash_generate

result = whisper_dflash_generate(
    draft, target, mel,
    max_new_tokens=10,   # just 10 tokens
    temperature=0.0,
    return_stats=True,
)
mx.eval(result.output_ids)

n_out = result.num_output_tokens
check("Generated tokens", n_out > 0, f"got {n_out}")
check("Output shape valid", result.output_ids.shape[0] == 1)
check("Has acceptance stats", len(result.acceptance_lengths) > 0,
      f"got {result.acceptance_lengths}")
check("TTFT measured", result.time_to_first_token is not None and result.time_to_first_token > 0)

print(f"  Generated {n_out} tokens, acceptance={result.acceptance_lengths}")
print(f"  TTFT={result.time_to_first_token*1000:.0f}ms, "
      f"per-token={result.time_per_output_token*1000:.1f}ms")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS:  {PASS} passed, {FAIL} failed  out of {PASS+FAIL} checks")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
