#!/usr/bin/env python3
"""
Focused KV cache benchmark — verify correctness & measure speedup.
Runs on whisper-tiny (fast) first, then whisper-large-v3-turbo.
"""

import time, sys, heapq, numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
import jiwer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny"
N_TRAIN = 10
N_EVAL = 10
EPOCHS = 20
PCA_RANK = 64 if "tiny" in MODEL else 128
D_DRAFT = 256 if "tiny" in MODEL else 512
ADAPTIVE_THRESHOLD = 0.15

def norm(t):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t))))

def compute_pca_basis(data, pca_rank, d_target):
    all_h = np.concatenate([np.array(d["true_hidden"]) for d in data], axis=0)
    X = all_h.reshape(-1, d_target)
    mean = np.mean(X, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    V = Vt[:pca_rank].T
    return mx.array(mean, dtype=mx.float32), mx.array(V, dtype=mx.float32)

def to_pca(h, mean, V): return (h - mean) @ V
def from_pca(z, mean, V): return z @ V.T + mean

class CorrectionDrafter(nn.Module):
    def __init__(self, d_draft, pca_rank, d_target, d_audio, num_taps=2):
        super().__init__()
        ctx_dim = num_taps * d_target + d_audio + pca_rank
        self.ctx_proj = nn.Linear(ctx_dim, d_draft)
        self.layer1 = nn.Linear(d_draft, d_draft)
        self.layer1_norm = nn.LayerNorm(d_draft)
        self.layer2 = nn.Linear(d_draft, d_draft)
        self.layer2_norm = nn.LayerNorm(d_draft)
        self.output_head = nn.Linear(d_draft, pca_rank, bias=False)

    def __call__(self, z_t, target_hidden, audio_summary):
        ctx = mx.concatenate([target_hidden, audio_summary, z_t], axis=-1)
        x = self.ctx_proj(ctx)
        r1 = x; x = self.layer1_norm(x); x = nn.gelu(self.layer1(x)); x = x + r1
        r2 = x; x = self.layer2_norm(x); x = nn.gelu(self.layer2(x)); x = x + r2
        return self.output_head(x)

def generate_both(target, drafter, tokenizer, mel, pca_mean, pca_V, static_k=1, max_tokens=150):
    """Run adaptive multi-path with and without KV cache, return (text_no_cache, text_with_cache, times)."""
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    
    results = {}
    for use_kv in [False, True]:
        mx.eval()  # clear cache
        tokens = [tokenizer.sot]
        accepted = 0
        attempts = 0
        kv_cache = None
        t0 = time.perf_counter()

        while len(tokens) < max_tokens:
            if use_kv and kv_cache is not None:
                inp = mx.array([[tokens[-1]]], dtype=mx.int32)
            else:
                inp = mx.array([tokens], dtype=mx.int32)

            logits, kv_cache, hidden_all = decoder_forward_with_hidden_states(
                target, inp, enc, kv_cache=kv_cache if use_kv else None,
                collect_hidden_states=True, return_cross_attention=False)

            h_t = hidden_all[-1][:, -1:, :]
            ctx_feats = mx.concatenate([hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
            z_t = to_pca(h_t, pca_mean, pca_V)
            delta_z = drafter(z_t, ctx_feats, audio_summary)
            h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V)
            draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
            draft_token = mx.argmax(draft_logits, axis=-1).item()

            running_rate = accepted / max(attempts, 1)
            k = 1 if static_k is not None else (1 if running_rate >= ADAPTIVE_THRESHOLD else 3)
            attempts += 1

            tgt_logits = logits[0, -1, :]
            tgt_probs = mx.softmax(tgt_logits).tolist()
            topk_idxs = heapq.nlargest(k, range(len(tgt_probs)), key=lambda i: tgt_probs[i])

            if draft_token in topk_idxs:
                tokens.append(draft_token)
                accepted += 1
            else:
                tokens.append(mx.argmax(logits[:, -1, :], axis=-1).item())

            if tokens[-1] == tokenizer.eot:
                break

        dt = time.perf_counter() - t0
        results[use_kv] = (tokenizer.decode(tokens), dt, accepted, len(tokens))

    return results

print(f"{'='*60}")
print(f"KV CACHE BENCHMARK — {MODEL}")
print(f"{'='*60}")

t0 = time.time()
target = load_target_model(MODEL)
tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
d_target = target.dims.n_text_state
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
print(f"Loaded in {time.time()-t0:.1f}s  d_model={d_target}")

# Training
print(f"\nTraining on {N_TRAIN} samples...")
train_data = []
for i in range(N_TRAIN):
    s = ds[i]
    audio = np.array(s["audio"]["array"], dtype=np.float32)
    text_tokens = tokenizer.encode(s["text"])
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
    mel_mx = mx.array(mel[None], dtype=mx.float32)
    labels = mx.concatenate([
        mx.array([[tokenizer.sot]], dtype=mx.int32),
        mx.array([text_tokens], dtype=mx.int32)], axis=1)
    enc_h = encoder_forward(target, mel_mx)
    audio_summ = mx.mean(enc_h, axis=1, keepdims=True)

    for t in range(1, labels.shape[1] - 1):
        inp_tok = labels[:, :t+1]
        _, _, h_all = decoder_forward_with_hidden_states(
            target, inp_tok, enc_h, collect_hidden_states=True, return_cross_attention=False)
        ctx = mx.concatenate([h_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
        h_t = h_all[-1][:, -1:, :]
        train_data.append({
            "h_t": mx.stop_gradient(h_t),
            "ctx": mx.stop_gradient(ctx),
            "audio": audio_summ,
            "true_token": labels[0, t+1],
        })

print(f"  {len(train_data)} datapoints")

# PCA
print(f"PCA (R={PCA_RANK})...", end=" ", flush=True)
pca_mean, pca_V = compute_pca_basis([{"true_hidden": d["h_t"]} for d in train_data], PCA_RANK, d_target)
for d in train_data: d["z_t"] = to_pca(d["h_t"], pca_mean, pca_V)
print("done")

# Train drafter
print(f"Training (E={EPOCHS})...")
drafter = CorrectionDrafter(D_DRAFT, PCA_RANK, d_target, d_target, num_taps=2)
_ = drafter(train_data[0]["z_t"], train_data[0]["ctx"], train_data[0]["audio"])

def loss_fn(m, d):
    delta_z = m(d["z_t"], d["ctx"], d["audio"])
    h_corrected = from_pca(d["z_t"] + delta_z, pca_mean, pca_V)
    draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
    ce = nn.losses.cross_entropy(draft_logits.reshape(1, -1), mx.array([d["true_token"]]), reduction="mean")
    return ce + 0.1 * mx.mean(mx.square(delta_z))

grad_fn = nn.value_and_grad(drafter, loss_fn)
opt = optim.Adam(learning_rate=1e-3)
for epoch in range(EPOCHS):
    loss_sum = 0.0
    for d in train_data:
        l, g = grad_fn(drafter, d)
        opt.update(drafter, g)
        mx.eval(drafter.parameters(), opt.state)
        loss_sum += l.item()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}/{EPOCHS} loss={loss_sum/len(train_data):.4f}")

# EVAL: compare KV vs no-KV
print(f"\n{'='*60}")
print(f"BENCHMARK — {N_EVAL} eval samples (static top-1)")
print(f"{'='*60}")

totals = {"no_cache": {"time": 0, "tok": 0}, "kv_cache": {"time": 0, "tok": 0}}
all_match = True

for i in range(N_TRAIN, N_TRAIN + N_EVAL):
    s = ds[i]
    audio = np.array(s["audio"]["array"], dtype=np.float32)
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
    mel_mx = mx.array(mel[None], dtype=mx.float32)
    ref = s["text"]

    res = generate_both(target, drafter, tokenizer, mel_mx, pca_mean, pca_V, static_k=1)

    text_no, dt_no, acc_no, ntok_no = res[False]
    text_yes, dt_yes, acc_yes, ntok_yes = res[True]

    totals["no_cache"]["time"] += dt_no
    totals["no_cache"]["tok"] += ntok_no
    totals["kv_cache"]["time"] += dt_yes
    totals["kv_cache"]["tok"] += ntok_yes

    w_no = jiwer.wer(norm(ref), norm(text_no)) if ref.strip() else 1.0
    w_yes = jiwer.wer(norm(ref), norm(text_yes)) if ref.strip() else 1.0

    match = "✓" if text_no == text_yes else "✗"
    if text_no != text_yes:
        all_match = False
    speedup = dt_no / dt_yes if dt_yes > 0 else 0
    print(f"  [{i}] no-cache={dt_no:.4f}s  kv-cache={dt_yes:.4f}s  "
          f"{speedup:.1f}x  WER={w_no:.4f}/{w_yes:.4f}  {match}")

# Summary
t_nc = totals["no_cache"]["time"]
t_kv = totals["kv_cache"]["time"]
tok_nc = totals["no_cache"]["tok"]
tok_kv = totals["kv_cache"]["tok"]
print(f"\n{'='*60}")
print(f"RESULTS — {N_EVAL} samples")
print(f"{'='*60}")
print(f"  No KV cache : {t_nc:.3f}s  ({tok_nc/t_nc:.0f} tok/s)")
print(f"  KV cache    : {t_kv:.3f}s  ({tok_kv/t_kv:.0f} tok/s)")
print(f"  Speedup     : {t_nc/t_kv:.1f}x")
print(f"  Correctness : {'ALL MATCH ✓' if all_match else 'MISMATCHES ✗'}")
print(f"  Total wall  : {time.time()-t0:.0f}s")
