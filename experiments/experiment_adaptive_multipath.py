#!/usr/bin/env python3
"""
ADAPTIVE MULTI-PATH GATE — Final validated architecture.

Gate logic:
  Track running acceptance rate.
  If rate < 15% → use top-3 (permissive, finds good alternatives)
  If rate >= 15% → tighten to top-1 (prevents runaway)

Everything else from validated Δz correction architecture:
  - Single-step correction head
  - No decoder skip
  - Decoder always advances context
"""

import time, heapq, numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
import jiwer

# Config
MODEL_NAME = None  # set in run()
PCA_RANK = None
D_DRAFT = None
D_TARGET = None
EPOCHS = 30
N_TRAIN = 20
N_EVAL = 20
LAMBDA_CE = 0.1
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


def to_pca(h, mean, V):
    return (h - mean) @ V


def from_pca(z, mean, V):
    return z @ V.T + mean


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


def generate_greedy(target, tokenizer, mel, max_tokens=150):
    enc = encoder_forward(target, mel)
    tokens = [tokenizer.sot]
    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        logits, _, _ = decoder_forward_with_hidden_states(
            target, inp, enc, collect_hidden_states=False, return_cross_attention=False)
        ntok = mx.argmax(logits[:, -1, :], axis=-1).item()
        tokens.append(ntok)
        if ntok == tokenizer.eot:
            break
    return tokenizer.decode(tokens)


def generate_adaptive(target, model, tokenizer, mel, max_tokens=150,
                      pca_mean=None, pca_V=None, static_k=None, use_kv_cache=True):
    """
    Adaptive multi-path gate with optional KV caching.
    If static_k is set → use that fixed K (baseline mode).
    If static_k is None → adaptive: top-3 if accept<15%, else top-1.
    """
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    accepted = 0
    attempts = 0  # total positions considered for gate

    # KV cache state (list of layer caches or None)
    kv_cache = None

    while len(tokens) < max_tokens:
        # KV cache: first call passes all tokens, subsequent calls pass only the last
        if use_kv_cache and kv_cache is not None:
            inp = mx.array([[tokens[-1]]], dtype=mx.int32)  # single new token
        else:
            inp = mx.array([tokens], dtype=mx.int32)        # full sequence

        logits, kv_cache, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc, kv_cache=kv_cache if use_kv_cache else None,
            collect_hidden_states=True, return_cross_attention=False)

        h_t = hidden_all[-1][:, -1:, :]
        ctx_feats = mx.concatenate([hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)

        z_t = to_pca(h_t, pca_mean, pca_V)
        delta_z = model(z_t, ctx_feats, audio_summary)
        z_corrected = z_t + delta_z
        h_corrected = from_pca(z_corrected, pca_mean, pca_V)

        draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
        draft_token = mx.argmax(draft_logits, axis=-1).item()

        # Determine K: adaptive or static
        if static_k is not None:
            k = static_k
        else:
            running_rate = accepted / max(attempts, 1)
            k = 1 if running_rate >= ADAPTIVE_THRESHOLD else 3

        attempts += 1

        tgt_logits = logits[0, -1, :]
        tgt_probs = mx.softmax(tgt_logits).tolist()
        topk_idxs = heapq.nlargest(k, range(len(tgt_probs)),
                                    key=lambda i: tgt_probs[i])

        if draft_token in topk_idxs:
            tokens.append(draft_token)
            accepted += 1
        else:
            greedy_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            tokens.append(greedy_token)

        if tokens[-1] == tokenizer.eot:
            break

    return tokenizer.decode(tokens), accepted, len(tokens)


def run_model(model_name, pca_rank, d_draft, n_train=20, n_eval=20):
    """Run full experiment for a given model."""
    global PCA_RANK, D_DRAFT, D_TARGET, N_TRAIN, N_EVAL
    PCA_RANK = pca_rank
    D_DRAFT = d_draft

    print(f"\n{'='*70}")
    print(f"ADAPTIVE MULTI-PATH GATE on {model_name}")
    print(f"  PCA R={pca_rank}, Draft dim={d_draft}")
    print(f"  Adaptive: top-3 if accept<{ADAPTIVE_THRESHOLD*100:.0f}%, else top-1")
    print(f"  {'='*70}")

    t0 = time.time()
    print(f"Loading model...", end=" ", flush=True)
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    D_TARGET = target.dims.n_text_state
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print(f"done ({time.time()-t0:.1f}s) [hidden={D_TARGET}]")

    # Training data
    print(f"Extracting training data ({n_train} samples)...")
    train_data = []
    for i in range(n_train):
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
            true_tok = labels[0, t+1]
            train_data.append({
                "h_t": mx.stop_gradient(h_t),
                "ctx": mx.stop_gradient(ctx),
                "audio": audio_summ,
                "true_token": true_tok,
            })
    print(f"  {len(train_data)} datapoints")

    # PCA
    print(f"PCA basis (R={pca_rank})...", end=" ", flush=True)
    pca_mean, pca_V = compute_pca_basis(
        [{"true_hidden": d["h_t"]} for d in train_data], pca_rank, D_TARGET)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"], pca_mean, pca_V)
    print("done")

    # Train
    print(f"Training (N={len(train_data)}, E={EPOCHS})...")
    model = CorrectionDrafter(d_draft, pca_rank, D_TARGET, D_TARGET, num_taps=2)
    _ = model(train_data[0]["z_t"], train_data[0]["ctx"], train_data[0]["audio"])

    def loss_fn(m, d):
        delta_z = m(d["z_t"], d["ctx"], d["audio"])
        z_corrected = d["z_t"] + delta_z
        h_corrected = from_pca(z_corrected, pca_mean, pca_V)
        draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
        ce = nn.losses.cross_entropy(draft_logits.reshape(1, -1),
                                      mx.array([d["true_token"]]), reduction="mean")
        return ce + LAMBDA_CE * mx.mean(mx.square(delta_z))

    grad_fn = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=1e-3)
    train_t0 = time.time()

    for epoch in range(EPOCHS):
        loss_sum = 0.0
        for d in train_data:
            l, g = grad_fn(model, d)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            loss_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:02d}/{EPOCHS} loss={loss_sum/len(train_data):.4f} "
                  f"({time.time()-train_t0:.0f}s)")

    # Greedy baseline
    print(f"\n  --- Greedy ({n_eval} eval) ---")
    gw = []
    for i in range(n_train, n_train + n_eval):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        text = generate_greedy(target, tokenizer, mel_mx)
        w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
        gw.append(w)
    gw_mean = np.mean(gw)
    print(f"  Mean WER={gw_mean:.4f}")

    # Test configs: static top-1, static top-3, adaptive
    results = {}
    for label, static_k in [("Static Top-1", 1), ("Static Top-3", 3), ("Adaptive", None)]:
        print(f"\n  --- {label} ---")
        ws, acs, toks = [], [], []
        for i in range(n_train, n_train + n_eval):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)
            text, acc, ntok = generate_adaptive(
                target, model, tokenizer, mel_mx,
                pca_mean=pca_mean, pca_V=pca_V, static_k=static_k)
            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w); acs.append(acc); toks.append(ntok)
            print(f"    [{i}] WER={w:.4f} accept={acc}/{ntok} ({acc/max(ntok,1)*100:.0f}%)")
        mw = np.mean(ws)
        ma = sum(acs)/max(1, sum(toks))*100
        results[label] = (mw, ma)
        delta = mw - gw_mean
        print(f"    -> WER={mw:.4f} ({delta:+.4f}) Accept={ma:.1f}%")

    # Summary
    print(f"\n  {'='*40}")
    print(f"  {model_name} — FINAL")
    print(f"  {'='*40}")
    print(f"  Greedy:     WER={gw_mean:.4f}")
    for label, (mw, ma) in results.items():
        delta = mw - gw_mean
        print(f"  {label:15s}: WER={mw:.4f} ({delta:+.4f}) Accept={ma:.1f}%")
    print()

    # KV cache benchmark (whisper-large only, first n_eval//2 samples)
    if "large" in model_name:
        n_kv = min(n_eval // 2, 10)
        print(f"\n  --- KV Cache Speed Benchmark ({n_kv} samples) ---")
        times_nocache, times_cache = [], []
        for i in range(n_train, n_train + n_kv):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)

            t0 = time.time()
            text_a, _, _ = generate_adaptive(
                target, model, tokenizer, mel_mx,
                pca_mean=pca_mean, pca_V=pca_V, static_k=1, use_kv_cache=False)
            dt_a = time.time() - t0

            t0 = time.time()
            text_b, _, _ = generate_adaptive(
                target, model, tokenizer, mel_mx,
                pca_mean=pca_mean, pca_V=pca_V, static_k=1, use_kv_cache=True)
            dt_b = time.time() - t0

            times_nocache.append(dt_a)
            times_cache.append(dt_b)
            match = "✓" if text_a == text_b else "✗"
            print(f"    [{i}] no-cache={dt_a:.3f}s  kv-cache={dt_b:.3f}s  "
                  f"speedup={dt_a/dt_b:.1f}x  {match}")
        mean_a = np.mean(times_nocache)
        mean_b = np.mean(times_cache)
        print(f"    -> Mean: no-cache={mean_a:.3f}s  kv-cache={mean_b:.3f}s  "
              f"speedup={mean_a/mean_b:.1f}x")
        print()

    return gw_mean, results


def run():
    # Whisper-tiny
    run_model(
        model_name="mlx-community/whisper-tiny",
        pca_rank=64,
        d_draft=256,
        n_train=20,
        n_eval=20,
    )

    # Whisper-large-v3-turbo
    run_model(
        model_name="mlx-community/whisper-large-v3-turbo",
        pca_rank=128,
        d_draft=512,
        n_train=10,
        n_eval=10,
    )


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"\nTotal wall: {time.time()-t_start:.0f}s")
