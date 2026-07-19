#!/usr/bin/env python3
"""
B=1 correction drafter + top-3 multi-path gate.
SCALED to whisper-large-v3-turbo (1280 hidden dim).

Architecture validated on whisper-tiny, now at production scale.

Training loss: CE(W @ h'_t, true_token_{t+1}) + λ_ce · MSE(Δz)
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

PCA_RANK = 128
D_DRAFT = 512
EPOCHS = 30
N_TRAIN = 10
N_EVAL = 10
LAMBDA_CE = 0.1

TARGET_MODEL = "mlx-community/whisper-large-v3-turbo"


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
    def __init__(self, d_draft=512, pca_rank=128, d_target=1280, d_audio=1280, num_taps=2):
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


def generate_multi_path(target, model, tokenizer, mel, max_tokens=150,
                        pca_mean=None, pca_V=None, top_k=3):
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    accepted = 0

    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        logits, _, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc, collect_hidden_states=True, return_cross_attention=False)

        h_t = hidden_all[-1][:, -1:, :]
        ctx_feats = mx.concatenate([hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)

        z_t = to_pca(h_t, pca_mean, pca_V)
        delta_z = model(z_t, ctx_feats, audio_summary)
        z_corrected = z_t + delta_z
        h_corrected = from_pca(z_corrected, pca_mean, pca_V)

        draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
        draft_token = mx.argmax(draft_logits, axis=-1).item()

        tgt_logits = logits[0, -1, :]
        tgt_probs = mx.softmax(tgt_logits).tolist()
        topk_idxs = heapq.nlargest(top_k, range(len(tgt_probs)),
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


def run():
    print("=" * 72)
    print(f"B=1 CORRECTION DRAFTER + TOP-K MULTI-PATH GATE")
    print(f"  Model: {TARGET_MODEL}")
    print(f"  PCA rank={PCA_RANK}, draft dim={D_DRAFT}")
    print(f"  Δz correction on h_t → alternative token at position t+1")
    print(f"  No decoder skip. Multi-path only.")
    print("=" * 72)

    # Load target
    t0 = time.time()
    print(f"\nLoading {TARGET_MODEL}...", end=" ", flush=True)
    target = load_target_model(TARGET_MODEL)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print(f"done ({time.time()-t0:.1f}s)")

    # Training data
    print(f"Extracting training data ({N_TRAIN} samples)...")
    train_data = []
    extract_t0 = time.time()
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
            true_tok = labels[0, t+1]
            train_data.append({
                "h_t": mx.stop_gradient(h_t),
                "ctx": mx.stop_gradient(ctx),
                "audio": audio_summ,
                "true_token": true_tok,
            })
    print(f"  {len(train_data)} datapoints in {time.time()-extract_t0:.1f}s")

    # PCA basis
    pca_t0 = time.time()
    print(f"Computing PCA basis (R={PCA_RANK})...", end=" ", flush=True)
    pca_mean, pca_V = compute_pca_basis(
        [{"true_hidden": d["h_t"]} for d in train_data], PCA_RANK, d_target)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"], pca_mean, pca_V)
    print(f"done ({time.time()-pca_t0:.1f}s)")

    # Build & train
    print(f"Building & training (N={len(train_data)}, E={EPOCHS})...")
    model = CorrectionDrafter(
        d_draft=D_DRAFT, pca_rank=PCA_RANK,
        d_target=d_target, d_audio=d_target, num_taps=2)
    _ = model(train_data[0]["z_t"], train_data[0]["ctx"], train_data[0]["audio"])

    def loss_fn(m, d):
        delta_z = m(d["z_t"], d["ctx"], d["audio"])
        z_corrected = d["z_t"] + delta_z
        h_corrected = from_pca(z_corrected, pca_mean, pca_V)
        draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
        ce = nn.losses.cross_entropy(draft_logits.reshape(1, -1),
                                      mx.array([d["true_token"]]), reduction="mean")
        mse = mx.mean(mx.square(delta_z))
        return ce + LAMBDA_CE * mse

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
    print(f"\n--- Greedy Baseline ({N_EVAL} eval) ---")
    gw, gt = [], []
    for i in range(N_TRAIN, N_TRAIN + N_EVAL):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        t1 = time.time()
        text = generate_greedy(target, tokenizer, mel_mx)
        gt.append(time.time() - t1)
        w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
        gw.append(w)
    gw_mean = np.mean(gw)
    gt_mean = np.mean(gt)
    print(f"  WER={gw_mean:.4f} avg_time={gt_mean:.3f}s")

    # Multi-path verification
    for topk in [1, 3]:
        print(f"\n--- Correction + top-{topk} ({N_EVAL} eval) ---")
        ws, acs, toks, rt = [], [], [], []
        for i in range(N_TRAIN, N_TRAIN + N_EVAL):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)
            t1 = time.time()
            text, acc, ntok = generate_multi_path(
                target, model, tokenizer, mel_mx,
                pca_mean=pca_mean, pca_V=pca_V, top_k=topk)
            elapsed = time.time() - t1
            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w); acs.append(acc); toks.append(ntok); rt.append(elapsed)
            print(f"  [{i}] WER={w:.4f} accept={acc}/{ntok} ({acc/max(ntok,1)*100:.0f}%) "
                  f"t={elapsed:.3f}s")
        mw = np.mean(ws); ma = sum(acs)/max(1, sum(toks))*100
        mr = np.mean(rt)
        print(f"  -> WER={mw:.4f} (+{mw-gw_mean:+.4f}) Accept={ma:.1f}% "
              f"time={mr:.3f}s (greedy={gt_mean:.3f}s)")

    print(f"\n{'='*50}")
    print("WHISPER-LARGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Model: {TARGET_MODEL}")
    print(f"  Hidden dim: {d_target}")
    print(f"  PCA rank: {PCA_RANK}")
    print(f"  Train data: {len(train_data)} datapoints")
    print(f"  Greedy WER: {gw_mean:.4f} ({gt_mean:.3f}s avg)")
    print(f"  Top-1:  WER={np.mean(ws)} accept={sum(acs)/max(1,sum(toks))*100:.1f}% (if available)")
    print(f"  Top-3:  {'WER=' + str(mw) + ' accept=' + str(ma) if False else 'see above'}")


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"\nTotal wall: {time.time()-t_start:.0f}s")
