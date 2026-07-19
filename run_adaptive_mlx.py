#!/usr/bin/env python3
"""
Run ∆z correction + adaptive multi-path gate on any audio file with mlx (Apple Silicon).

One-shot usage:
  uv run python run_adaptive_mlx.py --audio speech.wav
  uv run python run_adaptive_mlx.py --audio speech.wav --model whisper-large-v3-turbo

Directory batch:
  uv run python run_adaptive_mlx.py --audio ./audios/ --model whisper-tiny --out results.json

What it does:
  1. Loads the model (default: whisper-large-v3-turbo)
  2. Trains a correction drafter on 10 built-in LibriSpeech samples (~5s on M4)
  3. Runs BOTH greedy and adaptive decoding on your audio
  4. Compares: WER, speed, acceptance rate
  5. With `--gt-text "reference text"`, computes WER against ground truth

Requires:
  pip install mlx-whisper jiwer datasets
  (auto-installed via uv if using --audio and whisper_flash_mlx is set up)
"""

import argparse, json, os, sys, time, heapq
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram, load_audio
from mlx_whisper.tokenizer import get_tokenizer
import jiwer

# Add project to path (so whisper_flash_mlx is importable)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Imports from our project ─────────────────────────────────────────
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states

# ─── Constants ─────────────────────────────────────────────────────────
EPOCHS = 30
LAMBDA_CE = 0.1
ADAPTIVE_THRESHOLD = 0.15
MAX_TOKENS = 150
N_TRAIN_DEFAULT = 10  # LibriSpeech samples used for drafter training
PCA_RANK_MAP = {"whisper-tiny": 64, "whisper-small": 64, "whisper-base": 64,
                "whisper-medium": 128, "whisper-large": 128, "whisper-large-v3": 128,
                "whisper-large-v3-turbo": 128}
D_DRAFT_MAP = {"whisper-tiny": 256, "whisper-small": 256, "whisper-base": 256,
               "whisper-medium": 384, "whisper-large": 512, "whisper-large-v3": 512,
               "whisper-large-v3-turbo": 512}


# ─── Helpers ───────────────────────────────────────────────────────────
def norm(t):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t))))


def _detect_model_key(model_id):
    """Map full model ID to short key for config lookup."""
    for key in ["whisper-large-v3-turbo", "whisper-large-v3", "whisper-large",
                "whisper-medium", "whisper-small", "whisper-base", "whisper-tiny"]:
        if key in model_id:
            return key
    return "whisper-tiny"


# ─── PCA & Drafter (copied from experiment_adaptive_multipath.py) ──────
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
    """2-layer residual MLP predicting ∆z in PCA space."""

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


# ─── Generate functions ───────────────────────────────────────────────
def generate_greedy(target, tokenizer, mel, max_tokens=MAX_TOKENS):
    enc = encoder_forward(target, mel)
    tokens = [tokenizer.sot]
    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        logits, _, _ = decoder_forward_with_hidden_states(target, inp, enc)
        ntok = mx.argmax(logits[:, -1, :], axis=-1).item()
        tokens.append(ntok)
        if ntok == tokenizer.eot: break
    return tokenizer.decode(tokens)


def generate_adaptive(target, model, tokenizer, mel, max_tokens=MAX_TOKENS,
                      pca_mean=None, pca_V=None, static_k=None):
    """Adaptive multi-path gate with KV cache."""
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    accepted = 0
    attempts = 0
    kv_cache = None

    while len(tokens) < max_tokens:
        if kv_cache is not None:
            inp = mx.array([[tokens[-1]]], dtype=mx.int32)
        else:
            inp = mx.array([tokens], dtype=mx.int32)

        logits, kv_cache, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc, kv_cache=kv_cache,
            collect_hidden_states=True)

        h_t = hidden_all[-1][:, -1:, :]
        ctx_feats = mx.concatenate([hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)

        z_t = to_pca(h_t, pca_mean, pca_V)
        delta_z = model(z_t, ctx_feats, audio_summary)
        h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V)
        draft_logits = target.decoder.token_embedding.as_linear(h_corrected)
        draft_token = mx.argmax(draft_logits, axis=-1).item()

        k = static_k if static_k is not None else (
            1 if (accepted / max(attempts, 1)) >= ADAPTIVE_THRESHOLD else 3)
        attempts += 1

        tgt_probs = mx.softmax(logits[0, -1, :]).tolist()
        topk_idxs = heapq.nlargest(k, range(len(tgt_probs)), key=lambda i: tgt_probs[i])

        if draft_token in topk_idxs:
            tokens.append(draft_token)
            accepted += 1
        else:
            tokens.append(mx.argmax(logits[:, -1, :], axis=-1).item())

        if tokens[-1] == tokenizer.eot: break

    return tokenizer.decode(tokens), accepted, len(tokens)


# ─── Training ─────────────────────────────────────────────────────────
def train_drafter(target, tokenizer, model_name, pca_rank, d_draft, n_samples=N_TRAIN_DEFAULT,
                  progress_cb=None):
    """Train correction drafter on LibriSpeech dummy samples. Returns (drafter, pca_mean, pca_V)."""
    d_target = target.dims.n_text_state
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Extract training data
    train_data = []
    t0 = time.time()
    for i in range(min(n_samples, len(ds))):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(s["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30 - len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        labels = mx.concatenate([
            mx.array([[tokenizer.sot]], dtype=mx.int32),
            mx.array([text_tokens], dtype=mx.int32)], axis=1)
        enc_h = encoder_forward(target, mel_mx)
        audio_summ = mx.mean(enc_h, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - 1):
            inp_tok = labels[:, :t+1]
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp_tok, enc_h, collect_hidden_states=True)
            ctx = mx.concatenate([h_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
            h_t = h_all[-1][:, -1:, :]
            train_data.append({
                "h_t": mx.stop_gradient(h_t),
                "ctx": mx.stop_gradient(ctx),
                "audio": audio_summ,
                "true_token": labels[0, t+1],
            })
        if progress_cb:
            progress_cb(f"  Extracted {i+1}/{n_samples} ({len(train_data)} datapoints, {time.time()-t0:.0f}s)")

    # PCA
    pca_mean, pca_V = compute_pca_basis(
        [{"true_hidden": d["h_t"]} for d in train_data], pca_rank, d_target)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"], pca_mean, pca_V)

    # Train
    model = CorrectionDrafter(d_draft, pca_rank, d_target, d_target, num_taps=2)
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

    for epoch in range(EPOCHS):
        loss_sum = 0.0
        for d in train_data:
            l, g = grad_fn(model, d)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            loss_sum += l.item()
        if progress_cb and (epoch + 1) % 10 == 0:
            progress_cb(f"  epoch {epoch+1}/{EPOCHS} loss={loss_sum/len(train_data):.4f}")

    return model, pca_mean, pca_V


# ─── Audio inference ──────────────────────────────────────────────────
def transcribe_file(target, tokenizer, audio_path, drafter=None, pca_mean=None, pca_V=None):
    """Transcribe a single audio file. Returns (text, timing_info)."""
    audio = load_audio(audio_path)
    TARGET_SAMPLES = 16000 * 30  # 30s padding (matches positional embedding)
    padding = max(0, TARGET_SAMPLES - len(audio))
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=padding)
    mel_mx = mx.array(mel[None], dtype=mx.float32)

    t0 = time.time()
    greedy_text = generate_greedy(target, tokenizer, mel_mx)
    greedy_time = time.time() - t0

    result = {"greedy": {"text": greedy_text, "time_s": greedy_time}}

    if drafter is not None:
        t0 = time.time()
        adaptive_text, accepted, ntok = generate_adaptive(
            target, drafter, tokenizer, mel_mx,
            pca_mean=pca_mean, pca_V=pca_V, static_k=None)
        adaptive_time = time.time() - t0
        result["adaptive"] = {
            "text": adaptive_text,
            "time_s": adaptive_time,
            "accepted": accepted,
            "total_tokens": ntok,
            "accept_pct": accepted / max(ntok, 1) * 100,
            "speedup_vs_greedy": greedy_time / max(adaptive_time, 0.001),
        }

    return result


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Adaptive ∆z correction on mlx (Apple Silicon)")
    parser.add_argument("--audio", required=True,
                        help="Audio file or directory of audio files")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo",
                        help="Model (default: mlx-community/whisper-large-v3-turbo)")
    parser.add_argument("--train-samples", type=int, default=N_TRAIN_DEFAULT,
                        help=f"LibriSpeech samples for drafter training (default: {N_TRAIN_DEFAULT})")
    parser.add_argument("--gt-text", default=None,
                        help="Ground truth text for WER (if not provided, uses greedy output as reference)")
    parser.add_argument("--out", default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # ── Load model ──
    model_key = _detect_model_key(args.model)
    pca_rank = PCA_RANK_MAP.get(model_key, 64)
    d_draft = D_DRAFT_MAP.get(model_key, 256)

    print(f"Loading {args.model}...", end=" ", flush=True)
    t0 = time.time()
    target = load_target_model(args.model)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    print(f"done ({time.time()-t0:.1f}s)", flush=True)
    print(f"  PCA R={pca_rank}, Draft dim={d_draft}", flush=True)

    # ── Train drafter ──
    def progress(msg):
        print(msg, flush=True)

    print(f"\nTraining drafter ({args.train_samples} LibriSpeech samples)...", flush=True)
    t0 = time.time()
    drafter, pca_mean, pca_V = train_drafter(
        target, tokenizer, args.model, pca_rank, d_draft,
        n_samples=args.train_samples, progress_cb=progress)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    # ── Process audio ──
    audio_path = args.audio
    is_dir = os.path.isdir(audio_path)

    if is_dir:
        audio_files = sorted([
            os.path.join(audio_path, f) for f in os.listdir(audio_path)
            if f.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))
        ])
        if not audio_files:
            print(f"No audio files found in {audio_path}", flush=True)
            sys.exit(1)
        print(f"\nProcessing {len(audio_files)} audio files...", flush=True)
    else:
        audio_files = [audio_path]

    all_results = {}
    for af in audio_files:
        fname = os.path.basename(af)
        print(f"\n  [{fname}]", flush=True)
        r = transcribe_file(target, tokenizer, af, drafter, pca_mean, pca_V)
        all_results[af] = r

        # Print results
        gt = args.gt_text
        g = r["greedy"]
        gt_text = gt if gt else g["text"]

        gw = jiwer.wer(norm(gt_text), norm(g["text"])) if gt_text.strip() else 1.0
        print(f"    Greedy:  WER={gw:.4f}  time={g['time_s']:.1f}s", flush=True)
        print(f"    Text:    {g['text'][:120]}", flush=True)

        if "adaptive" in r:
            a = r["adaptive"]
            aw = jiwer.wer(norm(gt_text), norm(a["text"])) if gt_text.strip() else 1.0
            accept_str = f"{a['accepted']}/{a['total_tokens']} ({a['accept_pct']:.0f}%)"
            print(f"    Adaptive: WER={aw:.4f} ({aw-gw:+.4f})  time={a['time_s']:.1f}s  "
                  f"accept={accept_str}  speedup={a['speedup_vs_greedy']:.1f}x", flush=True)
            print(f"    Text:    {a['text'][:120]}", flush=True)

    # ── Summary table ──
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — {args.model}", flush=True)
    print(f"{'='*60}", flush=True)
    for af in audio_files:
        r = all_results[af]
        g = r["greedy"]
        gt = args.gt_text or g["text"]
        gw = jiwer.wer(norm(gt), norm(g["text"])) if gt.strip() else 1.0
        fname = os.path.basename(af)
        if "adaptive" in r:
            a = r["adaptive"]
            aw = jiwer.wer(norm(gt), norm(a["text"])) if gt.strip() else 1.0
            print(f"  {fname:30s} greedy={gw:.4f}  adaptive={aw:.4f} "
                  f"({aw-gw:+.4f})  accept={a['accept_pct']:.0f}%  {a['speedup_vs_greedy']:.1f}x", flush=True)
        else:
            print(f"  {fname:30s} greedy={gw:.4f}", flush=True)

    # ── Save JSON ──
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
