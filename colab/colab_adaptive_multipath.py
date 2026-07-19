#!/usr/bin/env python3
"""
Adaptive Δz Correction + Multi-Path Gate — Final Validation on Colab.
Runs whisper-large-v3-turbo via HuggingFace transformers on CUDA.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], capture_output=True)

import sys, time, heapq, numpy as np, torch
import torch.nn as nn
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import jiwer

# ─── Config ───────────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3-turbo"
PCA_RANK = 128
D_DRAFT = 512
EPOCHS = 30
N_TRAIN = 20
N_EVAL = 30
LAMBDA_CE = 0.1
ADAPTIVE_THRESHOLD = 0.15
MAX_TOKENS = 150
SAMPLE_RATE = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})", flush=True)


# ─── Helpers ───────────────────────────────────────────────────────────
def norm(t):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t))))


def compute_pca_basis_torch(hidden_states_list, pca_rank):
    """Fit PCA on list of [1, 1, d] tensors, return (mean, V) as torch tensors."""
    all_h = torch.cat([h.flatten(0, 1) for h in hidden_states_list], dim=0).float()  # [N, d]
    mean = all_h.mean(dim=0, keepdim=True)
    X = all_h - mean
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    V = Vt[:pca_rank].T.contiguous()
    return mean.to(device), V.to(device)


def to_pca(h, mean, V):
    return (h.float() - mean) @ V


def from_pca(z, mean, V):
    return z @ V.T + mean


# ─── MLP Correction Head (PyTorch) ────────────────────────────────────
class CorrectionDrafter(nn.Module):
    """2-layer MLP predicting Δz in PCA space. Same architecture as mlx version."""

    def __init__(self, d_draft=512, pca_rank=128, d_target=1280, d_audio=1280, num_taps=2):
        super().__init__()
        ctx_dim = num_taps * d_target + d_audio + pca_rank
        self.ctx_proj = nn.Linear(ctx_dim, d_draft)
        self.layer1 = nn.Linear(d_draft, d_draft)
        self.layer1_norm = nn.LayerNorm(d_draft)
        self.layer2 = nn.Linear(d_draft, d_draft)
        self.layer2_norm = nn.LayerNorm(d_draft)
        self.output_head = nn.Linear(d_draft, pca_rank, bias=False)

    def forward(self, z_t, target_hidden, audio_summary):
        ctx = torch.cat([target_hidden, audio_summary, z_t], dim=-1)
        x = self.ctx_proj(ctx)
        r1 = x; x = self.layer1_norm(x); x = torch.nn.functional.gelu(self.layer1(x)); x = x + r1
        r2 = x; x = self.layer2_norm(x); x = torch.nn.functional.gelu(self.layer2(x)); x = x + r2
        return self.output_head(x)


# ─── Inference ─────────────────────────────────────────────────────────
def proc_audio(processor, audio_array, model):
    """Process audio and return input_features with correct dtype."""
    dtype = next(model.parameters()).dtype
    x = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    return x.to(device, dtype=dtype)


def generate_greedy(model, processor, audio_array, max_tokens=MAX_TOKENS):
    """Standard greedy decoding returning decoded text."""
    input_features = proc_audio(processor, audio_array, model)
    with torch.no_grad():
        tokens = model.generate(input_features, max_length=max_tokens, num_beams=1)
    return processor.decode(tokens[0], skip_special_tokens=True)


def get_tokens_and_logits(model, processor, audio_array, max_tokens=MAX_TOKENS, skip_special=True):
    """Returns (decoded_text, token_id_list, logits_list, hidden_states_list)."""
    input_features = proc_audio(processor, audio_array, model)
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_features).last_hidden_state
    tokens = [model.config.decoder_start_token_id]
    logits_list = []
    hidden_states_list = []

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens], device=device)
            out = model.model.decoder(
                input_ids=inp,
                encoder_hidden_states=encoder_outputs,
                output_hidden_states=True,
            )
            logits = model.proj_out(out.last_hidden_state)  # [1, seq, vocab]
            logits_list.append(logits[:, -1:, :])  # last position only

            # Collect h_t (last-layer hidden at last position)
            # Whisper decoder returns (hidden_states_after_embed, h_layer0, h_layer1, ..., h_output)
            # last_hidden_state IS the output of the last decoder layer
            h_t = out.last_hidden_state[:, -1:, :]  # [1, 1, d]
            hidden_states_list.append(h_t)

            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == model.config.eos_token_id:
                break

    text = processor.decode(tokens, skip_special_tokens=skip_special)
    return text, tokens, logits_list, hidden_states_list


def generate_adaptive(target, processor, audio_array, drafter, pca_mean, pca_V,
                      static_k=None, max_tokens=MAX_TOKENS):
    """
    Adaptive multi-path verification.
    - static_k: if set, use fixed K. If None, adaptive (top-3→top-1 at 15%).
    Returns (decoded_text, accepted_count, total_tokens).
    """
    input_features = proc_audio(processor, audio_array, target)
    model_dtype = next(target.parameters()).dtype

    with torch.no_grad():
        enc_out = target.model.encoder(input_features).last_hidden_state

    audio_summary = enc_out.mean(dim=1, keepdim=True)  # [1, 1, d_audio]
    tokens = [target.config.decoder_start_token_id]
    accepted = 0
    attempts = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens], device=device)
            out = target.model.decoder(
                input_ids=inp,
                encoder_hidden_states=enc_out,
                output_hidden_states=True,
            )

            # Collect hidden states from tapped layers
            # WhisperDecoder: hidden_states[0] = embeds, hidden_states[1..L] = after each decoder layer
            L = len(out.hidden_states) - 1  # number of decoder layers
            h1 = out.hidden_states[1][:, -1:, :] if L >= 1 else out.last_hidden_state[:, -1:, :]
            h2 = out.hidden_states[2][:, -1:, :] if L >= 2 else out.last_hidden_state[:, -1:, :]

            h_t = out.last_hidden_state[:, -1:, :]  # last layer, last position
            ctx_feats = torch.cat([h1, h2], dim=-1)
            logits = target.proj_out(out.last_hidden_state)

            z_t = to_pca(h_t, pca_mean, pca_V)
            delta_z = drafter(z_t, ctx_feats, audio_summary)
            z_corrected = z_t + delta_z
            h_corrected = from_pca(z_corrected, pca_mean, pca_V)

            draft_logits = target.proj_out(h_corrected.to(model_dtype))
            draft_token = draft_logits[0, -1].argmax().item()

            # Gate
            attempts += 1
            k = static_k if static_k is not None else (1 if accepted / max(attempts, 1) >= ADAPTIVE_THRESHOLD else 3)

            tgt_logits = logits[0, -1, :]
            tgt_probs = torch.softmax(tgt_logits, dim=-1)
            topk_vals, topk_idxs = torch.topk(tgt_probs, k)

            if draft_token in topk_idxs.tolist():
                tokens.append(draft_token)
                accepted += 1
            else:
                greedy_token = logits[0, -1].argmax().item()
                tokens.append(greedy_token)

            if tokens[-1] == target.config.eos_token_id:
                break

    text = processor.decode(tokens, skip_special_tokens=True)
    return text, accepted, len(tokens)


# ─── Training Data Extraction ─────────────────────────────────────────
def extract_training_data(target, processor, ds, n_samples, device):
    """Extract (h_t, context features, audio summary, true token) pairs."""
    data = []
    t0 = time.time()

    for i in range(n_samples):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        input_features = proc_audio(processor, audio, target)

        # Get text tokens including start token
        text = s["text"]
        labels = processor(text=text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        labels = torch.cat([torch.tensor([target.config.decoder_start_token_id]), labels]).tolist()

        with torch.no_grad():
            enc_out = target.model.encoder(input_features).last_hidden_state

        audio_summ = enc_out.mean(dim=1, keepdim=True)

        for t_idx in range(1, len(labels) - 1):
            inp_tok = torch.tensor([labels[:t_idx+1]], device=device)
            dec_out = target.model.decoder(
                input_ids=inp_tok,
                encoder_hidden_states=enc_out,
                output_hidden_states=True,
            )
            L = len(dec_out.hidden_states) - 1
            h1 = dec_out.hidden_states[1][:, -1:, :] if L >= 1 else dec_out.last_hidden_state[:, -1:, :]
            h2 = dec_out.hidden_states[2][:, -1:, :] if L >= 2 else dec_out.last_hidden_state[:, -1:, :]
            ctx = torch.cat([h1, h2], dim=-1)
            h_t = dec_out.last_hidden_state[:, -1:, :]
            true_tok = labels[t_idx+1]

            data.append({
                "h_t": h_t.detach().cpu(),
                "ctx": ctx.detach().cpu(),
                "audio": audio_summ.detach().cpu(),
                "true_token": true_tok,
            })

        if (i + 1) % 5 == 0:
            print(f"    Extracted {i+1}/{n_samples} samples ({len(data)} datapoints, "
                  f"{time.time()-t0:.0f}s)", flush=True)

    return data


# ─── Main ──────────────────────────────────────────────────────────────
def run():
    print("=" * 72, flush=True)
    print("ADAPTIVE MULTI-PATH GATE — Colab Final Validation")
    print(f"  Model: {MODEL_ID}")
    print(f"  PCA R={PCA_RANK}, Draft dim={D_DRAFT}")
    print(f"  Adaptive: top-3 if accept<{ADAPTIVE_THRESHOLD*100:.0f}%, else top-1")
    print(f"  Train: {N_TRAIN} samples  Eval: {N_EVAL} samples")
    print("=" * 72, flush=True)

    # ── Load model & data ──
    t0 = time.time()
    print(f"\nLoading {MODEL_ID}...", end=" ", flush=True)
    target = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    target.eval()
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    print(f"done ({time.time()-t0:.1f}s)  "
          f"d_model={target.config.d_model}  vocab={target.config.vocab_size}", flush=True)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # ── Extract training data ──
    print(f"\nExtracting training data ({N_TRAIN} samples)...", flush=True)
    train_data = extract_training_data(target, processor, ds, N_TRAIN, device)
    print(f"  Total: {len(train_data)} datapoints in {time.time()-t0:.0f}s", flush=True)

    # ── PCA basis ──
    print("Computing PCA basis...", end=" ", flush=True)
    pca_pca_t0 = time.time()
    pca_mean, pca_V = compute_pca_basis_torch(
        [d["h_t"] for d in train_data], PCA_RANK)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"].to(device), pca_mean, pca_V)
    print(f"done ({time.time()-pca_pca_t0:.1f}s)", flush=True)

    # ── Train drafter ──
    print(f"\nTraining drafter (N={len(train_data)}, E={EPOCHS})...", flush=True)
    drafter = CorrectionDrafter(D_DRAFT, PCA_RANK, target.config.d_model, target.config.d_model, num_taps=2)
    drafter.to(device)
    drafter.train()

    opt = torch.optim.Adam(drafter.parameters(), lr=1e-3)
    model_dtype = next(target.parameters()).dtype
    train_t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for d in train_data:
            z_t = d["z_t"].to(device)
            ctx = d["ctx"].to(device)
            audio = d["audio"].to(device)
            true_tok = torch.tensor([d["true_token"]], device=device)

            delta_z = drafter(z_t, ctx, audio)
            z_corrected = z_t + delta_z
            h_corrected = from_pca(z_corrected, pca_mean, pca_V)
            draft_logits = target.proj_out(h_corrected.to(model_dtype))

            ce = torch.nn.functional.cross_entropy(draft_logits.view(1, -1), true_tok)
            mse = torch.mean(delta_z ** 2)
            loss = ce + LAMBDA_CE * mse

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:02d}/{EPOCHS} loss={epoch_loss/len(train_data):.4f} "
                  f"({time.time()-train_t0:.0f}s)", flush=True)

    drafter.eval()

    # ── Greedy baseline ──
    print(f"\n--- Greedy Baseline ({N_EVAL} eval) ---", flush=True)
    gw = []
    for i in range(N_TRAIN, N_TRAIN + N_EVAL):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        t1 = time.time()
        text = generate_greedy(target, processor, audio)
        elapsed = time.time() - t1
        w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
        gw.append(w)
        print(f"  [{i}] WER={w:.4f}  t={elapsed:.2f}s  text={text[:60]}", flush=True)
    gw_mean = np.mean(gw)
    print(f"  -> Mean WER={gw_mean:.4f}", flush=True)

    # ── Multi-path: static top-1, static top-3, adaptive ──
    configs = [
        ("Static Top-1", 1),
        ("Static Top-3", 3),
        ("Adaptive",     None),
    ]

    results = {}
    for label, static_k in configs:
        print(f"\n--- {label} ---", flush=True)
        ws, acs, toks = [], [], []
        for i in range(N_TRAIN, N_TRAIN + N_EVAL):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            t1 = time.time()
            text, acc, ntok = generate_adaptive(
                target, processor, audio, drafter, pca_mean, pca_V,
                static_k=static_k)
            elapsed = time.time() - t1
            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w); acs.append(acc); toks.append(ntok)
            accept_pct = acc / max(ntok, 1) * 100
            print(f"  [{i}] WER={w:.4f} accept={acc}/{ntok} ({accept_pct:.0f}%)  "
                  f"t={elapsed:.2f}s", flush=True)
        mw = np.mean(ws); ma = sum(acs)/max(1, sum(toks))*100
        delta = mw - gw_mean
        results[label] = (mw, ma)
        print(f"  -> WER={mw:.4f} ({delta:+.4f} vs greedy)  Accept={ma:.1f}%", flush=True)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RESULTS — {MODEL_ID}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Greedy:          WER={gw_mean:.4f}", flush=True)
    for label, (mw, ma) in results.items():
        delta = mw - gw_mean
        print(f"  {label:15s}: WER={mw:.4f} ({delta:+.4f})  Accept={ma:.1f}%", flush=True)
    print(f"  Training:        {N_TRAIN} samples, {len(train_data)} datapoints, {EPOCHS} epochs", flush=True)
    print(f"  Eval:            {N_EVAL} samples (LibriSpeech dummy-clean)", flush=True)
    print(f"{'='*60}", flush=True)

    return gw_mean, results


if __name__ == "__main__":
    t_start = time.time()
    gw, results = run()
    print(f"\nTotal wall: {time.time()-t_start:.0f}s", flush=True)

    # Return non-zero if adaptive WER is below greedy (signalling success)
    if "Adaptive" in results:
        adaptive_wer, _ = results["Adaptive"]
        if adaptive_wer <= gw:
            print(f"\n✅ ADAPTIVE WER ({adaptive_wer:.4f}) <= GREEDY WER ({gw:.4f}) — ARCHITECTURE VALIDATED", flush=True)
            sys.exit(0)
        else:
            print(f"\n⚠️  Adaptive WER ({adaptive_wer:.4f}) > Greedy ({gw:.4f})", flush=True)
            sys.exit(1)
