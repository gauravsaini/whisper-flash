#!/usr/bin/env python3
"""
MULTI-HEAD DRAFTING — Medusa-style multi-token speculative decoding.

Architecture:
  Shared backbone (existing CorrectionDrafter) → shared features
    ├── Head 0: ∆z in PCA space → W → token_{t+1}  (existing verified approach)
    ├── Head 1: MLP → vocab logits → token_{t+2}   (Medusa-style)
    └── Head 2: MLP → vocab logits → token_{t+3}   (Medusa-style)

Verification:
  Sequential KV-cache: verify d_{t+1}, then d_{t+2}, then d_{t+3}.
  Accept longest prefix passing top-k gate.

If N=3 all accepted → 3 decoder steps saved in 1 block → 3x speculative speedup.
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
EPOCHS = 30
N_TRAIN = 20
N_EVAL = 20
LAMBDA_CE = 0.1
MEDUSA_HIDDEN = 32   # hidden dim for Medusa heads 1,2
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


class MultiHeadDrafter(nn.Module):
    """
    Shared backbone + 3 heads (v2 — model-embedding output projection).
    
    Heads 1,2 project features into the model's embedding space (d_target-dim).
    The CALLER decodes to logits via target.decoder.token_embedding.as_linear().
    This avoids storing the massive (vocab_size, d_model) weight in the drafter.
    
    Head 0: ∆z in PCA space → decoded through model's W → token_{t+1}
    Head 1: features → d_target-dim embedding → decoded through model's W → token_{t+2}
    Head 2: features → d_target-dim embedding → decoded through model's W → token_{t+3}
    """

    def __init__(self, d_draft, pca_rank, d_target, d_audio, vocab_size,
                 medusa_hidden=None, num_taps=2):
        super().__init__()
        ctx_dim = num_taps * d_target + d_audio + pca_rank

        # Shared backbone
        self.ctx_proj = nn.Linear(ctx_dim, d_draft)
        self.layer1 = nn.Linear(d_draft, d_draft)
        self.layer1_norm = nn.LayerNorm(d_draft)
        self.layer2 = nn.Linear(d_draft, d_draft)
        self.layer2_norm = nn.LayerNorm(d_draft)

        # Head 0 — ∆z correction
        self.head0 = nn.Linear(d_draft, pca_rank, bias=False)

        # Heads 1,2 — project into model's embedding space
        # These output d_target-dim vectors that the model's W decodes to logits
        self.head1_to_emb = nn.Linear(d_draft, d_target)
        self.head1_norm = nn.LayerNorm(d_target)
        self.head2_to_emb = nn.Linear(d_draft, d_target)
        self.head2_norm = nn.LayerNorm(d_target)

    def __call__(self, z_t, target_hidden, audio_summary):
        """
        Returns:
          delta_z: PCA-space correction for head 0 (shape [1, 1, pca_rank])
          emb_1: d_target-dim embedding for head 1 (caller decodes via W)
          emb_2: d_target-dim embedding for head 2 (caller decodes via W)
        """
        ctx = mx.concatenate([target_hidden, audio_summary, z_t], axis=-1)
        x = self.ctx_proj(ctx)
        r1 = x; x = self.layer1_norm(x); x = nn.gelu(self.layer1(x)); x = x + r1
        r2 = x; x = self.layer2_norm(x); x = nn.gelu(self.layer2(x)); x = x + r2

        # Head 0: ∆z in PCA space
        delta_z = self.head0(x)

        # Head 1: features → d_target-dim embedding
        emb_1 = nn.gelu(self.head1_norm(self.head1_to_emb(x)))

        # Head 2: features → d_target-dim embedding
        emb_2 = nn.gelu(self.head2_norm(self.head2_to_emb(x)))

        return delta_z, emb_1, emb_2


def extract_triple_data(target, tokenizer, ds, n_train, device=None):
    """Extract (h_t, ctx, audio, token_{t+1}, token_{t+2}, token_{t+3}) triples."""
    data = []
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

        for t in range(1, labels.shape[1] - 3):  # need t+3
            inp_tok = labels[:, :t+1]
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp_tok, enc_h, collect_hidden_states=True, return_cross_attention=False)
            ctx = mx.concatenate([h_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
            h_t = h_all[-1][:, -1:, :]
            data.append({
                "h_t": mx.stop_gradient(h_t),
                "ctx": mx.stop_gradient(ctx),
                "audio": audio_summ,
                "tok1": labels[0, t+1],  # token_{t+1}
                "tok2": labels[0, t+2],  # token_{t+2}
                "tok3": labels[0, t+3],  # token_{t+3}
            })
    return data


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


def generate_multihead(target, drafter, tokenizer, mel, pca_mean, pca_V,
                       static_k=None, max_tokens=150):
    """
    Multi-head speculative decoding with sequential KV-cache verification.
    If static_k is set → use that fixed K.
    If static_k is None → adaptive: top-3 if accept<15%, else top-1.
    Returns (text, {head_i: (accepted, total) for each head}, speedup_info).
    """
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    stats = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # accepted, total per head
    attempts = 0
    accepted_total = 0
    blocks_used = 0  # how many multi-head blocks were attempted
    kv_cache = None

    while len(tokens) < max_tokens:
        # --- Draft phase (from h_t) ---
        if kv_cache is None:
            inp = mx.array([tokens], dtype=mx.int32)
        else:
            inp = mx.array([[tokens[-1]]], dtype=mx.int32)

        logits, kv_cache, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc, kv_cache=kv_cache,
            collect_hidden_states=True, return_cross_attention=False)

        h_t = hidden_all[-1][:, -1:, :]
        ctx_feats = mx.concatenate([hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
        z_t = to_pca(h_t, pca_mean, pca_V)
        delta_z, emb_1, emb_2 = drafter(z_t, ctx_feats, audio_summary)

        # Head 0: ∆z → corrected hidden → W → draft token_{t+1}
        h_corrected = from_pca(z_t + delta_z, pca_mean, pca_V)
        draft_logits_0 = target.decoder.token_embedding.as_linear(h_corrected)
        draft0 = mx.argmax(draft_logits_0, axis=-1).item()

        # Heads 1,2: embedding → W → draft tokens_{t+2, t+3}
        draft1 = mx.argmax(target.decoder.token_embedding.as_linear(emb_1), axis=-1).item()
        draft2 = mx.argmax(target.decoder.token_embedding.as_linear(emb_2), axis=-1).item()

        # Determine K
        if static_k is not None:
            k = static_k
        else:
            running_rate = accepted_total / max(attempts, 1)
            k = 1 if running_rate >= ADAPTIVE_THRESHOLD else 3
        attempts += 1

        # --- Verify phase (sequential with KV cache) ---
        # We need the decoder logits at the new position to check each draft.
        # tgt_logits = decoder logits at position len(tokens)-1  (predicts next token)
        tgt_logits = logits[0, -1, :]
        tgt_probs = mx.softmax(tgt_logits).tolist()
        topk_idxs = heapq.nlargest(k, range(len(tgt_probs)),
                                    key=lambda i: tgt_probs[i])

        accepted_any = False
        accepted_count = 0

        # Verify head 0 draft (token_{t+1})
        if draft0 in topk_idxs:
            tokens.append(draft0)
            accepted_total += 1
            stats[0][0] += 1
            stats[0][1] += 1
            accepted_any = True
            accepted_count = 1
            blocks_used += 1

            # If head 0 accepted, try head 1 (token_{t+2})
            # Run decoder with draft0 in context to get logits for position t+1
            kv_cache_next = kv_cache  # keep for sequential verification
            inp1 = mx.array([[draft0]], dtype=mx.int32)
            logits1, kv_cache_next, _ = decoder_forward_with_hidden_states(
                target, inp1, enc, kv_cache=kv_cache_next,
                collect_hidden_states=False, return_cross_attention=False)
            tgt_probs_1 = mx.softmax(logits1[0, -1, :]).tolist()
            topk_1 = heapq.nlargest(k, range(len(tgt_probs_1)),
                                     key=lambda i: tgt_probs_1[i])

            stats[1][1] += 1
            if draft1 in topk_1:
                tokens.append(draft1)
                accepted_total += 1
                stats[1][0] += 1
                accepted_count = 2
                kv_cache = kv_cache_next  # commit cache

                # If head 1 also accepted, try head 2 (token_{t+3})
                inp2 = mx.array([[draft1]], dtype=mx.int32)
                logits2, kv_cache_next, _ = decoder_forward_with_hidden_states(
                    target, inp2, enc, kv_cache=kv_cache_next,
                    collect_hidden_states=False, return_cross_attention=False)
                tgt_probs_2 = mx.softmax(logits2[0, -1, :]).tolist()
                topk_2 = heapq.nlargest(k, range(len(tgt_probs_2)),
                                         key=lambda i: tgt_probs_2[i])

                stats[2][1] += 1
                if draft2 in topk_2:
                    tokens.append(draft2)
                    accepted_total += 1
                    stats[2][0] += 1
                    accepted_count = 3
                    kv_cache = kv_cache_next
                else:
                    # Reject head 2, fall back to greedy for position t+3
                    greedy2 = mx.argmax(logits2[0, -1, :], axis=-1).item()
                    tokens.append(greedy2)
                    kv_cache = kv_cache_next  # still advance context
            else:
                # Reject head 1, fall back to greedy for position t+2
                kv_cache = kv_cache_next
                greedy1 = mx.argmax(logits1[0, -1, :], axis=-1).item()
                tokens.append(greedy1)
        else:
            # Reject head 0, fall back to greedy
            stats[0][1] += 1
            greedy0 = mx.argmax(logits[:, -1, :], axis=-1).item()
            tokens.append(greedy0)

        if tokens[-1] == tokenizer.eot:
            break

    return (tokenizer.decode(tokens),
            {k: (v[0], v[1], v[0]/max(v[1],1)*100) for k, v in stats.items()},
            accepted_total, len(tokens), blocks_used)


def run_experiment(model_name, pca_rank, d_draft, n_train=N_TRAIN, n_eval=N_EVAL):
    print(f"\n{'='*70}")
    print(f"MULTI-HEAD DRAFTING on {model_name}")
    print(f"  PCA R={pca_rank}, Draft dim={d_draft}, Medusa hidden={MEDUSA_HIDDEN}")
    print(f"  Heads: 0(∆z→token+1), 1(medusa→token+2), 2(medusa→token+3)")
    print(f"  {'='*70}")

    t0 = time.time()
    print(f"Loading model...", end=" ", flush=True)
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    vocab_size = target.dims.n_vocab
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print(f"done ({time.time()-t0:.1f}s)  d_model={d_target}  vocab={vocab_size}")

    # Training data (triples: t+1, t+2, t+3)
    print(f"\nExtracting training data ({n_train} samples)...")
    train_data = extract_triple_data(target, tokenizer, ds, n_train)
    print(f"  {len(train_data)} datapoints")

    # PCA
    print(f"PCA basis (R={pca_rank})...", end=" ", flush=True)
    pca_mean, pca_V = compute_pca_basis(
        [{"true_hidden": d["h_t"]} for d in train_data], pca_rank, d_target)
    for d in train_data:
        d["z_t"] = to_pca(d["h_t"], pca_mean, pca_V)
    print("done")

    # Build drafter
    print(f"Building MultiHeadDrafter...", end=" ", flush=True)
    try:
        drafter = MultiHeadDrafter(d_draft, pca_rank, d_target, d_target,
                                    vocab_size)
        _ = drafter(train_data[0]["z_t"], train_data[0]["ctx"], train_data[0]["audio"])
        n_params = sum(arr.size for v in drafter.parameters().values()
                       for arr in (v.values() if isinstance(v, dict) else [v]))
        print(f"{n_params} params", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise

    # ── Training ──
    print(f"Training (N={len(train_data)}, E={EPOCHS})...", flush=True)

    def loss_fn(m, d):
        delta_z, emb_1, emb_2 = m(d["z_t"], d["ctx"], d["audio"])
        h_corrected = from_pca(d["z_t"] + delta_z, pca_mean, pca_V)
        draft_logits_0 = target.decoder.token_embedding.as_linear(h_corrected)
        ce0 = nn.losses.cross_entropy(draft_logits_0.reshape(1, -1),
                                       mx.array([d["tok1"]]), reduction="mean")
        loss0 = ce0 + LAMBDA_CE * mx.mean(mx.square(delta_z))
        logits_1 = target.decoder.token_embedding.as_linear(emb_1)
        ce1 = nn.losses.cross_entropy(logits_1.reshape(1, -1),
                                       mx.array([d["tok2"]]), reduction="mean")
        logits_2 = target.decoder.token_embedding.as_linear(emb_2)
        ce2 = nn.losses.cross_entropy(logits_2.reshape(1, -1),
                                       mx.array([d["tok3"]]), reduction="mean")
        return loss0 + ce1 + ce2

    grad_fn = nn.value_and_grad(drafter, loss_fn)
    opt = optim.Adam(learning_rate=1e-3)
    train_t0 = time.time()

    for epoch in range(EPOCHS):
        loss_sum = 0.0
        for step_idx, d in enumerate(train_data):
            l, g = grad_fn(drafter, d)
            opt.update(drafter, g)
            mx.eval(drafter.parameters(), opt.state)
            loss_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:02d}/{EPOCHS} loss={loss_sum/len(train_data):.4f} "
                  f"({time.time()-train_t0:.0f}s)")
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:02d}/{EPOCHS} loss={loss_sum/len(train_data):.4f} "
                  f"({time.time()-train_t0:.0f}s)")

    # ── Greedy baseline ──
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

    # ── Multi-head eval ──
    for label, static_k in [("Static Top-1", 1), ("Adaptive", None)]:
        print(f"\n  --- {label} ---")
        ws, acpt_tot, toks_tot, blk_tot = [], 0, 0, 0
        head_stats_acc = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

        for i in range(n_train, n_train + n_eval):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)

            text, hstats, acpt, ntoks, nblocks = generate_multihead(
                target, drafter, tokenizer, mel_mx,
                pca_mean=pca_mean, pca_V=pca_V, static_k=static_k)

            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w)
            acpt_tot += acpt
            toks_tot += ntoks
            blk_tot += nblocks
            for hid, (ac, tl, _) in hstats.items():
                head_stats_acc[hid][0] += ac
                head_stats_acc[hid][1] += tl

            # Print per-sample: WER + head-level accept rates
            hstr = " ".join([f"H{hid}={ac}/{tl}({p:.0f}%)"
                           for hid, (ac, tl, p) in hstats.items()])
            print(f"    [{i}] WER={w:.4f} total={acpt}/{ntoks} "
                  f"blocks={nblocks}  {hstr}")

        mw = np.mean(ws)
        ma = acpt_tot / max(toks_tot, 1) * 100
        print(f"    -> WER={mw:.4f} ({mw-gw_mean:+.4f}) Accept={ma:.1f}%")
        print(f"    -> Head stats:")
        for hid in [0, 1, 2]:
            ac, tl = head_stats_acc[hid]
            rate = ac / max(tl, 1) * 100
            print(f"       Head {hid}: {ac}/{tl} ({rate:.1f}%)")

    print()
    return gw_mean


def run():
    # Step 1: Whisper-tiny (fast, verify heads 1,2 learn anything)
    run_experiment(
        model_name="mlx-community/whisper-tiny",
        pca_rank=64,
        d_draft=256,
        n_train=30,
        n_eval=10,  # fewer eval samples for speed
    )


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"\nTotal wall: {time.time()-t_start:.0f}s")
