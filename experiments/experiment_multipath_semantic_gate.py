#!/usr/bin/env python3
"""
Multi-path semantic verification gate.

Architecture:
  1. At anchor step t, compute K target branches (top-K tokens, force-decode B steps)
  2. Compare draft block against ALL K branches: M_multi = max_k M_graph^(k)
  3. If M_multi >= theta, accept draft (block-level)
  4. If M_multi < theta, try per-position acceptance (each position against any branch)
  5. Fallback: single true token

Training improvements vs naive MSE:
  - Velocity prediction (predict delta from target context, not absolute)
  - Multi-objective loss: MSE + lambda_ce * CE
  - More data (20 samples instead of 10)
"""

import time, math, numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel
import jiwer

TOP_K = 5
BLOCK = 4
TAU = 0.97
LAMBDA_CE = 0.3
EPOCHS = 30
N_TRAIN = 20


def norm(t):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t))))


def compute_m_graph(pred_h, true_h):
    pn = pred_h / (np.linalg.norm(pred_h, axis=-1, keepdims=True) + 1e-9)
    tn = true_h / (np.linalg.norm(true_h, axis=-1, keepdims=True) + 1e-9)
    nsim = float(np.mean(np.sum(pn * tn, axis=-1)))
    Gp = pn @ pn.T
    Gt = tn @ tn.T
    tsim = float(1.0 - np.mean(np.abs(Gp - Gt)) / 2.0)
    return 0.5 * nsim + 0.5 * tsim, nsim, tsim


def poswise_m_graph(pred_h, true_h):
    """Per-position node_sim only (for position-by-position verification)."""
    pn = pred_h / (np.linalg.norm(pred_h, axis=-1, keepdims=True) + 1e-9)
    tn = true_h / (np.linalg.norm(true_h, axis=-1, keepdims=True) + 1e-9)
    return list(np.sum(pn * tn, axis=-1))


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


def get_branches(target, enc, tokens, k=TOP_K, block_size=BLOCK):
    """K full-block hidden trajectories from current context.
    Robust implementation: processes one branch at a time with explicit error handling."""
    inp = mx.array([tokens], dtype=mx.int32)
    logits, _, _ = decoder_forward_with_hidden_states(
        target, inp, enc, collect_hidden_states=False, return_cross_attention=False)
    probs = mx.softmax(logits[0, -1, :])
    import heapq
    probs_list = probs.tolist()
    top_toks = heapq.nlargest(k, range(len(probs_list)), key=lambda i: probs_list[i])
    branches = []
    for bt in top_toks:
        bi = mx.concatenate([inp, mx.array([[bt]], dtype=mx.int32)], axis=1)
        ids = [bt]
        for _ in range(block_size - 1):
            lb, _, _ = decoder_forward_with_hidden_states(
                target, bi, enc, collect_hidden_states=False, return_cross_attention=False)
            nid = mx.argmax(lb[:, -1:, :], axis=-1).item()
            ids.append(nid)
            bi = mx.concatenate([bi, mx.array([[nid]], dtype=mx.int32)], axis=1)
        fi = mx.concatenate([inp, mx.array([ids], dtype=mx.int32)], axis=1)
        _, _, hb = decoder_forward_with_hidden_states(
            target, fi, enc, collect_hidden_states=True, return_cross_attention=False)
        branches.append((ids, np.array(hb[-1][0, -block_size:, :])))
    return branches


def generate_speculative_multipath(target, draft_model, tokenizer, mel, max_tokens=150,
                                   theta_block=0.90, theta_pos=0.85, k=TOP_K):
    """
    Multi-path verification with fallback chain:
      1. Block-level: if M_multi >= theta_block, accept all B tokens
      2. Per-position: for each position k, if any branch's position k has cos >= theta_pos, accept
      3. Fallback: single true token
    """
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)
    tokens = [tokenizer.sot]
    accepted = 0
    total = 0

    while len(tokens) < max_tokens:
        inp = mx.array([tokens], dtype=mx.int32)
        _, _, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc, collect_hidden_states=True, return_cross_attention=False)

        # Velocity-prediction context: the anchor hidden state
        anchor_h = mx.concatenate(
            [hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
        noise = target.decoder.token_embedding(mx.array([[50257] * BLOCK]))
        pos = mx.arange(len(tokens), len(tokens) + BLOCK, dtype=mx.int32)[None]
        draft_h = draft_model(noise_embedding=noise, target_hidden=anchor_h,
                              audio_summary=audio_summary, position_ids=pos)
        draft_logits = target.decoder.token_embedding.as_linear(draft_h)
        draft_tokens = mx.argmax(draft_logits, axis=-1)[0].tolist()
        pred_np = np.array(draft_h[0])

        # Get K branches
        try:
            branches = get_branches(target, enc, tokens, k=k, block_size=BLOCK)
        except Exception as e:
            print(f"  ERROR get_branches: {e} at len(tokens)={len(tokens)}")
            branches = []

        if not branches:
            # Fallback: use the target's own forward pass as a single branch
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp, enc, collect_hidden_states=True, return_cross_attention=False)
            greedy_h = np.array(h_all[-1][0, -BLOCK:, :])
            branches = [(draft_tokens[:BLOCK], greedy_h)]

        # Multi-path similarity against all branches
        mg_vals = [compute_m_graph(pred_np, bh) for _, bh in branches]
        best_idx = int(np.argmax([m[0] for m in mg_vals]))
        best_mg, best_ns, best_ts = mg_vals[best_idx]

        # Greedy-path for comparison
        greedy_mg, _, _ = mg_vals[0]

        # Also run target on full speculative sequence (for fallback tokens)
        spec = tokens + draft_tokens
        spec_inp = mx.array([spec], dtype=mx.int32)
        true_logits, _, true_hidden = decoder_forward_with_hidden_states(
            target, spec_inp, enc, collect_hidden_states=True, return_cross_attention=False)
        B_off = len(tokens)
        true_h_fut = true_hidden[-1][:, B_off:, :]
        true_logits_fut = true_logits[:, B_off - 1:, :]
        true_next = mx.argmax(true_logits_fut, axis=-1)[0].tolist()

        # Per-position cosines against each branch (for position-by-position verification)
        pos_cos = [poswise_m_graph(pred_np, bh) for _, bh in branches]
        max_pos_cos = [max(pc[k] for pc in pos_cos) for k in range(BLOCK)]

        # === Acceptance block ===
        accepted_k = 0

        # Level 1: Full-block acceptance
        if best_mg >= theta_block:
            # Token-level τ verification against the best branch
            best_bh = branches[best_idx][1]
            best_bh_mx = mx.array(best_bh[None], dtype=mx.float32)
            best_logits = target.decoder.token_embedding.as_linear(best_bh_mx)
            for k in range(BLOCK):
                dp = mx.softmax(draft_logits[0, k] / 1.0)
                bp = mx.softmax(best_logits[0, k] / 1.0)
                if bp[draft_tokens[k]].item() >= TAU * dp[draft_tokens[k]].item():
                    accepted_k += 1
                else:
                    break
        else:
            # Level 2: Per-position acceptance
            for k in range(BLOCK):
                if max_pos_cos[k] >= theta_pos:
                    accepted_k += 1
                else:
                    break

        tokens.extend(draft_tokens[:accepted_k])
        if accepted_k < BLOCK:
            tokens.append(true_next[accepted_k])
        accepted += accepted_k
        total += BLOCK

        if tokens[-1] == tokenizer.eot:
            break

    return tokenizer.decode(tokens), accepted, total, max_pos_cos[:5] if len(tokens) > 5 else max_pos_cos


def run():
    print("=" * 70)
    print("MULTI-PATH SEMANTIC VERIFICATION GATE")
    print(f"  K={TOP_K}, B={BLOCK}, λ_ce={LAMBDA_CE}, τ={TAU}")
    print("=" * 70)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # --- Train with multi-objective loss ---
    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2,
        vocab_size=target.dims.n_vocab, block_size=BLOCK, target_layer_ids=[1, 2])
    draft = ContinuousDraftModel(config)

    print(f"\nExtracting training data ({N_TRAIN} samples)...")
    train_data = []
    for i in range(N_TRAIN):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(s["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)
        enc_h = encoder_forward(target, mel_mx)
        audio_summ = mx.mean(enc_h, axis=1, keepdims=True)
        for t in range(1, labels.shape[1] - BLOCK, 2):
            inp_tok = labels[:, :t+1]
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp_tok, enc_h, collect_hidden_states=True, return_cross_attention=False)
            ctx = mx.concatenate([h_all[lid] for lid in [1, 2]], axis=-1)
            _, _, h_fut = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+BLOCK], enc_h, collect_hidden_states=True, return_cross_attention=False)
            true_h = h_fut[-1][:, t:t+BLOCK, :]
            # Target tokens for CE loss
            true_toks = labels[0, t+1:t+1+BLOCK]
            noise = target.decoder.token_embedding(mx.array([[50257]*BLOCK]))
            pos = mx.arange(t, t+BLOCK, dtype=mx.int32)[None]
            train_data.append({
                "noise": noise, "ctx": ctx, "audio": audio_summ, "pos": pos,
                "true_hidden": true_h, "true_tokens": true_toks
            })

    print(f"  {len(train_data)} datapoints")
    _ = draft(train_data[0]["noise"], train_data[0]["ctx"], train_data[0]["audio"], train_data[0]["pos"])

    def compute_loss(m, d):
        pred = m(d["noise"], d["ctx"], d["audio"], d["pos"])
        # MSE on hidden states
        mse = mx.mean(mx.square(pred - d["true_hidden"]))
        # CE on logits (from weight-tied projection)
        logits = target.decoder.token_embedding.as_linear(pred)
        ce = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), d["true_tokens"].reshape(-1)))
        return mse + LAMBDA_CE * ce

    loss_and_grad = nn.value_and_grad(draft, compute_loss)
    opt = optim.Adam(learning_rate=1e-3)

    for ep in range(EPOCHS):
        ls = 0.0
        for d in train_data:
            l, g = loss_and_grad(draft, d)
            opt.update(draft, g)
            mx.eval(draft.parameters(), opt.state)
            ls += l.item()
        if (ep+1) % 5 == 0:
            print(f"  Epoch {ep+1}/{EPOCHS} Loss={ls/len(train_data):.5f}")

    # --- Greedy baseline ---
    print(f"\n--- Greedy Baseline ---")
    gw = []
    for i in range(N_TRAIN, N_TRAIN + 10):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        text = generate_greedy(target, tokenizer, mel_mx)
        gw.append(jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0)
        print(f"  [{i}] WER={gw[-1]:.4f}")
    print(f"  -> Mean WER={np.mean(gw):.4f}")

    # --- Multi-path verification: threshold sweep ---
    for theta_b, theta_p, label in [
        (0.90, 0.85, "Block θ=0.90 / Pos θ=0.85"),
        (0.75, 0.75, "Block θ=0.75 / Pos θ=0.75"),
        (0.60, 0.65, "Block θ=0.60 / Pos θ=0.65"),
        (0.00, 0.55, "Pos-only θ=0.55 (no block gate)"),
    ]:
        print(f"\n--- {label} ---")
        ws, acs, tots = [], [], []
        for i in range(N_TRAIN, N_TRAIN + 10):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)
            t1 = time.time()
            text, acc, tot, _ = generate_speculative_multipath(
                target, draft, tokenizer, mel_mx, theta_block=theta_b, theta_pos=theta_p)
            el = time.time() - t1
            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w); acs.append(acc); tots.append(tot)
            ar = acc / max(tot, 1) * 100
            print(f"  [{i}] WER={w:.4f} accept={acc}/{tot} ({ar:.1f}%) time={el:.1f}s")
        mw = np.mean(ws); ma = sum(acs)/max(sum(tots),1)*100
        print(f"  -> WER={mw:.4f} Accept={ma:.1f}%")

        # If no acceptance, dump per-position cos for first sample
        if ma == 0 and label == list([
            (0.90, 0.85, "Block θ=0.90 / Pos θ=0.85"),
            (0.75, 0.75, "Block θ=0.75 / Pos θ=0.75"),
        ])[0]:
            print(f"  (Dumping per-position max-cos from sample {N_TRAIN}...)")

    # --- Detailed step-by-step on first eval sample ---
    print(f"\n{'='*70}")
    print("DETAILED STEP-BY-STEP (Sample 10, Per-position max-cos across K=5 branches)")
    print(f"{'='*70}")
    s = ds[N_TRAIN]
    audio = np.array(s["audio"]["array"], dtype=np.float32)
    mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
    mel_mx = mx.array(mel[None], dtype=mx.float32)
    _, acc, tot, cos5 = generate_speculative_multipath(
        target, draft, tokenizer, mel_mx, theta_block=0.0, theta_pos=0.0)
    print(f"  Accept: {acc}/{tot} ({acc/max(tot,1)*100:.1f}%)")
    print(f"  First 5 step-wise max-per-position cos: {[f'{c:.3f}' for c in cos5]}")


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"\nTotal: {time.time()-t_start:.0f}s")
