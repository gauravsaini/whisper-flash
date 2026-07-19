#!/usr/bin/env python3
"""
Multi-path verification for continuous speculative decoding.

Core idea: instead of demanding M_graph >= 0.95 against the SINGLE greedy path,
compute the target's top-K plausible continuations and accept the draft if it
matches ANY of them.

Architecture:
1. At anchor step t, get target logits → top-K tokens (K=5)
2. For each branch k, decode BLOCK steps to get candidate hidden trajectory H_k
3. Compute M_graph between draft H_draft and EACH H_k
4. Accept draft if max_k M_graph(H_draft, H_k) >= θ (e.g. 0.90)
5. Fallback: sequential token-level verification at τ=0.97

This directly addresses Experiment #4's finding that the target's own
alternatives diverge to ~0.60 cosine — we now check against ALL of them.
"""

import time, math, numpy as np
import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel
import jiwer

TOP_K = 5
BLOCK = 4
THETA = 0.90
TAU = 0.97
EPOCHS = 15

def gram_matrix(H):
    norms = np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9
    return (H / norms) @ (H / norms).T

def node_similarity(pred_h, true_h):
    pn = pred_h / (np.linalg.norm(pred_h, axis=-1, keepdims=True) + 1e-9)
    tn = true_h / (np.linalg.norm(true_h, axis=-1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(pn * tn, axis=-1)))

def topological_similarity(pred_h, true_h):
    Gp, Gt = gram_matrix(pred_h), gram_matrix(true_h)
    return float(1.0 - np.mean(np.abs(Gp - Gt)) / 2.0)

def compute_m_graph(pred_h, true_h):
    nsim = node_similarity(pred_h, true_h)
    tsim = topological_similarity(pred_h, true_h)
    return 0.5 * nsim + 0.5 * tsim, nsim, tsim

def get_top_k_branches(target, encoder_hidden, tokens, k=TOP_K, block_size=BLOCK):
    """
    Given the current context tokens, compute K alternative future hidden trajectories.
    
    Returns: list of (branch_tokens, branch_hidden) for each of the K paths.
    - branch_tokens: the full block of greedy tokens for this branch (B,)
    - branch_hidden: the hidden states (B, d_target)
    """
    inp = mx.array([tokens], dtype=mx.int32)
    logits, _, hidden_all = decoder_forward_with_hidden_states(
        target, inp, encoder_hidden,
        collect_hidden_states=True, return_cross_attention=False
    )

    # Get top-K tokens from the last logits
    probs = mx.softmax(logits[0, -1, :])
    sorted_idx = mx.argsort(-probs)
    top_k_tokens = sorted_idx[:k]

    branches = []
    for branch_tok in top_k_tokens:
        # The branch starts by forcing this token
        branch_tokens = [int(branch_tok)]
        branch_input = mx.concatenate([
            inp,
            mx.array([[int(branch_tok)]], dtype=mx.int32)
        ], axis=1)

        # Greedy decode the remaining block-1 positions
        for step in range(block_size - 1):
            logits_b, _, _ = decoder_forward_with_hidden_states(
                target, branch_input, encoder_hidden,
                collect_hidden_states=False, return_cross_attention=False
            )
            next_tok = mx.argmax(logits_b[:, -1:, :], axis=-1).item()
            branch_tokens.append(next_tok)
            branch_input = mx.concatenate([
                branch_input,
                mx.array([[next_tok]], dtype=mx.int32)
            ], axis=1)

        # Get hidden states for the full branch
        full_input = mx.concatenate([
            inp,
            mx.array([branch_tokens], dtype=mx.int32)
        ], axis=1)
        _, _, h_branch = decoder_forward_with_hidden_states(
            target, full_input, encoder_hidden,
            collect_hidden_states=True, return_cross_attention=False
        )
        branch_hidden = np.array(h_branch[-1][0, -block_size:, :])
        branches.append((branch_tokens, branch_hidden))

    return branches, hidden_all

def generate_speculative_multipath(
    target, draft_model, tokenizer, mel, max_tokens=150,
    block_size=BLOCK, theta=THETA, tau=TAU, k=TOP_K
):
    """Multi-path speculative decoding with fixed (correct) harness."""
    enc = encoder_forward(target, mel)
    audio_summary = mx.mean(enc, axis=1, keepdims=True)

    tokens = [tokenizer.sot]
    accepted_draft = 0
    total_draft = 0
    steps = 0
    multi_path_key_account = ""

    while len(tokens) < max_tokens:
        steps += 1
        inp = mx.array([tokens], dtype=mx.int32)
        logits, _, hidden_all = decoder_forward_with_hidden_states(
            target, inp, enc,
            collect_hidden_states=True, return_cross_attention=False
        )

        # Build context for draft model
        target_hidden = mx.concatenate(
            [hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1
        )

        # Generate draft block
        noise = target.decoder.token_embedding(mx.array([[50257] * block_size]))
        pos = mx.arange(len(tokens), len(tokens) + block_size, dtype=mx.int32)[None]
        draft_hidden = draft_model(noise_embedding=noise, target_hidden=target_hidden,
                                   audio_summary=audio_summary, position_ids=pos)
        draft_logits = target.decoder.token_embedding.as_linear(draft_hidden)
        draft_tokens = mx.argmax(draft_logits, axis=-1)[0].tolist()
        pred_np = np.array(draft_hidden[0])

        # Get top-K branches as candidate target paths
        branches, hidden_all = get_top_k_branches(target, enc, tokens, k=k, block_size=block_size)

        # Compute M_graph against each branch
        best_mg = -1.0
        best_nsim = 0.0
        best_tsim = 0.0
        best_path_idx = -1
        path_metrics = []

        for b_idx, (btoks, bh) in enumerate(branches):
            mg, ns, ts = compute_m_graph(pred_np, bh)
            path_metrics.append((mg, ns, ts))
            if mg > best_mg:
                best_mg = mg
                best_nsim = ns
                best_tsim = ts
                best_path_idx = b_idx

        # Get single greedy path M_graph for comparison
        greedy_tokens = [mx.argmax(
            target.decoder.token_embedding.as_linear(
                mx.array(branch_hidden[None])
            ), axis=-1)[0, -1].item()
            for _, branch_hidden in branches[:1]]
        # Actually just use the top-1 branch hidden
        g_mg, _, _ = compute_m_graph(pred_np, branches[0][1])

        if steps <= 5 or best_mg >= theta:
            cos_str = ", ".join(f"{c:.3f}" for c in [
                float(np.dot(pred_np[k], branches[0][1][k]) /
                      ((np.linalg.norm(pred_np[k])+1e-9)*(np.linalg.norm(branches[0][1][k])+1e-9)))
                for k in range(min(block_size, len(pred_np)))
            ])
            print(f"  Step@{steps}: M_graph_greedy={g_mg:.3f} best_multi={best_mg:.3f} (path={best_path_idx}) ns={best_nsim:.3f} ts={best_tsim:.3f} cos=[{cos_str}]")

        # Multi-path verification
        accepted_k = 0
        if best_mg >= theta:
            # Use the best-matching branch for token-level verification
            best_hidden = branches[best_path_idx][1]
            best_hidden_mx = mx.array(best_hidden[None], dtype=mx.float32)
            best_logits = target.decoder.token_embedding.as_linear(best_hidden_mx)

            for k_pos in range(block_size):
                draft_p = mx.softmax(draft_logits[0, k_pos] / 1.0)
                true_p = mx.softmax(best_logits[0, k_pos] / 1.0)
                if true_p[draft_tokens[k_pos]].item() >= tau * draft_p[draft_tokens[k_pos]].item():
                    accepted_k += 1
                else:
                    break
            if accepted_k > 0:
                multi_path_key_account += "+"

        # Fallback: always accept at least the correct next token
        # Run target on the full speculative sequence
        spec_tokens = tokens + draft_tokens
        spec_inp = mx.array([spec_tokens], dtype=mx.int32)
        true_logits_full, _, _ = decoder_forward_with_hidden_states(
            target, spec_inp, enc,
            collect_hidden_states=False, return_cross_attention=False
        )
        true_next = mx.argmax(true_logits_full[:, len(tokens):, :], axis=-1)[0].tolist()

        tokens.extend(draft_tokens[:accepted_k])
        tokens.append(true_next[len(tokens) - 1])
        accepted_draft += accepted_k
        total_draft += block_size

        if tokens[-1] == tokenizer.eot:
            break

    text = tokenizer.decode(tokens)
    return text, accepted_draft, total_draft, multi_path_key_account


def run():
    print("=" * 65)
    print("MULTI-PATH VERIFICATION (k=5, θ=0.90)")
    print("=" * 65)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state

    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2,
        vocab_size=target.dims.n_vocab, block_size=BLOCK, target_layer_ids=[1, 2]
    )
    draft = ContinuousDraftModel(config)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Train
    print(f"\nExtracting {len(ds)} samples...")
    train_data = []
    for i in range(10):
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(s["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)
        enc_h = encoder_forward(target, mel_mx)
        audio_summ = mx.mean(enc_h, axis=1, keepdims=True)
        for t in range(1, labels.shape[1]-BLOCK, 3):
            inp_tok = labels[:, :t+1]
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp_tok, enc_h, collect_hidden_states=True, return_cross_attention=False)
            ctx = mx.concatenate([h_all[lid] for lid in [1, 2]], axis=-1)
            _, _, h_fut = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+BLOCK], enc_h, collect_hidden_states=True, return_cross_attention=False)
            true_h = h_fut[-1][:, t:t+BLOCK, :]
            noise = target.decoder.token_embedding(mx.array([[50257]*BLOCK]))
            pos = mx.arange(t, t+BLOCK, dtype=mx.int32)[None]
            train_data.append({"noise": noise, "ctx": ctx, "audio": audio_summ,
                               "pos": pos, "true_hidden": true_h})

    print(f"  {len(train_data)} datapoints, training {EPOCHS} epochs...")
    _ = draft(train_data[0]["noise"], train_data[0]["ctx"], train_data[0]["audio"], train_data[0]["pos"])

    def mse_loss(m, noise, ctx, audio, pos, true_h):
        pred = m(noise, ctx, audio, pos)
        return mx.mean(mx.square(pred - true_h))

    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(draft, mse_loss)

    for ep in range(EPOCHS):
        ls = 0.0
        for d in train_data:
            l, g = loss_and_grad(draft, d["noise"], d["ctx"], d["audio"], d["pos"], d["true_hidden"])
            opt.update(draft, g)
            mx.eval(draft.parameters(), opt.state)
            ls += l.item()
        if (ep+1) % 5 == 0:
            print(f"  Epoch {ep+1}/{EPOCHS} Loss={ls/len(train_data):.5f}")

    # Evaluate multi-path vs greedy-path
    print(f"\n{'='*65}")
    print("EVALUATION: Multi-path vs Greedy-path verification")
    print(f"{'='*65}")

    for label, multi in [("Greedy Path (θ=0.95)", False), ("Multi-Path (θ=0.90)", True)]:
        print(f"\n  {label}:")
        total_wer, total_acc, total_dr = 0.0, 0, 0
        for i in range(10, 20):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)

            from experiment_fixed_harness import generate_speculative_fixed as gen_single
            text, acc, dr = gen_single(
                target, draft, tokenizer, mel_mx,
                max_tokens=150, block_size=BLOCK,
                m_graph_threshold=0.95 if not multi else 0.90,
                tau=TAU
            )

            total_acc += acc; total_dr += dr
            norm_pred = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
                jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(text))))
            norm_ref = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
                jiwer.ToLowerCase()(jiwar.ExpandCommonEnglishContractions()(s["text"]))))
            wer = jiwar.wer(norm_ref, norm_pred) if norm_ref else 1.0
            total_wer += wer
            print(f"    [{i}] WER={wer:.4f} accept={acc}/{dr} ({acc/max(dr,1)*100:.1f}%)")

        mw = total_wer / 10
        ma = total_acc / max(total_dr, 1) * 100
        print(f"  -> {label}: WER={mw:.4f}, Accept={ma:.2f}%")

    # Also run true multi-path on sample 10
    print(f"\n{'='*65}")
    print("TRUE MULTI-PATH: generating with K=5 branches")
    print(f"{'='*65}")
    for i in [10]:
        s = ds[i]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)

        t1 = time.time()
        text, acc, dr, flags = generate_speculative_multipath(
            target, draft, tokenizer, mel_mx, k=TOP_K, theta=THETA)
        elapsed = time.time() - t1

        norm_pred = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
            jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(text))))
        norm_ref = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
            jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(s["text"]))))
        wer = jiwer.wer(norm_ref, norm_ref) if norm_ref else 1.0
        print(f"  [{i}] WER={wer:.4f} accept={acc}/{dr} ({acc/max(dr,1)*100:.1f}%) time={elapsed:.1f}s")
        print(f"       Multi-path acceptances: {len(flags)} times")

    print(f"\nTotal: {time.time()-t0:.0f}s" if 't0' in dir() else "")

if __name__ == "__main__":
    run()
