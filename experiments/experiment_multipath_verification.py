#!/usr/bin/env python3
"""
Multi-path verification: accept draft tokens that match ANY of the target's 
top-K continuations. Clean implementation avoiding mlx argsort issues.

Core flow per verification step:
  1. Draft generates B hidden states, decoded to B tokens
  2. Run target on full speculative sequence → get logits for each position
  3. For each position k: target's top-K tokens from its own logits at that position
  4. Accept draft token at position k if it's in the target's top-K at that position
  5. Fallback: single greedy token on first position where draft doesn't match

This directly tests: can the drafter produce ANY token the target considers plausible?
"""

import time, math, heapq, numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_whisper.audio import log_mel_spectrogram
from mlx_whisper.tokenizer import get_tokenizer
from whisper_flash_mlx.target_model import load_target_model, encoder_forward, decoder_forward_with_hidden_states
from whisper_flash_mlx.draft_model import WhisperDFlashConfig, ContinuousDraftModel
import jiwer

BLOCK = 4
TAU = 0.97
TOP_K = [1, 3, 5]
LAMBDA_CE = 0.3
EPOCHS = 30
N_TRAIN = 20

def norm(t):
    return jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
        jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(t))))

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

def generate_speculative_multipath(target, draft_model, tokenizer, mel, max_tokens=150, top_k=5):
    """
    Draft-then-verify with top-k multi-path acceptance.
    Accept position k if draft_token[k] is in the target's top-k at that position.
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

        anchor_h = mx.concatenate(
            [hidden_all[lid][:, -1:, :] for lid in [1, 2]], axis=-1)
        noise = target.decoder.token_embedding(mx.array([[50257] * BLOCK]))
        pos = mx.arange(len(tokens), len(tokens) + BLOCK, dtype=mx.int32)[None]
        draft_h = draft_model(noise_embedding=noise, target_hidden=anchor_h,
                              audio_summary=audio_summary, position_ids=pos)
        draft_logits = target.decoder.token_embedding.as_linear(draft_h)
        draft_tokens = mx.argmax(draft_logits, axis=-1)[0].tolist()

        # Run target on full speculative input
        spec = tokens + draft_tokens
        spec_inp = mx.array([spec], dtype=mx.int32)
        true_logits, _, _ = decoder_forward_with_hidden_states(
            target, spec_inp, enc, collect_hidden_states=False, return_cross_attention=False)
        B_off = len(tokens)
        true_logits_fut = true_logits[:, B_off - 1:, :]
        true_next = mx.argmax(true_logits_fut, axis=-1)[0].tolist()

        # Acceptance: check if each draft token is in target's top-k at that position
        accepted_k = 0
        for k in range(BLOCK):
            probs_k = mx.softmax(true_logits_fut[0, k, :]).tolist()
            topk_idxs = heapq.nlargest(top_k, range(len(probs_k)), key=lambda i: probs_k[i])
            if draft_tokens[k] in topk_idxs:
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

    return tokenizer.decode(tokens), accepted, total

def run():
    print("=" * 70)
    print("MULTI-PATH VERIFICATION (top-k token matching)")
    print("=" * 70)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Train
    config = WhisperDFlashConfig(d_target=d_target, d_draft=256, num_layers=2,
        vocab_size=target.dims.n_vocab, block_size=BLOCK, target_layer_ids=[1, 2])
    draft = ContinuousDraftModel(config)

    print(f"\nTraining on {N_TRAIN} samples...")
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
            true_toks = labels[0, t+1:t+1+BLOCK]
            noise = target.decoder.token_embedding(mx.array([[50257]*BLOCK]))
            pos = mx.arange(t, t+BLOCK, dtype=mx.int32)[None]
            train_data.append({"noise": noise, "ctx": ctx, "audio": audio_summ,
                               "pos": pos, "true_hidden": true_h, "true_tokens": true_toks})
    print(f"  {len(train_data)} datapoints")

    _ = draft(train_data[0]["noise"], train_data[0]["ctx"], train_data[0]["audio"], train_data[0]["pos"])

    def compute_loss(m, d):
        pred = m(d["noise"], d["ctx"], d["audio"], d["pos"])
        mse = mx.mean(mx.square(pred - d["true_hidden"]))
        logits = target.decoder.token_embedding.as_linear(pred)
        ce = mx.mean(nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), d["true_tokens"].reshape(-1)))
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
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}/{EPOCHS} Loss={ls/len(train_data):.5f}")

    # Greedy baseline
    print(f"\n--- Greedy ---")
    gw = [jiwer.wer(norm(s["text"]), norm(generate_greedy(target, tokenizer,
        mx.array(log_mel_spectrogram(np.array(s["audio"]["array"], dtype=np.float32),
            n_mels=target.dims.n_mels, padding=16000*30-len(np.array(s["audio"]["array"], dtype=np.float32)))[None], dtype=mx.float32))))
        for s in [ds[i] for i in range(N_TRAIN, N_TRAIN+10)]]
    for i, w in enumerate(gw):
        print(f"  [{N_TRAIN+i}] WER={w:.4f}")
    print(f"  -> Mean WER={np.mean(gw):.4f}")

    # Sweep top-k values
    for topk in TOP_K:
        print(f"\n--- Multi-path (top-{topk} per position) ---")
        ws, acs, tots = [], [], []
        for i in range(N_TRAIN, N_TRAIN + 10):
            s = ds[i]
            audio = np.array(s["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000*30-len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)
            t1 = time.time()
            text, acc, tot = generate_speculative_multipath(
                target, draft, tokenizer, mel_mx, top_k=topk)
            el = time.time() - t1
            w = jiwer.wer(norm(s["text"]), norm(text)) if s["text"].strip() else 1.0
            ws.append(w); acs.append(acc); tots.append(tot)
            ar = acc / max(tot, 1) * 100
            print(f"  [{i}] WER={w:.4f} accept={acc}/{tot} ({ar:.1f}%) time={el:.1f}s")
        mw = np.mean(ws); ma = sum(acs)/max(sum(tots),1)*100
        print(f"  -> WER={mw:.4f} Accept={ma:.1f}%")


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"\nTotal: {time.time()-t_start:.0f}s")
