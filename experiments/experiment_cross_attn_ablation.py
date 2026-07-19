#!/usr/bin/env python3
"""
Experiment #2: Single-variable cross-attention ablation.

The ONLY change between two otherwise identical runs. Tests whether removing
audio_ctx from the DFlashAttention KV sequence changes live acceptance.

Config: ContinuousDraftModel, block_size=4, 2 layers, span-level graph verification
Exactly the setup from Checkpoint 17 / Exp 11 that achieved 51.52% acceptance.

Two models trained identically:
  A) audio_ctx = self.audio_proj(audio_summary)  — normal
  B) audio_ctx = zeros_like                       — ablated

Evaluated in live speculative loop with M_graph >= 0.95.
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

def make_ablation_model(config, zero_audio=False):
    """Create a ContinuousDraftModel with optional audio_ctx ablation."""
    model = ContinuousDraftModel(config)
    orig_call = model.__call__

    def ablated_call(noise_embedding, target_hidden, audio_summary, position_ids, mask=None):
        x = model.input_proj(noise_embedding) + model.pos_embed(position_ids)
        ctx = model.hidden_norm(model.fc(target_hidden))
        if zero_audio:
            audio_ctx = mx.zeros_like(model.audio_proj(audio_summary))
        else:
            audio_ctx = model.audio_proj(audio_summary)
        for layer in model.layers:
            x = layer(x, ctx, audio_ctx, mask=mask)
        x = model.norm(x)
        return model.continuous_head(x)

    model.__call__ = ablated_call
    return model

def mse_loss_fn(model, noise, target_hidden, audio_summary, position_ids, true_hidden):
    pred = model(noise, target_hidden, audio_summary, position_ids)
    return mx.mean(mx.square(pred - true_hidden))

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

def generate_speculative(model, target, tokenizer, mel, max_tokens=100,
                         block_size=4, m_graph_thresh=0.95, tau=0.97):
    encoder_hidden = encoder_forward(target, mel)
    audio_summary = mx.mean(encoder_hidden, axis=1, keepdims=True)

    tokens = [tokenizer.sot]
    accepted, total_draft, steps = 0, 0, 0
    seen_sot = False

    while len(tokens) < max_tokens:
        steps += 1
        inp = mx.array([tokens], dtype=mx.int32)
        _, _, hidden_all = decoder_forward_with_hidden_states(
            target, inp, encoder_hidden, collect_hidden_states=True, return_cross_attention=False)

        # Build target_hidden from tapped layers [1, 2]
        target_hidden = mx.concatenate([hidden_all[1][:, -1:, :], hidden_all[2][:, -1:, :]], axis=-1)

        noise = target.decoder.token_embedding(
            mx.array([[model.config.mask_token_id] * block_size]))
        pos_ids = mx.arange(len(tokens), len(tokens) + block_size, dtype=mx.int32)[None]

        draft_hidden = model(noise, target_hidden, audio_summary, pos_ids)
        draft_logits = target.decoder.token_embedding.as_linear(draft_hidden)
        draft_tokens = mx.argmax(draft_logits, axis=-1)[0].tolist()

        spec_tokens = tokens + draft_tokens
        spec_inp = mx.array([spec_tokens], dtype=mx.int32)
        _, _, true_all = decoder_forward_with_hidden_states(
            target, spec_inp, encoder_hidden, collect_hidden_states=True, return_cross_attention=False)
        true_future = true_all[-1][:, len(tokens):, :]
        true_logits = target.decoder.token_embedding.as_linear(true_all[-1])
        true_next = mx.argmax(true_logits, axis=-1)[0].tolist()

        pn = np.array(draft_hidden[0])
        tn = np.array(true_future[0])
        nsim = node_similarity(pn, tn)
        tsim = topological_similarity(pn, tn)
        mg = 0.5 * nsim + 0.5 * tsim

        if steps <= 5:
            per_step = ", ".join(
                f"{float(np.dot(pn[k], tn[k])/((np.linalg.norm(pn[k])+1e-9)*(np.linalg.norm(tn[k])+1e-9))):.3f}"
                for k in range(block_size))
            print(f"  Step@{steps}: M_graph={mg:.4f} node={nsim:.4f} topo={tsim:.4f} [{per_step}]")

        accepted_k = 0
        if mg >= m_graph_thresh:
            for k in range(block_size):
                dp = mx.softmax(draft_logits[0, k])
                tp = mx.softmax(true_logits[0, len(tokens) + k - 1])
                if tp[draft_tokens[k]].item() >= tau * dp[draft_tokens[k]].item():
                    accepted_k += 1
                else:
                    break

        tokens.extend(draft_tokens[:accepted_k])
        tokens.append(true_next[len(tokens) - 1])
        accepted += accepted_k
        total_draft += block_size

        if tokens[-1] == tokenizer.eot:
            break

    return tokenizer.decode(tokens), accepted, total_draft

def run():
    t0 = time.time()
    BLOCK = 4
    EPOCHS = 15

    print("=" * 65)
    print("EXP #2: SINGLE-VARIABLE CROSS-ATTENTION ABLATION")
    print("=" * 65)

    target = load_target_model("mlx-community/whisper-tiny")
    tokenizer = get_tokenizer(target.is_multilingual, num_languages=target.num_languages)
    d_target = target.dims.n_text_state

    config = WhisperDFlashConfig(
        d_target=d_target, d_draft=256, num_layers=2, vocab_size=target.dims.n_vocab,
        block_size=BLOCK, target_layer_ids=[1, 2]
    )

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    print(f"\nExtracting {len(ds)} samples...")
    train_data = []
    for i in range(10):
        sample = ds[i]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text_tokens = tokenizer.encode(sample["text"])
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel_mx = mx.array(mel[None], dtype=mx.float32)
        labels = mx.concatenate([mx.array([[tokenizer.sot]], dtype=mx.int32),
                                 mx.array([text_tokens], dtype=mx.int32)], axis=1)
        enc_h = encoder_forward(target, mel_mx)
        audio_summ = mx.mean(enc_h, axis=1, keepdims=True)

        for t in range(1, labels.shape[1] - BLOCK, 3):
            inp_tok = labels[:, :t+1]
            _, _, h_all = decoder_forward_with_hidden_states(
                target, inp_tok, enc_h, collect_hidden_states=True, return_cross_attention=False)
            ctx = mx.concatenate([h_all[lid] for lid in [1, 2]], axis=-1)
            _, _, h_fut = decoder_forward_with_hidden_states(
                target, labels[:, :t+1+BLOCK], enc_h, collect_hidden_states=True, return_cross_attention=False)
            true_h = h_fut[-1][:, t:t+BLOCK, :]
            noise = target.decoder.token_embedding(
                mx.array([[config.mask_token_id] * BLOCK]))
            pos = mx.arange(t, t + BLOCK, dtype=mx.int32)[None]
            train_data.append({"noise": noise, "ctx": ctx, "audio": audio_summ,
                               "pos": pos, "true_hidden": true_h})

    print(f"  {len(train_data)} datapoints")

    for label, zero_audio in [("A: Cross-Attn ON", False), ("B: Cross-Attn OFF", True)]:
        print(f"\n{'='*65}")
        print(f"Training {label}")
        print(f"{'='*65}")

        model = make_ablation_model(config, zero_audio=zero_audio)
        # Initialize params
        d0 = train_data[0]
        _ = model(d0["noise"], d0["ctx"], d0["audio"], d0["pos"])

        opt = optim.Adam(learning_rate=1e-3)
        loss_and_grad = nn.value_and_grad(model, mse_loss_fn)

        for ep in range(EPOCHS):
            ls = 0.0
            for d in train_data:
                l, g = loss_and_grad(model, d["noise"], d["ctx"], d["audio"], d["pos"], d["true_hidden"])
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
                ls += l.item()
            if (ep+1) % 5 == 0:
                print(f"  Epoch {ep+1}/{EPOCHS} Loss={ls/len(train_data):.5f}")

        print(f"\n  Evaluating {label} in live speculative loop...")
        import jiwer
        total_wer, total_acc, total_dr = 0.0, 0, 0
        for i in range(10, 20):
            sample = ds[i]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
            mel_mx = mx.array(mel[None], dtype=mx.float32)
            text, acc, dr = generate_speculative(model, target, tokenizer, mel_mx)
            total_acc += acc; total_dr += dr
            norm_pred = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
                jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(text))))
            norm_ref = jiwer.RemoveMultipleSpaces()(jiwer.Strip()(
                jiwer.ToLowerCase()(jiwer.ExpandCommonEnglishContractions()(sample["text"]))))
            wer = jiwer.wer(norm_ref, norm_pred) if norm_ref else 1.0
            total_wer += wer
            print(f"    [{i}] WER={wer:.4f} accept={acc}/{dr} ({acc/max(dr,1)*100:.1f}%)")

        mw = total_wer / 10
        ma = total_acc / max(total_dr, 1) * 100
        print(f"\n  -> {label}: WER={mw:.4f}, Accept={ma:.2f}%")

    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    run()
