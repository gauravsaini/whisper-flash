"""P39: Trained Linear Probe for Adaptive Decoder Depth Early-Exit.

Trains a lightweight linear probe (shape 384x1) on intermediate hidden states
to predict if the intermediate layer's prediction matches the final layer's
prediction. If matched, we exit early and skip later layers.
"""

import time
import json
import tempfile
import os
import sys
import io
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pyarrow as pa
from jiwer import wer as _wer

from whisper_flash_mlx.target_model import (
    decoder_forward_with_hidden_states,
    load_target_model,
)
from whisper_flash_mlx.utils import sample
from mlx_whisper.tokenizer import get_tokenizer

EOS_ID, SOT_ID = 50257, 50258
DUMMY_ARROW = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"


class EarlyExitProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.probe = nn.Linear(d_model, 1)

    def __call__(self, x):
        return self.probe(x)


def load_dummy(n):
    with pa.memory_map(DUMMY_ARROW) as src:
        try:
            t = pa.ipc.open_file(src).read_all()
        except Exception:
            t = pa.ipc.open_stream(src).read_all()
    return t.to_pylist()[:n]


def wav_bytes_to_mel(b):
    arr, sr = sf.read(io.BytesIO(b))
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    from mlx_whisper.audio import log_mel_spectrogram
    mel = log_mel_spectrogram(arr, n_mels=80, padding=16000 * 30 - len(arr))
    return mx.array(mel)[None]


import soundfile as sf


def greedy_collect(model, mel, max_new=200):
    """Greedy-generate and return (seq, per-step list of (hiddens, y_inter, y_final))."""
    enc = model.encoder(mel)
    mx.eval(enc)
    seq = [SOT_ID]
    kv = None
    data = []
    
    ln = model.decoder.ln
    head = model.decoder.token_embedding.as_linear
    
    while len(seq) < max_new:
        inp = mx.array([seq], dtype=mx.int32) if kv is None else mx.array([[seq[-1]]], dtype=mx.int32)
        logits, kv, hiddens = decoder_forward_with_hidden_states(
            model, inp, enc, kv_cache=kv, collect_hidden_states=True)
        tok = sample(logits[:, -1:, :], 0.0).item()
        
        # Check agreement for all intermediate layers
        step_agreement = []
        for li in range(1, len(model.decoder.blocks)):
            h_int = hiddens[li]
            y_inter = mx.argmax(head(ln(h_int[:, -1, :])), axis=-1).item()
            step_agreement.append((h_int[:, -1, :], y_inter, tok))
            
        data.append(step_agreement)
        seq.append(tok)
        if tok == EOS_ID:
            break
            
    del enc, kv
    mx.clear_cache()
    return seq, data


import mlx.optimizers as optim

def train_probe(model, layer_idx, rows, epochs=100):
    d = model.dims.n_text_state
    probe = EarlyExitProbe(d)
    opt = optim.Adam(learning_rate=0.01)
    
    Xs, Ys = [], []
    for r in rows:
        mel = wav_bytes_to_mel(r["audio"]["bytes"])
        _, data = greedy_collect(model, mel)
        for step_agreement in data:
            h_val, y_inter, y_final = step_agreement[layer_idx - 1]
            Xs.append(h_val[0])
            Ys.append(1.0 if y_inter == y_final else 0.0)
            
    X = mx.stack(Xs)
    Y = mx.array(Ys, dtype=mx.float32)[:, None]
    mx.eval(X, Y)
    print(f"  layer {layer_idx}: {X.shape[0]} pairs, agreement rate: {mx.mean(Y).item()*100:.1f}%")

    loss_fn = nn.losses.binary_cross_entropy

    def loss_step(model_probe, x, y):
        pred = model_probe(x)
        return mx.mean(loss_fn(pred, y))
        
    loss_and_grad = mx.value_and_grad(loss_step)

    for ep in range(epochs):
        loss, grad = loss_and_grad(probe, X, Y)
        opt.update(probe, grad)
        mx.eval(probe, loss)
        if ep % 20 == 0:
            print(f"    ep {ep} loss={loss.item():.4f}")

    preds = mx.sigmoid(probe(X)) > 0.5
    acc = mx.mean(preds == (Y > 0.5)).item()
    print(f"  layer {layer_idx} TRAIN probe accuracy={acc*100:.1f}%")
    return probe, acc


def decode_early_exit(model, mel, probe, exit_layer, threshold, max_new=448):
    enc = model.encoder(mel)
    mx.eval(enc)
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv, _ = decoder_forward_with_hidden_states(model, dec, enc, kv_cache=None)
    tok = sample(logits[:, -1:, :], 0.0).item()
    seq = [SOT_ID, tok]
    exited = 0
    ln = model.decoder.ln
    head = model.decoder.token_embedding.as_linear
    
    while len(seq) < max_new:
        inp = mx.array([[seq[-1]]], dtype=mx.int32)
        pos = len(seq)
        
        # We only run the blocks up to the exit layer!
        if probe is not None and exit_layer < len(model.decoder.blocks):
            # Run first blocks up to exit_layer
            h = model.decoder.token_embedding(inp) + model.decoder.positional_embedding[pos : pos + 1]
            if kv is None:
                kv = [None] * len(model.decoder.blocks)
                
            for e in range(exit_layer + 1):
                h, kv[e], _ = model.decoder.blocks[e](
                    h, enc, mask=model.decoder._mask, kv_cache=kv[e]
                )
                
            # Run probe on the last token's representation
            state_vec = h[:, -1, :]
            probe_prob = mx.sigmoid(probe(state_vec)).item()
            
            if probe_prob >= threshold:
                # Exit early! Project intermediate state to logits
                logits_step = head(ln(h))
                tok = int(mx.argmax(logits_step, axis=-1).item())
                exited += 1
            else:
                # Run remaining layers
                for e in range(exit_layer + 1, len(model.decoder.blocks)):
                    h, kv[e], _ = model.decoder.blocks[e](
                        h, enc, mask=model.decoder._mask, kv_cache=kv[e]
                    )
                logits_step = head(ln(h))
                tok = int(mx.argmax(logits_step, axis=-1).item())
        else:
            # Standard decoding
            logits, kv, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv, offset=pos
            )
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            
        seq.append(tok)
        if tok == EOS_ID:
            break
            
    del enc, kv
    mx.clear_cache()
    return seq, exited


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny-mlx"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = load_dummy(10)
    train_rows = rows[:n_train]
    eval_rows = rows[n_train:n_train + 5]
    model = load_target_model(repo)
    L = len(model.decoder.blocks)
    
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )

    refs = [r["text"].strip().lower() for r in eval_rows]

    # Evaluate Baseline (Standard 4-layer model)
    hyps_base = []
    t0 = time.perf_counter()
    for r in eval_rows:
        mel = wav_bytes_to_mel(r["audio"]["bytes"])
        seq, _ = decode_early_exit(model, mel, None, -1, 0.0)
        toks = [t for t in seq[1:] if t < 50257]
        hyps_base.append(tokenizer.decode(toks).lower())
    dt_base = time.perf_counter() - t0
    wer_base = float(np.mean([_wer(refs[i], hyps_base[i]) for i in range(len(refs))]))
    print(f"\nBaseline (full)  time={dt_base:.3f}s wer={wer_base:.4f}")

    # Sweep each exit layer and threshold
    for li in range(1, L):
        print(f"\n{'='*60}\nEvaluating Exit Layer {li}/{L-1}\n{'='*60}")
        pr, agree = train_probe(model, li, train_rows)
        
        for threshold in [0.5, 0.8, 0.95]:
            hyps = []
            exit_counts = []
            total_steps = []
            t0 = time.perf_counter()
            for r in eval_rows:
                mel = wav_bytes_to_mel(r["audio"]["bytes"])
                seq, exited = decode_early_exit(model, mel, pr, li, threshold)
                toks = [t for t in seq[1:] if t < 50257]
                hyps.append(tokenizer.decode(toks).lower())
                exit_counts.append(exited)
                total_steps.append(len(seq) - 1)
            dt = time.perf_counter() - t0
            w = float(np.mean([_wer(refs[i], hyps[i]) for i in range(len(refs))]))
            exit_pct = (sum(exit_counts) / sum(total_steps)) * 100 if sum(total_steps) > 0 else 0
            sp = dt_base / dt
            lossless = "YES" if w <= wer_base + 0.01 else "NO"
            print(f"  th={threshold:<4} time={dt:.3f}s speedup={sp:.3f}x wer={w:.4f} exit_pct={exit_pct:.1f}% lossless={lossless}")


if __name__ == "__main__":
    main()
