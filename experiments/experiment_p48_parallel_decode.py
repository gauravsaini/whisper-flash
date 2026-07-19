"""P48: Parallel multi-chunk decoding with stride-8 compressed encoder outputs.
"""

import time, sys, os, json
import numpy as np
import mlx.core as mx
import mlx_whisper
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram
import soundfile as sf
import io
import pyarrow as pa

from whisper_flash_mlx.target_model import decoder_forward_with_hidden_states
from whisper_flash_mlx.utils import sample

DUMMY = "/Users/ektasaini/.cache/huggingface/datasets/hf-internal-testing___librispeech_asr_dummy/clean/0.0.0/5be91486e11a2d616f4ec5db8d3fd248585ac07a/librispeech_asr_dummy-validation.arrow"


def build_long_audio():
    """Build long audio from concatenated dummy samples."""
    with pa.memory_map(DUMMY) as src:
        try: table = pa.ipc.open_file(src).read_all()
        except: table = pa.ipc.open_stream(src).read_all()
    rows = table.to_pylist()
    sigs, sr = [], None
    for r in rows:
        s, rs = sf.read(io.BytesIO(r["audio"]["bytes"]))
        if sr is None: sr = rs
        if rs != sr: continue
        sigs.append(s)
    os.makedirs("/tmp/flashbench", exist_ok=True)
    path = "/tmp/flashbench/long.wav"
    if not os.path.exists(path):
        sf.write(path, np.concatenate(sigs), sr)
    return path, sr


def load_chunks(audio_path, chunk_sec=30, max_chunks=None, n_mels=80):
    """Split audio into ~chunk_sec chunks, return (mel_arrays, sr)."""
    arr, sr = sf.read(audio_path)
    if arr.ndim == 2: arr = arr.mean(axis=1)
    total = len(arr)
    chunk_n = chunk_sec * sr
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_n, total)
        seg = np.ascontiguousarray(arr[start:end], dtype=np.float32)
        pad = chunk_n - len(seg)
        mel = log_mel_spectrogram(seg, n_mels=n_mels, padding=pad)
        chunks.append(mx.array(mel))
        start = end
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks, sr


def encode_chunks(model, chunks, stride=8):
    """Encode all chunks with stride reduction, return stacked tensor [K, T', D]."""
    encs = []
    for mel in chunks:
        e = model.encoder(mel[None])
        T = e.shape[1]; Tt = (T // stride) * stride
        e_st = mx.mean(e[:, :Tt, :].reshape(1, Tt // stride, stride, e.shape[-1]), axis=2)
        mx.eval(e_st)
        encs.append(e_st)
    stacked = mx.concatenate(encs, axis=0)
    return stacked


def decode_sequential(model, enc_chunks, max_new=448):
    """Decode chunks one-by-one with correct single-token-per-step."""
    e = enc_chunks[0]  # use first to infer shape
    enc_list = [e[:, None, :]] if e.ndim == 2 else [e[None]] if e.ndim == 3 else enc_chunks
    # enc_chunks is a list of [T', D] tensors
    hyps = []
    t0 = time.perf_counter()
    for enc in enc_chunks:
        if enc.ndim == 2:
            enc = enc[None]
        seq = [50258]
        kv = None
        while len(seq) < max_new:
            inp = mx.array([seq], dtype=mx.int32) if kv is None else mx.array([[seq[-1]]], dtype=mx.int32)
            logits, kv, _ = decoder_forward_with_hidden_states(model, inp, enc, kv_cache=kv, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0).item()
            seq.append(tok)
            if tok == 50257:
                break
        hyps.append(seq)
    t1 = time.perf_counter()
    return hyps, t1 - t0


def decode_parallel(model, enc_stacked, max_new=448):
    """Decode K chunks simultaneously. enc_stacked: [K, T', D]."""
    K = enc_stacked.shape[0]
    seqs = [[50258] for _ in range(K)]
    active = [True] * K
    kv = None
    t0 = time.perf_counter()
    for step in range(max_new - 1):
        if not any(active):
            break
        if kv is None:
            inp = mx.array([[50258]] * K, dtype=mx.int32)
        else:
            inp = mx.array([[seqs[i][-1] if active[i] else 50257] for i in range(K)], dtype=mx.int32)
        logits, kv, _ = decoder_forward_with_hidden_states(
            model, inp, enc_stacked, kv_cache=kv, collect_hidden_states=False, offset=step)
        for i in range(K):
            if active[i]:
                tok = sample(logits[i:i+1, -1:, :], 0.0).item()
                seqs[i].append(tok)
                if tok == 50257:
                    active[i] = False
    t1 = time.perf_counter()
    return seqs, t1 - t0


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/whisper-tiny-mlx"
    max_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    model = load_model(repo, mx.float16)
    n_mels = model.dims.n_mels

    audio_path, sr = build_long_audio()
    chunks, _ = load_chunks(audio_path, chunk_sec=30, max_chunks=max_chunks or None, n_mels=n_mels)
    K = len(chunks)
    print(f"Chunks: {K}  ({K * 30}s audio equivalent)", flush=True)

    enc_stacked = encode_chunks(model, chunks, stride=8)
    print(f"Encoded: {enc_stacked.shape}", flush=True)
    enc_list = [enc_stacked[i:i+1] for i in range(K)]

    # Warmup
    _ = decode_sequential(model, [enc_list[0]], max_new=50)
    _ = decode_parallel(model, enc_stacked[:1], max_new=50)
    mx.clear_cache()

    # Benchmark sequential
    s_seqs, t_seq = decode_sequential(model, enc_list, max_new=200)
    print(f"Sequential: {t_seq:.3f}s  ({K} chunks)", flush=True)

    mx.clear_cache()

    # Benchmark parallel
    p_seqs, t_par = decode_parallel(model, enc_stacked, max_new=200)
    print(f"Parallel:   {t_par:.3f}s  ({K} chunks)", flush=True)
    print(f"Speedup:    {t_seq / t_par:.3f}x", flush=True)

    # Verify correctness (first N tokens where both have data)
    matches = sum(
        1 for i in range(K)
        if s_seqs[i][:min(len(s_seqs[i]), len(p_seqs[i]))] == p_seqs[i][:min(len(s_seqs[i]), len(p_seqs[i]))]
    )
    print(f"Match:      {matches}/{K} ({100*matches//K}%)", flush=True)

    # Save results
    result = {
        "experiment": "p48",
        "model": repo,
        "chunks": K,
        "t_seq": round(t_seq, 3),
        "t_par": round(t_par, 3),
        "speedup": round(t_seq / t_par, 3),
        "match": f"{matches}/{K}"
    }
    res_path = "/Users/ektasaini/Desktop/whisper-flash/results/p48_result.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {res_path}", flush=True)


if __name__ == "__main__":
    main()
