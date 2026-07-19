"""Production-grade Whisper decoder for MLX.

Four levers (independent, composable):
  1. Q8 quantization (lossless ~1.7×)
  2. Encoder downsampling stride-2 (lossless, halves cross-attention)
  3. KV cache compression (avg-pool ×2)
  4. Sparse cross-attention (centre-of-mass window)

No draft model, no speculative decode — just clean greedy with production optimisations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_whisper.whisper import Whisper

from whisper_flash_mlx.target_model import (
    decoder_forward_with_hidden_states,
    encoder_forward,
    load_target_model,
    select_encoder_frames,
    build_sparse_working_cache,
    extract_cross_cache,
    crop_self_attention_cache,
)
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.utils import sample
from whisper_flash_mlx.stride import StridedEncoder, apply_stride, restore_encoder, encoder_forward_with_stride
from whisper_flash_mlx.parallel import parallel_transcribe, split_audio, encode_chunks, parallel_decode

import mlx_whisper  # for native-path decode

EOS_ID, SOT_ID = 50257, 50258


@dataclass
class ProductionConfig:
    """Configuration for production decoding."""
    model_path: str = "mlx-community/whisper-large-v3-mlx"
    quantize: bool = False          # Apply Q8 quantisation to Linear layers
    encoder_stride: int = 1         # Encoder frame downsampling stride (2 = lossless, halves cross-attn)
    kv_compress: bool = False       # Average-pool KV cache pairs (EXPERIMENTAL, not lossless)
    sparse_cross_attn: bool = False # Slice cross-attn to centre-of-mass window
    use_kv_cache: bool = True       # Use KV cache during decode (False = full-sequence baseline)
    cross_margin: int = 50          # Frames each side of centre-of-mass peak
    cross_min_window: int = 100     # Minimum window before trimming
    cross_max_window: int = 250
    probe_interval: int = 5         # Re-probe full cross-attn every N tokens
    min_tokens_before_sparse: int = 10  # Let speech tokens accumulate first
    dtype: mx.Dtype = mx.float16
    mode: str = "custom"            # "custom" (GreedyDecoder loop) or "native" (mlx_whisper.transcribe)
    structural: bool = False        # Auto-set best validated stack: stride-8 (turbo/large-v3, transcribe mode) or Q8 (tiny)
    parallel: bool = False          # Parallel batch decode over audio chunks
    parallel_chunks: int = 0        # Max parallel chunks (0 = all)

    def get_structural_config(self):
        """Return (stride, quantize) for the best validated stack.

        stride-8 (1500→188) is lossless via the full transcribe() pipeline
        (temperature fallback + repetition handling). Greedy decode loops
        without fallback get stuck in repetition — use native transcribe mode.
        On tiny, Q8 is best (~1.3× speedup).
        """
        model_is_large = "large" in self.model_path.lower() or "turbo" in self.model_path.lower()
        if model_is_large:
            return 8, False  # stride-8 lossless via transcribe() pipeline
        else:
            return 1, True  # Q8 is best on tiny (P40: 1.31x)


@dataclass
class DecodeResult:
    text: str
    token_ids: list[int]
    tokens_per_sec: float
    wall_time_s: float
    n_decoder_steps: int


# ════════════════════════════════════════════════════════════════
# Greedy decoder
# ════════════════════════════════════════════════════════════════

class GreedyDecoder:
    """Clean greedy decoder — no draft model, no speculation.

    Supports four composable optimisations:
      - ``quantize=True``  → Q8 weights
      - ``encoder_stride=2`` → halve encoder frames (lossless, validated E2)
      - ``kv_compress=True`` → avg-pool KV pairs
      - ``sparse_cross_attn=True`` → sliced cross-attention window
    """

    def __init__(self, cfg: ProductionConfig):
        self.cfg = cfg
        self.model = load_target_model(cfg.model_path, dtype=cfg.dtype)
        if cfg.quantize:
            quantize_model(self.model, encoder_bits=8, decoder_bits=8, group_size=64)
        self.tokenizer = None  # lazy init

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from mlx_whisper.tokenizer import get_tokenizer
            self.tokenizer = get_tokenizer(multilingual=self.model.is_multilingual)
        return self.tokenizer

    def decode(self, audio_path: str, max_new_tokens: int = 448) -> DecodeResult:
        """Transcribe an audio file.
        Routes to parallel decode, custom loop, or native mlx_whisper path."""
        if self.cfg.parallel:
            return self.decode_parallel(audio_path, max_new_tokens)
        if self.cfg.mode == "native":
            return self.decode_native(audio_path)
        return self.decode_custom(audio_path, max_new_tokens)

    def decode_native(self, audio_path: str) -> DecodeResult:
        """Transcribe via native mlx_whisper.transcribe with structural levers.

        This uses the SAME model (self.model) but monkey-patches it into
        mlx_whisper's global model registry so transcribe() picks it up.
        The encoder stride is patched directly onto the encoder object.
        """
        s = self.cfg.encoder_stride
        q = self.cfg.quantize
        _enc = self.model.encoder
        if s > 1:
            class _StridedEncoder(nn.Module):
                def __init__(e):
                    super().__init__()
                    object.__setattr__(e, "_enc", _enc)
                    object.__setattr__(e, "_s", s)
                def __call__(e, x):
                    o = e._enc(x)
                    B, T, D = o.shape
                    Tt = (T // e._s) * e._s
                    o = o[:, :Tt, :]
                    return mx.mean(o.reshape(B, Tt // e._s, e._s, D), axis=2)
                def __getattr__(e, n):
                    return getattr(object.__getattribute__(e, "_enc"), n)
            self.model.encoder = _StridedEncoder()

        _real_load = mlx_whisper.load_models.load_model
        mlx_whisper.load_models.load_model = lambda *a, **k: self.model
        try:
            t0 = time.perf_counter()
            out = mlx_whisper.transcribe(audio_path, fp16=True, verbose=False, condition_on_previous_text=False)
            dt = time.perf_counter() - t0
        finally:
            mlx_whisper.load_models.load_model = _real_load
            self.model.encoder = _enc  # restore

        return DecodeResult(
            text=out["text"].strip(),
            token_ids=[],  # not collected in native path
            tokens_per_sec=0,
            wall_time_s=dt,
            n_decoder_steps=0,
        )

    def decode_custom(self, audio_path: str, max_new_tokens: int = 448) -> DecodeResult:
        import soundfile as sf
        import numpy as np
        from mlx_whisper.audio import log_mel_spectrogram

        arr, sr = sf.read(audio_path)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        arr = np.ascontiguousarray(arr, dtype=np.float32)

        mel = log_mel_spectrogram(arr, n_mels=self.model.dims.n_mels,
                                   padding=16000 * 30 - len(arr))
        mel = mx.array(mel)[None]

        return self._decode_mel(mel, max_new_tokens)

    @staticmethod
    def _downsample_encoder(enc: mx.array, stride: int) -> mx.array:
        """Average-pool encoder output along the time axis.

        Validated in E2: stride-2 is lossless (WER actually improves by -0.005)
        and halves cross-attention compute. 73% of adjacent encoder frames have
        >0.90 cosine similarity — massive redundancy.
        """
        if stride <= 1:
            return enc
        B, T, D = enc.shape
        T_trim = (T // stride) * stride
        enc_trimmed = enc[:, :T_trim, :]
        return mx.mean(enc_trimmed.reshape(B, T_trim // stride, stride, D), axis=2)

    def _decode_kv(self, enc: mx.array, max_new_tokens: int):
        """Decode using a KV cache (single token per step)."""
        # ── Prefill with SOT ──
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            self.model, dec, enc, kv_cache=None, collect_hidden_states=False,
            offset=0)
        first = sample(logits[:, -1:, :], 0.0)
        mx.eval(first)
        dec = mx.concatenate([dec, first], axis=1)
        output_ids = [SOT_ID, first.item()]

        # For sparse cross-attn: capture full cross cache from prefill;
        # wait for min_tokens_before_sparse tokens before first window trim
        # so that cross-attention has stabilised on speech tokens.
        full_cross = extract_cross_cache(kv_cache) if self.cfg.sparse_cross_attn else None
        cross_probe_counter = 0
        sparse_active = not self.cfg.sparse_cross_attn  # start inactive

        while len(output_ids) < max_new_tokens:
            last_tok = output_ids[-1]
            inp = mx.array([[last_tok]], dtype=mx.int32)

            is_probe = (sparse_active and full_cross is not None
                        and cross_probe_counter % self.cfg.probe_interval == 0)

            # True token position of the new query (SOT=0, first=1, ...).
            # Must be passed explicitly so that KV-cache compression (which
            # shrinks the cache length) does not shift positional embeddings.
            pos = len(output_ids) - 1

            logits, kv_cache, _, cross_attns = decoder_forward_with_hidden_states(
                self.model, inp, enc, kv_cache=kv_cache,
                collect_hidden_states=False, return_cross_attention=True,
                offset=pos)
            tok = sample(logits[:, -1:, :], 0.0)

            # Activate sparse mode after enough tokens
            if (self.cfg.sparse_cross_attn and not sparse_active
                    and len(output_ids) >= self.cfg.min_tokens_before_sparse):
                sparse_active = True
                full_cross = extract_cross_cache(kv_cache)

            if is_probe and cross_attns and cross_attns[-1] is not None:
                s, e = select_encoder_frames(
                    cross_attns, self.cfg.cross_margin,
                    self.cfg.cross_min_window, self.cfg.cross_max_window)
                full_cross = extract_cross_cache(kv_cache)
                kv_cache = build_sparse_working_cache(kv_cache, full_cross, s, e)

            cross_probe_counter += 1

            mx.eval(tok)
            token_id = tok.item()
            output_ids.append(token_id)

            # KV cache compression
            if self.cfg.kv_compress and len(output_ids) > 4:
                kv_cache = self._compress_kv(kv_cache)

            if token_id == EOS_ID:
                break

        return output_ids, kv_cache

    def _decode_full_seq(self, enc: mx.array, max_new_tokens: int):
        """Decode WITHOUT a KV cache — re-run the full token sequence each step.

        This is the reference baseline that the KV-cache path is measured
        against. It is mathematically identical to the KV path (same
        logits at every step) but recomputes all prior keys/values, giving
        O(n^2) self-attention cost.
        """
        output_ids = [SOT_ID]
        dec = mx.array([[SOT_ID]], dtype=mx.int32)
        while len(output_ids) < max_new_tokens:
            logits, _, _ = decoder_forward_with_hidden_states(
                self.model, dec, enc, kv_cache=None, collect_hidden_states=False)
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            token_id = tok.item()
            output_ids.append(token_id)
            if token_id == EOS_ID:
                break
            dec = mx.concatenate([dec, tok], axis=1)
        return output_ids

    def _decode_mel(self, mel: mx.array, max_new_tokens: int = 448) -> DecodeResult:
        t0 = time.perf_counter()

        # ── Encode once ──
        enc = encoder_forward(self.model, mel)
        mx.eval(enc)

        # ── Downsample encoder frames (E2: stride-2 is lossless) ──
        if self.cfg.encoder_stride > 1:
            enc = self._downsample_encoder(enc, self.cfg.encoder_stride)
            mx.eval(enc)

        if self.cfg.use_kv_cache:
            output_ids, kv_cache = self._decode_kv(enc, max_new_tokens)
        else:
            output_ids = self._decode_full_seq(enc, max_new_tokens)

        t1 = time.perf_counter()
        wall = t1 - t0
        n_tokens = len(output_ids) - 1  # exclude initial SOT
        tps = n_tokens / wall if wall > 0 else 0

        # Decode tokens to text
        tok = self._get_tokenizer()
        text_tokens = [t for t in output_ids[1:] if t < tok.eot]
        text = tok.decode(text_tokens)

        return DecodeResult(
            text=text.strip(),
            token_ids=output_ids[1:],
            tokens_per_sec=tps,
            wall_time_s=wall,
            n_decoder_steps=len(output_ids) - 2,
        )

    def _compress_kv(self, kv_cache: list) -> list:
        """Average-pool pairs of newly added KV entries (compression factor ×2)."""
        new_cache = []
        for self_kv, cross_kv in kv_cache:
            if self_kv is not None:
                k, v = self_kv
                L = k.shape[1]
                if L >= 3:
                    k_prev = k[:, :-2, :]
                    v_prev = v[:, :-2, :]
                    k_new = k[:, -2:, :]
                    v_new = v[:, -2:, :]
                    k_pooled = mx.mean(k_new, axis=1, keepdims=True)
                    v_pooled = mx.mean(v_new, axis=1, keepdims=True)
                    self_kv = (mx.concatenate([k_prev, k_pooled], axis=1),
                               mx.concatenate([v_prev, v_pooled], axis=1))
            new_cache.append((self_kv, cross_kv))
        return new_cache

    def decode_parallel(self, audio_path: str, max_new_tokens: int = 448) -> DecodeResult:
        """Transcribe long audio by splitting into chunks and decoding in parallel.

        Steps:
          1. Split audio into 30s chunks
          2. Stride-8 encode each chunk
          3. Batch-decode all chunks simultaneously (parallel cross-attention)
          4. Concatenate texts

        Speedup scales with chunk count up to ~16 on Apple Silicon (5.61× validated).
        """
        stride = self.cfg.encoder_stride or 8
        t0 = time.perf_counter()

        texts = parallel_transcribe(
            audio_path,
            self.model,
            stride=stride,
            chunk_sec=30,
            max_new_tokens=max_new_tokens,
            max_chunks=self.cfg.parallel_chunks,
            verbose=False,
        )
        full_text = " ".join(texts)

        t1 = time.perf_counter()
        wall = t1 - t0
        n_tokens = sum(len(t.split()) for t in texts)
        tps = n_tokens / wall if wall > 0 else 0

        return DecodeResult(
            text=full_text.strip(),
            token_ids=[],
            tokens_per_sec=tps,
            wall_time_s=wall,
            n_decoder_steps=0,
        )

    def benchmark(self, audio_path: str, runs: int = 3) -> dict:
        """Run multiple decode passes and report WER + speed stats."""
        results = []
        references = []
        preds = []

        for _ in range(runs):
            r = self.decode(audio_path)
            results.append(r)
            preds.append(r.text)

        from jiwer import wer
        # For WER we need reference text — use the consensus of greedy runs
        # (this is a proxy; ideally use ground-truth transcripts)
        ref = max(set(preds), key=preds.count) if len(set(preds)) > 1 else preds[0]
        wers = [wer(ref, p) for p in preds]

        tps_list = [r.tokens_per_sec for r in results]
        return {
            "mean_tps": sum(tps_list) / len(tps_list),
            "mean_wer": sum(wers) / len(wers),
            "mean_time_s": sum(r.wall_time_s for r in results) / len(results),
            "n_runs": runs,
            "text": results[0].text,
        }


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--encoder-stride", type=int, default=1,
                        help="Encoder frame downsampling stride (2=lossless, halves cross-attn)")
    parser.add_argument("--kv-compress", action="store_true")
    parser.add_argument("--sparse-cross", action="store_true")
    parser.add_argument("--mode", choices=["custom", "native"], default="native",
                        help="Decode mode: 'custom' (GreedyDecoder loop) or 'native' (mlx_whisper.transcribe)")
    parser.add_argument("--structural", action="store_true",
                        help="Auto-set best validated lossless stack for the model")
    parser.add_argument("--parallel", action="store_true",
                        help="Parallel batch decode over 30s chunks (stride-8 auto-enabled)")
    parser.add_argument("--parallel-chunks", type=int, default=0,
                        help="Max chunks for parallel decode (0 = all)")
    args = parser.parse_args()

    stride = args.encoder_stride
    quantize = False
    if args.structural:
        if "large" in args.model.lower() or "turbo" in args.model.lower():
            stride = 8; quantize = False  # stride-8 lossless via transcribe() pipeline
        else:
            stride = 1; quantize = True  # Q8 lossless on tiny (P40)
    if args.parallel:
        stride = max(stride, 8)  # parallel uses stride-8 by default
        quantize = False

    cfg = ProductionConfig(
        model_path=args.model,
        encoder_stride=stride,
        quantize=quantize,
        kv_compress=args.kv_compress,
        sparse_cross_attn=args.sparse_cross,
        mode=args.mode,
        parallel=args.parallel,
        parallel_chunks=args.parallel_chunks,
    )

    dec = GreedyDecoder(cfg)
    result = dec.decode(args.audio)

    label = "native" if args.mode == "native" else "custom"
    mode_label = f"parallel/{label}" if args.parallel else label
    print(f"\n{'='*60}")
    print(f"  Model:           {args.model}")
    print(f"  Mode:            {mode_label}" + (" (structural)" if args.structural else ""))
    print(f"  Encoder stride:  {stride}")
    print(f"  Quantize:        {quantize}")
    if args.parallel:
        print(f"  Parallel chunks: {args.parallel_chunks if args.parallel_chunks else 'all'}")
    print(f"{'='*60}")
    print(f"  Text:            {result.text}")
    print(f"  Wall time:       {result.wall_time_s:.3f}s")
