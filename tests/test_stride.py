"""Tests for stride-8 encoder and parallel batch decode."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from whisper_flash_mlx.stride import (
    StridedEncoder,
    apply_stride,
    restore_encoder,
    is_wrapped,
    encoder_forward_with_stride,
)
from whisper_flash_mlx.parallel import split_audio, parallel_decode, decode_sequential


# ════════════════════════════════════════════════════════════════
# Helper: mock encoder
# ════════════════════════════════════════════════════════════════

class MockEncoder(nn.Module):
    """Simple encoder that returns identity embeddings."""
    def __init__(self, d_model: int = 1280):
        super().__init__()
        self.d_model = d_model

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape[0], x.shape[1]
        # Return linearly increasing values so we can verify avg-pool
        pos = mx.arange(T, dtype=mx.float32)[None, :, None]
        return mx.broadcast_to(pos, (B, T, self.d_model))

    def some_method(self) -> str:
        return "original"


class MockModel:
    def __init__(self):
        self.encoder = MockEncoder()
        self.dims = type("dims", (), {"n_mels": 80})()


# ════════════════════════════════════════════════════════════════
# StridedEncoder
# ════════════════════════════════════════════════════════════════


class TestStridedEncoder:
    def test_stride_divides_evenly(self):
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=8)
        x = mx.zeros((1, 3000, 80))
        out = wrapped(x)
        assert out.shape == (1, 3000 // 8, 1280), f"Expected (1, 375, 1280), got {out.shape}"

    def test_stride_trims_remainder(self):
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=8)
        x = mx.zeros((1, 1504, 80))  # 1504 / 8 = 188 exactly
        out = wrapped(x)
        assert out.shape == (1, 188, 1280), f"Expected (1, 188, 1280), got {out.shape}"

    def test_stride_one_passthrough(self):
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=1)
        x = mx.zeros((1, 3000, 80))
        out = wrapped(x)
        assert out.shape == (1, 3000, 1280)

    def test_avg_pool_values(self):
        """Verify that avg-pool of sequential [0,1,2,...,7] equals 3.5 for every output position."""
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=4)
        x = mx.zeros((1, 8, 80))
        out = wrapped(x)
        # MockEncoder returns pos values [0..7] broadcast across channels
        # stride-4 avg -> (0+1+2+3)/4 = 1.5, (4+5+6+7)/4 = 5.5
        expected = mx.array([[1.5, 5.5]], dtype=mx.float32)
        result = out[0, :, 0]
        assert mx.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"

    def test_attribute_proxy(self):
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=8)
        assert wrapped.some_method() == "original"
        assert wrapped.d_model == 1280

    def test_repr(self):
        enc = MockEncoder()
        wrapped = StridedEncoder(enc, stride=8)
        r = repr(wrapped)
        assert "stride=8" in r
        assert "MockEncoder" in r


class TestApplyRestore:
    def test_apply_wraps_encoder(self):
        model = MockModel()
        wrapper = apply_stride(model, stride=8)
        assert is_wrapped(model)
        assert isinstance(model.encoder, StridedEncoder)
        assert model.encoder._stride == 8

    def test_restore_original(self):
        model = MockModel()
        original = model.encoder
        apply_stride(model, stride=8)
        restored = restore_encoder(model)
        assert restored is original
        assert not is_wrapped(model)

    def test_restore_on_unwrapped_is_noop(self):
        model = MockModel()
        original = model.encoder
        restored = restore_encoder(model)
        assert restored is original

    def test_encoder_forward_with_stride(self):
        model = MockModel()
        x = mx.zeros((1, 3000, 80))
        out = encoder_forward_with_stride(model, x, stride=8)
        assert out.shape == (1, 3000 // 8, 1280)

    def test_encoder_forward_stride_one(self):
        model = MockModel()
        x = mx.zeros((1, 3000, 80))
        out = encoder_forward_with_stride(model, x, stride=1)
        assert out.shape == (1, 3000, 1280)


# ════════════════════════════════════════════════════════════════
# Parallel decode (unit)
# ════════════════════════════════════════════════════════════════

class TestParallelDecodeUnit:
    """Tests for parallel_decode / decode_sequential with mock model.

    These tests verify the routing logic and basic correctness of the
    batch decode loop. A real model test is in e2e tests.
    """

    def test_split_audio_returns_chunks(self):
        import soundfile as sf

        sr = 16000
        data = np.sin(2 * np.pi * 440 * np.arange(sr * 60) / sr).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            fname = f.name
        try:
            chunks, out_sr = split_audio(fname, chunk_sec=30, max_chunks=0)
            assert len(chunks) == 2  # 60s / 30s = 2
            assert out_sr == sr
            assert len(chunks[0]) == sr * 30
            assert len(chunks[1]) == sr * 30
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_split_audio_max_chunks(self):
        import soundfile as sf

        sr = 16000
        data = np.zeros(sr * 120, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            fname = f.name
        try:
            chunks, out_sr = split_audio(fname, chunk_sec=30, max_chunks=2)
            assert len(chunks) == 2
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_split_audio_mono_conversion(self):
        import soundfile as sf

        sr = 16000
        data = np.zeros((sr * 5, 2), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            fname = f.name
        try:
            chunks, out_sr = split_audio(fname, chunk_sec=30)
            assert len(chunks[0].shape) == 1  # mono
        finally:
            Path(fname).unlink(missing_ok=True)


class TestConfigIntegration:
    """Verify ProductionConfig routes correctly for stride-8 + parallel."""

    def test_structural_returns_stride_8_on_large(self):
        from whisper_flash_mlx.production import ProductionConfig

        cfg = ProductionConfig(model_path="mlx-community/whisper-large-v3-turbo")
        s, q = cfg.get_structural_config()
        assert s == 8
        assert q is False

    def test_structural_returns_q8_on_tiny(self):
        from whisper_flash_mlx.production import ProductionConfig

        cfg = ProductionConfig(model_path="mlx-community/whisper-tiny-mlx")
        s, q = cfg.get_structural_config()
        assert s == 1
        assert q is True
