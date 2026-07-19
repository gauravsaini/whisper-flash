"""whisper-flash: clean greedy ASR with production optimisations (Q8, KV cache).

Minimal usage:
    from whisper_flash_mlx import transcribe

    result = transcribe("audio.wav")
    print(result.text)

    result = transcribe("audio.wav", model="turbo", quantize=True)
    print(result.text)  # 3.94x faster than large-v3-fp16
"""

from .production import GreedyDecoder, ProductionConfig, DecodeResult
from .stride import StridedEncoder, apply_stride, restore_encoder, is_wrapped, encoder_forward_with_stride
from .parallel import parallel_transcribe, split_audio, parallel_decode, decode_sequential

MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

def transcribe(
    audio_path: str,
    model: str = "turbo",
    quantize: bool = True,
    max_new_tokens: int = 448,
) -> DecodeResult:
    """Transcribe audio with Whisper-Flash production decoder.

    Args:
        audio_path: Path to WAV/MP3/FLAC/ogg file.
        model: Model size or HuggingFace repo ID.
               Shorthand names: "tiny", "large-v3", "turbo" (default).
        quantize: Apply Q8 quantization (lossless, ~1.2-1.3x speedup).
        max_new_tokens: Max tokens to generate.

    Returns:
        DecodeResult with .text, .token_ids, .tokens_per_sec, .wall_time_s.
    """
    model_path = MODEL_ALIASES.get(model, model)
    cfg = ProductionConfig(
        model_path=model_path,
        quantize=quantize,
    )
    dec = GreedyDecoder(cfg)
    return dec.decode(audio_path, max_new_tokens=max_new_tokens)


__all__ = [
    "GreedyDecoder",
    "ProductionConfig",
    "DecodeResult",
    "transcribe",
    "StridedEncoder",
    "apply_stride",
    "restore_encoder",
    "is_wrapped",
    "encoder_forward_with_stride",
    "parallel_transcribe",
    "split_audio",
    "parallel_decode",
    "decode_sequential",
]
