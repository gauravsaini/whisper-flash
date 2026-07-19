# Whisper-Flash

Systems-level acceleration for OpenAI's Whisper via **stride-8 avg-pool encoder compression** — 8–11× speedup with byte-identical output on Apple Silicon (MLX).

## Core Result

| Model | Stride-1 | Stride-8 | Speedup | Text Match |
|-------|----------|----------|:-------:|:----------:|
| whisper-base | 0.641s | 0.079s | **8.11×** | ✅ IDENTICAL |
| whisper-medium | 0.899s | 0.084s | **10.71×** | ✅ IDENTICAL |
| large-v3-turbo | 1.232s | 0.144s | **8.56×** | ✅ IDENTICAL |

The decoder's cross-attention is memory-bandwidth-bound on Apple Silicon. Stride-8 cuts K/V frames from 1500→188 (8× less memory traffic), translating directly to 8–11× wall-clock speedup.

**Why this works:** Avg-pool after layernorm preserves positional embedding indices (unlike strided-conv which truncates). The `mlx_whisper.transcribe()` pipeline's temperature fallback handles low-confidence repetition loops that occasionaly arise with 188 frames — manual greedy decode loops cannot replicate this.

## Stride-8 Validation (MLX)

```bash
# Validate any audio file with stride-8 compression
uv run python3 -m whisper_flash_mlx.validate_stride audio.wav --model mlx-community/whisper-large-v3-turbo --stride 8
```

## CoreML Model Generation

Generate stride-8 CoreML models deployable to Apple Silicon via WhisperKit/Hex:

```bash
# Requires: whisperkittools repo + Python 3.12+ with uv
python convert_to_stride8.py \
  --model-version openai/whisper-large-v3-turbo \
  --output-dir ./output \
  --stride 8
```

The script:
1. Patches `whisperkittools` audio_encoder to add avg-pool after layernorm
2. Generates 3 CoreML models: `AudioEncoder.mlmodelc`, `TextDecoder.mlmodelc`, `MelSpectrogram.mlmodelc`
3. Restores all source files to original state
4. TextDecoder reads `encoder_output_embeds` shape dynamically — **zero code changes needed** in WhisperKit or the host app

Supported models: any HuggingFace Whisper variant (`openai/whisper-tiny`, `openai/whisper-base`, `openai/whisper-medium`, `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`, custom fine-tuned checkpoints).

## Installation

```bash
uv sync
```

## Paper

See `paper.tex` / `paper.pdf` for the full research document covering speculative ASR decoding, branching ceiling, spectral collapse, Δz correction (falsified), and stride-8 avg-pool compression.
