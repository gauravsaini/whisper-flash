"""Generate training dataset by running teacher-forced Whisper decoding — MLX version.

For each utterance in LibriSpeech:
  1. Compute mel spectrogram
  2. Run encoder → cache encoder_hidden_states
  3. Compute audio_summary = mean_pool(encoder_hidden)
  4. Run teacher-forced decoder with ground-truth tokens + output_hidden_states
  5. Extract decoder hidden states at tapped layers
  6. Save per-utterance: audio_summary, decoder_feats, token_ids

Runs entirely on-device via MLX — no GPU VM required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from .target_model import (
    decoder_forward_with_hidden_states,
    encoder_forward,
    load_target_model,
)
from .utils import build_target_layer_ids, extract_context_feature


def generate_dataset(
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    dataset_name: str = "librispeech_asr",
    dataset_config: str = "clean",
    dataset_split: str = "train.100",
    output_dir: str = "data/train",
    target_layer_ids: list[int] | None = None,
    max_samples: int | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """Generate training data from teacher-forced Whisper decoding (MLX).

    Args:
        model_name: HuggingFace model/repo identifier for mlx-whisper.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration (e.g. "clean").
        dataset_split: Dataset split (e.g. "train.100").
        output_dir: Directory to save NPZ files.
        target_layer_ids: Decoder layers to tap. Defaults to [8, 16, 24].
        max_samples: Limit number of samples (for debugging).
    """
    from datasets import load_dataset
    from mlx_whisper.audio import log_mel_spectrogram

    if target_layer_ids is None:
        target_layer_ids = [8, 16, 24]

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{dataset_config}/{dataset_split}")
    print(f"Target layers: {target_layer_ids}")

    # Load model
    print("Loading model...")
    model = load_target_model(model_name)

    # We need a tokenizer — use the whisper tokenizer
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages,
    )

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(ds)))
    if num_shards > 1:
        indices = [i for i in indices if i % num_shards == shard_id]
        print(f"Sharding dataset: processing {len(indices)} / {len(ds)} samples (shard {shard_id} of {num_shards})")
    else:
        print(f"Processing {len(ds)} utterances...")

    for idx in tqdm(indices):
        sample = ds[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        text = sample["text"]

        # Compute mel spectrogram using mlx-whisper's utility
        # mlx-whisper log_mel_spectrogram returns (frames, n_mels)
        # We must pad it to exactly 30 seconds for the positional embedding
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=16000 * 30 - len(audio))
        mel = mx.array(mel)[None]  # (1, n_mels, T)

        # Get ground-truth token ids (teacher forcing)
        text_tokens = tokenizer.encode(text)
        token_ids = mx.array([text_tokens], dtype=mx.int32)

        # Prepend SOT token
        sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
        decoder_input_ids = mx.concatenate([sot, token_ids[:, :-1]], axis=1)
        labels = token_ids

        # Run encoder
        encoder_hidden = encoder_forward(model, mel)

        # Audio summary
        audio_summary = mx.mean(encoder_hidden, axis=1)  # (1, d_model)

        # Run teacher-forced decoder with hidden state collection
        logits, _, all_hidden = decoder_forward_with_hidden_states(
            model, decoder_input_ids, encoder_hidden,
            kv_cache=None,
            collect_hidden_states=True,
        )

        # Extract hidden states from tapped layers
        decoder_feats = extract_context_feature(
            all_hidden, target_layer_ids
        )  # (1, seq_len, num_taps * d_model)

        mx.eval(audio_summary, decoder_feats, labels, decoder_input_ids)

        # Save as NPZ (float16 to save space)
        np.savez_compressed(
            output_path / f"sample_{idx:06d}.npz",
            audio_summary=np.array(audio_summary.astype(mx.float16)),
            decoder_feats=np.array(decoder_feats.astype(mx.float16)),
            token_ids=np.array(labels),
            decoder_input_ids=np.array(decoder_input_ids),
        )

    print(f"Saved {len(ds)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DFlash training dataset (MLX)")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard ID to process")
    args = parser.parse_args()

    generate_dataset(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.config,
        dataset_split=args.split,
        output_dir=args.output_dir,
        target_layer_ids=args.layers,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )


if __name__ == "__main__":
    main()
