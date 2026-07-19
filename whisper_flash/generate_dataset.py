"""Generate training dataset by running teacher-forced Whisper decoding.

For each utterance in LibriSpeech:
  1. Compute mel spectrogram
  2. Run encoder → cache encoder_hidden_states
  3. Compute audio_summary = mean_pool(encoder_hidden)
  4. Run teacher-forced decoder with ground-truth tokens + output_hidden_states
  5. Extract decoder hidden states at tapped layers
  6. Save per-utterance: audio_summary, decoder_feats, token_ids
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .utils import extract_context_feature, build_target_layer_ids, get_device


def generate_dataset(
    model_name: str = "openai/whisper-large-v3",
    dataset_name: str = "librispeech_asr",
    dataset_config: str = "clean",
    dataset_split: str = "train.100",
    output_dir: str = "data/train",
    target_layer_ids: list[int] | None = None,
    max_samples: int | None = None,
    device: torch.device | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """Generate training data from teacher-forced Whisper decoding.

    Args:
        model_name: HuggingFace model identifier.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration (e.g. "clean").
        dataset_split: Dataset split (e.g. "train.100").
        output_dir: Directory to save NPZ files.
        target_layer_ids: Decoder layers to tap. Defaults to [8, 16, 24].
        max_samples: Limit number of samples (for debugging).
        device: Torch device to use.
    """
    from datasets import load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device is None:
        device = get_device()

    if target_layer_ids is None:
        target_layer_ids = [8, 16, 24]

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{dataset_config}/{dataset_split}")
    print(f"Target layers: {target_layer_ids}")

    # Load model and processor
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

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
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        # Compute mel spectrogram
        inputs = processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device)
        if device.type == "cuda":
            inputs = inputs.to(torch.float16)

        # Get ground-truth token ids (teacher forcing)
        forced_decoder_ids = processor.tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(device)

        # Prepend decoder_start_token_id
        bos = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )
        decoder_input_ids = torch.cat([bos, forced_decoder_ids[:, :-1]], dim=1)
        labels = forced_decoder_ids

        with torch.no_grad():
            # Run encoder
            encoder_outputs = model.model.encoder(inputs)
            encoder_hidden = encoder_outputs.last_hidden_state

            # Audio summary
            audio_summary = encoder_hidden.mean(dim=1)  # (1, d_model)

            # Run teacher-forced decoder
            outputs = model(
                encoder_outputs=(encoder_hidden,),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract hidden states from tapped layers
            decoder_feats = extract_context_feature(
                outputs.decoder_hidden_states, target_layer_ids
            )  # (1, seq_len, num_taps * d_model)

        # Save as NPZ (float16 to save space)
        np.savez_compressed(
            output_path / f"sample_{idx:06d}.npz",
            audio_summary=audio_summary.cpu().to(torch.float16).numpy(),
            decoder_feats=decoder_feats.cpu().to(torch.float16).numpy(),
            token_ids=labels.cpu().numpy(),
            decoder_input_ids=decoder_input_ids.cpu().numpy(),
        )

    print(f"Saved {len(ds)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DFlash training dataset")
    parser.add_argument("--model", default="openai/whisper-large-v3")
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
