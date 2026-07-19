#!/usr/bin/env python3
"""
Whisper-Flash Real Pipeline — End-to-End on Apple Silicon
=========================================================

This script runs the FULL DFlash speculative decoding pipeline on REAL audio:

  1. Dataset Generation  — real LibriSpeech speech → teacher-forced Whisper traces
  2. Training            — train a DFlash draft model from those traces
  3. Evaluation          — baseline vs DFlash on held-out real audio (WER, speedup, acceptance)
  4. Single-File Demo    — transcribe one utterance side-by-side for qualitative check

Uses whisper-tiny by default (fast, ~39MB). Switch to whisper-large-v3-turbo
for production-quality results.

Usage:
    uv run python run_real_pipeline.py                          # full pipeline
    uv run python run_real_pipeline.py --skip-generate          # reuse existing data
    uv run python run_real_pipeline.py --skip-generate --skip-train  # eval only
    uv run python run_real_pipeline.py --model mlx-community/whisper-large-v3-turbo --train-samples 500 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def print_header(title: str):
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def print_step(step: int, total: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────
# Step 1: Generate dataset from real audio
# ─────────────────────────────────────────────────────────────────────────

def generate_real_dataset(
    model_name: str,
    output_dir: str,
    train_samples: int,
    eval_samples: int,
    target_layer_ids: list[int] | None = None,
):
    """Generate training + eval datasets from real LibriSpeech audio."""
    from datasets import load_dataset
    from mlx_whisper.audio import log_mel_spectrogram
    from mlx_whisper.tokenizer import get_tokenizer
    from tqdm import tqdm

    from whisper_flash_mlx.target_model import (
        decoder_forward_with_hidden_states,
        encoder_forward,
        load_target_model,
    )
    from whisper_flash_mlx.utils import extract_context_feature

    print(f"Model: {model_name}")
    print(f"Train samples: {train_samples}, Eval samples: {eval_samples}")

    # Load model
    print("Loading Whisper model...")
    model = load_target_model(model_name)

    if target_layer_ids is None:
        n_layers = model.dims.n_text_layer
        if n_layers <= 4:
            target_layer_ids = [1, 1, 1]  # tiny model has very few layers
        else:
            from whisper_flash_mlx.utils import build_target_layer_ids
            target_layer_ids = build_target_layer_ids(n_layers, 3)

    print(f"Target layer IDs: {target_layer_ids}")

    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages,
    )

    # Load real speech data
    dataset_id = "openslr/librispeech_asr"
    print(f"Loading {dataset_id} (clean, train split)...")
    try:
        ds_train = load_dataset(dataset_id, "clean", split="train.100")
    except Exception:
        # Fallback: use the test split for training too (small-scale run)
        print("  train.100 not available, using test split for training")
        ds_train = load_dataset(dataset_id, "clean", split="test")

    print(f"Loading {dataset_id} (clean, test split for eval)...")
    ds_test = load_dataset(dataset_id, "clean", split="test")

    # Process training data
    train_dir = Path(output_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ds, out_dir, n_samples in [
        ("Training", ds_train, train_dir, train_samples),
        ("Eval", ds_test, eval_dir, eval_samples),
    ]:
        n_samples = min(n_samples, len(ds))
        print(f"\nGenerating {split_name} data ({n_samples} real utterances)...")

        for idx in tqdm(range(n_samples), desc=split_name):
            out_file = out_dir / f"sample_{idx:06d}.npz"
            if out_file.exists():
                continue  # skip already generated

            sample = ds[idx]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            text = sample["text"]

            # Compute mel spectrogram, pad to 30s
            mel = log_mel_spectrogram(
                audio, n_mels=model.dims.n_mels,
                padding=16000 * 30 - len(audio),
            )
            mel_mx = mx.array(mel)[None]  # (1, frames, n_mels)

            # Tokenize ground truth
            text_tokens = tokenizer.encode(text)
            token_ids = mx.array([text_tokens], dtype=mx.int32)

            # Prepend SOT
            sot = mx.array([[tokenizer.sot]], dtype=mx.int32)
            decoder_input_ids = mx.concatenate([sot, token_ids[:, :-1]], axis=1)
            labels = token_ids

            # Encode
            encoder_hidden = encoder_forward(model, mel_mx)

            # Audio summary
            audio_summary = mx.mean(encoder_hidden, axis=1)  # (1, d_model)

            # Teacher-forced decode with hidden states
            logits, _, all_hidden = decoder_forward_with_hidden_states(
                model, decoder_input_ids, encoder_hidden,
                kv_cache=None,
                collect_hidden_states=True,
            )

            # Extract context features
            decoder_feats = extract_context_feature(
                all_hidden, target_layer_ids,
            )

            mx.eval(audio_summary, decoder_feats, labels, decoder_input_ids)

            np.savez_compressed(
                out_file,
                audio_summary=np.array(audio_summary.astype(mx.float16)),
                decoder_feats=np.array(decoder_feats.astype(mx.float16)),
                token_ids=np.array(labels),
                decoder_input_ids=np.array(decoder_input_ids),
            )

        print(f"  {split_name} data saved to {out_dir} ({n_samples} files)")


# ─────────────────────────────────────────────────────────────────────────
# Step 2: Train
# ─────────────────────────────────────────────────────────────────────────

def train_draft_model(
    model_name: str,
    data_dir: str,
    output_dir: str,
    epochs: int,
    block_size: int,
    lr: float,
):
    """Train the DFlash draft model on real speech traces."""
    from whisper_flash_mlx.train import train

    train_dir = str(Path(data_dir) / "train")
    val_dir = str(Path(data_dir) / "eval")

    if not Path(val_dir).exists() or not list(Path(val_dir).glob("sample_*.npz")):
        val_dir = None

    train(
        data_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        block_size=block_size,
        anchors_per_seq=16,
        gamma=7.0,
        grad_accumulation=4,
        save_every=max(1, epochs // 5),
        max_keep=3,
    )


# ─────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate on real audio
# ─────────────────────────────────────────────────────────────────────────

def evaluate_real(
    model_name: str,
    checkpoint_dir: str,
    block_size: int,
    eval_samples: int,
):
    """Run real evaluation: baseline vs DFlash on LibriSpeech test."""
    from whisper_flash_mlx.evaluate import evaluate

    # Find checkpoint
    ckpt_dir = Path(checkpoint_dir)
    candidates = ["best_model.safetensors", "final_model.safetensors"]
    ckpt_path = None
    for c in candidates:
        if (ckpt_dir / c).exists():
            ckpt_path = str(ckpt_dir / c)
            break

    if ckpt_path is None:
        # Find latest epoch checkpoint
        epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.safetensors"))
        if epoch_ckpts:
            ckpt_path = str(epoch_ckpts[-1])

    if ckpt_path is None:
        print("ERROR: No checkpoint found!")
        return None

    print(f"Using checkpoint: {ckpt_path}")

    # Run evaluation on real LibriSpeech test data
    metrics = evaluate(
        checkpoint_path=ckpt_path,
        model_name=model_name,
        dataset_name="openslr/librispeech_asr",
        dataset_config="clean",
        dataset_split="test",
        block_size=block_size,
        max_samples=eval_samples,
        temperature=0.0,
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Step 4: Qualitative single-file transcription demo
# ─────────────────────────────────────────────────────────────────────────

def transcribe_demo(
    model_name: str,
    checkpoint_dir: str,
    block_size: int,
):
    """Transcribe a real LibriSpeech utterance: baseline vs DFlash side-by-side."""
    from datasets import load_dataset
    from mlx_whisper.audio import log_mel_spectrogram
    from mlx_whisper.tokenizer import get_tokenizer

    from whisper_flash_mlx.evaluate import baseline_generate, load_draft_model
    from whisper_flash_mlx.generate import whisper_dflash_generate
    from whisper_flash_mlx.target_model import load_target_model

    # Load model
    target = load_target_model(model_name)
    tokenizer = get_tokenizer(
        target.is_multilingual, num_languages=target.num_languages,
    )

    # Find checkpoint
    ckpt_dir = Path(checkpoint_dir)
    ckpt_path = None
    for c in ["best_model.safetensors", "final_model.safetensors"]:
        if (ckpt_dir / c).exists():
            ckpt_path = str(ckpt_dir / c)
            break
    if ckpt_path is None:
        epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.safetensors"))
        if epoch_ckpts:
            ckpt_path = str(epoch_ckpts[-1])

    if ckpt_path is None:
        print("  No checkpoint found, skipping demo.")
        return

    draft = load_draft_model(ckpt_path)
    if block_size is not None:
        draft.block_size = block_size

    # Load a real test utterance
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test")

    # Pick 3 diverse samples
    sample_indices = [0, len(ds) // 2, len(ds) - 1]

    for i, idx in enumerate(sample_indices):
        sample = ds[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        ref_text = sample["text"]
        duration_s = len(audio) / 16000

        mel = log_mel_spectrogram(
            audio, n_mels=target.dims.n_mels,
            padding=16000 * 30 - len(audio),
        )
        mel_mx = mx.array(mel)[None]

        print(f"\n  ── Utterance {i+1} ({duration_s:.1f}s audio) ──")
        print(f"  Reference:  {ref_text}")

        # Baseline
        t0 = time.perf_counter()
        bl = baseline_generate(target, mel_mx)
        bl_time = time.perf_counter() - t0
        bl_tokens = np.array(bl["output_ids"][0]).tolist()
        bl_text = tokenizer.decode(bl_tokens)
        print(f"  Baseline:   {bl_text}")
        print(f"              ({bl['num_tokens']} tokens, {bl_time*1000:.0f}ms, "
              f"{bl['num_tokens']/bl_time:.0f} tok/s)")

        # DFlash
        t0 = time.perf_counter()
        df = whisper_dflash_generate(
            draft, target, mel_mx,
            temperature=0.0,
            return_stats=True,
        )
        df_time = time.perf_counter() - t0
        df_tokens = np.array(df.output_ids[0]).tolist()
        df_text = tokenizer.decode(df_tokens)
        mean_acc = np.mean(df.acceptance_lengths) if df.acceptance_lengths else 0
        print(f"  DFlash:     {df_text}")
        print(f"              ({df.num_output_tokens} tokens, {df_time*1000:.0f}ms, "
              f"mean_accept={mean_acc:.2f})")

        match = "✅ EXACT MATCH" if bl_text == df_text else "⚠️  MISMATCH"
        print(f"  Result:     {match}")


# ─────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper-Flash: Real end-to-end pipeline on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run_real_pipeline.py
  uv run python run_real_pipeline.py --model mlx-community/whisper-large-v3-turbo
  uv run python run_real_pipeline.py --skip-generate --skip-train
  uv run python run_real_pipeline.py --train-samples 500 --epochs 20
        """,
    )

    parser.add_argument(
        "--model", default="mlx-community/whisper-tiny",
        help="MLX Whisper model (default: whisper-tiny for speed)",
    )
    parser.add_argument(
        "--data-dir", default="data/real",
        help="Directory for real dataset (default: data/real)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints_real",
        help="Directory for checkpoints (default: checkpoints_real)",
    )
    parser.add_argument(
        "--train-samples", type=int, default=200,
        help="Number of real utterances for training (default: 200)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=50,
        help="Number of real utterances for evaluation (default: 50)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs (default: 10)",
    )
    parser.add_argument(
        "--block-size", type=int, default=4,
        help="Speculative block size (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip dataset generation (reuse existing data)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (use existing checkpoint)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip quantitative evaluation",
    )
    parser.add_argument(
        "--skip-demo", action="store_true",
        help="Skip qualitative transcription demo",
    )

    args = parser.parse_args()
    total_steps = 4 - sum([args.skip_generate, args.skip_train, args.skip_eval, args.skip_demo])
    current_step = 0
    t_start = time.perf_counter()

    print_header("Whisper-Flash: Real End-to-End Pipeline")
    print(f"  Model:           {args.model}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  Train samples:   {args.train_samples}")
    print(f"  Eval samples:    {args.eval_samples}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Block size:      {args.block_size}")

    # ── Step 1: Generate ──
    if not args.skip_generate:
        current_step += 1
        print_step(current_step, total_steps, "Generate Dataset from Real Audio")

        generate_real_dataset(
            model_name=args.model,
            output_dir=args.data_dir,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
        )
    else:
        print("\n⏭  Skipping dataset generation (--skip-generate)")

    # ── Step 2: Train ──
    if not args.skip_train:
        current_step += 1
        print_step(current_step, total_steps, "Train DFlash Draft Model")

        train_draft_model(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.checkpoint_dir,
            epochs=args.epochs,
            block_size=args.block_size,
            lr=args.lr,
        )
    else:
        print("\n⏭  Skipping training (--skip-train)")

    # ── Step 3: Evaluate ──
    metrics = None
    if not args.skip_eval:
        current_step += 1
        print_step(current_step, total_steps, "Evaluate on Real Audio (LibriSpeech test)")

        metrics = evaluate_real(
            model_name=args.model,
            checkpoint_dir=args.checkpoint_dir,
            block_size=args.block_size,
            eval_samples=args.eval_samples,
        )

        if metrics:
            # Save metrics
            metrics_path = Path(args.checkpoint_dir) / "real_eval_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n  Metrics saved to {metrics_path}")
    else:
        print("\n⏭  Skipping evaluation (--skip-eval)")

    # ── Step 4: Demo ──
    if not args.skip_demo:
        current_step += 1
        print_step(current_step, total_steps, "Qualitative Transcription Demo")

        transcribe_demo(
            model_name=args.model,
            checkpoint_dir=args.checkpoint_dir,
            block_size=args.block_size,
        )
    else:
        print("\n⏭  Skipping demo (--skip-demo)")

    # ── Summary ──
    total_time = time.perf_counter() - t_start
    print_header(f"Pipeline Complete  ({total_time:.0f}s total)")

    if metrics:
        print(f"  Baseline WER:          {metrics['baseline_wer']*100:.2f}%")
        print(f"  DFlash WER:            {metrics['dflash_wer']*100:.2f}%")
        print(f"  Exact match:           {metrics['exact_match_rate']*100:.1f}%")
        print(f"  Speedup:               {metrics['speedup']:.2f}x")
        print(f"  Mean acceptance:       {metrics['mean_acceptance_length']:.2f}")
        print(f"  Baseline throughput:   {metrics['baseline_tps']:.1f} tok/s")
        print(f"  DFlash throughput:     {metrics['dflash_tps']:.1f} tok/s")

    print(f"\n  Checkpoint dir:  {args.checkpoint_dir}/")
    print(f"  Data dir:        {args.data_dir}/")
    print()


if __name__ == "__main__":
    main()
