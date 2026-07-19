"""Benchmark: standard greedy decoding vs DFlash speculative decoding — MLX version.

Reports WER, latency, throughput, acceptance length, and memory usage.
Runs entirely on Apple Silicon via MLX.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from .draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from .generate import whisper_dflash_generate
from .target_model import (
    decoder_forward_with_hidden_states,
    encoder_forward,
    load_target_model,
)
from .utils import sample


def load_draft_model(
    checkpoint_path: str,
    config_path: str | None = None,
) -> WhisperDFlashDraftModel:
    """Load a trained draft model from checkpoint.

    Args:
        checkpoint_path: Path to .safetensors weights.
        config_path: Path to config JSON. If None, looks for config alongside weights.

    Returns:
        Loaded WhisperDFlashDraftModel.
    """
    ckpt_path = Path(checkpoint_path)

    # Find config
    if config_path is None:
        # Try common naming patterns
        parent = ckpt_path.parent
        for candidate in [
            parent / "best_config.json",
            parent / "final_config.json",
            parent / f"config_{ckpt_path.stem.replace('checkpoint_', '')}.json",
        ]:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is None:
        raise FileNotFoundError(
            f"Could not find config JSON for {checkpoint_path}. "
            "Please specify --config-path."
        )

    with open(config_path) as f:
        config_data = json.load(f)
    # Remove non-config keys
    config_data.pop("epoch", None)
    config_data.pop("val_acc", None)

    config = WhisperDFlashConfig(**config_data)
    model = WhisperDFlashDraftModel(config)
    model.load_weights(checkpoint_path)
    model.eval()
    print(f"Loaded draft model from {checkpoint_path}")
    return model


def baseline_generate(
    target,
    mel: mx.array,
    max_new_tokens: int = 448,
    temperature: float = 0.0,
) -> dict:
    """Standard greedy decoding (no speculation) for baseline comparison."""

    t0 = time.perf_counter()

    # Encode
    encoder_hidden = encoder_forward(target, mel)
    mx.eval(encoder_hidden)

    # Decode token by token
    decoder_ids = mx.array([[50258]], dtype=mx.int32)  # SOT

    t_first = None
    kv_cache = None
    eos_token_id = 50257  # EOT

    for step in range(max_new_tokens):
        if kv_cache is None:
            input_tokens = decoder_ids
        else:
            input_tokens = decoder_ids[:, -1:]

        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            target, input_tokens, encoder_hidden,
            kv_cache=kv_cache,
            collect_hidden_states=False,
        )

        next_token = sample(logits[:, -1:, :], temperature)
        mx.eval(next_token)
        decoder_ids = mx.concatenate([decoder_ids, next_token], axis=1)

        if t_first is None:
            t_first = time.perf_counter() - t0

        if next_token.item() == eos_token_id:
            break

    total_time = time.perf_counter() - t0
    num_tokens = decoder_ids.shape[1] - 1  # Exclude SOT

    return {
        "output_ids": decoder_ids,
        "num_tokens": num_tokens,
        "total_time": total_time,
        "time_to_first_token": t_first,
        "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
    }


def evaluate(
    checkpoint_path: str = "checkpoints/best_model.safetensors",
    config_path: str | None = None,
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    dataset_name: str = "librispeech_asr",
    dataset_config: str = "clean",
    dataset_split: str = "test",
    block_size: int | None = None,
    max_samples: int | None = None,
    temperature: float = 0.0,
):
    """Run evaluation comparing baseline vs DFlash decoding on Apple Silicon."""
    from datasets import load_dataset
    from mlx_whisper.audio import log_mel_spectrogram
    from mlx_whisper.tokenizer import get_tokenizer

    try:
        from jiwer import wer as compute_wer
    except ImportError:
        print("Warning: jiwer not installed, WER will not be computed")
        compute_wer = None

    print(f"Model: {model_name}")

    # Load models
    print("Loading target model...")
    target = load_target_model(model_name)

    tokenizer = get_tokenizer(
        target.is_multilingual, num_languages=target.num_languages,
    )

    print("Loading draft model...")
    draft = load_draft_model(checkpoint_path, config_path)
    if block_size is not None:
        draft.block_size = block_size

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Results storage
    baseline_results = []
    dflash_results = []
    baseline_texts = []
    dflash_texts = []
    reference_texts = []

    print(f"\nEvaluating {len(ds)} utterances...")
    for idx in tqdm(range(len(ds))):
        sample_data = ds[idx]
        audio = np.array(sample_data["audio"]["array"], dtype=np.float32)
        ref_text = sample_data["text"]
        reference_texts.append(ref_text)

        # pad to 30s
        mel = log_mel_spectrogram(audio, n_mels=target.dims.n_mels, padding=16000 * 30 - len(audio))
        mel = mx.array(mel)[None]  # (1, 3000, 80)

        # --- Baseline ---
        bl = baseline_generate(target, mel, temperature=temperature)
        bl_tokens = np.array(bl["output_ids"][0]).tolist()
        bl_text = tokenizer.decode(bl_tokens)
        baseline_results.append(bl)
        baseline_texts.append(bl_text)

        # --- DFlash ---
        df = whisper_dflash_generate(
            draft, target, mel,
            temperature=temperature,
            return_stats=True,
        )
        df_tokens = np.array(df.output_ids[0]).tolist()
        df_text = tokenizer.decode(df_tokens)
        dflash_results.append(df)
        dflash_texts.append(df_text)

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Throughput
    bl_tps = np.mean([r["tokens_per_second"] for r in baseline_results])
    df_tps = np.mean([1.0 / r.time_per_output_token for r in dflash_results])
    print(f"\nBaseline throughput:  {bl_tps:.1f} tok/s")
    print(f"DFlash throughput:   {df_tps:.1f} tok/s")
    print(f"Speedup:             {df_tps / bl_tps:.2f}x")

    # Latency
    bl_lat = np.mean([r["total_time"] for r in baseline_results])
    df_lat = np.mean([
        r.time_to_first_token + r.time_per_output_token * r.num_output_tokens
        for r in dflash_results
    ])
    print(f"\nBaseline avg latency: {bl_lat*1000:.1f} ms")
    print(f"DFlash avg latency:   {df_lat*1000:.1f} ms")

    # Acceptance lengths
    all_acc = []
    for r in dflash_results:
        all_acc.extend(r.acceptance_lengths)

    mean_acc = 0.0
    hist = []
    if all_acc:
        mean_acc = np.mean(all_acc)
        print(f"\nMean acceptance length: {mean_acc:.2f}")
        B = draft.block_size
        hist = [all_acc.count(b) / len(all_acc) * 100 for b in range(B + 1)]
        print(f"Acceptance histogram:  {[f'{h:.1f}%' for h in hist]}")

    # WER
    bl_wer = 0.0
    df_wer = 0.0
    exact_match_rate = 0.0
    if compute_wer is not None and reference_texts:
        refs_lower = [t.lower() for t in reference_texts]
        bl_wer = compute_wer(refs_lower, [t.lower() for t in baseline_texts])
        df_wer = compute_wer(refs_lower, [t.lower() for t in dflash_texts])
        print(f"\nBaseline WER: {bl_wer*100:.2f}%")
        print(f"DFlash WER:   {df_wer*100:.2f}%")

        # Check exact match
        exact_match = sum(
            1 for bl, df in zip(baseline_texts, dflash_texts) if bl == df
        )
        exact_match_rate = exact_match / len(baseline_texts)
        print(f"Exact match:  {exact_match}/{len(baseline_texts)} "
              f"({exact_match_rate*100:.1f}%)")

    print("\n" + "=" * 70)

    return {
        "baseline_tps": float(bl_tps),
        "dflash_tps": float(df_tps),
        "speedup": float(df_tps / bl_tps) if bl_tps > 0 else 0.0,
        "baseline_latency_ms": float(bl_lat * 1000),
        "dflash_latency_ms": float(df_lat * 1000),
        "mean_acceptance_length": float(mean_acc),
        "acceptance_histogram": [float(h) for h in hist],
        "baseline_wer": float(bl_wer),
        "dflash_wer": float(df_wer),
        "exact_match_rate": float(exact_match_rate),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DFlash for Whisper (MLX)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.safetensors")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config_path,
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.config,
        dataset_split=args.split,
        block_size=args.block_size,
        max_samples=args.max_samples,
        temperature=args.temperature,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
