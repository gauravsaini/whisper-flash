"""Benchmark: standard greedy decoding vs DFlash speculative decoding.

Reports WER, latency, throughput, acceptance length, and memory usage.
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from .generate import whisper_dflash_generate
from .utils import get_device, sample


def load_draft_model(
    checkpoint_path: str,
    device: torch.device,
) -> WhisperDFlashDraftModel:
    """Load a trained draft model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = WhisperDFlashDraftModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded draft model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.inference_mode()
def baseline_generate(
    target,
    input_features: torch.FloatTensor,
    max_new_tokens: int = 448,
    temperature: float = 0.0,
) -> dict:
    """Standard greedy decoding (no speculation) for baseline comparison."""
    device = input_features.device

    # Encode
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    encoder_outputs = target.model.encoder(input_features)
    encoder_hidden = encoder_outputs.last_hidden_state

    # Decode token by token
    decoder_ids = torch.tensor(
        [[target.config.decoder_start_token_id]], device=device
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = None

    past_key_values = None
    for step in range(max_new_tokens):
        output = target(
            encoder_outputs=(encoder_hidden,),
            decoder_input_ids=decoder_ids if past_key_values is None else decoder_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values

        next_token = sample(output.logits[:, -1:, :], temperature)
        decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

        if t_first is None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter() - t0

        if next_token.item() == target.config.eos_token_id:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    num_tokens = decoder_ids.shape[1] - 1  # Exclude BOS
    return {
        "output_ids": decoder_ids,
        "num_tokens": num_tokens,
        "total_time": total_time,
        "time_to_first_token": t_first,
        "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
    }


def evaluate(
    checkpoint_path: str = "checkpoints/best_model.pt",
    model_name: str = "openai/whisper-large-v3",
    dataset_name: str = "librispeech_asr",
    dataset_config: str = "clean",
    dataset_split: str = "test",
    block_size: int | None = None,
    max_samples: int | None = None,
    temperature: float = 0.0,
    device: torch.device | None = None,
):
    """Run evaluation comparing baseline vs DFlash decoding."""
    from datasets import load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    try:
        from jiwer import wer as compute_wer
    except ImportError:
        print("Warning: jiwer not installed, WER will not be computed")
        compute_wer = None

    if device is None:
        device = get_device()

    print(f"Device: {device}")
    print(f"Model: {model_name}")

    # Load models
    print("Loading target model...")
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name)
    target = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=target_dtype,
    ).to(device).eval()

    print("Loading draft model...")
    draft = load_draft_model(checkpoint_path, device)
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
        audio = sample_data["audio"]["array"]
        sr = sample_data["audio"]["sampling_rate"]
        ref_text = sample_data["text"]
        reference_texts.append(ref_text)

        inputs = processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device).to(target_dtype)

        # --- Baseline ---
        bl = baseline_generate(target, inputs, temperature=temperature)
        bl_text = processor.batch_decode(
            bl["output_ids"], skip_special_tokens=True
        )[0]
        baseline_results.append(bl)
        baseline_texts.append(bl_text)

        # --- DFlash ---
        df = whisper_dflash_generate(
            draft, target, inputs,
            temperature=temperature,
            return_stats=True,
        )
        df_text = processor.batch_decode(
            df.output_ids, skip_special_tokens=True
        )[0]
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
    parser = argparse.ArgumentParser(description="Evaluate DFlash for Whisper")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--model", default="openai/whisper-large-v3")
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
