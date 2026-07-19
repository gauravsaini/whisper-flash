"""E5: Speculative N-gram Decoding — Discrete token speculation via lookup table.

The entire TIMELINE chased continuous manifold drafting, which hit the branching
ceiling. This experiment takes the opposite approach: use a dead-simple n-gram
lookup table as the "drafter". No neural model, no training, zero overhead.

Hypothesis: Common English phrases ("the United States", "thank you very much",
"in the morning") appear frequently in speech. A trigram lookup table can draft
4 tokens ahead, verified by a single batched decoder forward pass. Each accepted
draft saves 1-3 decoder calls.

Usage:
    uv run python experiments/experiment_ngram_speculation.py
    uv run python experiments/experiment_ngram_speculation.py --model mlx-community/whisper-large-v3-mlx
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flash_mlx.target_model import (
    load_target_model,
    encoder_forward,
    decoder_forward_with_hidden_states,
)
from whisper_flash_mlx.quantization import quantize_model
from whisper_flash_mlx.utils import sample

EOS_ID, SOT_ID = 50257, 50258


# ════════════════════════════════════════════════════════════════
# N-gram table builder
# ════════════════════════════════════════════════════════════════

class NgramDrafter:
    """Dead-simple n-gram lookup drafter. Zero parameters, zero training.

    Maintains a trigram → next-K-tokens table. At decode time, checks if the
    last N tokens match any entry and proposes a draft.

    Two sources of n-grams:
    1. Static: built from a text corpus (common English phrases)
    2. Dynamic: built on-the-fly from tokens decoded so far in this utterance
    """

    def __init__(self, context_len: int = 3, max_draft: int = 4, tokenizer=None):
        self.context_len = context_len
        self.max_draft = max_draft
        self.tokenizer = tokenizer
        # Static table: context_tuple → list of continuation tokens
        self.static_table: dict[tuple, list[int]] = {}
        # Dynamic table: built per-utterance
        self.dynamic_table: dict[tuple, list[int]] = {}

    def build_static_table(self, token_sequences: list[list[int]]):
        """Build static n-gram table from a list of token sequences."""
        for seq in token_sequences:
            for i in range(len(seq) - self.context_len - self.max_draft):
                context = tuple(seq[i:i + self.context_len])
                continuation = seq[i + self.context_len:i + self.context_len + self.max_draft]
                if context not in self.static_table:
                    self.static_table[context] = continuation

    def build_static_from_text(self, texts: list[str]):
        """Build static table from raw text strings (tokenizes them first)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text-based table building")
        sequences = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            sequences.append(tokens)
        self.build_static_table(sequences)

    def reset_dynamic(self):
        """Reset dynamic table for a new utterance."""
        self.dynamic_table = {}

    def update_dynamic(self, decoded_tokens: list[int]):
        """Update dynamic table with tokens decoded so far."""
        seq = decoded_tokens
        for i in range(len(seq) - self.context_len - 1):
            context = tuple(seq[i:i + self.context_len])
            max_cont = min(self.max_draft, len(seq) - i - self.context_len)
            continuation = seq[i + self.context_len:i + self.context_len + max_cont]
            if len(continuation) > 0:
                self.dynamic_table[context] = continuation

    def draft(self, context_tokens: list[int]) -> list[int] | None:
        """Look up a draft continuation for the given context.

        Returns:
            List of draft token IDs, or None if no match.
        """
        if len(context_tokens) < self.context_len:
            return None

        context = tuple(context_tokens[-self.context_len:])

        # Check dynamic table first (utterance-specific patterns)
        if context in self.dynamic_table:
            return self.dynamic_table[context]

        # Then static table
        if context in self.static_table:
            return self.static_table[context]

        return None


# ════════════════════════════════════════════════════════════════
# Speculative decode with n-gram drafting
# ════════════════════════════════════════════════════════════════

def greedy_decode_baseline(model, mel: mx.array, max_new_tokens: int = 448) -> tuple[list[int], float, int]:
    """Standard greedy decode. Returns (tokens, wall_time, n_decoder_calls)."""
    t0 = time.perf_counter()
    n_calls = 0

    enc = encoder_forward(model, mel)
    mx.eval(enc)

    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    n_calls += 1
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]

    while len(output_ids) < max_new_tokens:
        inp = mx.array([[output_ids[-1]]], dtype=mx.int32)
        logits, kv_cache, _ = decoder_forward_with_hidden_states(
            model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
        n_calls += 1
        tok = sample(logits[:, -1:, :], 0.0)
        mx.eval(tok)
        token_id = tok.item()
        output_ids.append(token_id)
        if token_id == EOS_ID:
            break

    wall = time.perf_counter() - t0
    return output_ids, wall, n_calls


def speculative_ngram_decode(
    model,
    mel: mx.array,
    drafter: NgramDrafter,
    max_new_tokens: int = 448,
    use_dynamic: bool = True,
    top_k_verify: int = 1,  # 1 = exact match, 3 = top-3 multi-path
) -> tuple[list[int], float, int, dict]:
    """Speculative decode using n-gram drafting.

    At each step:
    1. Check if the last N tokens match an n-gram entry
    2. If yes, draft K tokens and verify by running them through the decoder
    3. Accept the longest prefix where draft matches target's greedy (or top-k)
    4. If no match, fall back to standard greedy decode

    Returns:
        (tokens, wall_time, n_decoder_calls, stats_dict)
    """
    t0 = time.perf_counter()
    n_calls = 0
    stats = {
        "n_draft_attempts": 0,
        "n_draft_accepted_tokens": 0,
        "n_draft_rejected": 0,
        "n_greedy_steps": 0,
        "draft_lengths": [],
        "accepted_lengths": [],
    }

    enc = encoder_forward(model, mel)
    mx.eval(enc)

    # Prefill
    dec = mx.array([[SOT_ID]], dtype=mx.int32)
    logits, kv_cache, _ = decoder_forward_with_hidden_states(
        model, dec, enc, kv_cache=None, collect_hidden_states=False)
    n_calls += 1
    first = sample(logits[:, -1:, :], 0.0)
    mx.eval(first)
    output_ids = [SOT_ID, first.item()]

    if use_dynamic:
        drafter.reset_dynamic()

    while len(output_ids) < max_new_tokens:
        if output_ids[-1] == EOS_ID:
            break

        # Update dynamic table with what we've decoded so far
        if use_dynamic and len(output_ids) > drafter.context_len + 1:
            drafter.update_dynamic(output_ids)

        # Try to get a draft
        draft = drafter.draft(output_ids)

        if draft is not None and len(draft) > 0:
            stats["n_draft_attempts"] += 1
            stats["draft_lengths"].append(len(draft))

            # Verify draft tokens one by one using KV cache
            # We verify by running each draft token through the decoder
            # and checking if the target agrees
            accepted = 0
            temp_kv = kv_cache  # Don't modify kv_cache until we know how many to accept

            for di, draft_token in enumerate(draft):
                # Run the PREVIOUS token (last accepted) through decoder to get logits for THIS position
                inp = mx.array([[output_ids[-1] if di == 0 else draft[di-1]]], dtype=mx.int32)
                logits_v, temp_kv, _ = decoder_forward_with_hidden_states(
                    model, inp, enc, kv_cache=temp_kv, collect_hidden_states=False)
                n_calls += 1
                mx.eval(logits_v)

                # Get target's prediction
                if top_k_verify == 1:
                    target_token = mx.argmax(logits_v[:, -1, :], axis=-1).item()
                    if draft_token == target_token:
                        accepted += 1
                        output_ids.append(draft_token)
                    else:
                        # Draft doesn't match — use target's token instead
                        output_ids.append(target_token)
                        kv_cache = temp_kv
                        break
                else:
                    # Top-K verification
                    top_k_logits = mx.sort(logits_v[:, -1, :], axis=-1)[:, -top_k_verify:]
                    top_k_indices = mx.argsort(logits_v[:, -1, :], axis=-1)[:, -top_k_verify:]
                    top_k_ids = set(top_k_indices[0].tolist())

                    if draft_token in top_k_ids:
                        accepted += 1
                        output_ids.append(draft_token)
                    else:
                        # Reject — use greedy token
                        target_token = mx.argmax(logits_v[:, -1, :], axis=-1).item()
                        output_ids.append(target_token)
                        kv_cache = temp_kv
                        break
            else:
                # All draft tokens accepted — need one more call for the next token
                inp = mx.array([[draft[-1]]], dtype=mx.int32)
                logits_v, temp_kv, _ = decoder_forward_with_hidden_states(
                    model, inp, enc, kv_cache=temp_kv, collect_hidden_states=False)
                n_calls += 1
                next_tok = sample(logits_v[:, -1:, :], 0.0)
                mx.eval(next_tok)
                output_ids.append(next_tok.item())
                kv_cache = temp_kv

            if accepted > 0:
                stats["n_draft_accepted_tokens"] += accepted
                stats["accepted_lengths"].append(accepted)
                kv_cache = temp_kv
            else:
                stats["n_draft_rejected"] += 1

        else:
            # No draft available — standard greedy step
            stats["n_greedy_steps"] += 1
            inp = mx.array([[output_ids[-1]]], dtype=mx.int32)
            logits, kv_cache, _ = decoder_forward_with_hidden_states(
                model, inp, enc, kv_cache=kv_cache, collect_hidden_states=False)
            n_calls += 1
            tok = sample(logits[:, -1:, :], 0.0)
            mx.eval(tok)
            token_id = tok.item()
            output_ids.append(token_id)

    wall = time.perf_counter() - t0
    return output_ids, wall, n_calls, stats


# ════════════════════════════════════════════════════════════════
# Common English phrases for static table
# ════════════════════════════════════════════════════════════════

COMMON_PHRASES = [
    "the united states of america",
    "ladies and gentlemen",
    "thank you very much",
    "once upon a time",
    "in the morning",
    "in the evening",
    "at the same time",
    "on the other hand",
    "for example",
    "as a matter of fact",
    "i would like to",
    "we are going to",
    "it is important to",
    "one of the most",
    "in order to",
    "as well as",
    "at the end of the day",
    "the fact that",
    "in the first place",
    "on behalf of",
    "with respect to",
    "as a result",
    "in addition to",
    "the beginning of the",
    "the end of the",
    "the middle of the",
    "the top of the",
    "the bottom of the",
    "a number of",
    "a lot of",
    "the rest of the",
    "the last of the",
    "the first of the",
    "and so on",
    "and so forth",
    "more or less",
    "sooner or later",
    "from time to time",
    "step by step",
    "little by little",
    "side by side",
    "day by day",
    "one by one",
    "in front of",
    "in the middle of",
    "at the beginning of",
    "at the end of",
    "according to the",
    "with regard to",
    "in terms of",
    "the purpose of this",
    "the reason for this",
    "as far as",
    "it would be",
    "there would be",
    "it could be",
    "that would be",
    "this is the",
    "that is the",
    "what is the",
    "how do you",
    "do you think",
    "i don't know",
    "i don't think",
    "you know what",
]


# ════════════════════════════════════════════════════════════════
# WER computation
# ════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def run_experiment(args):
    print(f"\n{'='*70}")
    print(f"  E5: Speculative N-gram Decoding")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Context len: {args.context_len}, Max draft: {args.max_draft}")
    print(f"  Q8: {args.q8}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    model = load_target_model(args.model)
    if args.q8:
        quantize_model(model, encoder_bits=8, decoder_bits=8, group_size=64)

    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=model.is_multilingual)

    from datasets import load_dataset as hf_load
    from mlx_whisper.audio import log_mel_spectrogram
    ds = hf_load("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.select(range(min(args.n_samples, len(ds))))

    def make_mel(audio_arr):
        arr = np.ascontiguousarray(audio_arr, dtype=np.float32)
        if len(arr) > 16000 * 30:
            arr = arr[:16000 * 30]
        mel = log_mel_spectrogram(arr, n_mels=model.dims.n_mels,
                                   padding=16000 * 30 - len(arr))
        return mx.array(mel)[None]

    def tokens_to_text(tids):
        text_tokens = [t for t in tids[1:] if t < 50257]
        return tokenizer.decode(text_tokens).strip()

    # ── Build n-gram drafter ──
    drafter = NgramDrafter(
        context_len=args.context_len,
        max_draft=args.max_draft,
        tokenizer=tokenizer,
    )

    # Build static table from common phrases
    print(f"Building static n-gram table from {len(COMMON_PHRASES)} phrases...")
    drafter.build_static_from_text(COMMON_PHRASES)
    print(f"  Static table: {len(drafter.static_table)} entries")

    # Also build from training set references (first half of dataset)
    n_train = min(args.n_samples // 2, len(ds))
    train_texts = [ds[i]["text"] for i in range(n_train)]
    drafter.build_static_from_text(train_texts)
    print(f"  After adding {n_train} training references: {len(drafter.static_table)} entries")

    # ── Run experiments ──
    configs = [
        ("greedy_baseline", {}),
        ("ngram_static_only", {"use_dynamic": False, "top_k": 1}),
        ("ngram_dynamic", {"use_dynamic": True, "top_k": 1}),
        ("ngram_dynamic_top3", {"use_dynamic": True, "top_k": 3}),
    ]

    all_results = {}
    eval_start = n_train  # Evaluate on second half to avoid data leakage

    for config_name, config in configs:
        print(f"\n── Config: {config_name} ──")
        sample_results = []
        total_time = 0.0
        total_calls = 0
        total_stats = defaultdict(int)
        total_stats["draft_lengths"] = []
        total_stats["accepted_lengths"] = []

        for i in range(eval_start, len(ds)):
            sample_data = ds[i]
            mel = make_mel(sample_data["audio"]["array"])
            ref = sample_data["text"]

            if config_name == "greedy_baseline":
                tids, wall, n_calls = greedy_decode_baseline(model, mel)
                stats = {}
            else:
                tids, wall, n_calls, stats = speculative_ngram_decode(
                    model, mel, drafter,
                    use_dynamic=config.get("use_dynamic", True),
                    top_k_verify=config.get("top_k", 1),
                )

            hyp = tokens_to_text(tids)
            wer = compute_wer(ref, hyp)

            sample_results.append({
                "sample_idx": i,
                "wer": wer,
                "n_tokens": len(tids) - 1,
                "wall_time": wall,
                "n_decoder_calls": n_calls,
                "stats": stats,
            })
            total_time += wall
            total_calls += n_calls

            for k, v in stats.items():
                if isinstance(v, list):
                    total_stats[k].extend(v)
                elif isinstance(v, (int, float)):
                    total_stats[k] += v

            if i < eval_start + 3 or wer > 0.5:
                n_accepted = stats.get("n_draft_accepted_tokens", 0)
                n_attempts = stats.get("n_draft_attempts", 0)
                print(f"  [{i:2d}] WER={wer:.4f} | {len(tids)-1} toks | "
                      f"{wall:.3f}s | {n_calls} calls | "
                      f"drafts={n_attempts} accepted={n_accepted}")

        n_eval = len(ds) - eval_start
        mean_wer = sum(r["wer"] for r in sample_results) / n_eval
        mean_time = total_time / n_eval
        mean_calls = total_calls / n_eval
        total_tokens = sum(r["n_tokens"] for r in sample_results)

        all_results[config_name] = {
            "mean_wer": mean_wer,
            "mean_time_s": mean_time,
            "total_time_s": total_time,
            "mean_decoder_calls": mean_calls,
            "total_tokens": total_tokens,
            "n_eval": n_eval,
            "agg_stats": {k: v for k, v in total_stats.items()
                         if not isinstance(v, list)},
            "draft_lengths": total_stats.get("draft_lengths", []),
            "accepted_lengths": total_stats.get("accepted_lengths", []),
        }

        draft_acc = total_stats.get("n_draft_accepted_tokens", 0)
        draft_att = total_stats.get("n_draft_attempts", 0)
        greedy_steps = total_stats.get("n_greedy_steps", 0)

        print(f"  ▶ Mean WER: {mean_wer:.4f} | Mean time: {mean_time:.3f}s | "
              f"Mean calls: {mean_calls:.1f}")
        if draft_att > 0:
            print(f"    Drafts: {draft_att} attempts, {draft_acc} tokens accepted, "
                  f"{greedy_steps} greedy steps")
            if total_stats.get("accepted_lengths"):
                mean_accept = sum(total_stats["accepted_lengths"]) / len(total_stats["accepted_lengths"])
                print(f"    Avg accepted length: {mean_accept:.2f} tokens/draft")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    baseline = all_results.get("greedy_baseline", {})
    baseline_time = baseline.get("mean_time_s", 1.0)
    baseline_wer = baseline.get("mean_wer", 0.0)
    baseline_calls = baseline.get("mean_decoder_calls", 1.0)

    print(f"\n  {'Config':<25} {'WER':>8} {'ΔWER':>8} {'Time(s)':>8} "
          f"{'Speedup':>8} {'Calls':>8} {'CallSave':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, res in all_results.items():
        delta_wer = res["mean_wer"] - baseline_wer
        speedup = baseline_time / res["mean_time_s"] if res["mean_time_s"] > 0 else 0
        call_save = 1 - (res["mean_decoder_calls"] / baseline_calls) if baseline_calls > 0 else 0
        print(f"  {name:<25} {res['mean_wer']:>8.4f} {delta_wer:>+8.4f} "
              f"{res['mean_time_s']:>8.3f} {speedup:>8.2f}x "
              f"{res['mean_decoder_calls']:>8.1f} {call_save:>+8.1%}")

    # Save
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "ngram_speculation.json"

    save_data = {k: {kk: vv for kk, vv in v.items()
                     if kk not in ("draft_lengths", "accepted_lengths")}
                 for k, v in all_results.items()}
    save_data["config"] = {
        "model": args.model,
        "n_samples": args.n_samples,
        "context_len": args.context_len,
        "max_draft": args.max_draft,
        "q8": args.q8,
        "static_table_size": len(drafter.static_table),
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E5: Speculative N-gram Decoding")
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Total samples (first half=train, second half=eval)")
    parser.add_argument("--context-len", type=int, default=3,
                        help="N-gram context length (trigram=3)")
    parser.add_argument("--max-draft", type=int, default=4,
                        help="Maximum draft tokens per attempt")
    parser.add_argument("--q8", action="store_true",
                        help="Apply Q8 quantization")
    run_experiment(parser.parse_args())
