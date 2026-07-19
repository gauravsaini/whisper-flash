"""CLI entry point: python -m whisper_flash_mlx <audio> [options]."""

import argparse
import sys

from . import transcribe, MODEL_ALIASES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Whisper-Flash: clean greedy ASR with production optimisations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  whisper-flash audio.wav\n"
            "  whisper-flash audio.wav --model tiny\n"
            "  whisper-flash audio.wav --model large-v3 --no-quantize\n"
            "  whisper-flash audio.wav --max-new-tokens 200\n"
        ),
    )
    parser.add_argument("audio", help="Path to audio file (WAV/MP3/FLAC/ogg)")
    parser.add_argument(
        "--model", "-m", default="turbo",
        choices=list(MODEL_ALIASES.keys()) + [f"mlx-community/whisper-{s}-mlx" for s in
            ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
             "large-v3-turbo"]],
        help="Model size or HF repo ID (default: turbo)",
    )
    parser.add_argument(
        "--quantize", default=True, action=argparse.BooleanOptionalAction,
        help="Apply Q8 quantization (default: on)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=448,
        help="Max tokens to generate (default: 448)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run 3 passes and report timing",
    )

    args = parser.parse_args(argv)

    if args.benchmark:
        import time
        times = []
        for run in range(3):
            t0 = time.perf_counter()
            r = transcribe(args.audio, model=args.model,
                           quantize=args.quantize,
                           max_new_tokens=args.max_new_tokens)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        print(f"\n{'='*55}")
        print(f"  Model:       {args.model}")
        print(f"  Q8:          {args.quantize}")
        print(f"  Time (avg):  {avg:.3f}s")
        print(f"  RT factor:   {avg/30:.2f}x")
        print(f"  Tokens/sec:  {r.tokens_per_sec:.1f}")
        print(f"  Steps:       {r.n_decoder_steps}")
        print(f"  Text:        {r.text[:80]}")
        print(f"{'='*55}")
    else:
        r = transcribe(args.audio, model=args.model,
                       quantize=args.quantize,
                       max_new_tokens=args.max_new_tokens)
        print(r.text)


if __name__ == "__main__":
    main()
