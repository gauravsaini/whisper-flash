"""Q8 group-wise affine quantization for MLX Whisper models.

Replaces nn.Linear layers with nn.QuantizedLinear using MLX's native
quantiser.  Stores int8 weights + fp16 scales with efficient Metal
dequantisation kernel.

Usage:
    model = load_target_model("mlx-community/whisper-tiny-mlx")
    quantize_model(model, bits=8, group_size=64)
"""

from __future__ import annotations

from typing import Optional

import mlx.nn as nn

from mlx_whisper.whisper import Whisper


def _quantize_linear(lin: nn.Linear, bits: int, group_size: int) -> nn.QuantizedLinear:
    """Convert an ``nn.Linear`` to ``nn.QuantizedLinear`` (MLX native)."""
    return nn.QuantizedLinear.from_linear(lin, group_size, bits, mode="affine")


def quantize_model(
    model: Whisper,
    encoder_bits: Optional[int] = 8,
    decoder_bits: Optional[int] = 8,
    group_size: int = 64,
):
    """Replace ``nn.Linear`` layers with ``nn.QuantizedLinear`` in-place.

    Uses manual DFS traversal via ``children()`` to avoid MLX's
    ``named_modules`` recursion issues.
    """
    _replace_linears_manual(model.encoder, encoder_bits, group_size)
    _replace_linears_manual(model.decoder, decoder_bits, group_size)


def _replace_linears_manual(module: object, bits: Optional[int], group_size: int):
    if bits is None:
        return

    to_replace = []
    to_recurse = []

    if hasattr(module, "children") and callable(getattr(module, "children")):
        try:
            children_map = module.children()
            if isinstance(children_map, dict):
                for name, child in children_map.items():
                    if isinstance(child, nn.Linear):
                        to_replace.append((name, child, module))
                    elif isinstance(child, nn.Module):
                        to_recurse.append(child)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, nn.Module):
                                to_recurse.append(item)
        except Exception:
            pass

    for name, lin, parent in to_replace:
        q = _quantize_linear(lin, bits, group_size)
        setattr(parent, name, q)

    for child_mod in to_recurse:
        _replace_linears_manual(child_mod, bits, group_size)


# ── Utility ─────────────────────────────────────────────────────

def _walk_modules(model: object) -> list[nn.Module]:
    """BFS over model tree via children()."""
    seen = set()
    out = []
    stack = [model]
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if isinstance(m, (nn.Linear, nn.QuantizedLinear)):
            out.append(m)
            continue
        if hasattr(m, "children") and callable(getattr(m, "children")):
            try:
                cm = m.children()
                if isinstance(cm, dict):
                    for child in cm.values():
                        if isinstance(child, (nn.Module, nn.Linear, nn.QuantizedLinear)):
                            stack.append(child)
                        elif isinstance(child, list):
                            stack.extend(item for item in child
                                         if isinstance(item, nn.Module))
            except Exception:
                pass
    return out


def count_params(model: Whisper) -> dict:
    """Count total and quantized params."""
    total = 0
    quant = 0
    for m in _walk_modules(model):
        if isinstance(m, nn.QuantizedLinear):
            quant += m.weight.size
            total += m.weight.size
        elif isinstance(m, nn.Linear):
            total += m.weight.size
    return {"total": total, "quantized": quant}


def print_quantization_stats(model: Whisper):
    """Count parameters and estimate memory savings."""
    stats = count_params(model)
    total = stats["total"]
    quant = stats["quantized"]
    fp16_bytes = total * 2
    q8_bytes = quant * 1 + (total - quant) * 2
    print(f"  Total params:  {total:,}")
    print(f"  Quantised:     {quant:,} ({100*quant/total:.0f}%)")
    print(f"  Memory (fp16): {fp16_bytes/1024/1024:.1f} MB")
    print(f"  Memory (Q8):   {q8_bytes/1024/1024:.1f} MB")
    print(f"  Savings:       {(1 - q8_bytes/fp16_bytes)*100:.0f}%")


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    from whisper_flash_mlx.target_model import load_target_model
    model = load_target_model(args.model)
    print(f"Model: {args.model}")
    print(f"Bits:  {args.bits}")
    print(f"Group: {args.group_size}")
    print("Before quantization:")
    print_quantization_stats(model)
    quantize_model(model, args.bits, args.bits, args.group_size)
    print("After quantization:")
    print_quantization_stats(model)
