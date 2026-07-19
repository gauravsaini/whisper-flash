"""Train the DFlash draft model for Whisper speculative decoding — MLX version.

Training recipe following DFlash/HunyuanOCR, running entirely on Apple Silicon:
  - Target model is frozen; only draft model is trained
  - For each sequence, sample n anchor positions
  - Each anchor produces a block-drafting task
  - Position-weighted cross-entropy loss with exponential decay
  - Uses MLX's value_and_grad for functional-style gradient computation
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
from tqdm import tqdm

from .draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from .target_model import get_token_embedding, load_target_model, project_to_logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DFlashTrainDataset:
    """Dataset of pre-computed Whisper traces for DFlash training.

    Each sample is an NPZ file containing:
      - audio_summary: (1, d_model) float16
      - decoder_feats: (1, seq_len, num_taps * d_model) float16
      - token_ids: (1, seq_len) int
      - decoder_input_ids: (1, seq_len) int
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 8,
        anchors_per_seq: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.anchors_per_seq = anchors_per_seq
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if not self.files:
            raise ValueError(f"No sample files found in {data_dir}")
        print(f"Found {len(self.files)} training samples in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx])
        audio_summary = mx.array(data["audio_summary"].astype(np.float32))  # (1, d_model)
        decoder_feats = mx.array(data["decoder_feats"].astype(np.float32))  # (1, seq_len, feat_dim)
        token_ids = mx.array(data["token_ids"].astype(np.int64))  # (1, seq_len)
        decoder_input_ids = mx.array(data["decoder_input_ids"].astype(np.int64))  # (1, seq_len)

        # Remove batch dimension
        audio_summary = audio_summary.squeeze(0)        # (d_model,)
        decoder_feats = decoder_feats.squeeze(0)        # (seq_len, feat_dim)
        token_ids = token_ids.squeeze(0)                # (seq_len,)
        decoder_input_ids = decoder_input_ids.squeeze(0)  # (seq_len,)

        seq_len = token_ids.shape[0]
        B = self.block_size

        # Sample anchor positions
        max_anchor = seq_len - B
        if max_anchor < 1:
            max_anchor = 1

        n_anchors = min(self.anchors_per_seq, max_anchor)
        anchors = sorted(random.sample(range(max_anchor), n_anchors))

        # Build training blocks
        blocks_input_ids = []
        blocks_labels = []
        blocks_positions = []

        for a_j in anchors:
            # Block input
            block_input = decoder_input_ids[a_j: a_j + B]
            if block_input.shape[0] < B:
                pad = mx.zeros((B - block_input.shape[0],), dtype=mx.int32)
                block_input = mx.concatenate([block_input, pad])

            # Block labels
            block_label = token_ids[a_j: a_j + B]
            if block_label.shape[0] < B:
                pad = mx.full((B - block_label.shape[0],), -100, dtype=mx.int32)
                block_label = mx.concatenate([block_label, pad])

            # Position ids
            block_pos = mx.arange(a_j, a_j + B)

            blocks_input_ids.append(block_input)
            blocks_labels.append(block_label)
            blocks_positions.append(block_pos)

        return {
            "audio_summary": audio_summary,
            "decoder_feats": decoder_feats,
            "blocks_input_ids": mx.stack(blocks_input_ids),
            "blocks_labels": mx.stack(blocks_labels),
            "blocks_positions": mx.stack(blocks_positions),
            "anchors": anchors,
            "seq_len": seq_len,
        }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def position_weighted_loss(
    logits: mx.array,
    labels: mx.array,
    gamma: float = 7.0,
) -> mx.array:
    """Position-weighted cross-entropy loss from DFlash.

    Weight w_k = exp(-k / gamma) for position k in the block.
    Position 0 (the anchor token) gets weight 0 (we predict from position 1).

    Args:
        logits: (n_anchors, B, vocab_size)
        labels: (n_anchors, B)
        gamma: Exponential decay factor.

    Returns:
        Scalar loss.
    """
    n_anchors, B, vocab_size = logits.shape

    # Compute per-position weights: w_0 = 0, w_k = exp(-k/gamma) for k >= 1
    positions = mx.arange(B).astype(mx.float32)
    weights = mx.exp(-positions / gamma)
    # Zero out anchor position using a mask (MLX arrays are immutable)
    mask = mx.array([0.0] + [1.0] * (B - 1))
    weights = weights * mask


    # Cross-entropy per position
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Manual cross-entropy: -log_softmax[label]
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)

    # Gather log probs at label positions
    # Handle ignore_index=-100 by clamping to 0 for gathering
    safe_labels = mx.maximum(labels_flat, 0)
    # One-hot gather
    one_hot = mx.zeros_like(log_probs)
    # Use take_along_axis equivalent
    ce = -mx.take_along_axis(log_probs, safe_labels[:, None], axis=1).squeeze(1)
    ce = ce.reshape(n_anchors, B)

    # Apply position weights and valid mask
    valid_mask = (labels != -100) & (weights[None, :] > 0)
    weighted_ce = ce * weights[None, :]
    # Zero out invalid positions
    weighted_ce = weighted_ce * valid_mask

    n_valid = mx.sum(valid_mask)
    # Avoid division by zero
    return mx.where(n_valid > 0, mx.sum(weighted_ce) / n_valid, mx.array(0.0))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def prune_checkpoints(output_dir: Path, max_keep: int | None, extension: str = "safetensors"):
    """Keep only the latest max_keep checkpoints in output_dir."""
    if max_keep is None or max_keep <= 0:
        return

    checkpoint_files = list(output_dir.glob(f"checkpoint_epoch*.{extension}"))
    checkpoints = []
    for f in checkpoint_files:
        try:
            epoch = int(f.stem.replace("checkpoint_epoch", ""))
            checkpoints.append((epoch, f))
        except ValueError:
            continue

    checkpoints.sort()
    if len(checkpoints) > max_keep:
        to_delete = checkpoints[:-max_keep]
        for epoch, f in to_delete:
            try:
                f.unlink()
                print(f"Pruned old checkpoint: {f}")
                if extension == "safetensors":
                    cfg_file = f.parent / f"config_epoch{epoch}.json"
                    if cfg_file.exists():
                        cfg_file.unlink()
                        print(f"Pruned old config: {cfg_file}")
            except Exception as e:
                print(f"Error pruning checkpoint {f}: {e}")


def train(
    data_dir: str = "data/train",
    val_dir: str | None = None,
    output_dir: str = "checkpoints",
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    block_size: int = 8,
    anchors_per_seq: int = 16,
    gamma: float = 7.0,
    grad_accumulation: int = 4,
    save_every: int = 1,
    max_keep: int | None = None,
):
    """Train the DFlash draft model entirely on Apple Silicon via MLX.

    Args:
        data_dir: Directory containing NPZ training data.
        val_dir: Optional validation data directory.
        output_dir: Directory to save checkpoints.
        model_name: Target Whisper model (for embedding/lm_head).
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        block_size: DFlash block size B.
        anchors_per_seq: Number of anchor positions per sequence.
        gamma: Position weight decay factor.
        grad_accumulation: Gradient accumulation steps.
        save_every: Save checkpoint every N epochs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Block size: {block_size}, Anchors/seq: {anchors_per_seq}")

    # Load target model (frozen, for embeddings and lm_head only)
    print("Loading target model (frozen)...")
    target = load_target_model(model_name)

    # Create draft model
    config = WhisperDFlashConfig(
        d_target=target.dims.n_text_state,
        num_target_layers=target.dims.n_text_layer,
        vocab_size=target.dims.n_vocab,
        max_target_positions=target.dims.n_text_ctx,
        block_size=block_size,
    )
    draft_model = WhisperDFlashDraftModel(config)

    # TRICK 1: Initialize draft model from target model's last layers
    # Target model has d_model (e.g. 384 for tiny). Draft FFN should match target FFN.
    # The paper says: initialized from the last 5 decoder layers of the target model.
    target_layers = target.decoder.blocks[-config.num_layers:]
    for i, t_layer in enumerate(target_layers):
        # We can copy the FFN weights which are the largest parameter blocks.
        # mlx models use 'mlp.0.weight' and 'mlp.2.weight' for FFN
        if "mlp.0.weight" in t_layer.parameters():
            draft_model.layers[i].fc1.update({"weight": t_layer.mlp[0].weight, "bias": t_layer.mlp[0].bias})
            draft_model.layers[i].fc2.update({"weight": t_layer.mlp[2].weight, "bias": t_layer.mlp[2].bias})

    mx.eval(draft_model.parameters())
    print(f"Draft model parameters: {draft_model.count_params():,}")

    # Dataset
    train_dataset = DFlashTrainDataset(data_dir, block_size, anchors_per_seq)

    val_dataset = None
    if val_dir and Path(val_dir).exists():
        val_dataset = DFlashTrainDataset(val_dir, block_size, anchors_per_seq)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    # Define the loss function for a single anchor
    def single_anchor_loss(draft_model, block_emb, ctx, audio_summary, positions, labels):
        """Compute loss for a single anchor block."""
        hidden = draft_model(
            noise_embedding=block_emb,
            target_hidden=ctx,
            audio_summary=audio_summary,
            position_ids=positions,
        )  # (1, B, d_target)

        logits = project_to_logits(target, hidden)  # (1, B, vocab_size)
        return logits

    # Training
    best_val_acc = 0.0
    global_step = 0

    # Accumulated gradients (tree of arrays, same shape as model params)
    accumulated_grads = None

    for epoch in range(epochs):
        draft_model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs}")
        for step_idx, idx in enumerate(pbar):
            batch = train_dataset[idx]

            audio_summary = batch["audio_summary"][None, None, :]  # (1, 1, d_model)
            decoder_feats = batch["decoder_feats"][None, :]  # (1, seq_len, feat_dim)
            blocks_input_ids = batch["blocks_input_ids"]   # (n_anchors, B)
            blocks_labels = batch["blocks_labels"]         # (n_anchors, B)
            blocks_positions = batch["blocks_positions"]   # (n_anchors, B)
            anchors = batch["anchors"]

            n_anchors = blocks_input_ids.shape[0]

            n_anchors = blocks_input_ids.shape[0]

            # -----------------------------------------------------
            # TRICK 2: Parallelizing Anchor Training with Masking
            # -----------------------------------------------------
            max_anchor = max(anchors)
            if max_anchor == 0:
                max_anchor = 1

            ctx_list = []
            mask_list = []
            for j in range(n_anchors):
                a_j = anchors[j]
                if a_j > 0:
                    c = decoder_feats[0, :a_j, :]
                    pad = max_anchor - a_j
                    if pad > 0:
                        c = mx.concatenate([c, mx.zeros((pad, c.shape[-1]))], axis=0)
                    m = mx.concatenate([mx.zeros((a_j,)), mx.full((max_anchor - a_j,), -1e9)], axis=0)
                else:
                    c = mx.zeros((max_anchor, decoder_feats.shape[-1]))
                    m = mx.full((max_anchor,), -1e9)
                    
                ctx_list.append(c)
                mask_list.append(m)

            batched_ctx = mx.stack(ctx_list) # (n_anchors, max_anchor, d_target)
            ctx_mask = mx.stack(mask_list) # (n_anchors, max_anchor)

            audio_mask = mx.zeros((n_anchors, 1))
            draft_mask = mx.zeros((n_anchors, block_size))
            full_mask = mx.concatenate([audio_mask, ctx_mask, draft_mask], axis=1) # (n_anchors, kv_len)
            full_mask = full_mask[:, None, None, :] # (n_anchors, 1, 1, kv_len)

            batched_emb = get_token_embedding(target, blocks_input_ids) # (n_anchors, B, d_target)
            audio_summary_batched = mx.repeat(audio_summary, n_anchors, axis=0) # (n_anchors, 1, d_target)

            def loss_fn(model):
                hidden = model(
                    noise_embedding=batched_emb,
                    target_hidden=batched_ctx,
                    audio_summary=audio_summary_batched,
                    position_ids=blocks_positions,
                    mask=full_mask,
                )
                all_logits = project_to_logits(target, hidden)
                return position_weighted_loss(all_logits, blocks_labels, gamma), all_logits

            (loss, all_logits), grads = nn.value_and_grad(draft_model, loss_fn)(draft_model)
            mx.eval(loss, grads, all_logits)

            total_loss += loss.item()

            # Per-token accuracy (excluding position 0 and padding)
            preds = mx.argmax(all_logits[:, 1:, :], axis=-1)
            labels_check = blocks_labels[:, 1:]
            valid = labels_check != -100
            if mx.sum(valid).item() > 0:
                correct = mx.sum((preds == labels_check) & valid).item()
                total_correct += correct
                total_tokens += mx.sum(valid).item()

            # Gradient accumulation
            if accumulated_grads is None:
                # Scale grads by 1/grad_accumulation
                accumulated_grads = tree_map(
                    lambda g: g / grad_accumulation, grads
                )
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g / grad_accumulation,
                    accumulated_grads, grads,
                )

            # Gradient step
            if (step_idx + 1) % grad_accumulation == 0:
                # Clip gradients
                grad_norm = mx.sqrt(
                    sum(mx.sum(g * g).item() for _, g in tree_flatten(accumulated_grads))
                )
                if grad_norm > 1.0:
                    accumulated_grads = tree_map(
                        lambda g: g / grad_norm, accumulated_grads
                    )

                optimizer.update(draft_model, accumulated_grads)
                mx.eval(draft_model.parameters(), optimizer.state)
                accumulated_grads = None
                global_step += 1

                # Warmup
                if global_step <= warmup_steps:
                    warmup_lr = lr * global_step / warmup_steps
                    optimizer.learning_rate = warmup_lr

            avg_loss = total_loss / (step_idx + 1)
            avg_acc = total_correct / max(total_tokens, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

        epoch_loss = total_loss / len(train_dataset)
        epoch_acc = total_correct / max(total_tokens, 1)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, token_acc={epoch_acc:.3f}")

        # Validation
        if val_dataset is not None:
            val_acc = validate(draft_model, target, val_dataset, block_size, gamma)
            print(f"  Val token_acc={val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                draft_model.save_weights(str(output_path / "best_model.safetensors"))
                _save_config(config, output_path / "best_config.json", epoch + 1, val_acc)
                print(f"  Saved best model (val_acc={val_acc:.3f})")

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            draft_model.save_weights(
                str(output_path / f"checkpoint_epoch{epoch+1}.safetensors")
            )
            _save_config(config, output_path / f"config_epoch{epoch+1}.json", epoch + 1)
            prune_checkpoints(output_path, max_keep, extension="safetensors")

    # Save final model
    draft_model.save_weights(str(output_path / "final_model.safetensors"))
    _save_config(config, output_path / "final_config.json", epochs)
    print(f"Training complete. Models saved to {output_path}")


def _save_config(config: WhisperDFlashConfig, path: Path, epoch: int, val_acc: float | None = None):
    """Save config + metadata as JSON."""
    import json
    from dataclasses import asdict
    data = asdict(config)
    data["epoch"] = epoch
    if val_acc is not None:
        data["val_acc"] = val_acc
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def validate(
    draft_model: WhisperDFlashDraftModel,
    target,
    val_dataset: DFlashTrainDataset,
    block_size: int,
    gamma: float,
) -> float:
    """Compute validation token accuracy."""
    draft_model.eval()
    total_correct = 0
    total_tokens = 0

    for idx in tqdm(range(len(val_dataset)), desc="Validation", leave=False):
        batch = val_dataset[idx]

        audio_summary = batch["audio_summary"][None, None, :]
        decoder_feats = batch["decoder_feats"][None, :]
        blocks_input_ids = batch["blocks_input_ids"]
        blocks_labels = batch["blocks_labels"]
        blocks_positions = batch["blocks_positions"]
        anchors = batch["anchors"]

        n_anchors = blocks_input_ids.shape[0]
        for j in range(n_anchors):
            a_j = anchors[j]
            if a_j > 0:
                ctx = decoder_feats[:, :a_j, :]
            else:
                ctx = decoder_feats[:, :1, :] * 0

            block_emb = get_token_embedding(target, blocks_input_ids[j:j+1])
            hidden = draft_model(
                noise_embedding=block_emb,
                target_hidden=ctx,
                audio_summary=audio_summary,
                position_ids=blocks_positions[j:j+1],
            )
            logits = project_to_logits(target, hidden)  # (1, B, vocab_size)

            preds = mx.argmax(logits[0, 1:, :], axis=-1)
            labels = blocks_labels[j, 1:]
            valid = labels != -100
            if mx.sum(valid).item() > 0:
                total_correct += mx.sum((preds == labels) & valid).item()
                total_tokens += mx.sum(valid).item()

    draft_model.train()
    return total_correct / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train DFlash draft model (MLX)")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=7.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-keep", type=int, default=None, help="Max checkpoints to keep")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        block_size=args.block_size,
        anchors_per_seq=args.anchors,
        gamma=args.gamma,
        grad_accumulation=args.grad_accum,
        max_keep=args.max_keep,
    )


if __name__ == "__main__":
    main()
