"""Train the DFlash draft model for Whisper speculative decoding.

Training recipe following DFlash/HunyuanOCR:
  - Target model is frozen; only draft model is trained
  - For each sequence, sample n anchor positions
  - Each anchor produces a block-drafting task
  - Position-weighted cross-entropy loss with exponential decay
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from .utils import get_device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DFlashTrainDataset(Dataset):
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
        audio_summary = torch.from_numpy(data["audio_summary"].astype(np.float32))  # (1, d_model)
        decoder_feats = torch.from_numpy(data["decoder_feats"].astype(np.float32))  # (1, seq_len, feat_dim)
        token_ids = torch.from_numpy(data["token_ids"].astype(np.int64))  # (1, seq_len)
        decoder_input_ids = torch.from_numpy(data["decoder_input_ids"].astype(np.int64))  # (1, seq_len)

        # Remove batch dimension
        audio_summary = audio_summary.squeeze(0)       # (d_model,)
        decoder_feats = decoder_feats.squeeze(0)       # (seq_len, feat_dim)
        token_ids = token_ids.squeeze(0)               # (seq_len,)
        decoder_input_ids = decoder_input_ids.squeeze(0)  # (seq_len,)

        seq_len = token_ids.shape[0]
        B = self.block_size

        # Sample anchor positions: each anchor a_j means we predict tokens[a_j:a_j+B]
        # The anchor token itself is the last accepted token (decoder_input_ids[a_j])
        # Labels are token_ids[a_j:a_j+B]
        max_anchor = seq_len - B
        if max_anchor < 1:
            max_anchor = 1

        n_anchors = min(self.anchors_per_seq, max_anchor)
        anchors = sorted(random.sample(range(max_anchor), n_anchors))

        # Build training blocks
        blocks_input_ids = []   # (n_anchors, B) — decoder_input_ids at anchor positions
        blocks_labels = []      # (n_anchors, B) — target token_ids
        blocks_positions = []   # (n_anchors, B) — position indices
        blocks_context = []     # (n_anchors, ctx_len, feat_dim) — but variable length

        # For simplicity, we use the full decoder_feats up to the anchor
        # and pad/truncate context to a fixed length
        for a_j in anchors:
            # Block input: [decoder_input_ids[a_j], mask, mask, ..., mask]
            block_input = decoder_input_ids[a_j: a_j + B]
            if block_input.shape[0] < B:
                block_input = torch.cat([
                    block_input,
                    torch.zeros(B - block_input.shape[0], dtype=torch.long),
                ])

            # Block labels: token_ids[a_j:a_j+B]
            block_label = token_ids[a_j: a_j + B]
            if block_label.shape[0] < B:
                block_label = torch.cat([
                    block_label,
                    torch.full((B - block_label.shape[0],), -100, dtype=torch.long),
                ])

            # Position ids
            block_pos = torch.arange(a_j, a_j + B)

            blocks_input_ids.append(block_input)
            blocks_labels.append(block_label)
            blocks_positions.append(block_pos)

        return {
            "audio_summary": audio_summary,           # (d_model,)
            "decoder_feats": decoder_feats,            # (seq_len, feat_dim)
            "blocks_input_ids": torch.stack(blocks_input_ids),    # (n_anchors, B)
            "blocks_labels": torch.stack(blocks_labels),          # (n_anchors, B)
            "blocks_positions": torch.stack(blocks_positions),    # (n_anchors, B)
            "anchors": torch.tensor(anchors),                     # (n_anchors,)
            "seq_len": seq_len,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length sequences.

    For simplicity, process one sample at a time (batch_size=1 at the
    dataloader level; effective batch comes from n_anchors per sequence).
    """
    assert len(batch) == 1, "Use batch_size=1; effective batching via anchors"
    return batch[0]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def position_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 7.0,
) -> torch.Tensor:
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
    positions = torch.arange(B, device=logits.device, dtype=torch.float)
    weights = torch.exp(-positions / gamma)
    weights[0] = 0.0  # Don't penalize the anchor position

    # Cross-entropy per position
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    ce = nn.functional.cross_entropy(
        logits_flat, labels_flat, reduction="none", ignore_index=-100
    )
    ce = ce.reshape(n_anchors, B)

    # Apply position weights
    weighted_ce = ce * weights.unsqueeze(0)

    # Average over valid positions
    valid_mask = (labels != -100) & (weights.unsqueeze(0) > 0)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return weighted_ce[valid_mask].mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def prune_checkpoints(output_dir: Path, max_keep: int | None, extension: str = "pt"):
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
            except Exception as e:
                print(f"Error pruning checkpoint {f}: {e}")


def train(
    data_dir: str = "data/train",
    val_dir: str | None = None,
    output_dir: str = "checkpoints",
    model_name: str = "openai/whisper-large-v3",
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    block_size: int = 8,
    anchors_per_seq: int = 16,
    gamma: float = 7.0,
    grad_accumulation: int = 4,
    save_every: int = 1,
    device: torch.device | None = None,
    max_keep: int | None = None,
):
    """Train the DFlash draft model.

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
        device: Torch device.
    """
    from transformers import WhisperForConditionalGeneration

    if device is None:
        device = get_device()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Block size: {block_size}, Anchors/seq: {anchors_per_seq}")

    # Load target model (frozen, for embeddings and lm_head only)
    print("Loading target model (frozen)...")
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    target = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=target_dtype,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # Create draft model
    config = WhisperDFlashConfig(
        d_target=target.config.d_model,
        num_target_layers=target.config.decoder_layers,
        vocab_size=target.config.vocab_size,
        max_target_positions=target.config.max_target_positions,
        block_size=block_size,
    )
    draft_model = WhisperDFlashDraftModel(config).to(device)
    print(f"Draft model parameters: {draft_model.num_parameters:,}")

    # Dataset and dataloader
    train_dataset = DFlashTrainDataset(data_dir, block_size, anchors_per_seq)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    val_loader = None
    if val_dir and Path(val_dir).exists():
        val_dataset = DFlashTrainDataset(val_dir, block_size, anchors_per_seq)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        draft_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs // grad_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )

    # Training
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        draft_model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            audio_summary = batch["audio_summary"].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, d_model)
            decoder_feats = batch["decoder_feats"].unsqueeze(0).to(device)  # (1, seq_len, feat_dim)
            blocks_input_ids = batch["blocks_input_ids"].to(device)  # (n_anchors, B)
            blocks_labels = batch["blocks_labels"].to(device)        # (n_anchors, B)
            blocks_positions = batch["blocks_positions"].to(device)  # (n_anchors, B)
            anchors = batch["anchors"]

            n_anchors = blocks_input_ids.shape[0]

            # Process each anchor block
            all_logits = []
            for j in range(n_anchors):
                a_j = anchors[j].item()

                # Context: decoder_feats up to anchor
                ctx = decoder_feats[:, :a_j, :] if a_j > 0 else decoder_feats[:, :1, :] * 0

                # Embed block input using target's embedding
                block_emb = target.model.decoder.embed_tokens(
                    blocks_input_ids[j:j+1]  # (1, B)
                )  # (1, B, d_target)

                # Run draft model
                hidden = draft_model(
                    noise_embedding=block_emb,
                    target_hidden=ctx,
                    audio_summary=audio_summary,
                    position_ids=blocks_positions[j:j+1],  # (1, B)
                )  # (1, B, d_target)

                # Project to logits
                logits = target.proj_out(hidden)  # (1, B, vocab_size)
                all_logits.append(logits.squeeze(0))  # (B, vocab_size)

            all_logits = torch.stack(all_logits)  # (n_anchors, B, vocab_size)

            # Compute loss
            loss = position_weighted_loss(all_logits, blocks_labels, gamma)
            loss = loss / grad_accumulation
            loss.backward()

            total_loss += loss.item() * grad_accumulation

            # Per-token accuracy (excluding position 0 and padding)
            with torch.no_grad():
                preds = all_logits[:, 1:, :].argmax(dim=-1)  # (n_anchors, B-1)
                labels = blocks_labels[:, 1:]                  # (n_anchors, B-1)
                valid = labels != -100
                if valid.sum() > 0:
                    total_correct += (preds[valid] == labels[valid]).sum().item()
                    total_tokens += valid.sum().item()

            # Gradient step
            if (step + 1) % grad_accumulation == 0:
                nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Warmup
                if global_step <= warmup_steps:
                    warmup_lr = lr * global_step / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

            avg_loss = total_loss / (step + 1)
            avg_acc = total_correct / max(total_tokens, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.3f}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = total_correct / max(total_tokens, 1)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, token_acc={epoch_acc:.3f}")

        # Validation
        if val_loader is not None:
            val_acc = validate(draft_model, target, val_loader, device, block_size, gamma)
            print(f"  Val token_acc={val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": draft_model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                }, output_path / "best_model.pt")
                print(f"  Saved best model (val_acc={val_acc:.3f})")

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = output_path / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "model_state_dict": draft_model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            prune_checkpoints(output_path, max_keep, extension="pt")

    # Save final model
    torch.save({
        "model_state_dict": draft_model.state_dict(),
        "config": config,
        "epoch": epochs,
    }, output_path / "final_model.pt")
    print(f"Training complete. Models saved to {output_path}")


@torch.no_grad()
def validate(
    draft_model: WhisperDFlashDraftModel,
    target: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    block_size: int,
    gamma: float,
) -> float:
    """Compute validation token accuracy."""
    draft_model.eval()
    total_correct = 0
    total_tokens = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        audio_summary = batch["audio_summary"].unsqueeze(0).unsqueeze(0).to(device)
        decoder_feats = batch["decoder_feats"].unsqueeze(0).to(device)
        blocks_input_ids = batch["blocks_input_ids"].to(device)
        blocks_labels = batch["blocks_labels"].to(device)
        blocks_positions = batch["blocks_positions"].to(device)
        anchors = batch["anchors"]

        n_anchors = blocks_input_ids.shape[0]
        for j in range(n_anchors):
            a_j = anchors[j].item()
            ctx = decoder_feats[:, :a_j, :] if a_j > 0 else decoder_feats[:, :1, :] * 0

            block_emb = target.model.decoder.embed_tokens(blocks_input_ids[j:j+1])
            hidden = draft_model(
                noise_embedding=block_emb,
                target_hidden=ctx,
                audio_summary=audio_summary,
                position_ids=blocks_positions[j:j+1],
            )
            logits = target.proj_out(hidden)  # (1, B, vocab_size)

            preds = logits[0, 1:, :].argmax(dim=-1)
            labels = blocks_labels[j, 1:]
            valid = labels != -100
            if valid.sum() > 0:
                total_correct += (preds[valid] == labels[valid]).sum().item()
                total_tokens += valid.sum().item()

    draft_model.train()
    return total_correct / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train DFlash draft model")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--model", default="openai/whisper-large-v3")
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
