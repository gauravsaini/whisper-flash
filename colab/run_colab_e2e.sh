#!/bin/bash
set -e

# Install uv on the Colab VM
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies and sync project
uv sync

echo "======================================"
echo "1. Generating Dataset (100 samples)"
echo "======================================"
uv run whisper-flash-dataset --max-samples 100 --model openai/whisper-tiny

echo "======================================"
echo "2. Training Draft Model (2 epochs)"
echo "======================================"
uv run whisper-flash-train --epochs 2 --block-size 8 --anchors 8 --model openai/whisper-tiny

echo "======================================"
echo "3. Evaluating (50 samples)"
echo "======================================"
uv run whisper-flash-eval --checkpoint checkpoints/best_model.pt --max-samples 50 --block-size 8 --model openai/whisper-tiny
