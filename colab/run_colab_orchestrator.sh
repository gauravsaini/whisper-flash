#!/bin/bash
set -e

echo "1. Packaging local code..."
tar -czvf whisper-flash.tar.gz pyproject.toml whisper_flash/

echo "2. Starting Colab Session (T4 GPU)..."
colab new -s dflash-e2e --gpu T4

echo "3. Uploading code..."
colab upload -s dflash-e2e whisper-flash.tar.gz

echo "4. Executing E2E pipeline on Colab..."
colab exec -s dflash-e2e << 'EOF'
set -e
echo "Extracting code..."
tar -xzvf whisper-flash.tar.gz

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Syncing dependencies..."
uv sync

echo "======================================"
echo "Generating Dataset (100 samples)"
echo "======================================"
uv run whisper-flash-dataset --max-samples 100 --model openai/whisper-tiny

echo "======================================"
echo "Training Draft Model (2 epochs)"
echo "======================================"
uv run whisper-flash-train --epochs 2 --block-size 8 --anchors 8 --model openai/whisper-tiny

echo "======================================"
echo "Evaluating (50 samples)"
echo "======================================"
uv run whisper-flash-eval --checkpoint checkpoints/best_model.pt --max-samples 50 --block-size 8 --model openai/whisper-tiny
EOF

echo "5. Stopping Colab Session..."
colab stop -s dflash-e2e

echo "Done!"
