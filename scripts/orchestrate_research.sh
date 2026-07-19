#!/bin/bash
set -e

echo "1. Packaging local code..."
tar -czvf whisper-flash.tar.gz pyproject.toml whisper_flash/ run_batch.py artifacts/colab/manifest.json

echo "2. Starting Colab Session (T4 GPU)..."
colab new -s dflash-research --gpu T4

echo "3. Uploading code..."
colab upload -s dflash-research whisper-flash.tar.gz /content/whisper-flash.tar.gz

echo "4. Executing batch queue on Colab..."
colab exec -s dflash-research --timeout 3600 << 'EOF'
import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/content")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

print("Extracting code...")
run_cmd("tar -xzvf whisper-flash.tar.gz")

print("Installing uv...")
run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:{os.environ['PATH']}"

print("Syncing dependencies...")
run_cmd("uv sync")

print("Running batch framework...")
run_cmd("uv run run_batch.py")
EOF

echo "5. Downloading artifacts (Download Contract)..."
mkdir -p artifacts/results
colab download -s dflash-research /content/artifacts/results/batch_summary.json artifacts/results/batch_summary.json || echo "Warning: Failed to download summary"
colab download -s dflash-research /content/artifacts/results/job_1_b8_result.json artifacts/results/job_1_b8_result.json || echo "Warning: Failed to download job 1 result"

echo "6. Stopping Colab Session..."
colab stop -s dflash-research

echo "Batch research complete! Artifacts saved to artifacts/results/"
