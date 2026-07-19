import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("Installing uv...")
    run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:{os.environ['PATH']}"
    
    print("Syncing dependencies...")
    run_cmd("uv sync")
    
    print("======================================")
    print("1. Generating Dataset (100 samples)")
    print("======================================")
    run_cmd("uv run whisper-flash-dataset --max-samples 100 --model openai/whisper-tiny")
    
    print("======================================")
    print("2. Training Draft Model (2 epochs)")
    print("======================================")
    run_cmd("uv run whisper-flash-train --epochs 2 --block-size 8 --anchors 8 --model openai/whisper-tiny")
    
    print("======================================")
    print("3. Evaluating (50 samples)")
    print("======================================")
    run_cmd("uv run whisper-flash-eval --checkpoint checkpoints/best_model.pt --max-samples 50 --block-size 8 --model openai/whisper-tiny")

if __name__ == "__main__":
    main()
