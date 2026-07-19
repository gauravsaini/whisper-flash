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

def main():
    # Note: Redirecting HF cache to Google Drive causes FileNotFoundError due to Colab Drive FUSE sync lag.
    # We will keep the cache local.
    
    print("Extracting code...")
    run_cmd("tar -xzvf /content/whisper-flash.tar.gz -C /content")

    print("Installing uv...")
    run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] = f"{os.environ['HOME']}/.local/bin:{os.environ['PATH']}"

    print("Syncing dependencies...")
    run_cmd("uv sync")

    print("Running batch framework...")
    run_cmd("uv run run_batch.py")

if __name__ == "__main__":
    main()
