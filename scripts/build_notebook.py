import json
from pathlib import Path

def make_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source and source[-1] == "\n":
            source.pop()
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

def main():
    cells = []
    
    # 1. Introduction
    cells.append(make_cell("markdown", """# Whisper-DFlash Speculative Decoding Benchmark

This notebook demonstrates the end-to-end implementation of the DFlash speculative decoding pipeline for `openai/whisper-large-v3` (or `openai/whisper-tiny` for testing)."""))

    # 2. Drive mount block
    cells.append(make_cell("markdown", "### Step 1: Mount Google Drive\nMounting Google Drive allows us to extract the workspace and save all generated datasets and checkpoints directly to your Drive."))
    cells.append(make_cell("code", """from google.colab import drive
drive.mount('/content/drive')"""))

    # 3. Workspace Setup & Unzipping
    cells.append(make_cell("markdown", "### Step 2: Unzip Workspace & Setup Environment\nWe extract the zip file explicitly into a safe folder (`/content/drive/MyDrive/whisper-flash-workspace`) and set the python paths so modules resolve correctly."))
    cells.append(make_cell("code", """import os
import zipfile
import sys

# Define the root directory where we want the project to live
workspace_dir = '/content/drive/MyDrive/whisper-flash-workspace'
os.makedirs(workspace_dir, exist_ok=True)

# 1. Locate zip file and extract it safely into the workspace folder
zip_path = "/content/whisper-flash.zip"
gdrive_zip_path = "/content/drive/MyDrive/whisper-flash.zip"

extracted = False
if os.path.exists(zip_path):
    print(f"Extracting {zip_path} to {workspace_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(workspace_dir)
    extracted = True
elif os.path.exists(gdrive_zip_path):
    print(f"Extracting {gdrive_zip_path} to {workspace_dir}...")
    with zipfile.ZipFile(gdrive_zip_path, 'r') as zip_ref:
        zip_ref.extractall(workspace_dir)
    extracted = True
else:
    print("No zip file found. Assuming workspace is already unzipped.")

# 2. Automatically find the actual project root (where pyproject.toml is)
project_root = workspace_dir
for root, dirs, files in os.walk(workspace_dir):
    if 'pyproject.toml' in files:
        project_root = root
        break

# 3. Change directory using Jupyter %cd magic
print(f"\\nChanging working directory to: {project_root}")
%cd {project_root}

# 4. EXPLICITLY fix Colab PYTHONPATH to ensure `python -m whisper_flash...` works
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

print("\\nCurrent directory contents:")
!ls -la

# 5. Install required packages
!pip install -q datasets soundfile jiwer librosa transformers tqdm accelerate torchcodec"""))

    # 4. Verify Codebase
    cells.append(make_cell("markdown", "### Step 3: Verify Codebase Imports\nEnsure the unzipped package modules import correctly."))
    cells.append(make_cell("code", """import torch
from whisper_flash.utils import get_device
from whisper_flash.draft_model import WhisperDFlashConfig, WhisperDFlashDraftModel
from whisper_flash.generate import whisper_dflash_generate

device = get_device()
print(f"Import check successful! Device: {device}")"""))

    # 5. Run Dataset Generation
    cells.append(make_cell("markdown", "### Step 4: Run Dataset Generation & Save to Google Drive\nWe extract context vectors from Whisper for 100 audio samples to train our DFlash model. The generated `.npz` files are saved directly in your Google Drive under `./data/train` for persistence."))
    cells.append(make_cell("code", """# Run dataset generation (outputs saved directly to Google Drive)
!python -m whisper_flash.generate_dataset --max-samples 100 --model openai/whisper-tiny --dataset openslr/librispeech_asr --output-dir ./data/train"""))

    # 6. Train DFlash Draft Model
    cells.append(make_cell("markdown", "### Step 5: Train DFlash Draft Model\nWe train the DFlash draft model for 2 epochs on the extracted features. The trained model checkpoint will be saved directly to your Google Drive under `./checkpoints/best_model.pt`."))
    cells.append(make_cell("code", """# Train model (checkpoints saved directly to Google Drive)
!python -m whisper_flash.train --epochs 2 --block-size 8 --anchors 8 --model openai/whisper-tiny --data-dir ./data/train --checkpoint-dir ./checkpoints"""))

    # 7. Benchmark and Evaluate DFlash
    cells.append(make_cell("markdown", "### Step 6: Benchmark and Evaluate DFlash\nWe evaluate the model comparing throughput speedup, latency reduction, and Word Error Rate (WER) consistency on the test split."))
    cells.append(make_cell("code", """# Run benchmark
!python -m whisper_flash.evaluate --checkpoint ./checkpoints/best_model.pt --max-samples 50 --block-size 8 --model openai/whisper-tiny --dataset openslr/librispeech_asr"""))

    # 8. Interactive Speculative Generation
    cells.append(make_cell("markdown", """### Step 7: Interactive Speculative Generation
Load a custom audio sample from the test set and witness DFlash Speculative Decoding live!"""))
    cells.append(make_cell("code", """import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper_flash.generate import whisper_dflash_generate
from whisper_flash.evaluate import load_draft_model, baseline_generate

# Setup devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load models
target = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", torch_dtype=target_dtype).to(device).eval()
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
draft = load_draft_model("./checkpoints/best_model.pt", device)

# Load a test sample
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
sample_data = ds[0]
audio = sample_data["audio"]["array"]
sr = sample_data["audio"]["sampling_rate"]

# Process input mel features
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device).to(target_dtype)

# Run Speculative Decoding
df_out = whisper_dflash_generate(draft, target, inputs, return_stats=True)
df_text = processor.batch_decode(df_out.output_ids, skip_special_tokens=True)[0]

# Print stats
print("\\nGenerated Text:", df_text)
print(f"Average tokens accepted per step: {sum(df_out.acceptance_lengths) / len(df_out.acceptance_lengths):.2f}")
print(f"Time per output token: {df_out.time_per_output_token*1000:.1f} ms")
"""))

    # Notebook final assembly
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    out_path = Path("whisper_dflash_benchmark.ipynb")
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook at: {out_path.absolute()}")

if __name__ == "__main__":
    main()
