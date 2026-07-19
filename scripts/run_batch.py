import json
import subprocess
import sys
import gc
from pathlib import Path
import traceback

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def main():
    manifest_path = Path("artifacts/colab/manifest.json")
    summary_path = Path("artifacts/results/batch_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    summary = {"jobs": []}
    
    for job in manifest["jobs"]:
        job_id = job["id"]
        print(f"\n{'='*50}\nStarting Job: {job_id}\n{'='*50}")
        job_result_path = Path(f"artifacts/results/{job_id}_result.json")
        job_status = {"id": job_id, "status": "failed", "error": None}
        
        try:
            # 1. Dataset Generation
            run_cmd(f"uv run python -m whisper_flash.generate_dataset --max-samples {job['train_samples']} --model {job['model']} --dataset {job['dataset']}")
            
            # 2. Train
            run_cmd(f"uv run python -m whisper_flash.train --epochs {job['epochs']} --block-size {job['block_size']} --anchors {job['anchors']} --model {job['model']}")
            
            # 3. Evaluate
            run_cmd(f"uv run python -m whisper_flash.evaluate --checkpoint checkpoints/best_model.pt --max-samples {job['max_samples']} --block-size {job['block_size']} --model {job['model']} --dataset {job['dataset']} --output-json {job_result_path}")
            
            job_status["status"] = "success"
        except Exception as e:
            traceback.print_exc()
            job_status["error"] = str(e)
            
        summary["jobs"].append(job_status)
        
        # Write summary continuously
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Cleanup memory between jobs
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
            
    print(f"\nBatch processing complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
