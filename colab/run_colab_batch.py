#!/usr/bin/env python3
"""
Colab batch runner for adaptive Δz correction evaluation.

Usage:
  python run_colab_batch.py --manifest batch_manifest.json [--gpu T4] [--dry-run]

This provisions a Colab GPU VM, sequentially runs each manifest entry,
downloads JSON artifacts, and cleans up.

Manifest format (JSON):
[
  {"name": "tiny-10", "model": "openai/whisper-tiny", "train": 10, "eval": 20, "gates": "top1,adaptive"},
  {"name": "turbo-20", "model": "openai/whisper-large-v3-turbo", "train": 20, "eval": 50, "gates": "top1,top3,adaptive", "kv_cache": true}
]
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

# The evaluator is currently tracked in-repo under a .bak suffix.
SCRIPT = "colab_eval_adaptive.py.bak"
RESULTS_DIR = Path("results/colab")
MANIFEST_PATH = "batch_manifest.json"


def ensure_auth():
    """Verify colab CLI is authenticated."""
    r = subprocess.run(["colab", "sessions"], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        print(f"❌ colab CLI not authenticated. Run: colab sessions", flush=True)
        print(f"   Error: {r.stderr[:200]}", flush=True)
        sys.exit(1)
    print("✅ colab CLI authenticated", flush=True)


def run_batch(manifest, gpu, dry_run, session_name):
    """Run all manifest entries sequentially on one Colab session."""
    ensure_auth()

    # Create results dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set per-entry artifact paths
    for entry in manifest:
        fname = f"{entry['name']}.json"
        entry["_artifact"] = str(RESULTS_DIR / fname)

    # Provision VM
    if not dry_run:
        print(f"\n🚀 Provisioning Colab VM ({gpu or 'CPU'})...", flush=True)
        cmd = ["colab", "new", "-s", session_name]
        if gpu:
            cmd.extend(["--gpu", gpu])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"❌ Failed to provision VM: {r.stderr[:500]}", flush=True)
            print(f"   Stderr: {r.stderr}", flush=True)
            print(f"   Stdout: {r.stdout}", flush=True)
            sys.exit(1)
        print(f"✅ VM provisioned (session: {session_name})", flush=True)
        print(f"   {r.stdout.strip()}", flush=True)

        # Install packages
        print("📦 Installing dependencies...", flush=True)
        subprocess.run(["colab", "install", "-s", session_name,
                        "torch", "transformers", "datasets", "jiwer", "accelerate"],
                       timeout=120)
    else:
        print(f"\n🚀 [DRY RUN] Would provision Colab VM ({gpu or 'CPU'})", flush=True)

    # Run each entry
    total = len(manifest)
    for i, entry in enumerate(manifest):
        name = entry["name"]
        artifact = entry["_artifact"]
        train_s = entry["train"]
        eval_s = entry["eval"]
        gates = entry.get("gates", "top1,adaptive")
        model = entry["model"]

        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{total}] {name} — {model} train={train_s} eval={eval_s}", flush=True)
        print(f"{'='*60}", flush=True)

        args = [
            "--model", model,
            "--train-samples", str(train_s),
            "--eval-samples", str(eval_s),
            "--gate-modes", gates,
            "--out", f"/content/results/{name}.json",
        ]
        if entry.get("dataset"):
            args.extend(["--dataset", entry["dataset"]])
        if entry.get("dataset_config"):
            args.extend(["--dataset-config", entry["dataset_config"]])
        if entry.get("dataset_split"):
            args.extend(["--dataset-split", entry["dataset_split"]])
        if entry.get("kv_cache"):
            args.append("--kv-cache")
        if entry.get("pca_rank"):
            args.extend(["--pca-rank", str(entry["pca_rank"])])
        if entry.get("d_draft"):
            args.extend(["--d-draft", str(entry["d_draft"])])

        if dry_run:
            print(f"   [DRY RUN] Would run: colab exec -s {session_name} -f {SCRIPT} {' '.join(args)}", flush=True)
            print(f"   [DRY RUN] Would download: /content/results/{name}.json → {artifact}", flush=True)
            continue

        # Create remote results dir
        subprocess.run(["colab", "exec", "-s", session_name],
                       input="import os; os.makedirs('/content/results', exist_ok=True)\n",
                       text=True, timeout=15)

        # Run evaluation
        t0 = time.time()
        r = subprocess.run(
            ["colab", "exec", "-s", session_name, "-f", SCRIPT, "--"] + args,
            capture_output=True, text=True, timeout=3600)  # 1h per run
        elapsed = time.time() - t0

        # Print output
        if r.stdout:
            print(r.stdout[-3000:], flush=True)  # last 3000 chars
        if r.stderr:
            # colab CLI chatter goes to stderr; filter the script's stderr from it
            for line in r.stderr.splitlines():
                if "[colab]" not in line:
                    print(line, flush=True)

        if r.returncode == 0:
            print(f"   ✅ {name} completed in {elapsed:.0f}s", flush=True)
        else:
            print(f"   ⚠️  {name} finished with code {r.returncode} in {elapsed:.0f}s", flush=True)

        # Download artifact
        subprocess.run(
            ["colab", "download", "-s", session_name,
             f"/content/results/{name}.json", str(artifact)],
            capture_output=True, timeout=30)
        if os.path.exists(artifact):
            print(f"   📥 Downloaded → {artifact}", flush=True)
        else:
            print(f"   ⚠️  Artifact not found at {artifact}", flush=True)

    # Cleanup
    if not dry_run:
        print(f"\n🧹 Stopping session {session_name}...", flush=True)
        subprocess.run(["colab", "stop", "-s", session_name], capture_output=True, timeout=30)
        print("✅ Session stopped", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"BATCH COMPLETE — {total} runs", flush=True)
    print(f"{'='*60}", flush=True)
    for entry in manifest:
        ap = entry["_artifact"]
        if os.path.exists(ap):
            with open(ap) as f:
                d = json.load(f)
            r = d["results"]
            gw = r["greedy"]["mean_wer"]
            print(f"  {entry['name']:20s} greedy={gw:.4f}", flush=True)
            for mode, info in r["modes"].items():
                print(f"    {mode:15s} WER={info['mean_wer']:.4f} ({info['wer_delta']:+.4f})  "
                      f"accept={info['accept_pct']:.1f}%", flush=True)
        else:
            print(f"  {entry['name']:20s} ❌ no results", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Colab batch runner")
    parser.add_argument("--manifest", default=MANIFEST_PATH,
                        help="Path to batch manifest JSON")
    parser.add_argument("--gpu", default="T4",
                        help="GPU type: T4, L4, A100 (default: T4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    parser.add_argument("--session", default="flash-eval",
                        help="Colab session name")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"📋 Loaded manifest: {args.manifest} ({len(manifest)} entries)", flush=True)
    run_batch(manifest, args.gpu, args.dry_run, args.session)


if __name__ == "__main__":
    main()
