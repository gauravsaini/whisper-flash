#!/bin/bash
# run_remote.sh — Copy experiment script and run on remote Mac via SSH
set -e

REMOTE="ektasaini@192.168.68.90"
KEY="/Users/ektasaini/Desktop/id_ed25519"
SCRIPT="experiment_id53.py"
REMOTE_DIR="/Users/ektasaini/Desktop/whisper-flash"

echo "Copying $SCRIPT to remote..."
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY" "$SCRIPT" "$REMOTE:$REMOTE_DIR/"

echo "Running $SCRIPT on remote..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY" "$REMOTE" \
  "cd $REMOTE_DIR && ~/.local/bin/uv run python $SCRIPT"
