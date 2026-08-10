#!/bin/bash

set -e

CONFIG="../configs/evaluate/colbert.yaml"
CKPT_ROOT="/root/autodl-tmp/ckpt/colbert/states"

for ckpt_dir in $(find "$CKPT_ROOT" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
    MODEL_PATH="${ckpt_dir}/model.safetensors"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Skip ${ckpt_dir}: model.safetensors not found."
        continue
    fi

    echo "======================================"
    echo "Evaluating: ${MODEL_PATH}"
    echo "======================================"

    sed -i "s|ckpt_path:.*|ckpt_path: ${MODEL_PATH}|" "$CONFIG"

    python ../evaluate.py --config "$CONFIG"

done

echo "All checkpoints have been evaluated."