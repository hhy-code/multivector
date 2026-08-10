#!/usr/bin/env bash

set -uo pipefail
shopt -s nullglob

CONFIG_DIR="../configs/evaluate"
DATASET_PATH="/root/data/beir/trec_2020"
SKIP_MODELS=( "colbert" "constbert" "memtoken" )

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME="$(basename "$CONFIG_FILE" .yaml)"

    SHOULD_SKIP=false
    for SKIP_NAME in "${SKIP_MODELS[@]}"; do
        if [ "$CONFIG_NAME" = "$SKIP_NAME" ]; then
            SHOULD_SKIP=true
            break
        fi
    done

    if [ "$SHOULD_SKIP" = true ]; then
        echo "Skipping model: $CONFIG_NAME"
        echo
        continue
    fi

    echo "========================================"
    echo "Evaluating ${CONFIG_NAME}"
    echo "========================================"

    sed -i "s|dataset_path: .*|dataset_path: $DATASET_PATH|" "$CONFIG_FILE"

    python ../evaluate.py --config "$CONFIG_FILE" --no-progress
done

sudo shutdown -h now
