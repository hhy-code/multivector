#!/bin/bash

set -e

DATA_ROOT="/root/data/beir"
# SKIP_DATASETS=( "msmarco" "trec_2019" "nq" "quora" "hotpotqa" "dbpedia-entity" )
# SKIP_DATASETS=( "webis-touche2020" )
SKIP_DATASETS=( "msmarco" "trec_2019" )

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_name>"
    echo "Example: $0 colbert"
    exit 1
fi

CONFIG_NAME=$1
CONFIG_FILE="../configs/evaluate/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo

for DATASET in "$DATA_ROOT"/*; do
    if [ -d "$DATASET" ]; then
        DATASET_NAME=$(basename "$DATASET")

        SHOULD_SKIP=false
        for SKIP_NAME in "${SKIP_DATASETS[@]}"; do
            if [ "$DATASET_NAME" = "$SKIP_NAME" ]; then
                SHOULD_SKIP=true
                break
            fi
        done

        if [ "$SHOULD_SKIP" = true ]; then
            echo "Skipping dataset: $DATASET_NAME"
            echo
            continue
        fi

        echo "========================================"
        echo "Evaluating dataset: $DATASET_NAME"
        echo "========================================"

        sed -i "s|dataset_path: .*|dataset_path: $DATASET|" "$CONFIG_FILE"

        if [ "$CONFIG_NAME" = "tokenpruning_idf" ]; then
            sed -i "s|idf_path: .*|idf_path: $DATASET/idf.tsv|" "$CONFIG_FILE"  # only for tokenpruning_idf
        fi

        python ../evaluate.py --config "$CONFIG_FILE" --no-progress

        echo
    fi
done

echo "All datasets have been evaluated."

shutdown
