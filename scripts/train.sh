#!/bin/bash

MODEL_NAME=$1
NUM_DEVICES=$2

if [ -z "$MODEL_NAME" ] || [ -z "$NUM_DEVICES" ]; then
    echo "Usage: $0 <model_name> <device_num>"
    echo "Example: $0 colbert 2"
    exit 1
fi

torchrun --nproc_per_node=$NUM_DEVICES ../train.py --config ../configs/train/${MODEL_NAME}.yaml

shutdown