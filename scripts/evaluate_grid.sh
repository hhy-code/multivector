#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Names correspond to configs/evaluate/<name>.yaml.
# MODELS=(colbert leapmv tokenpruning_idf tokenpruning_attn tokenpruning_first meanpooling_fixed meanpooling_adaptive)
MODELS=(meanpooling_adaptive)
# DATASETS=(
#   arguana dbpedia-entity fiqa hotpotqa nfcorpus nq
#   quora scidocs scifact trec-covid webis-touche2020
# )
DATASETS=(arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020)  # 小的那组
# DATASETS=(dbpedia-entity hotpotqa nq)  # 大的那组
DATA_ROOT="/mnt/HDD1/home/lyy25/data/beir"

for model in "${MODELS[@]}"; do
  CONFIG="configs/evaluate/$model.yaml"
  for dataset in "${DATASETS[@]}"; do
    echo "Evaluating $model / $dataset"
    sed -i -E "s|^([[:space:]]*)dataset_path:.*|\1dataset_path: $DATA_ROOT/$dataset|" "$CONFIG"

    if [ "$model" = "tokenpruning_idf" ]; then
      sed -i -E "s|^([[:space:]]*)idf_path:.*|\1idf_path: $DATA_ROOT/$dataset/idf.tsv|" "$CONFIG"
    fi

    python -u evaluate.py --config "$CONFIG" --no-progress
  done
done

echo "All evaluations finished."
