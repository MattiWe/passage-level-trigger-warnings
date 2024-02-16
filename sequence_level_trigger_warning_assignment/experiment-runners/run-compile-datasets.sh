#!/bin/bash -e

# python3 main.py parse-ls \
#     --ls-primary "../../resources/annotation-results/ls-primary.json" \
#     --ls-additional "../../resources/annotation-results/ls-addition-matti.json" \
#     --ls-additional "../../resources/annotation-results/ls-addition-students.json" \
#     --output-path "../../resources/generated/dataset.jsonl"

for DSI in "43" "44" "45" "46" "47"
do
    python3 main.py prepare-experiments \
        --dataset-path "../../resources/generated/dataset.jsonl" \
        --output-path "../../resources/generated/datasets-$DSI" \
        --prompt-path "../../resources/prompt" \
        --seed $DSI
done