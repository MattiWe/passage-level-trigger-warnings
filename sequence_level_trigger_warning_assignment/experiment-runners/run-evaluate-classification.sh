#!/bin/bash -e

python3 main.py evaluate-classification \
    --dataset-path "../../resources/generated/dataset.jsonl"\
    --bert-predictions-path "../../resources/classification-results/predictions" 
