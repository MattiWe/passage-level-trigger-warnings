#!/bin/bash -e

python3 main.py annotation-statistics \
    --dataset-path "../../resources/generated"\
    --annotations-path "../../resources/annotation-results"\
    --keywords-path "../../resources/keywords"
