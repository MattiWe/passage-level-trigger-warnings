#!/bin/bash
CACHE_DIR=".cache/huggingface/hub"
DATASETS="../../resources/generated/datasets"
SAVEPOINT="../../resources/models"
KEYFILE="../../resources/prompt/gpt-secret-sentence-triggers.txt"

for DSI in "43" "44" "45" "46" "47" 
do
    for SPLIT in "ood-majority" "id-minority" "ood-minority" "id-majority" 
    do
        echo "gpt3"
        CHECKPOINT="gpt-3.5-turbo-0125"
        NAME="acl24-gpt3-5-turbo-${SPLIT}-$DSI"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS-$DSI/$SPLIT/test-prompts" \
        --savepoint "$SAVEPOINT-$DSI/$NAME" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "12"

        # ---

        echo "gpt4"
        CHECKPOINT="gpt-4-0125-preview"
        NAME="acl24-gpt4-turbo-${SPLIT}-$DSI"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS-$DSI/$SPLIT/test-prompts" \
        --savepoint "$SAVEPOINT-$DSI/$NAME" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "12"
    done
done