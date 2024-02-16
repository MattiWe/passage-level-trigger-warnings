#!/bin/bash -e

CACHE_DIR=".cache/huggingface/hub"
DATASETS="../../resources/generated/datasets"
SAVEPOINT="../../resources/models"
KEYFILE="../../resources/prompt/gpt-secret-sentence-triggers.txt"

for DSI in  "43" "44" "45" "46" "47"
do
    for SPLIT in "id-minority" "ood-minority" "id-majority" "ood-majority"
    do

        echo "llama2 7 default"
        DECODING="default"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS/prompt-ablation" \
        --savepoint "$SAVEPOINT/$CHECKPOINT/prompt-ablation" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "6" \
        --decoding-strategy $DECODING

        echo "llama2 13 default"
        CHECKPOINT="meta-llama/Llama-2-13b-chat-hf"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS/prompt-ablation" \
        --savepoint "$SAVEPOINT/$CHECKPOINT/prompt-ablation" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "2" \
        --decoding-strategy $DECODING

        # ---

        echo "mixtral chat"
        CHECKPOINT="mistralai/Mixtral-8x7B-Instruct-v0.1"
        DECODING="chat"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS/prompt-ablation" \
        --savepoint "$SAVEPOINT/$CHECKPOINT-chat/prompt-ablation" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "12" \
        --decoding-strategy $DECODING

        echo "mistral-chat on $SPLIT for DSI $DSI"
        python3 main.py fewshot-experiment \
        --checkpoint $CHECKPOINT \
        --test "$DATASETS-$DSI/$SPLIT/test-prompts" \
        --savepoint "$SAVEPOINT-$DSI/$CHECKPOINT-chat/$SPLIT" \
        --cache-dir $CACHE_DIR \
        --keyfile $KEYFILE \
        --batches "1" \
        --decoding-strategy "chat"
    done
done


