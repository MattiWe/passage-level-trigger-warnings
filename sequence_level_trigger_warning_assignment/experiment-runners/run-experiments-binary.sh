#!/bin/bash -e

EP=20
BATCHES=12
MODE="binary"
CHECKPOINT="roberta-base"
DATASETS="../../resources/generated/datasets"
SAVEPOINT="../../resources/models"

for DSI in  "43" "44" "45" "46" "47"
do
    LR="1e-5"
    SPLIT="ood-majority"
    NAME="acl24-fanbert-${MODE}-${SPLIT}-${LR}lr-${EP}ep-$DSI"
    python3 main.py finetune-experiments \
    --checkpoint $CHECKPOINT \
    --training "$DATASETS-$DSI/$SPLIT/binary" \
    --validation "$DATASETS-$DSI/$SPLIT/test-binary" \
    --test "$DATASETS-$DSI/$SPLIT/test-binary" \
    --savepoint "$SAVEPOINT-$DSI/$NAME" \
    --epochs $EP \
    --lr $LR \
    --batches $BATCHES \
    --name "${NAME}" \
    --mode "$MODE" \
    --cache-dir $CACHE_DIR \
    --no-wandb

    LR="2e-5"
    SPLIT="ood-minority"
    NAME="acl24-fanbert-${MODE}-${SPLIT}-${LR}lr-${EP}ep-$DSI"
    python3 main.py finetune-experiments \
    --checkpoint $CHECKPOINT \
    --training "$DATASETS-$DSI/$SPLIT/binary" \
    --validation "$DATASETS-$DSI/$SPLIT/test-binary" \
    --test "$DATASETS-$DSI/$SPLIT/test-binary" \
    --savepoint "$SAVEPOINT-$DSI/$NAME" \
    --epochs $EP \
    --lr $LR \
    --batches $BATCHES \
    --name "${NAME}" \
    --mode "$MODE" \
    --cache-dir $CACHE_DIR \
    --no-wandb

    LR="1e-5"
    SPLIT="id-majority"
    NAME="acl24-fanbert-${MODE}-${SPLIT}-${LR}lr-${EP}ep-$DSI"
    python3 main.py finetune-experiments \
    --checkpoint $CHECKPOINT \
    --training "$DATASETS-$DSI/$SPLIT/binary" \
    --validation "$DATASETS-$DSI/$SPLIT/test-binary" \
    --test "$DATASETS-$DSI/$SPLIT/test-binary" \
    --savepoint "$SAVEPOINT-$DSI/$NAME" \
    --epochs $EP \
    --lr $LR \
    --batches $BATCHES \
    --name "${NAME}" \
    --mode "$MODE" \
    --cache-dir $CACHE_DIR \
    --no-wandb

    LR="5e-5"
    SPLIT="id-minority"
    NAME="acl24-fanbert-${MODE}-${SPLIT}-${LR}lr-${EP}ep-$DSI"
    python3 main.py finetune-experiments \
    --checkpoint $CHECKPOINT \
    --training "$DATASETS-$DSI/$SPLIT/binary" \
    --validation "$DATASETS-$DSI/$SPLIT/test-binary" \
    --test "$DATASETS-$DSI/$SPLIT/test-binary" \
    --savepoint "$SAVEPOINT-$DSI/$NAME" \
    --epochs $EP \
    --lr $LR \
    --batches $BATCHES \
    --name "${NAME}" \
    --mode "$MODE" \
    --cache-dir $CACHE_DIR \
    --no-wandb
done
