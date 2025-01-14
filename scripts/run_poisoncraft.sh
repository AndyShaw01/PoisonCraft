#!/bin/bash

RETRIEVER="bge-small"
PYTHON_EXP_SCRIPT="experiments/poisoncraft_exp.py"

DOMAIN_INDEX=$1
DATASET="nq" # msmarco
SHADOW_FILE="./datasets/${DATASET}/domain/train_domains_14/domain_${DOMAIN_INDEX}.jsonl"

ADV_LENGTH=$2
ATTACK_BATCH_SIZE=4
LOSS_THRESHOLD=0.2

if [ "$RETRIEVER" = "contriever" ]; then
    MODEL_PATH="facebook/contriever"
elif [ "$RETRIEVER" = "simcse" ]; then
    MODEL_PATH="princeton-nlp/unsup-simcse-bert-base-uncased"
elif [ "$RETRIEVER" = "bge-small" ]; then
    MODEL_PATH="BAAI/bge-small-en"
fi

LOG_PATH="logs/${RETRIEVER}_on_${DATASET}/domain_${DOMAIN_INDEX}_control_${ADV_LENGTH}"

mkdir -p "$LOG_PATH"

export CUDA_VISIBLE_DEVICES=$3
echo "Using GPU $3"

python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH \
    --loss_threshold $LOSS_THRESHOLD \
    --adv_string_length $ADV_LENGTH \
    --batch_size $ATTACK_BATCH_SIZE \
    --shadow_queries_path $SHADOW_FILE \
    --attack_target $DATASET \
    --retriever $RETRIEVER \
    --domain_index $DOMAIN_INDEX > "$LOG_PATH/gcg.log" 2>&1
echo "python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH --loss_threshold $LOSS_THRESHOLD --adv_string_length $ADV_LENGTH --batch_size $ATTACK_BATCH_SIZE --shadow_queries_path $SHADOW_FILE --domain_index $DOMAIN_INDEX --attack_target $DATASET"
