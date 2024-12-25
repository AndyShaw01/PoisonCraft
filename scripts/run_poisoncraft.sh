#!/bin/bash

RETRIEVER="ance"
PYTHON_EXP_SCRIPT="Experiments/gcg_exp.py"
PYTHON_PRE_SCRIPT="Experiments/data_process.py"

DOMAIN_INDEX=$1
DATASET="msmarco" # msmarco
SHADOW_FILE="./Datasets/${DATASET}/domain/train_domains_14/domain_${DOMAIN_INDEX}.jsonl"

CONTROL_LENGTH=$2
ATTACK_BATCH_SIZE=4
LOSS_THRESHOLD=0.2

if [ "$MODEL" = "contriever" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
elif [ "$RETRIEVER" = "ance" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/ance"
elif [ "$RETRIEVER" = "simcse" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/simcse"
fi

LOG_PATH="Logs/${MODEL}_on_${DATASET}/domain_${GROUP_INDEX}_control_${CONTROL_LENGTH}"

LOG_PATH_PRE="Logs/tmp_data/GCG_${GROUP_MODE}_${GROUP_INDEX}_cross"

mkdir -p "$LOG_PATH"
mkdir -p "$LOG_PATH_PRE"

export CUDA_VISIBLE_DEVICES=$3
echo "Using GPU $3"

python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH \
    --loss_threshold $LOSS_THRESHOLD \
    --control_string_length $CONTROL_LENGTH \
    --attack_batch_size $ATTACK_BATCH_SIZE \
    --shadow_queries_path $SHADOW_FILE \
    --domain_index $DOMAIN_INDEX # > "$LOG_PATH/gcg_${RUN_MODE}.log" 2>&1
echo "python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH --loss_threshold $LOSS_THRESHOLD --control_string_length $CONTROL_LENGTH --attack_batch_size $ATTACK_BATCH_SIZE --train_queries_path $TRAIN_FILE --topk $TOPK "
