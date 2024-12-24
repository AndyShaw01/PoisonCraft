#!/bin/bash

RETRIEVER="ance"
PYTHON_EXP_SCRIPT="Experiments/gcg_exp.py"
PYTHON_PRE_SCRIPT="Experiments/data_process.py"

GROUP_MODE="category" # random, category
GROUP_INDEX=$1
MODE="all"
DATASET="msmarco" # msmarco
TRAIN_FILE="./Datasets/${DATASET}/domain/train_domains_14/domain_${GROUP_INDEX}.jsonl"

CONTROL_LENGTH=$2
ATTACK_BATCH_SIZE=4
LOSS_THRESHOLD=0.2

if [ "$RETRIEVER" = "t5-base" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$RETRIEVER" = "MPNetModel" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
elif [ "$MODEL" = "contriever" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
elif [ "$RETRIEVER" = "ance" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/ance"
elif [ "$RETRIEVER" = "simcse" ]; then
    MODEL_PATH="/data1/shaoyangguang/offline_model/simcse"
fi

LOG_PATH="Logs/${MODEL}_Attack_${DATASET}/domain_${GROUP_INDEX}_control_${CONTROL_LENGTH}"

LOG_PATH_PRE="Logs/tmp_data/GCG_${GROUP_MODE}_${GROUP_INDEX}_cross"

mkdir -p "$LOG_PATH"
mkdir -p "$LOG_PATH_PRE"

export CUDA_VISIBLE_DEVICES=$3
echo "Using GPU $3"

python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
    --loss_threshold $LOSS_THRESHOLD \
    --attack_mode $MODE \
    --control_string_length $CONTROL_LENGTH \
    --group_mode $GROUP_MODE \
    --attack_batch_size $ATTACK_BATCH_SIZE \
    --train_queries_path $TRAIN_FILE \
    --group_index $GROUP_INDEX # > "$LOG_PATH/gcg_${RUN_MODE}.log" 2>&1
echo "python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE --control_string_length $CONTROL_LENGTH --group_mode $GROUP_MODE --attack_batch_size $ATTACK_BATCH_SIZE --train_queries_path $TRAIN_FILE --group_index $GROUP_INDEX --topk $TOPK "
