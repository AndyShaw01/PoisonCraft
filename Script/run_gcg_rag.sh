#!/bin/bash

MODEL="contriever"
PYTHON_EXP_SCRIPT="Experiments/gcg_exp.py"

RUN_MODE="Run" #Test
GROUP_MODE="category" # random, category
GROUP_INDEX=$1
MODE="all"

TRAIN_FILE="./Dataset/nq/category/categorized_jsonl_files_14_train_recheck/category_${GROUP_INDEX}.jsonl"

ADD_EOS=False

CONTROL_LENGTH=50
ATTACK_BATCH_SIZE=4

if [ "$MODEL" = "t5-base" ]; then
    LOSS_THRESHOLD=0.015
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$MODEL" = "MPNetModel" ]; then
    LOSS_THRESHOLD=0.025
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
elif [ "$MODEL" = "contriever" ]; then
    LOSS_THRESHOLD=0.2
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
fi

if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL}/GCG-EOS"
else
    LOG_PATH="Logs/${MODEL}/GCG-Batch-${ATTACK_BATCH_SIZE}-${GROUP_MODE}-category_${GROUP_INDEX}_cross_recheck"
fi

LOG_PATH_PRE="Logs/tmp_data/GCG-Batch-${ATTACK_BATCH_SIZE}-${GROUP_MODE}-category_${GROUP_INDEX}_cross"

mkdir -p "$LOG_PATH"
mkdir -p "$LOG_PATH_PRE"

export CUDA_VISIBLE_DEVICES=$2
echo "Using GPU $2"
# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

if [ "$RUN_MODE" = "Test" ]; then
    python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG  \
        --loss_threshold $LOSS_THRESHOLD \
        --attack_mode $MODE \
        --group_mode $GROUP_MODE \
        --attack_batch_size $ATTACK_BATCH_SIZE \
        --control_string_length $CONTROL_LENGTH \
        --group_index $GROUP_INDEX \
        --train_queries_path $TRAIN_FILE \
else
    if [ "$MODEL" = "t5-base" ]; then
        python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
            --loss_threshold $LOSS_THRESHOLD \
            --attack_mode $MODE \
            --control_string_length $CONTROL_LENGTH > "$LOG_PATH/gcg_${RUN_MODE}.log" 2>&1
    elif [ "$MODEL" = "MPNetModel" ]; then
        python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
            --loss_threshold $LOSS_THRESHOLD \
            --attack_mode $MODE \
            --control_string_length $CONTROL_LENGTH > "$LOG_PATH/gcg_${RUN_MODE}.log" 2>&1
    elif [ "$MODEL" = "contriever" ]; then
        python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
            --loss_threshold $LOSS_THRESHOLD \
            --attack_mode $MODE \
            --control_string_length $CONTROL_LENGTH \
            --group_mode $GROUP_MODE \
            --attack_batch_size $ATTACK_BATCH_SIZE \
            --train_queries_path $TRAIN_FILE \
            --group_index $GROUP_INDEX > "$LOG_PATH/gcg_${RUN_MODE}.log" 2>&1
        echo "python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE --control_string_length $CONTROL_LENGTH --group_mode $GROUP_MODE --attack_batch_size $ATTACK_BATCH_SIZE --train_queries_path $TRAIN_FILE --group_index $GROUP_INDEX --topk $TOPK "
    fi
fi
    
