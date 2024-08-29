#!/bin/bash

MODEL="contriever"
PYTHON_EXP_SCRIPT="Experiments/gcg_exp.py"
PYTHON_PRE_SCRIPT="Experiments/data_process.py"
RUN_MODE="Run" #Test
MODE="all"

ADD_EOS=False
TOPK_LIST=10,20,50
CONTROL_LENGTH=40

if [ "$MODEL" = "t5-base" ]; then
    LOSS_THRESHOLD=0.015
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$MODEL" = "MPNetModel" ]; then
    LOSS_THRESHOLD=0.025
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
elif [ "$MODEL" = "contriever" ]; then
    LOSS_THRESHOLD=0.4
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
fi

if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL}/GCG-EOS"
else
    LOG_PATH="Logs/${MODEL}/GCG-Batch"
fi

LOG_PATH_PRE="Logs/tmp_data/"

mkdir -p "$LOG_PATH"
mkdir -p "$LOG_PATH_PRE"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

for TOPK in $(echo $TOPK_LIST | sed "s/,/ /g")
do
    python -u "$PYTHON_PRE_SCRIPT" --k $TOPK > "$LOG_PATH_PRE/get_ground_truth_${TOPK}.log" 2>&1
    if [ "$RUN_MODE" = "Test" ]; then
        python -u "$PYTHON_EXP_SCRIPT" \
            --model_path $MODEL_PATH $ADD_EOS_FLAG  \
            --loss_threshold $LOSS_THRESHOLD \
            --attack_mode $MODE \
            --topk $TOPK \
            --control_string_length $CONTROL_LENGTH 
    else
        if [ "$MODEL" = "t5-base" ]; then
            python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
                --loss_threshold $LOSS_THRESHOLD \
                --attack_mode $MODE \
                --control_string_length $CONTROL_LENGTH \
                --topk $TOPK > "$LOG_PATH/gcg_${RUN_MODE}_top${TOPK}.log" 2>&1
        elif [ "$MODEL" = "MPNetModel" ]; then
            python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
                --loss_threshold $LOSS_THRESHOLD \
                --attack_mode $MODE \
                --control_string_length $CONTROL_LENGTH \
                --topk $TOPK > "$LOG_PATH/gcg_${RUN_MODE}_top${TOPK}.log" 2>&1
        elif [ "$MODEL" = "contriever" ]; then
            python -u "$PYTHON_EXP_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG \
                --loss_threshold $LOSS_THRESHOLD \
                --attack_mode $MODE \
                --control_string_length $CONTROL_LENGTH \
                --topk $TOPK > "$LOG_PATH/gcg_${RUN_MODE}_top${TOPK}.log" 2>&1
        fi
    fi
    echo TOPK: $TOPK
done