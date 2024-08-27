#!/bin/bash

# MODEL="t5-base"
# MODEL="MPNetModel"
MODEL="contriever"
PYTHON_SCRIPT="Experiments/gcg_exp.py"
RUN_MODE="Test" #Test
MODE="all"

ADD_EOS=False
TOPK=90

if [ "$MODEL" = "t5-base" ]; then
    LOSS_THRESHOLD=0.015
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$MODEL" = "MPNetModel" ]; then
    LOSS_THRESHOLD=0.025
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
elif [ "$MODEL" = "contriever" ]; then
    LOSS_THRESHOLD=0.1
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
fi

if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL}/GCG_EOS"
else
    LOG_PATH="Logs/${MODEL}/GCG"
fi

mkdir -p "$LOG_PATH"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

if [ "$RUN_MODE" = "Test" ]; then
    python -u "$PYTHON_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG  --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE
else
    if [ "$MODEL" = "t5-base" ]; then
        python -u "$PYTHON_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG  --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_t5.log" 2>&1
    elif [ "$MODEL" = "MPNetModel" ]; then
        python -u "$PYTHON_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG  --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_mp.log" 2>&1
    elif [ "$MODEL" = "contriever" ]; then
        python -u "$PYTHON_SCRIPT" --model_path $MODEL_PATH $ADD_EOS_FLAG  --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE> "$LOG_PATH/gcg_all_top100_2.log" 2>&1
    fi
fi
