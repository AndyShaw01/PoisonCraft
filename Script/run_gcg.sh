#!/bin/bash

# MODEL="t5-base"
# MODEL="MPNetModel"
MODEL="contriever"
PYTHON_SCRIPT="Experiments/gcg_exp.py"
RUN_MODE="Run" #Test
MODE="all"

ADD_EOS=False
RUN_INDEX=1
INDEX=0

if [ "$MODEL" = "t5-base" ]; then
    LOSS_THRESHOLD=0.015
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$MODEL" = "MPNetModel" ]; then
    LOSS_THRESHOLD=0.025
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
elif [ "$MODEL" = "contriever" ]; then
    LOSS_THRESHOLD=0.02
    MODEL_PATH="/data1/shaoyangguang/offline_model/contriever"
fi

if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL}/GCG_${RUN_INDEX}"
else
    LOG_PATH="Logs/${MODEL}/GCG-${RUN_INDEX}"
fi

mkdir -p "$LOG_PATH"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

if [ "$RUN_MODE" = "Test" ]; then
    python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE
else
    if [ "$MODEL" = "t5-base" ]; then
        python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_t5.log" 2>&1
    elif [ "$MODEL" = "MPNetModel" ]; then
        python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_mp.log" 2>&1
    elif [ "$MODEL" = "contriever" ]; then
        python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD --attack_mode $MODE> "$LOG_PATH/gcg_all_top100_2.log" 2>&1
    fi
fi
