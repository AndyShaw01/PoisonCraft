#!/bin/bash

MODEL="t5-base"
# MODEL="MPNetModel"
PYTHON_SCRIPT="Experiments/gcg_exp.py"
ADD_EOS=False
RUN_INDEX=1
INDEX=0
if [ "$MODEL" = "t5-base" ]; then
    LOSS_THRESHOLD=0.05
    MODEL_PATH="/data1/shaoyangguang/offline_model/t5-base"
elif [ "$MODEL" = "MPNetModel" ]; then
    LOSS_THRESHOLD=0.1
    MODEL_PATH="/data1/shaoyangguang/offline_model/MPNetModel"
fi

if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL_PATH}/GCG_eos-${RUN_INDEX}"
else
    LOG_PATH="Logs/${MODEL_PATH}/GCG-${RUN_INDEX}"
fi

mkdir -p "$LOG_PATH"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

if [ "$MODEL" = "t5-base" ]; then
    python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_5*.log" 2>&1
elif [ "$MODEL" = "MPNetModel" ]; then
    python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX --loss_threshold $LOSS_THRESHOLD > "$LOG_PATH/gcg_5!.log" 2>&1
fi

# python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX