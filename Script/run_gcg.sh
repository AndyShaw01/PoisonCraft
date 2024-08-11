#!/bin/bash

PYTHON_SCRIPT="Experiments/gcg_exp.py"
MODEL_PATH="/data1/shaoyangguang/offline_model/SBERT"
ADD_EOS=False
RUN_INDEX=1
INDEX=0

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


python -u "$PYTHON_SCRIPT" --index $INDEX --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX 