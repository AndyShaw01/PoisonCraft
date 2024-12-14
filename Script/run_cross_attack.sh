#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
# export CUDA_VISIBLE_DEVICES=1,2,3

PYTHON_SCRIPT="Experiments/cross_attack_baseline_pj.py"
TARGET_DATASET="nq" # nq, hotpotqa
DEVICE=1
RETRIEVER="simcse" # contriever simcse ance
BASELINE="prompt_injection"
# TARGET_THRESHOLD=4,9,19,49
LOG_PATH="Logs/baseline/${BASELINE}/${RETRIEVER}/${TARGET_DATASET}"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --target_dataset $TARGET_DATASET --retriever $RETRIEVER --device $DEVICE > $LOG_PATH/test.log  2>&1 &