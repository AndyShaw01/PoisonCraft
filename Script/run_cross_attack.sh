#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
# export CUDA_VISIBLE_DEVICES=1,2,3

PYTHON_SCRIPT="Experiments/cross_attack_ablation.py"
# PYTHON_SCRIPT="Experiments/cross_attack_baseline_pr.py"
TARGET_DATASET="nq" # nq, hotpotqa
DEVICE=3
RETRIEVER="contriever" # contriever simcse ance
# BASELINE="None"
# TARGET_THRESHOLD=4,9,19,49
# LOG_PATH="Logs/baseline/${BASELINE}/${RETRIEVER}/${TARGET_DATASET}"
# LOG_PATH="Logs/main_result/${RETRIEVER}/${TARGET_DATASET}"
LOG_PATH="Logs/abliation/no_adv_freq/${RETRIEVER}/${TARGET_DATASET}"
mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --target_dataset $TARGET_DATASET --retriever $RETRIEVER --device $DEVICE #> $LOG_PATH/test.log  2>&1 &