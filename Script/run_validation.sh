#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=1
PYTHON_SCRIPT="Experiments/validation_debug.py"

RETRIEVER="contriever"
DATASET="nq"
TOPK=4
LOG_PATH="./Logs/ablation/attack/${RETRIEVER}/${DATASET}/top${TOPK+1}" 

mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --retriever $RETRIEVER --eval_dataset $DATASET --top_k $TOPK> "$LOG_PATH/main_result_add.log" 2>&1