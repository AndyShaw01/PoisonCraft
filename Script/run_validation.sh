#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 
export CUDA_VISIBLE_DEVICES=3
PYTHON_SCRIPT="Experiments/validation_debug.py"

RETRIEVER="contriever" # contriever simcse
DATASET="nq" # msmarco, hotpotqa nq
TOPK=4
ABLATION="None"
BASELINE="None" # prompt_injection, poisonedrag
URL="agent4sports"
# LOG_PATH="./Logs/baseline/attack/${BASELINE}/${RETRIEVER}/${DATASET}/top5/" 
# LOG_PATH="./Logs/ablation/attack/no_adv_freq/${RETRIEVER}/${DATASET}/top5/" 
LOG_PATH="./Logs/sens/attack/${RETRIEVER}/${DATASET}/${URL}/top5/" 

mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT --retriever $RETRIEVER --eval_dataset $DATASET --top_k $TOPK --baseline_method $BASELINE --ablation_method $ABLATION  --url $URL > "$LOG_PATH/main.log" 2>&1 &