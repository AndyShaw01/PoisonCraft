#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/classify_queries.py"
LOG_PATH="Logs/classify_queries"
MODE="test"
DATASET="hotpotqa"

mkdir -p "$LOG_PATH" 

python -u $PYTHON_SCRIPT --mode $MODE --dataset $DATASET > $LOG_PATH/1122_classify_queries_${DATASET}_${MODE}.log 2>&1
