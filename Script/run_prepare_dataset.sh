#!/bin/bash

export all_proxy=socks5://192.168.112.1:7890 

PYTHON_SCRIPT="Experiments/classify_queries.py"
LOG_PATH="Logs/classify_queries"

mkdir -p "$LOG_PATH"

python -u $PYTHON_SCRIPT > $LOG_PATH/classify_queries_msmarco_test_1.log 2>&1 &
