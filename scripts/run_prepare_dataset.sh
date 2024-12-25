#!/bin/bash

PYTHON_SCRIPT_DOWNLOAD_DATASET="experiments/download_dataset.py"

# download dataset


# extract the beir test dataset

# extract the shadow dataset and the test dataset

# get the domain of the queries in the shadow dataset
export all_proxy=socks5://192.168.112.1:7890

PYTHON_SCRIPT="Experiments/classify_queries_by_domain.py"
LOG_PATH="Logs/classify_queries"
MODE="test"
DATASET="hotpotqa"

mkdir -p "$LOG_PATH" 

python -u $PYTHON_SCRIPT --mode $MODE --dataset $DATASET > $LOG_PATH/classify_queries_${DATASET}_${MODE}.log 2>&1
