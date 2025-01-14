#!/bin/bash

PYTHON_SCRIPT_PREPARE_DATASET="experiments/prepare_datasets.py"
PYTHON_SCRIPT_CLASSIFY_QUERIES="experiments/classify_queries_by_domain.py"

LOG_PATH="Logs/classify_queries"
MODE="test"
DATASETS=("nq" "msmarco" "hotpotqa")

mkdir -p "$LOG_PATH"

# Run prepare_datasets.py
python -u $PYTHON_SCRIPT_PREPARE_DATASET > $LOG_PATH/prepare_datasets.log 2>&1

# Run classify_queries_by_domain.py for each dataset
for DATASET in "${DATASETS[@]}"; do
    python -u $PYTHON_SCRIPT_CLASSIFY_QUERIES --mode $MODE --dataset $DATASET > $LOG_PATH/classify_queries_${DATASET}_${MODE}.log 2>&1
done
