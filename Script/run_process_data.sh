#!/bin/bash

PYTHON_SCRIPT="Experiments/data_process.py"

LOG_PATH="Logs/tmp_data/"

mkdir -p "$LOG_PATH"

python -u "$PYTHON_SCRIPT" > "$LOG_PATH/get_ground_truth.log" 2>&1