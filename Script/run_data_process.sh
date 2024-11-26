#!/bin/bash

PYTHON_PRE_SCRIPT="Experiments/data_process_align.py"
GROUP_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
TOPK_LIST=4
TARGET_DATASET="msmarco" # msmarco, hotpotqa
MODE="test" # test
for GROUP_INDEX in $(echo $GROUP_LIST | sed "s/,/ /g")
do
    for TOP_K in $(echo $TOPK_LIST | sed "s/,/ /g")
    do
        TEST_FILE="./Datasets/${TARGET_DATASET}/domain/${MODE}_domains_14/domain_${GROUP_INDEX}.jsonl"
        echo "Processing $TEST_FILE"
        LOG_PATH_PRE="Logs/main_cross_attack/${TARGET_DATASET}-${MODE}"
        mkdir -p "$LOG_PATH_PRE"
        python -u "$PYTHON_PRE_SCRIPT" --k $TOP_K --category $GROUP_INDEX --train_queries_path $TEST_FILE # > "$LOG_PATH_PRE/get_ground_truth_${TOP_K}_.log" 2>&1
    done
done

# 等待所有后台任务完成
wait
