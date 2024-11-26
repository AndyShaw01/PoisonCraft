#!/bin/bash

PYTHON_PRE_SCRIPT="Experiments/data_process_align.py"
GROUP_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
TOPK_LIST=10
for GROUP_INDEX in $(echo $GROUP_LIST | sed "s/,/ /g")
do
    for TOP_K in $(echo $TOPK_LIST | sed "s/,/ /g")
    do
        TEST_FILE="./Dataset/nq/category/categorized_jsonl_files_14_test_recheck/category_${GROUP_INDEX}.jsonl"
        LOG_PATH_PRE="Logs/tmp_data/GCG-Batch-${ATTACK_BATCH_SIZE}-${GROUP_MODE}-category_${GROUP_INDEX}-test-recheck"
        mkdir -p "$LOG_PATH_PRE"
        python -u "$PYTHON_PRE_SCRIPT" --k $TOP_K --category $GROUP_INDEX --train_queries_path $TEST_FILE> "$LOG_PATH_PRE/get_ground_truth_${TOP_K}.log" 2>&1
    done
done

# 等待所有后台任务完成
wait
