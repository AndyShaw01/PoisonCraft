#!/bin/bash

PYTHON_PRE_SCRIPT="experiments/process_data.py"
DOMAIN_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
TOPK_LIST=4,9,19,49
TARGET_DATASET="nq" # msmarco, hotpotqa
MODE="test" # test
RETRIEVER="contriever" # contriever ance
for DOMAIN_INDEX in $(echo $DOMAIN_LIST | sed "s/,/ /g")
do
    for TOP_K in $(echo $TOPK_LIST | sed "s/,/ /g")
    do
        TARGET_FILE="./Datasets/${TARGET_DATASET}/domain/${MODE}_domains_14/domain_${GROUP_INDEX}.jsonl"
        echo "Processing $TARGET_FILE"
        LOG_PATH_PRE="Logs/main_result/${RETRIEVER}/${TARGET_DATASET}/${TARGET_DATASET}-${MODE}"
        mkdir -p "$LOG_PATH_PRE"
        python -u "$PYTHON_PRE_SCRIPT" --dataset $TARGET_DATASET \
            --k $TOP_K --domain $DOMAIN_INDEX --target_queries_path $TARGET_FILE --retriever $RETRIEVER # > "$LOG_PATH_PRE/get_ground_truth_${TOP_K}_.log" 2>&1
    done
done