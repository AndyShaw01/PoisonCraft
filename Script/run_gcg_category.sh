#!/bin/bash

BASH_SCRIPT="./Script/run_gcg_rag.sh"
GROUP_LIST="1,2,3,4,5,6,7"
GPU_LIST="1,2,3,4"
for GROUP_NUM in $(echo $GROUP_LIST | sed "s/,/ /g")
do
    if [ "$GROUP_NUM" = "1" ]; then
        GPU_NUM=1
    elif [ "$GROUP_NUM" = "2" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "3" ]; then
        GPU_NUM=3
    elif [ "$GROUP_NUM" = "4" ]; then
        GPU_NUM=4
    elif [ "$GROUP_NUM" = "5" ]; then
        GPU_NUM=1
    elif [ "$GROUP_NUM" = "6" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "7" ]; then
        GPU_NUM=3
    fi
    bash $BASH_SCRIPT $GROUP_NUM $GPU_NUM &
    echo "Group $GROUP_NUM is running on GPU $GPU_NUM"
done

# 等待所有后台任务完成
wait
