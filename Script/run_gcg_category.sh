#!/bin/bash

BASH_SCRIPT="./Script/run_gcg_rag.sh"
GROUP_LIST="1,2,3,4,5,6,7,8,9,10,11,12,13,14"
GPU_LIST="0,1,2,3"
for GROUP_NUM in $(echo $GROUP_LIST | sed "s/,/ /g")
do
    if [ "$GROUP_NUM" = "1" ]; then
        GPU_NUM=0
    elif [ "$GROUP_NUM" = "2" ]; then
        GPU_NUM=1
    elif [ "$GROUP_NUM" = "3" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "4" ]; then
        GPU_NUM=3
    elif [ "$GROUP_NUM" = "5" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "6" ]; then
        GPU_NUM=1
    elif [ "$GROUP_NUM" = "7" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "8" ]; then
        GPU_NUM=3
    elif [ "$GROUP_NUM" = "9" ]; then
        GPU_NUM=0
    elif [ "$GROUP_NUM" = "10" ]; then
        GPU_NUM=1
    elif [ "$GROUP_NUM" = "11" ]; then
        GPU_NUM=0
    elif [ "$GROUP_NUM" = "12" ]; then
        GPU_NUM=3
    elif [ "$GROUP_NUM" = "13" ]; then
        GPU_NUM=2
    elif [ "$GROUP_NUM" = "14" ]; then
        GPU_NUM=3
    fi
    bash $BASH_SCRIPT $GROUP_NUM $GPU_NUM &
    echo "Group $GROUP_NUM is running on GPU $GPU_NUM"
done

# 等待所有后台任务完成
wait
