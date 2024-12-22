import pandas as pd
import numpy as np
import argparse
import os
import glob
import pdb

def get_static_result(file_path):
    # 读取总行数
    with open(file_path, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
    return total_lines


def main(args):
    file_paths = glob.glob(os.path.join(args.folder_path, '*.jsonl'))
    nums = []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        num = get_static_result(file_path)
        nums.append(num)
    all_nums = sum(nums)
    print(nums)
    # 计算占比
    ratios = [round(num / all_nums, 4)for num in nums]
    print(ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument("--folder_path", type=str, default='./Dataset/nq/category/categorized_jsonl_files_14/')
    
    args = parser.parse_args()

    main(args)