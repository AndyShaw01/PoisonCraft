import pandas as pd
import glob
import json

# 读取JSONL文件并提取目标的 _id 列表
jsonl_file = "../Dataset/hotpotqa/test_queries_add_class_14_recheck.jsonl"  # 替换为您的 JSONL 文件路径
target_ids = set()

with open(jsonl_file, 'r') as f:
    for line in f:
        record = json.loads(line)
        target_ids.add(record['_id'])

# 定义TSV文件路径和模式
file_path = "datasets/hotpotqa/qrels/*.tsv"  # 替换为TSV文件的路径

# 获取所有符合条件的文件路径
all_files = glob.glob(file_path)

# 读取并合并所有TSV文件
df_list = []
for filename in all_files:
    df = pd.read_csv(filename, sep='\t')
    df_list.append(df)

# 合并所有DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# 根据目标 query-id 进行筛选
filtered_df = merged_df[merged_df['query-id'].isin(target_ids)]

# 去重后的 query-id 数量
unique_query_ids_count = filtered_df['query-id'].nunique()

print(f"筛选后去重的 query-id 数量为: {unique_query_ids_count}")

# 如果需要保存筛选后的数据
filtered_df.to_csv("./datasets/hotpotqa/all_test.tsv", sep='\t', index=False)

print("筛选后的数据已保存到 'filtered_merged_file.tsv'")
