# Read tsv file from the file_path

import pandas as pd
import json
import argparse
import pdb
def main(args):
    # 读取 TSV 文件并去重
    df = pd.read_csv(args.id_file_path, sep="\t")
    df = df.drop_duplicates(subset=["query-id"])
    print(df.shape)
    
    # 提取 query-id 列为集合，提高查找速度
    query_id_set = set(df["query-id"])

    # 读取 JSONL 文件并筛选符合条件的 JSON 项
    selected_queries = []
    with open(args.target_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # pdb.set_trace()
            if int(data["_id"]) in query_id_set:
                # pdb.set_trace()
                selected_queries.append(data)

    # 保存到新的 JSONL 文件中
    with open("./Datasets/msmarco/selected_queries.jsonl", "w") as f:
        for query in selected_queries:
            f.write(json.dumps(query) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_file_path", type=str, default="./Datasets/msmarco/qrels/dev.tsv")
    parser.add_argument("--target_file_path", type=str, default="./Datasets/msmarco/queries.jsonl")
    args = parser.parse_args()

    main(args)
