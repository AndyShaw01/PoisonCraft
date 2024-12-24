# Read tsv file from the file_path

import pandas as pd
import json
import argparse
import pdb
def main(args):
    # Read the TSV file and drop duplicates
    df = pd.read_csv(args.id_file_path, sep="\t")
    df = df.drop_duplicates(subset=["query-id"])
    print(df.shape)
    
    query_id_set = set(df["query-id"])

    # Read the JSONL file and select the queries with the query ids
    selected_queries = []
    with open(args.target_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if int(data["_id"]) in query_id_set:
                selected_queries.append(data)
                
    with open("./Datasets/msmarco/selected_queries.jsonl", "w") as f:
        for query in selected_queries:
            f.write(json.dumps(query) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_file_path", type=str, default="./Datasets/msmarco/qrels/dev.tsv")
    parser.add_argument("--target_file_path", type=str, default="./Datasets/msmarco/queries.jsonl")
    args = parser.parse_args()

    main(args)
