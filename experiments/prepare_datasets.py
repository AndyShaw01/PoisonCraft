import os
import json
import random
import pandas as pd

from beir import util

def get_beir_datasets(datasets, out_dir):
    """
    Download and unzip the specified datasets.

    Args:
        datasets (list): List of dataset names.
        out_dir (str): Output directory path.
    """
    for dataset in datasets:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = os.path.join(out_dir, dataset)
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)
    os.system(f'rm {out_dir}/*.zip')

def generate_selected_queries(id_file_path, target_file_path, output_file_path):
    """
    Read the TSV file and remove duplicates, then select the queries with these query IDs from the JSONL file and save them to a new JSONL file.

    Args:
        id_file_path (str): Path to the TSV file.
        target_file_path (str): Path to the JSONL file.
        output_file_path (str): Path to the output JSONL file.
    """
    df = pd.read_csv(id_file_path, sep="\t")
    df = df.drop_duplicates(subset=["query-id"])
    query_id_set = set(df["query-id"])

    selected_queries = []
    with open(target_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if int(data["_id"]) in query_id_set:
                selected_queries.append(data)
                
    with open(output_file_path, "w") as f:
        for query in selected_queries:
            f.write(json.dumps(query) + "\n")


if __name__ == "__main__":
    datasets = ['nq', 'msmarco', 'hotpotqa']
    out_dir = os.path.join(os.getcwd(), "datasets_test")

    # Download and unzip datasets
    get_beir_datasets(datasets, out_dir)

    for dataset in datasets:
        id_file_path = f"./Datasets/{dataset}/qrels/dev.tsv"
        target_file_path = f"./Datasets/{dataset}/queries.jsonl"
        output_file_path = f"./datasets_test/{dataset}/selected_queries.jsonl"
        
        # Generate selected_queries.jsonl
        generate_selected_queries(id_file_path, target_file_path, output_file_path)

        original_queries_file_path = output_file_path
        train_queries_file_path = f"./datasets_test/{dataset}/train_queries.jsonl"
        test_queries_file_path = f"./datasets_test/{dataset}/test_queires.jsonl"
        data = []
        # Load data from file
        with open(f'{original_queries_file_path}', 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        # Randomly shuffle data
        random.shuffle(data)

        # Set the ratio to 2:8
        split_idx = int(0.2 * len(data))
        train_set = data[:split_idx]
        test_set = data[split_idx:]

        # Output training set and test set
        with open(train_queries_file_path, 'w') as train_file:
            for entry in train_set:
                train_file.write(json.dumps(entry) + '\n')

        with open(test_queries_file_path, 'w') as test_file:
            for entry in test_set:
                test_file.write(json.dumps(entry) + '\n')

        print(f"Train-datasets size: {len(train_set)}")
        print(f"Test-datasets  size: {len(test_set)}")