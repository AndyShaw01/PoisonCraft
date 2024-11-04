import json
import random



if __name__ == '__main__':
    dataset = "msmarco" # msmarco, nq, hotpotqa
    original_queries_file_path = f"./Dataset/{dataset}/queries.jsonl"
    train_queries_file_path = f"./Dataset/{dataset}/train_queries.jsonl"
    test_queries_file_path = f"./Dataset/{dataset}/test_queires.jsonl"
    data = []
    # Load data from file
    with open(f'{original_queries_file_path}', 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Randomly shuffle
    random.shuffle(data)

    # Set the ratio to 2:8
    split_idx = int(0.2 * len(data))
    train_set = data[:split_idx]
    test_set = data[split_idx:]

    # Output training set and test set
    with open(train_queries_file_path, 'w') as train_file:
        for entry in train_set:
            train_file.write(json.dumps(entry) + '\n')

    # Save the test set to a jsonl file
    with open(test_queries_file_path, 'w') as test_file:
        for entry in test_set:
            test_file.write(json.dumps(entry) + '\n')

    print(f"Train-datasets size: {len(train_set)}")
    print(f"Test-datasets  size: {len(test_set)}")