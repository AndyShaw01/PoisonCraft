import pandas as pd

import os
import json
import pdb
import argparse


def get_matched_bar(results_file_path, k):
    """
    Based on the results file, get the kth matched bar for each query.
    """
    with open(results_file_path, 'r') as f:
        data = json.load(f)
        
    kth_similarity = []
    for test_name in data:
        test_data = data[test_name]
        sorted_similarity = sorted(test_data.values(), reverse=True)
        if k < 0 or k >= len(sorted_similarity):
            raise IndexError(f"Index {k} is out of bounds for '{test_name}' with length {len(sorted_similarity)}.")
        kth_similarity.append({
            'test_name': test_name,
            f'matched_bar_{k}': sorted_similarity[k]
        })
        print("K:\t", sorted_similarity[k])
        
    # kth_similarity_df = pd.DataFrame.from_dict(kth_similarity, orient='index', columns=[f'matched_bar_{k}'])
    kth_similarity_df = pd.DataFrame(kth_similarity)
    save_path = f'./Datasets/nq/ground_truth_topk_{args.retriever}/{k}_th_similarity.csv'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    kth_similarity_df.to_csv(save_path, index=False)

    return kth_similarity_df

def main(args):
    # Read results file
    kth_similarity_df = get_matched_bar(args.results_file_path, args.k)

    # Read training queries by jsonl
    queries_id = []
    with open(args.train_queries_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries_id.append(data['_id'])
    
    selected_df = kth_similarity_df[kth_similarity_df['test_name'].isin(queries_id)]
    # selected_df.to_csv(args.ground_truth_file_path, index=False)
    save_path = f'./Datasets/nq/ground_truth_topk_{args.retriever}/ground_truth_top_{args.k}_domain_{args.category}.csv'
    print(f"Saving to {save_path}")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    selected_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--results_file_path', type=str, default='./Datasets/nq/nq-contriever-msmarco.json', help='The path to the results file')  
    # parser.add_argument('--ground_truth_file_path', type=str, default='./Dataset/nq/ground_truth_bar.csv', help='The path to the ground truth file')
    parser.add_argument('--train_queries_path', type=str, default='./Datasets/nq/test_queries.jsonl', help='The path to the train queries file')
    parser.add_argument('--k', type=int, default=49, help='The number of top k results to consider')
    parser.add_argument('--category', type=int, default=1, help='The category of the queries')
    parser.add_argument('--dataset', choices=['hotpotqa', 'msmarco', 'nq'], default='nq', help='The dataset to process')
    parser.add_argument('--retriever', choices=['contriever', 'contriever-msmarco', 'ance'], default='contriever-msmarco', help='The retriever to process')

    args = parser.parse_args()
    args.results_file_path = f'./Datasets/{args.dataset}/{args.dataset}-{args.retriever}.json'
    main(args)  