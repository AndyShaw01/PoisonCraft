import pandas as pd
import json
import os
import argparse
import pdb

def get_matched_bar(results_file_path, k):
    """
    Based on the results file, get the kth matched bar for each query.

    Args:
        results_file_path (str): The path to the results file
        k (int): The number of top k results to consider

    Returns:
        pd.DataFrame: The kth similarity for each query
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
        
    kth_similarity_df = pd.DataFrame(kth_similarity)
    return kth_similarity_df

def main(args):
    # Get the kth similarity for each query
    kth_similarity_df = get_matched_bar(args.results_file_path, args.k)

    # Load the train queries
    queries_id = []
    with open(args.target_queries_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries_id.append(data['_id'])
    # sort the kth similarity by the order of the queries
    selected_df = kth_similarity_df[kth_similarity_df['test_name'].isin(queries_id)]
    selected_df['id_order'] = pd.Categorical(selected_df['test_name'], categories=queries_id, ordered=True)
    sorted_df = selected_df.sort_values('id_order').drop(columns='id_order')

    file_path = f'./datasets/{args.dataset}/ground_truth_topk_{args.retriever}/ground_truth_top_{args.k}_domain_{args.domain}.csv'
    if not os.path.exists(os.path.dirname(file_path)): 
        os.makedirs(os.path.dirname(file_path))

    sorted_df.to_csv(file_path, index=False)
    print("The aligned results have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the top-k similarity score for each test query')
    parser.add_argument('--results_file_path', type=str, default='./datasets/nq/nq-contriever.json', help='The path to the beir evaluation results file')  
    parser.add_argument('--target_queries_path', type=str, default='./datasets/nq/test_queries.jsonl', help='The path to the train queries file')
    parser.add_argument('--k', type=int, default=19, help='The number of top k results to consider')
    parser.add_argument('--domain', type=int, default=1, help='The category of the queries')
    parser.add_argument('--dataset', choices=['hotpotqa', 'msmarco', 'nq'], default='nq', help='The dataset to process')
    parser.add_argument('--retriever', choices=['contriever', 'contriever-msmarco', 'ance', 'openai','simcse', 'openai-002', 'openai_3-small','openai_3-large', 'bge-small'], default='bge-small', help='The retriever to process')

    args = parser.parse_args()

    args.results_file_path = f'./datasets/{args.dataset}/{args.dataset}-{args.retriever}-dot.json'
    

    main(args)  
