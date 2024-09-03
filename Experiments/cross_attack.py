import sys
import os
import argparse
import pdb
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel

def get_suffix_db(category_list, threshold_list, attack_info):
    suffix_db = {}
    suffix_all = {}
    all_list = []
    for category in category_list:
        suffix_db[category] = {}
        for threshold in threshold_list:
            candidate_file = f'./Results/improve_exp_0902/batch-4/category_{category}/results_top_{threshold}.csv'
            df = pd.read_csv(candidate_file)
            attack_suffix = [attack_info + ' ' + x for x in df['control_suffix'].tolist()]
            suffix_db[category][threshold] = attack_suffix
            suffix_all[category] = attack_suffix
            all_list += attack_suffix

    return suffix_db, suffix_all, all_list
    

def main(args):
    # Load the sentence embedding model
    embedding_model = SentenceEmbeddingModel(args.model_path)
    embedding_model.to(embedding_model.device)

    # Load Attack DB and embed them
    category_list = eval(args.category_list)
    threshold_list = eval(args.threshold_list)
    attack_db, attack_all, all_list = get_suffix_db(category_list, threshold_list,args.attack_info)

    # Load the ground truth
    if args.mode == 'target':
        # Load Queries
        queries_path = args.queries_folder + f"/category_{args.target_category}.jsonl"
        df_queries = pd.read_json(queries_path, lines=True)
        queries = df_queries['text'].tolist()
        queries_embedding = embedding_model(queries)

        # Calculate the similarity
        attack_embedding = embedding_model(attack_all[args.target_category])
        similarity = torch.matmul(attack_embedding, queries_embedding.t())

        ground_truth_path = f'./Dataset/nq/ground_truth_test/ground_truth_top_{args.target_threshold}_category_{args.target_category}.csv'
        ground_truth_df = pd.read_csv(ground_truth_path)[f'matched_bar_{args.target_threshold}']
        ground_truth = torch.tensor(ground_truth_df, device='cuda:0')
        jailbreak_num = sum(similarity[0]>ground_truth)
        print(f"ASN: {jailbreak_num}, ASR: {jailbreak_num/len(queries)}")
    elif args.mode == 'all_category':
        for i in range(len(category_list)):
            # Load Queries
            queries_path = args.queries_folder + f"/category_{category_list[i]}.jsonl"
            df_queries = pd.read_json(queries_path, lines=True)
            queries = df_queries['text'].tolist()
            queries_embedding = embedding_model(queries)

            # Calculate the similarity
            attack_embedding = embedding_model(all_list)
            similarity = torch.matmul(attack_embedding, queries_embedding.t())
    
            ground_truth_path = f'./Dataset/nq/ground_truth_test/ground_truth_top_{args.target_threshold}_category_{category_list[i]}.csv'
            ground_truth_df = pd.read_csv(ground_truth_path)[f'matched_bar_{args.target_threshold}']
            ground_truth = torch.tensor(ground_truth_df, device='cuda:0')
            jailbreak_num = sum(similarity[0]>ground_truth)
            print(f"Category: {category_list[i]}, ASN: {jailbreak_num}, ASR: {jailbreak_num/len(queries)}")
    else:
        raise ValueError("Invalid mode")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_list", type=str, default="[1,2,3,4,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--threshold_list", type=str, default="[10,20,50]")
    parser.add_argument("--target_category", type=int, default=8)
    parser.add_argument("--target_threshold", type=int, default=10)
    parser.add_argument("--mode", type=str, default="all_category")
    parser.add_argument("--attack_info", type=str, default="the website is www.baidu.com")
    parser.add_argument("--queries_folder", type=str, default="./Dataset/nq/category/categorized_jsonl_files_14_test")
    parser.add_argument("--model_path", type=str, default="/data1/shaoyangguang/offline_model/contriever")
    parser.add_argument("--ground_truth_file", type=str, default="./Dataset/nq/ground_truth/ground_truth_top_10_category_8.csv")
    
    args = parser.parse_args()
    main(args)