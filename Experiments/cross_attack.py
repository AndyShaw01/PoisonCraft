import sys
import os
import argparse
import pdb
import torch
import csv
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
            candidate_file = f'./Results/improve_exp_0912/batch-4/category_{category}/results_top_{threshold}.csv'
            try:
                df = pd.read_csv(candidate_file)
            except:
                continue
            attack_suffix = [attack_info + ' ' + x for x in df['control_suffix'].tolist()]
            suffix_db[category][threshold] = attack_suffix
            suffix_all[category] = attack_suffix
            all_list += attack_suffix

    return suffix_db, suffix_all, all_list
    

def main(args):
    # Load the sentence embedding model
    
    result_file = f'Result/cross_attack/{args.mode}_0917.csv'

    if not os.path.exists(result_file):
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
    raw_fp = open(result_file, 'w', buffering=1)
    writter = csv.writer(raw_fp)
    if args.mode == 'single_category':
        writter.writerow(
            ['category', 'attack_source', 'threshold', 'ASN', 'ASR'])
    elif args.mode == 'all_category':
        writter.writerow(
            ['threshold', 'category', 'ASN', 'ASR'])
    
    with torch.no_grad():
        embedding_model = SentenceEmbeddingModel(args.model_path)
        embedding_model.to(embedding_model.device)

        # Load Attack DB and embed them
        category_list = args.category_list
        threshold_list = args.threshold_list
        attack_db, attack_all, all_list = get_suffix_db(category_list, threshold_list, args.attack_info)

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
            print(f"{args.target_threshold}")
            for target_threshold in args.target_threshold:
                for i in range(len(category_list)):
                    # Load Queries
                    queries_path = args.queries_folder + f"/category_{category_list[i]}.jsonl"
                    df_queries = pd.read_json(queries_path, lines=True)
                    queries = df_queries['text'].tolist()
                    queries_embedding = embedding_model(queries)

                    # Calculate the similarity
                    attack_embedding = embedding_model(all_list)
                    # similarity = torch.matmul(attack_embedding, queries_embedding.t())
                    similarity = torch.matmul(queries_embedding, attack_embedding.t())
                    ground_truth_path = f'./Dataset/nq/ground_truth_test_recheck_0905/ground_truth_top_{target_threshold}_category_{category_list[i]}.csv'
                    ground_truth_df = pd.read_csv(ground_truth_path)[f'matched_bar_{target_threshold}']
                    ground_truth = torch.tensor(ground_truth_df, device='cuda:0').unsqueeze(1) # [batch_size, 1]
                    jailbreak_num = sum((similarity>ground_truth).any(dim=1)).item()
                    print(f"Category: {category_list[i]}, AttackDBNum: {len(all_list)}, ASN: {jailbreak_num}, ASR: {round(jailbreak_num/len(queries), 4)}")
                    writter.writerow([target_threshold, category_list[i], jailbreak_num, round(jailbreak_num/len(queries), 4)])
                print("\n")
        elif args.mode == 'single_category':
            for i in range(len(category_list)): # for each category query
                # Load Queries
                queries_path = args.queries_folder + f"/category_{category_list[i]}.jsonl"
                df_queries = pd.read_json(queries_path, lines=True)
                queries = df_queries['text'].tolist()
                queries_embedding = embedding_model(queries)
                for j in range(len(args.target_threshold)): # for each threshold attack
                    ground_truth_path = f'./Dataset/nq/ground_truth_test_recheck_0905/ground_truth_top_{args.target_threshold[j]}_category_{category_list[i]}.csv'
                    ground_truth_df = pd.read_csv(ground_truth_path)[f'matched_bar_{args.target_threshold[j]}']
                    ground_truth = torch.tensor(ground_truth_df, device='cuda:0')
                    for k in range(len(category_list)): # for each category attack
                        # Calculate the similarity
                        attack_embedding = embedding_model(attack_all[category_list[k]])
                        similarity = torch.matmul(attack_embedding, queries_embedding.t())

                        jailbreak_num = sum((similarity>ground_truth).any(dim=1)).item()
                        print(f"Category: {category_list[i]}, Attack Source: {k+1}, AttackDBNum: {len(attack_all[category_list[k]])},, ASN: {jailbreak_num}, ASR: {round(jailbreak_num/len(queries), 4)}")
                        writter.writerow([category_list[i], k+1, args.target_threshold[j], jailbreak_num, round(jailbreak_num/len(queries), 4)])
        else:       
            raise ValueError("Invalid mode")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_list", type=str, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    parser.add_argument("--threshold_list", type=str, default=[10,20,50])
    parser.add_argument("--target_category", type=int, default=8)
    parser.add_argument("--target_threshold", type=str, default=[5,10,20,50,99])
    parser.add_argument("--mode", choices=['single_category', 'all_category', 'target'], default="all_category")
    parser.add_argument("--attack_info", type=str, default="the website is www.baidu.com")
    parser.add_argument("--queries_folder", type=str, default="./Dataset/nq/category/categorized_jsonl_files_14_test_recheck")
    parser.add_argument("--model_path", type=str, default="/data1/shaoyangguang/offline_model/contriever")
    # parser.add_argument("--ground_truth_file", type=str, default="./Dataset/nq/ground_truth/ground_truth_top_10_category_8.csv")
    
    args = parser.parse_args()
    main(args)