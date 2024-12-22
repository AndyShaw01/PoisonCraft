import os
import sys
import pdb
import json
import torch   
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from embedding.SentenceEmbedding import SentenceEmbeddingModel

def is_nan(value):
    try:
        return np.isnan(value)
    except TypeError:
        # 如果类型不支持 isnan，则返回 False
        return False

def get_asr(candidate_file, ground_truth_file, queries_file, threshold):
    try:
        df = pd.read_csv(candidate_file)
    except:
        return
    ground_truth_df = pd.read_csv(ground_truth_file)[f'matched_bar_{threshold}']
    ground_truth = torch.tensor(ground_truth_df, device='cuda:0')
    for i in range(len(df)):
        attack_control_suffix = df['control_suffix'].iloc[i]
        if is_nan(attack_control_suffix):
            continue
        df_queries = pd.read_json(queries_file, lines=True)
        questions = df_queries['text'].tolist()

        sentence = f'the website is www.baidu.com {attack_control_suffix}'

        model_path = "/data1/shaoyangguang/offline_model/contriever"
        similarity_model = SentenceEmbeddingModel(model_path)
        similarity_model.to(similarity_model.device)

        sentence_embedding = similarity_model(sentence)
        questions_embedding = similarity_model(questions)

        similarity = torch.matmul(sentence_embedding, questions_embedding.t())
        jailbreak_num = sum(similarity[0]>ground_truth)
        print(f"{i}, ASN: {jailbreak_num}, ASR: {jailbreak_num/len(questions)}")
        # secend_max = torch.topk(similarity[0], 8).values[7]
        # mean_similarity = torch.mean(similarity)
def main(args):
    # args.candidate_file = f'./Results/improve_exp/batch-4/category_{args.category}/results_top_{args.threshold}.csv'
    # args.ground_truth_file = f'./Dataset/nq/ground_truth/ground_truth_top_{args.threshold}_category_{args.category}.csv'
    # args.queries_file = f'./Dataset/nq/category/categorized_jsonl_files_14/category_{args.category}.jsonl'
    category_list = [4, 6, 8, 11, 12]
    threshold_list = [10, 20, 50]
    for category in category_list:
        for threshold in threshold_list:
            print(f"Category: {category}, Threshold: {threshold}")
            candidate_file = f'./Results/improve_exp/batch-4/category_{category}/results_top_{threshold}.csv'
            ground_truth_file = f'./Dataset/nq/ground_truth/ground_truth_top_{threshold}_category_{category}.csv'
            queries_file = f'./Dataset/nq/category/categorized_jsonl_files_14/category_{category}.jsonl'
            get_asr(candidate_file, ground_truth_file, queries_file, threshold)
            print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get attack suffix')
    parser.add_argument('--category', type=int, default=3, help='The category')
    parser.add_argument('--threshold', type=int, default=10, help='The threshold')
    parser.add_argument('--candidate_file', type=str, default='./Results/improve_exp_0830/batch-4/category_6/results_top_10.csv', help='The candidate file')
    parser.add_argument('--queries_file', type=str, default='./Dataset/nq/category/categorized_jsonl_files/category_6.jsonl', help='The queries file')
    parser.add_argument('--ground_truth_file', type=str, default='./Dataset/nq/ground_truth/ground_truth_top_10_category_6.jsonl', help='The ground truth file')

    args = parser.parse_args()
    main(args)
