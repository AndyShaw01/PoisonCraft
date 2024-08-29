import os
import sys
import pdb
import json
import torch   
import argparse
import pandas as pd
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel

def main(args):
    df = pd.read_csv(args.candidate_file)
    for i in range(len(df)):
        attack_control_suffix = df['control_suffix'].iloc[i]
        df_queries = pd.read_json(args.queries_file, lines=True)
        questions = df_queries['text'].tolist()

        sentence = f'the website is www.baidu.com {attack_control_suffix}'

        model_path = "/data1/shaoyangguang/offline_model/contriever"
        similarity_model = SentenceEmbeddingModel(model_path)
        similarity_model.to(similarity_model.device)

        sentence_embedding = similarity_model(sentence)
        questions_embedding = similarity_model(questions)

        similarity = torch.matmul(sentence_embedding, questions_embedding.t())
        print(similarity)
        # 记录similarity中第二大的值
        # secend_max = torch.topk(similarity[0], 2).values[1]
        # 记录similarity中最大的值
        # max_max = torch.topk(similarity[0], 1).values[0]
        # # print(f"{i}, {secend_max}")
        # print(f"{i}, {max_max}")
        # # 判断，如果 similarity中若有大于0.9的值，则此时，记录下来
        # if len(similarity > 0.9)>3:
        #     print("The index is: ", i)
        #     print("The attack suffix is: ", attack_control_suffix)
        #     print("The similarity is: ", similarity)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get attack suffix')
    parser.add_argument('--candidate_file', type=str, default='./Results/pre_exp/results_top_50.csv', help='The candidate file')
    parser.add_argument('--queries_file', type=str, default='./Dataset/nq/train_queries_test.jsonl', help='The queries file')

    args = parser.parse_args()

    main(args)
