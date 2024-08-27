import os
import pandas as pd
import pdb
import argparse
from GCG.gcg import GCG
import csv
import json
# from GCG.utils.templates import get_eos

def gcg_attack(args):

    question = pd.read_csv('Dataset/question.csv')['text'].tolist()[args.index]

    args.question = question   
    print("The question sentence is: ", question)
    args.product_threshold = 0.8
    gcg = GCG(args)
    target_sentence = pd.read_csv('./Dataset/infovector.csv')['text'].tolist()[args.index]
    optim_prompts, steps, _ = gcg.run(target_sentence)
    print("The target sentence is: ", target_sentence)
    print("The optimized prompts are: ", optim_prompts)
    print("The number of steps is: ", steps)

def gcg_attack_all(args):

    with open(args.train_queries_path, 'r') as f:
        queries_id = []
        queries_text = []
        for line in f:
            data = json.loads(line)
            queries_id.append(data['_id'])
            queries_text.append(data['text'])
    
    ground_truth = pd.read_csv(f'./Dataset/nq/ground_truth/ground_truth_top_{args.topk}.csv')

    args.save_path = f"./Results/pre_exp/results_top_{args.topk}.csv"
    print("The save path is: ", args.save_path)
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    gcg = GCG(args)

    for i in range(len(queries_id)):
        args.index = i
        gcg.question = queries_text[i]
        gcg.product_threshold = ground_truth[ground_truth['test_name'] == queries_id[i]].values[0][1]
        gcg.run(args.target)