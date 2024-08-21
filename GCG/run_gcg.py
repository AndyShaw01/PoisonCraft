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

    gcg = GCG(args)
    target_sentence = pd.read_csv('./Dataset/infovector.csv')['text'].tolist()[args.index]
    optim_prompts, steps, _ = gcg.run(target_sentence)
    print("The target sentence is: ", target_sentence)
    print("The optimized prompts are: ", optim_prompts)
    print("The number of steps is: ", steps)

def gcg_attack_all(args):

    # Read the question sentence from jsonl file to queries
    with open(args.train_queries_path, 'r') as f:
        queries_id = []
        queries_text = []
        for line in f:
            data = json.loads(line)
            queries_id.append(data['_id'])
            queries_text.append(data['text'])
    
    # Read ground truth file
    ground_truth = pd.read_csv(args.ground_truth_path)
    
    gcg = GCG(args)
    # For loop queries_id, ground_truth_values
    suffix_list = []
    optim_prompts_list = []
    for i in range(len(queries_id)):
        args.index = i
        gcg.question = queries_text[i]
        gcg.product_threshold = ground_truth[ground_truth['test_name'] == queries_id[i]].values[0][1]
        optim_prompts, steps, suffix = gcg.run(args.target)
        suffix_list.append(suffix)
        optim_prompts_list.append(optim_prompts)
    
    # Save suffix_list to csv
    with open(f'{args.save_path}/init/suffix.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(suffix_list)

    with open(f'{args.save_path}/init/optim_prompts_list', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(optim_prompts_list)