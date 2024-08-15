import os
import pandas as pd
import pdb
import argparse
from GCG.gcg import GCG
import csv
# from GCG.utils.templates import get_eos

def gcg_attack(args):

    question = pd.read_csv('Dataset/question.csv')['text'].tolist()[args.index]

    args.question = question   
    print("The question sentence is: ", question)

    gcg = GCG(args)
    target_sentence = pd.read_csv('./Dataset/infovector.csv')['text'].tolist()[args.index]
    optim_prompts, steps = gcg.run(target_sentence)
    print("The target sentence is: ", target_sentence)
    print("The optimized prompts are: ", optim_prompts)
    print("The number of steps is: ", steps)
