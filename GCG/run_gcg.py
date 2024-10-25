import os
import pandas as pd
import pdb
import argparse
from GCG.gcg import GCG
import csv
import json
import random
# from GCG.utils.templates import get_eos
class RepeatBatchSampler:
    def __init__(self, train_queries_path, attack_batch_size, repeat_degree, group_index, control_string_length, target):
        self.train_queries_path = train_queries_path
        self.attack_batch_size = attack_batch_size
        self.repeat_degree = repeat_degree
        self.group_index = group_index
        self.control_string_length = control_string_length
        self.target = target
        self.queries_id = []
        self.queries_text = []
        self.batches = []

        self._load_data()
        self._create_sample_pool()

    def _load_data(self):
        with open(self.train_queries_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.queries_id.append(data['_id'])
                self.queries_text.append(data['text'])

    def _create_sample_pool(self):
        self.sample_pool = self.queries_text * self.repeat_degree
    
    def create_batches(self):
        total_batches = len(self.sample_pool) // self.attack_batch_size
        for _ in range(total_batches):
            batch = random.sample(self.sample_pool, self.attack_batch_size)
            self.batches.append(batch)

        # 处理剩余样本
        remaining_samples = len(self.sample_pool) % self.attack_batch_size
        if remaining_samples > 0:
            batch = random.sample(self.sample_pool, remaining_samples)
            self.batches.append(batch)

    # Using:
    # args = {
    #     'train_queries_path': 'path/to/your/data.json',
    #     'attack_batch_size': 5,
    #     'group_index': 1,
    #     'control_string_length': 10,
    #     'target': 'your_target_here'
    # }
    # batch_sampler = BatchSampler(
    #     args['train_queries_path'],
    #     args['attack_batch_size'],
    #     args['group_index'],
    #     args['control_string_length'],
    #     args['target']
    # )
    # batch_sampler.create_batches()
    # batch_sampler.run_batches()
    # print("Batch processing complete.")
class BatchSampler:
    def __init__(self, train_queries_path, attack_batch_size, group_index, control_string_length, target):
        self.train_queries_path = train_queries_path
        self.attack_batch_size = attack_batch_size
        self.group_index = group_index
        self.control_string_length = control_string_length
        self.target = target
        self.queries_id = []
        self.queries_text = []
        self.batches = []

        self._load_data()

    def _load_data(self):
        with open(self.train_queries_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.queries_id.append(data['_id'])
                self.queries_text.append(data['text'])

    def create_batches(self, repeat_times=1):
        for i in range(repeat_times):
            # 随机打乱样本顺序
            random.shuffle(self.queries_text)
            # 将样本按batch_size分组
            batches = [self.queries_text[i:i + self.attack_batch_size] for i in range(0, len(self.queries_text), self.attack_batch_size)]
            self.batches.extend(batches)
        return self.batches

    # def run_batches(self):
    #     gcg = GCG(self)  # 确保GCG类接受当前实例
    #     for i, batch in enumerate(self.batches):
    #         self.index = i
    #         gcg.question = batch
    #         gcg.run(self.target)
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

    # with open(args.train_queries_path, 'r') as f:
    #     queries_id = []
    #     queries_text = []
    #     for line in f:
    #         data = json.loads(line)
    #         queries_id.append(data['_id'])
    #         queries_text.append(data['text'])
    
    if args.attack_batch_size > 1:
        args.save_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}.csv"
    else:
        args.save_path = f"./Results/improve_gcg/single/results.csv"
    print("The save path is: ", args.save_path)

    # if not os.path.exists(os.path.dirname(args.save_path)):
    #     os.makedirs(os.path.dirname(args.save_path))

    # gcg = GCG(args)
    # for i in range(len(queries_id)//args.attack_batch_size):
    #     args.index = i
    #     gcg.question = queries_text[i*args.attack_batch_size:(i+1)*args.attack_batch_size]
    #     gcg.run(args.target)
    batch_sampler = BatchSampler(args.train_queries_path, 
                                 args.attack_batch_size, 
                                 args.group_index, 
                                 args.control_string_length, 
                                 args.target)
    batch_sampler.create_batches(repeat_times=4)
    gcg = GCG(args)
    for i, batch in enumerate(batch_sampler.batches):
        args.index = i
        gcg.question = batch
        gcg.run(args.target)
        print(f"Batch {i} processing complete.")