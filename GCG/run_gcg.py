import os
import pandas as pd
import json
import random
import multiprocessing
import time
from GCG.gcg import GCG

# Set the start method to 'spawn' to avoid the error:
multiprocessing.set_start_method('spawn', force=True)

console_lock = multiprocessing.Lock()

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

    def create_batches(self, epoch_times=1):
        for _ in range(epoch_times):
            # Randomly shuffle the queries
            random.shuffle(self.queries_text)
            # Batch the queries
            batches = [self.queries_text[i:i + self.attack_batch_size] for i in range(0, len(self.queries_text), self.attack_batch_size)]
            self.batches.extend(batches)
        return self.batches

def gcg_attack_batch(args, batches, epoch_index):
    """
    GCG attack on a batch of queries.
    """
    temp_save_path = f"./Results/{args.attack_target}/batch-{args.attack_batch_size}/domain_{args.group_index}/results_{args.control_string_length}_epoch_{epoch_index}.csv"
    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
    args.save_path = temp_save_path
    gcg = GCG(args)
    for i, batch in enumerate(batches):
        gcg.index = i
        gcg.question = batch
        gcg.run(args.target)
    with console_lock:
        print(f"Epoch {epoch_index} processing complete. Results saved to {temp_save_path}")

def run_epoch(args, epoch_index):
    with console_lock:
        print(f"Starting epoch {epoch_index}")

    batch_sampler = BatchSampler(args.train_queries_path, 
                                 args.attack_batch_size, 
                                 args.group_index, 
                                 args.control_string_length, 
                                 args.target)
    batches = batch_sampler.create_batches(epoch_times=1) 
    gcg_attack_batch(args, batches, epoch_index)
    
    with console_lock:
        print(f"Epoch {epoch_index} processing complete.")

def merge_results(args, epoch_times):
    combined_results_path = f"./Results/{args.attack_target}/batch-{args.attack_batch_size}/domain_{args.group_index}/combined_results_{args.control_string_length}.csv"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    all_dataframes = []
    for epoch_index in range(epoch_times):
        epoch_result_path = f"./Results/{args.attack_target}/batch-{args.attack_batch_size}/domain_{args.group_index}/results_{args.control_string_length}_epoch_{epoch_index}.csv"
        if os.path.exists(epoch_result_path):
            df = pd.read_csv(epoch_result_path)
            all_dataframes.append(df)
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(combined_results_path, index=False)
    
    print(f"All epoch results combined into {combined_results_path}")

def gcg_attack_all(args, epoch_times=1):
    """
    GCG attack on all queries.
    """
    if epoch_times > 1:

        processes = []
        for epoch_index in range(epoch_times):
            p = multiprocessing.Process(target=run_epoch, args=(args, epoch_index))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  # 设置超时时间，例如 600 秒

        print("All epochs processing complete.")

        merge_results(args, epoch_times)
    else:
        # not parallel
        run_epoch(args, 0)

    # 合并结果
    

def gcg_attack(args):
    """
    GCG attack on a single query.
    """
    gcg = GCG(args)
    gcg.run(args.target)
    print(f"Results saved to {args.save_path}")