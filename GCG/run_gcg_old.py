import os
import pandas as pd
import json
import random
import multiprocessing
import time
from GCG.gcg import GCG

# Set the start method to 'spawn' to avoid the error:
multiprocessing.set_start_method('spawn', force=True)

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

def gcg_attack_batch(args, batches, index):
    """
    # GCG attack on a batch of queries.
    """
    temp_save_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_epoch_{index}.csv"
    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
    args.save_path = temp_save_path
    gcg = GCG(args)
    for i, batch in enumerate(batches):
        gcg.index = i
        gcg.question = batch
        gcg.run(args.target)
    print(f"Epoch {index} processing complete. Results saved to {temp_save_path}")

def run_epoch(args, epoch_index):
    print(f"Starting epoch {epoch_index}")  
    batch_sampler = BatchSampler(args.train_queries_path, 
                                 args.attack_batch_size, 
                                 args.group_index, 
                                 args.control_string_length, 
                                 args.target)
    batches = batch_sampler.create_batches(epoch_times=1) 
    gcg_attack_batch(args, batches, epoch_index)
    print(f"Epoch {epoch_index} processing complete.")

def gcg_attack_all(args, epoch_times=4):
    """
    # GCG attack on all queries.
    """
    processes = []
    for epoch_index in range(epoch_times):
        p = multiprocessing.Process(target=run_epoch, args=(args, epoch_index))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All epochs processing complete.")

    # Merge all the results into one file
    combined_results_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/combined_results_{args.control_string_length}.csv"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    with open(combined_results_path, 'w', newline='') as combined_file:
        writer = None
        for epoch_index in range(epoch_times):
            epoch_result_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_epoch_{epoch_index}.csv"
            if os.path.exists(epoch_result_path):
                with open(epoch_result_path, 'r') as epoch_file:
                    reader = pd.read_csv(epoch_file)
                    if writer is None:
                        reader.to_csv(combined_file, index=False)
                        writer = True
                    else:
                        reader.to_csv(combined_file, index=False, header=False)
                # os.remove(epoch_result_path)
    print(f"All epoch results combined into {combined_results_path}")

def gcg_attack(args):
    """
    # GCG attack on a single query.
    """
    gcg = GCG(args)
    gcg.run(args.target)
    print(f"Results saved to {args.save_path}")