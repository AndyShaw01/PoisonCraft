import os
import pandas as pd
import json
import random
import concurrent.futures
import pdb
from GCG.gcg import GCG

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
        for _ in range(repeat_times):
            # 随机打乱样本顺序
            random.shuffle(self.queries_text)
            # 将样本按batch_size分组
            batches = [self.queries_text[i:i + self.attack_batch_size] for i in range(0, len(self.queries_text), self.attack_batch_size)]
            self.batches.extend(batches)
        return self.batches

def gcg_attack_batch(args, batch, index):
    """
    Attack a single batch of questions using GCG.
    """
    # Set the save path for the results of this batch
    args.save_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_batch_{index}.csv"
    gcg = GCG(args)
    args.index = index
    gcg.question = batch
    gcg.run(args.target)
    print(f"Batch {index} processing complete. Results saved to {args.save_path}")

def gcg_attack_all(args):
    pdb.set_trace()
    """
    Attack all batches of questions using GCG.
    """
    # Create a batch sampler
    batch_sampler = BatchSampler(args.train_queries_path, 
                                 args.attack_batch_size, 
                                 args.group_index, 
                                 args.control_string_length, 
                                 args.target)
    batch_sampler.create_batches(repeat_times=3)

    # Attack all batches using multiple processes
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, batch in enumerate(batch_sampler.batches):
            futures.append(executor.submit(gcg_attack_batch, args, batch, i))

        # Wait for all processes to complete
        concurrent.futures.wait(futures)
        print("All batches processing complete.")

    # Combine all batch results into a single file
    combined_results_path = f"./Results/improve_gcg_parallel/batch-{args.attack_batch_size}/category_{args.group_index}/combined_results_{args.control_string_length}.csv"
    with open(combined_results_path, 'w', newline='') as combined_file:
        writer = None
        for i in range(len(batch_sampler.batches)):
            batch_result_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_batch_{i}.csv"
            if os.path.exists(batch_result_path):
                with open(batch_result_path, 'r') as batch_file:
                    reader = pd.read_csv(batch_file)
                    if writer is None:
                        reader.to_csv(combined_file, index=False)
                        writer = True
                    else:
                        reader.to_csv(combined_file, index=False, header=False)
    print(f"All batch results combined into {combined_results_path}")
