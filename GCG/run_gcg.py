import os
import pandas as pd
import json
import random
import multiprocessing
import time
from GCG.gcg import GCG

# 设置多进程启动方式为'spawn'，在Linux上通常更稳定
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
    针对一个批次进行对抗生成。
    """
    # 为每个进程设置独立的保存路径
    temp_save_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_repeat_{index}.csv"
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
    
    args.save_path = temp_save_path
    args.index = index
    args.question = batch
    
    gcg = GCG(args)
    gcg.run(args.target)
    print(f"Repeat {index} processing complete. Results saved to {temp_save_path}")

def run_repeat(args, repeat_index):
    print(f"Starting repeat {repeat_index}")  # 添加打印，检查是否进入函数
    batch_sampler = BatchSampler(args.train_queries_path, 
                                 args.attack_batch_size, 
                                 args.group_index, 
                                 args.control_string_length, 
                                 args.target)
    batches = batch_sampler.create_batches(repeat_times=1)
    for i, batch in enumerate(batches):
        print(f"Running batch {i} for repeat {repeat_index}")  # 添加打印，检查每个batch的运行
        gcg_attack_batch(args, batch, repeat_index)
    print(f"Repeat {repeat_index} processing complete.")

def gcg_attack_all(args, repeat_times=4):
    """
    针对不同的repeat次数分别创建批次并进行对抗生成。
    每个repeat的结果将由不同的进程并行执行。
    """
    # 使用multiprocessing对不同的repeat进行处理
    processes = []
    for repeat_index in range(repeat_times):
        p = multiprocessing.Process(target=run_repeat, args=(args, repeat_index))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All repeats processing complete.")

    # 合并所有结果文件
    combined_results_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/combined_results_{args.control_string_length}.csv"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    with open(combined_results_path, 'w', newline='') as combined_file:
        writer = None
        for repeat_index in range(repeat_times):
            repeat_result_path = f"./Results/improve_gcg_test/batch-{args.attack_batch_size}/category_{args.group_index}/results_{args.control_string_length}_repeat_{repeat_index}.csv"
            if os.path.exists(repeat_result_path):
                with open(repeat_result_path, 'r') as repeat_file:
                    reader = pd.read_csv(repeat_file)
                    if writer is None:
                        reader.to_csv(combined_file, index=False)
                        writer = True
                    else:
                        reader.to_csv(combined_file, index=False, header=False)
                # 删除临时文件
                os.remove(repeat_result_path)
    print(f"All repeat results combined into {combined_results_path}")

def gcg_attack(args):
    """
    针对所有的query进行对抗生成。
    """
    gcg = GCG(args)
    gcg.run(args.target)
    print(f"Results saved to {args.save_path}")