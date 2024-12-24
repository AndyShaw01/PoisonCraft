import os
import pandas as pd
import json
import random
import multiprocessing
import logging
import argparse

from gcg import GCG

multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')


class Config:
    """
    Configuration class for the GCG attack.

    Attributes:
        train_queries_path (str): Path to the train queries file.
        attack_target (str): The target of the attack.
        attack_batch_size (int): Number of queries to attack in each batch.
        domain_index (int): Index of the domain to attack.
        control_string_length (int): Length of the control string.
    """
    def __init__(self, train_queries_path, attack_target, attack_batch_size, domain_index, control_string_length):
        self.train_queries_path = train_queries_path
        self.attack_target = attack_target
        self.attack_batch_size = attack_batch_size
        self.domain_index = domain_index
        self.control_string_length = control_string_length

    def get_results_path(self, epoch_index, batch_index):
        return (
            f"./Results/{self.attack_target}/batch-{self.attack_batch_size}/"
            f"domain_{self.domain_index}/results_{self.control_string_length}_"
            f"epoch_{epoch_index}_batch_{batch_index}.csv"
        )


class BatchSampler:
    """
    Batch sampler

    Attributes:
        train_queries_path (str): Path to the train queries file.
        attack_batch_size (int): Number of queries to attack in each batch.
        queries_text (list): List of queries text.
    """
    def __init__(self, train_queries_path, attack_batch_size):
        self.train_queries_path = train_queries_path
        self.attack_batch_size = attack_batch_size
        self.queries_text = []
        self._load_data()

    def _load_data(self):
        with open(self.train_queries_path, 'r') as f:
            self.queries_text = [json.loads(line)['text'] for line in f]

    def __iter__(self):
        random.shuffle(self.queries_text)
        for i in range(0, len(self.queries_text), self.attack_batch_size):
            yield self.queries_text[i:i + self.attack_batch_size]


def save_results_path(config, epoch_index, batch_index):
    save_path = config.get_results_path(epoch_index, batch_index)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path


def gcg_attack_batch(config, args, batch, epoch_index, batch_index):
    save_path = save_results_path(config, epoch_index, batch_index)
    args.save_path = save_path
    gcg = GCG(args)
    gcg.question = batch
    gcg.index = batch_index
    gcg.run(args.target)
    logging.info(f"Epoch {epoch_index}, Batch {batch_index}: Results saved to {save_path}")


def run_epoch(args, epoch_index):
    config = Config(
        args.train_queries_path, args.attack_target, args.attack_batch_size,
        args.domain_index, args.control_string_length
    )
    logging.info(f"Starting epoch {epoch_index}")
    batch_sampler = BatchSampler(config.train_queries_path, config.attack_batch_size)
    for batch_index, batch in enumerate(batch_sampler):
        gcg_attack_batch(config, args, batch, epoch_index, batch_index)
    logging.info(f"Epoch {epoch_index} completed.")


def merge_results(config, epoch_times):
    combined_results_path = f"./Results/{config.attack_target}/batch-{config.attack_batch_size}/domain_{config.domain_index}/combined_results_{config.control_string_length}.csv"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    all_dataframes = []
    for epoch_index in range(epoch_times):
        epoch_dir = f"./Results/{config.attack_target}/batch-{config.attack_batch_size}/domain_{config.domain_index}"
        for file in os.listdir(epoch_dir):
            if file.startswith(f"results_{config.control_string_length}_epoch_{epoch_index}"):
                df = pd.read_csv(os.path.join(epoch_dir, file))
                all_dataframes.append(df)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(combined_results_path, index=False)
        logging.info(f"All results combined into {combined_results_path}")
    else:
        logging.warning("No results found to combine.")


def gcg_attack(args, epoch_times=1):
    config = Config(
        args.train_queries_path, args.attack_target, args.attack_batch_size,
        args.domain_index, args.control_string_length
    )
    if epoch_times > 1:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for epoch_index in range(epoch_times):
                pool.apply_async(run_epoch, args=(args, epoch_index))
            pool.close()
            pool.join()
        logging.info("All epochs completed.")
        merge_results(config, epoch_times)
    else:
        run_epoch(args, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GCG attack on a set of queries.")
    parser.add_argument("--train_queries_path", type=str, help="Path to the train queries file.")
    parser.add_argument("--attack_target", type=str, help="The target of the attack.")
    parser.add_argument("--attack_batch_size", type=int, help="Number of queries to attack in each batch.")
    parser.add_argument("--domain_index", type=int, help="Index of the domain to attack.")
    parser.add_argument("--control_string_length", type=int, help="Length of the control string.")
    parser.add_argument("--target", type=str, help="The target of the attack.")
    parser.add_argument("--epoch_times", type=int, default=1, help="Number of epochs to run.")
    args = parser.parse_args()

    gcg_attack(args, args.epoch_times)
