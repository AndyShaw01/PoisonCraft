import os
import pandas as pd
import json
import random
import multiprocessing
import logging
import argparse
import pdb

from src.poisoncraft.gcg import GCG

multiprocessing.set_start_method('spawn', force=True)
console_lock = multiprocessing.Lock()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')


class Config:
    """
    Configuration class for the GCG attack.

    Attributes:
        shadow_queries_path (str): Path to the shadow queries file.
        attack_target (str): The target of the attack.
        batch_size (int): Number of queries to attack in each batch.
        domain_index (int): Index of the domain to attack.
        adv_string_length (int): Length of the adversarial string.
    """
    def __init__(self, shadow_queries_path, attack_target, batch_size, domain_index, adv_string_length, retriever):
        self.shadow_queries_path = shadow_queries_path
        self.attack_target = attack_target
        self.batch_size = batch_size
        self.domain_index = domain_index
        self.adv_string_length = adv_string_length
        self.retriever = retriever

    def get_results_path(self, epoch_index):
        return (
            f"./results/adv_suffix/{self.retriever}/{self.attack_target}/batch-{self.batch_size}/"
            f"domain_{self.domain_index}/results_{self.adv_string_length}_"
            f"epoch_{epoch_index}.csv"
        )


class BatchSampler:
    """
    Batch sampler, used to sample batches of shadow queries.

    Attributes:
        shadow_queries_path (str): Path to the shadow queries file.
        batch_size (int): Number of queries to attack in each batch.
        queries_text (list): List of queries text.
    """
    def __init__(self, shadow_queries_path, batch_size):
        self.shadow_queries_path = shadow_queries_path
        self.batch_size = batch_size
        self.queries_text = []
        self._load_data()

    def _load_data(self):
        with open(self.shadow_queries_path, 'r') as f:
            self.queries_text = [json.loads(line)['text'] for line in f]

    def __iter__(self):
        random.shuffle(self.queries_text)
        for i in range(0, len(self.queries_text), self.batch_size):
            yield self.queries_text[i:i + self.batch_size]


def save_results_path(config, epoch_index):
    """
    Save the results path

    Args:
        config (Config): Configuration object
        epoch_index (int): Index of the epoch

    Returns:
        str: Save path
    """
    save_path = config.get_results_path(epoch_index)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path


def gcg_attack_batch(config, args, epoch_index):
    """
    Run GCG attack on a batch of queries

    Args:
        config (Config): Configuration object
        args (argparse.Namespace): Arguments
        batch (list): List of queries
        epoch_index (int): Index of the epoch

    Returns:
        None
    """
    save_path = save_results_path(config, epoch_index)
    args.save_path = save_path
    gcg = GCG(args)
    gcg.run(args.attack_info)
    logging.info(f"Epoch {epoch_index}, Batch {args.index}: Results saved to {save_path}")


def run_epoch(args, epoch_index):
    """
    Run an epoch of the GCG attack
    """
    config = Config(
        args.shadow_queries_path, args.attack_target, args.batch_size,
        args.domain_index, args.adv_string_length, args.retriever
    )
    logging.info(f"Starting epoch {epoch_index}")
    batch_sampler = BatchSampler(config.shadow_queries_path, config.batch_size)
    for batch_index, batch in enumerate(batch_sampler):
        args.question = batch
        args.index = batch_index
        gcg_attack_batch(config, args, epoch_index)
    logging.info(f"Epoch {epoch_index} completed.")


def merge_results(config, epoch_times):
    """
    Merge the same adversarial string length results into one file from different epochs
    """
    combined_results_path = f"./Results/{config.attack_target}/batch-{config.batch_size}/domain_{config.domain_index}/combined_results_{config.adv_string_length}.csv"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    all_dataframes = []
    for epoch_index in range(epoch_times):
        epoch_dir = f"./Results/{config.attack_target}/batch-{config.batch_size}/domain_{config.domain_index}"
        for file in os.listdir(epoch_dir):
            if file.startswith(f"results_{config.adv_string_length}_epoch_{epoch_index}"):
                df = pd.read_csv(os.path.join(epoch_dir, file))
                all_dataframes.append(df)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv(combined_results_path, index=False)
        logging.info(f"All results combined into {combined_results_path}")
    else:
        logging.warning("No results found to combine.")


def gcg_attack(args, epoch_times=1):
    """
    Run the GCG attack
    """
    config = Config(
        args.shadow_queries_path, args.attack_target, args.batch_size,
        args.domain_index, args.adv_string_length, args.retriever
    )
    if epoch_times > 1:
        processes = []
        for epoch_index in range(epoch_times):
            p = multiprocessing.Process(target=run_epoch, args=(args, epoch_index))
            p.start()
            processes.append(p)
        for p in processes:
            p.join() 
        logging.info("All epochs completed.")
        merge_results(config, epoch_times)
    else:
        run_epoch(args, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GCG attack on a set of queries.")
    parser.add_argument("--shadow_queries_path", type=str, help="Path to the train queries file.")
    parser.add_argument("--attack_target", type=str, help="The target of the attack.")
    parser.add_argument("--batch_size", type=int, help="Number of queries to attack in each batch.")
    parser.add_argument("--domain_index", type=int, help="Index of the domain to attack.")
    parser.add_argument("--adv_string_length", type=int, help="Length of the control string.")
    parser.add_argument("--epoch_times", type=int, default=1, help="Number of epochs to run.")
    args = parser.parse_args()

    gcg_attack(args, args.epoch_times)
