import os
import sys
import csv
import pandas as pd
import argparse
import numpy as np

import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel

from sklearn.cluster import KMeans, DBSCAN

class DomainSeedSampler:
    def __init__(self, args):
        self.args = args
        self.domain_control_suffix_dict = {}
        self.domain_embeddings_dict = {}
        self.domain_clusters_dict = {}
        self.domain_sampled_seeds_dict = {}
        self.embedding_model = SentenceEmbeddingModel(args.model_path)
        self.embedding_model.to(self.embedding_model.device)
        self.embedding_model.eval()

    def load_domain_control_suffix(self):
        domain_suffix_folder = self.args.domain_suffix_folder
        domain_control_suffix_dict = {}
        for domain_index in self.args.domain_list:
            domain_folder = os.path.join(domain_suffix_folder, f"category_{domain_index}")
            if not os.path.exists(domain_folder):
                print(f"Warning: {domain_folder} does not exist.")
                continue
            csv_files = [f for f in os.listdir(domain_folder) if f.startswith('results_') and f.endswith('.csv')]
            domain_control_suffix_list = []
            for csv_file in csv_files:
                csv_path = os.path.join(domain_folder, csv_file)
                df = pd.read_csv(csv_path)
                if 'control_suffix' in df.columns:
                    domain_control_suffix_list.append(df['control_suffix'])
                else:
                    print(f"Warning: 'control_suffix' column not found in {csv_path}")
            if domain_control_suffix_list:
                domain_control_suffix_series = pd.concat(domain_control_suffix_list, axis=0).reset_index(drop=True)
                domain_control_suffix_dict[domain_index] = domain_control_suffix_series
            else:
                print(f"Warning: No valid control_suffix columns found for domain {domain_index}")
        self.domain_control_suffix_dict = domain_control_suffix_dict
        print(f"Processed {len(domain_control_suffix_dict)} domains successfully.")
    
    def cluster_domain_seeds(self):
        for domain_index, control_suffix_series in self.domain_control_suffix_dict.items():
            embeddings = np.zeros((len(control_suffix_series), 768))
            chunk_size = 512
            for start_idx in range(0, len(control_suffix_series), chunk_size):
                end_idx = min(start_idx + chunk_size, len(control_suffix_series))
                chunk = control_suffix_series[start_idx:end_idx]
                chunk_embeddings = self.embedding_model(chunk.tolist()).detach().cpu().numpy()
                embeddings[start_idx:end_idx] = chunk_embeddings
            self.domain_embeddings_dict[domain_index] = embeddings
            print(f"Generated embeddings for domain {domain_index}")
        cluster_num_per_domain = self.args.cluster_num_per_domain
        for domain_index, embeddings in self.domain_embeddings_dict.items():
            if self.args.cluster_methods == 'kmeans':
                clustering_model = KMeans(n_clusters=cluster_num_per_domain, random_state=42)
                clustering_model.fit(embeddings)
                unique, counts = np.unique(clustering_model.labels_, return_counts=True)
                cluster_counts = dict(zip(unique, counts))
                print(f"Clustered domain {domain_index} into {cluster_num_per_domain} clusters using {self.args.cluster_methods}")
                print(f"Cluster counts for domain {domain_index}: {cluster_counts}")
            elif self.args.cluster_methods == 'dbscan':
                clustering_model = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
                unique, counts = np.unique(clustering_model.labels_, return_counts=True)
                cluster_counts = dict(zip(unique, counts))
                print(f"Clustered domain {domain_index} into {counts} clusters using {self.args.cluster_methods}")
                print(f"Cluster counts for domain {domain_index}: {cluster_counts}")
            else:
                raise ValueError("Unsupported clustering method selected.")
            self.domain_clusters_dict[domain_index] = clustering_model.labels_
    
    def sample_seeds_from_clusters(self):
        for domain_index, labels in self.domain_clusters_dict.items():
            unique_labels = set(labels)
            control_suffix_series = self.domain_control_suffix_dict[domain_index]
            sampled_seeds = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
                cluster_size = len(cluster_indices)
                sample_size = max(1, cluster_size // 4)
                sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)
                sampled_seeds.extend([control_suffix_series.iloc[i] for i in sampled_indices])
            self.domain_sampled_seeds_dict[domain_index] = sampled_seeds
            print(f"Sampled {len(sampled_seeds)} seeds from domain {domain_index}")
    def get_sampled_results(self):
        self.load_domain_control_suffix()
        self.cluster_domain_seeds()
        self.sample_seeds_from_clusters()

        cluster_result_path = self.args.cluster_result_path
        # transfer the results to csv file, each line is the domain index and the seed: domain_index,seed
        with open(cluster_result_path, 'w', newline='', encoding='utf-8') as f:
            f.write("domain_id,control_suffix\n")
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC) 
            writer.writerow(["domain_id", "control_suffix"])
            for domain_index, seeds in self.domain_sampled_seeds_dict.items():
                for seed in seeds:
                    writer.writerow([domain_index, str(seed)])  
                    
        print(f"Clustered seeds written to {cluster_result_path}")
        return self.domain_sampled_seeds_dict


def main(args):
    sampler = DomainSeedSampler(args)
    domain_sampled_seeds_dict = sampler.get_sampled_results()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--domain_list', type=str, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], help='The index of the target domain')
    parser.add_argument('--embedder', type=str, choices=['contriever', 'openai'], default='contriever', help='The index of the target domain')
    parser.add_argument('--cluster_methods', type=str, choices=['kmeans', 'dbscan'], default='kmeans', help='The index of the target domain')
    parser.add_argument('--cluster_num_per_domain', type=int, default=5, help='The index of the target domain')
    parser.add_argument('--model_path', type=str, default='/data1/shaoyangguang/offline_model/contriever', help='target model path')
    parser.add_argument('--seeds_initial_path', type=str, default='/data1/shaoyangguang/TransferAttack/seed_initial.csv', help='seed initial path')
    parser.add_argument('--cluster_result_path', type=str, default='./Result/transfer_attack/cluster_sample_25.csv', help='cluster path')
    parser.add_argument('--domain_suffix_folder', type=str, default='Results_from_A800/part_results/Results/improve_gcg_test/batch-4/', help='domain result folder')

    args = parser.parse_args()

    main(args)
