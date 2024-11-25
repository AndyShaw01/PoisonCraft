import os
import sys
import pandas as pd
import argparse
import numpy as np

import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from SentenceEmbedding.SentenceEmbedding import SentenceEmbeddingModel

from sklearn.cluster import KMeans, DBSCAN

def main(args):
    # Stage1 : Load the seed initial from different domain
    domain_suffix_folder = args.domain_suffix_folder
    domain_control_suffix_dict = {}
    
    # 遍历domain_suffix_folder，遍历所有子文件夹
    for domain_index in range(1, 15):  # 遍历 domain_1, domain_2, ..., domain_14
        domain_folder = os.path.join(domain_suffix_folder, f"category_{domain_index}")
        if not os.path.exists(domain_folder):
            print(f"Warning: {domain_folder} does not exist.")
            continue
        
        # 获取该 domain 文件夹中的所有 .csv 文件
        csv_files = [f for f in os.listdir(domain_folder) if f.startswith('results_') and f.endswith('.csv')]
        domain_control_suffix_list = []

        # 读取每一个 CSV 文件并提取 control_suffix 列
        for csv_file in csv_files:
            csv_path = os.path.join(domain_folder, csv_file)
            df = pd.read_csv(csv_path)
            if 'control_suffix' in df.columns:
                domain_control_suffix_list.append(df['control_suffix'])
            else:
                print(f"Warning: 'control_suffix' column not found in {csv_path}")
        
        # 将当前 domain 中的所有 control_suffix 列合并成一个列表，并存入字典中
        if domain_control_suffix_list:
            domain_control_suffix_series = pd.concat(domain_control_suffix_list, axis=0).reset_index(drop=True)
            domain_control_suffix_dict[domain_index] = domain_control_suffix_series
        else:
            print(f"Warning: No valid control_suffix columns found for domain {domain_index}")
    
    # 将 domain_control_suffix_dict 存储，供后续 stage 使用
    print(f"Processed {len(domain_control_suffix_dict)} domains successfully.")
    # Stage2 : Cluster the seed initial
    if args.embedder == 'contriever':
        model = SentenceEmbeddingModel(args.model_path)
        model.to(model.device)
        model.eval()
    elif args.embedder == 'openai':
        # You would load the OpenAI model here if implemented
        model = SentenceEmbeddingModel(args.model_path)  # Placeholder
        model.to(model.device)
    else:
        raise ValueError("Unsupported embedder selected.")

    domain_embeddings_dict = {}
    for domain_index, control_suffix_series in domain_control_suffix_dict.items():
        pdb.set_trace()
        # 获取每个 domain 的 control_suffix 的 embedding
        embeddings_list = []
        chunk_size = 512  # 定义每个块的大小
        for start_idx in range(0, len(control_suffix_series), chunk_size):
            end_idx = min(start_idx + chunk_size, len(control_suffix_series))
            chunk = control_suffix_series[start_idx:end_idx]
            chunk_embeddings = model(chunk.tolist())
            embeddings_list.extend(chunk_embeddings)
        domain_embeddings_dict[domain_index] = np.array(embeddings_list)
        print(f"Generated embeddings for domain {domain_index}")
    cluster_num_per_domain = args.cluster_num_per_domain
    pdb.set_trace()
    domain_clusters_dict = {}
    for domain_index, embeddings in domain_embeddings_dict.items():
        if args.cluster_methods == 'kmeans':
            clustering_model = KMeans(n_clusters=cluster_num_per_domain, random_state=42)
        elif args.cluster_methods == 'dbscan':
            clustering_model = DBSCAN()
        else:
            raise ValueError("Unsupported clustering method selected.")
        
        # 进行聚类
        clustering_model.fit(embeddings)
        domain_clusters_dict[domain_index] = clustering_model.labels_
        print(f"Clustered domain {domain_index} into {cluster_num_per_domain} clusters using {args.cluster_methods}")
    pdb.set_trace()
    # Stage3 : Sample the seed from the cluster
    domain_sampled_seeds_dict = {}
    for domain_index, labels in domain_clusters_dict.items():
        unique_labels = set(labels)
        domain_embeddings = domain_embeddings_dict[domain_index]
        sampled_seeds = []
        
        # 对每个类心采样 50% 的点
        for label in unique_labels:
            if label == -1:  # 忽略噪声点（仅对 DBSCAN 适用）
                continue
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            cluster_size = len(cluster_indices)
            sample_size = max(1, cluster_size // 2)  # 采样 50%，至少采样 1 个
            sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            sampled_seeds.extend([domain_embeddings[i] for i in sampled_indices])
        pdb.set_trace()
        domain_sampled_seeds_dict[domain_index] = sampled_seeds
        print(f"Sampled {len(sampled_seeds)} seeds from domain {domain_index}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--domain_list', type=str, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], help='The index of the target domain')
    parser.add_argument('--embedder', type=str, choices=['contriever', 'openai'], default='contriever', help='The index of the target domain')
    parser.add_argument('--cluster_methods', type=str, choices=['kmeans', 'dbscan'], default='kmeans', help='The index of the target domain')
    parser.add_argument('--cluster_num_per_domain', type=int, default=10, help='The index of the target domain')
    parser.add_argument('--model_path', type=str, default='/data1/shaoyangguang/offline_model/contriever', help='target model path')
    parser.add_argument('--seeds_initial_path', type=str, default='/data1/shaoyangguang/TransferAttack/seed_initial.csv', help='seed initial path')
    parser.add_argument('--cluster_result_path', type=str, default='/data1/shaoyangguang/TransferAttack/cluster.csv', help='cluster path')
    parser.add_argument('--domain_suffix_folder', type=str, default='Results_from_A800/part_results/Results/improve_gcg_test/batch-4/', help='domain result folder')

    args = parser.parse_args()

    main(args)
