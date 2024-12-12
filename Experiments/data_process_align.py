import pandas as pd
import json
import argparse
import pdb

def get_matched_bar(results_file_path, k):
    """
    Based on the results file, get the kth matched bar for each query.
    """ 
    with open(results_file_path, 'r') as f:
        data = json.load(f)
        
    kth_similarity = []
    for test_name in data:
        test_data = data[test_name]
        sorted_similarity = sorted(test_data.values(), reverse=True)
        if k < 0 or k >= len(sorted_similarity):
            raise IndexError(f"Index {k} is out of bounds for '{testname}' with length {len(sorted_similarity)}.")
        kth_similarity.append({
            'test_name': test_name,
            f'matched_bar_{k}': sorted_similarity[k]
        })
        print("K:\t", sorted_similarity[k])
        
    kth_similarity_df = pd.DataFrame(kth_similarity)
    return kth_similarity_df

def main(args):
    # 获取第 k 个相似度的匹配数据
    kth_similarity_df = get_matched_bar(args.results_file_path, args.k)

    # save the kth similarity to csv
    # kth_similarity_df.to_csv(f'./Datasets/{args.dataset}/ground_truth_topk/{args.dataset}_top_{args.k}_category_{args.category}.csv', index=False)

    # 读取 JSONL 文件的 _id 顺序
    queries_id = []
    with open(args.train_queries_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries_id.append(data['_id'])
    # 根据 _id 过滤并对 CSV 数据进行排序
    selected_df = kth_similarity_df[kth_similarity_df['test_name'].isin(queries_id)]
    selected_df['id_order'] = pd.Categorical(selected_df['test_name'], categories=queries_id, ordered=True)
    sorted_df = selected_df.sort_values('id_order').drop(columns='id_order')
    # 保存对齐后的文件
    sorted_df.to_csv(f'./Datasets/{args.dataset}/ground_truth_topk_{args.retriever}/ground_truth_top_{args.k}_domain_{args.category}.csv', index=False)
    print("对齐后的结果已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--results_file_path', type=str, default='./Dataset/nq/nq-contriever.json', help='The path to the results file')  
    parser.add_argument('--train_queries_path', type=str, default='./Dataset/nq/test_queries.jsonl', help='The path to the train queries file')
    parser.add_argument('--k', type=int, default=19, help='The number of top k results to consider')
    parser.add_argument('--category', type=int, default=1, help='The category of the queries')
    parser.add_argument('--dataset', choices=['hotpotqa', 'msmarco', 'nq'], default='nq', help='The dataset to process')
    parser.add_argument('--retriever', choices=['contriever', 'contriever-msmarco', 'ance', 'openai'], default='openai', help='The retriever to process')

    args = parser.parse_args()

    args.results_file_path = f'./Datasets/{args.dataset}/{args.dataset}-{args.retriever}.json'
    

    main(args)  
