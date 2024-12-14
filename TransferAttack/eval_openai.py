import json
import openai
import numpy as np
from tqdm import tqdm

# 设置 OpenAI API 密钥
openai.api_key = "sk-proj-***"  # 请替换为您的实际 API 密钥

# 定义获取嵌入的函数
def get_embeddings(texts, model="text-embedding-3-small"):
    """
    批量获取文本的嵌入。
    """
    embeddings = []
    batch_size = 1000  # 根据 OpenAI 的限制调整批量大小
    for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入"):
        batch = texts[i:i+batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [item['embedding'] for item in response['data']]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# 读取 query.jsonl 并构建查询映射
def load_queries(query_file):
    """
    读取 query.jsonl 文件，返回 {query_id: query_text} 的字典。
    """
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
    return queries

# 读取 corpus.jsonl 并构建文档映射
def load_corpus(corpus_file):
    """
    读取 corpus.jsonl 文件，返回 {doc_id: doc_text} 的字典。
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = item["text"]
    return corpus

# 读取 nq-contriever.json 并提取每个查询的前 50 个文档
def load_contriever_results(contriever_file, top_k=50):
    """
    读取 nq-contriever.json 文件，返回 {query_id: [doc_id1, doc_id2, ...]} 的字典。
    只保留前 top_k 个文档。
    """
    with open(contriever_file, 'r', encoding='utf-8') as f:
        contriever_data = json.load(f)
    
    top_docs = {}
    for query_id, docs in contriever_data.items():
        # 根据分数排序并取前 top_k
        sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
        top_docs[query_id] = [doc_id for doc_id, score in sorted_docs[:top_k]]
    return top_docs

# 计算余弦相似度
def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度。
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    return np.dot(b_norm, a_norm)

# 主函数
def main():
    # 文件路径
    contriever_file = "./Datasets/nq/nq-contriever.json"
    query_file = "./Datasets/nq/query.jsonl"
    corpus_file = "corpus.jsonl"
    output_file = "nq-openai.json"
    
    print("加载 Contriever 结果...")
    contriever_results = load_contriever_results(contriever_file, top_k=50)
    
    print("加载查询数据...")
    queries = load_queries(query_file)
    
    print("加载文档数据...")
    corpus = load_corpus(corpus_file)
    
    # 准备保存最终结果
    final_results = {}
    
    print("开始处理每个查询...")
    for query_id, doc_ids in tqdm(contriever_results.items(), desc="处理查询"):
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"警告: 查询 {query_id} 未找到对应的文本。")
            continue
        
        # 获取对应的文档文本
        docs_text = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            doc_text = corpus.get(doc_id, "")
            if doc_text:
                docs_text.append(doc_text)
                valid_doc_ids.append(doc_id)
            else:
                print(f"警告: 文档 {doc_id} 未找到对应的文本。")
        
        if not docs_text:
            print(f"警告: 查询 {query_id} 没有找到有效的文档。")
            continue
        
        # 计算查询和文档的嵌入
        texts = [query_text] + docs_text
        embeddings = get_embeddings(texts)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        # 根据相似度重新排序文档
        sorted_indices = np.argsort(-similarities)  # 从高到低排序
        sorted_doc_ids = [valid_doc_ids[idx] for idx in sorted_indices]
        
        # 保存结果
        final_results[query_id] = sorted_doc_ids
    
    # 保存最终结果到 JSON 文件
    print("保存最终的排序结果到文件...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print("完成！")

if __name__ == "__main__":
    main()
