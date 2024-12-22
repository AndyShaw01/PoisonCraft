import json
import openai
import numpy as np
from tqdm import tqdm
import pdb

# 设置 OpenAI API 密钥
openai.api_key = "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"  # 请替换为您的实际 API 密钥

# 定义获取嵌入的函数
def get_embeddings(texts, model="text-embedding-ada-002"):
    """
    批量获取文本的嵌入。
    """
    embeddings = []
    batch_size = 61  # 根据 OpenAI 的限制调整批量大小
    for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入"):
        batch = texts[i:i + batch_size]
        try:
            response = openai.Embedding.create(input=batch, model=model)
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        except openai.error.InvalidRequestError as e:
            print(f"InvalidRequestError occurred at batch {i//batch_size + 1}: {e}")
            print(f"Invalid input batch: {batch}")
            batch_embeddings = np.zeros((len(batch), 1536))  # 1536 是默认的嵌入维度
            embeddings.extend(batch_embeddings)
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
    
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
def load_contriever_results(contriever_file, top_k=60):
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
        top_docs[query_id] = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    return top_docs

# 计算文档与查询之间的相似度
def rank_docs_with_openai(query_id, query_text, doc_ids, corpus, model="text-embedding-ada-002"):
    """
    使用 OpenAI 模型计算每个文档与查询的相似度，并对文档进行排序。
    """
    
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    # pdb.set_trace()
    texts = [query_text] + doc_texts
    embeddings = get_embeddings(texts, model=model)
    
    # 获取查询和文档的嵌入
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]
    
    # 计算相似度 (余弦相似度)
    similarities = np.dot(query_embedding, np.array(doc_embeddings).T)
    similarities = similarities.flatten()
        # 获取每个查询与所有文档之间的相似度，选择最大相似度
    # similarity_scores = np.max(similarities, axis=1)  # 假设选择每个查询的最大相似度
    # pdb.set_trace()
    # 将文档ID和相似度分数配对
    # ranked_docs = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
    ranked_docs = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
    # ranked_docs = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
    
    return ranked_docs

# 主函数：重新排序并保存结果
def rerank_and_save(contriever_file, corpus_file, query_file, output_file):
    """
    重新排序 nq-contriever.json 文件中的文档，生成基于 OpenAI 模型的排序结果并保存为 output_file。
    """
    # 1. 加载数据
    queries = load_queries(query_file)
    corpus = load_corpus(corpus_file)
    contriever_results = load_contriever_results(contriever_file)
    
    # 2. 为每个查询重新排序
    reranked_results = {}
    for query_id, doc_ids in contriever_results.items():
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"警告: 查询 {query_id} 未找到对应的文本。")
            continue
        
        reranked_docs = rank_docs_with_openai(query_id, query_text, doc_ids, corpus)
        
        # 保存新的排序结果
        reranked_results[query_id] = {doc_id: score for doc_id, score in reranked_docs}
    
    # 3. 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reranked_results, f, ensure_ascii=False, indent=4)

# 调用主函数，重新排序并保存结果
rerank_and_save('./Datasets/nq/nq-contriever.json', './Datasets/nq/corpus.jsonl', './Datasets/nq/test_queries.jsonl', './Datasets/nq/nq-openai.json')
