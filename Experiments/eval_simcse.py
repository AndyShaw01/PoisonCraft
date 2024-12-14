import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# 加载 SimCSE 模型
def load_simcse_model(model_path="/data1/shaoyangguang/offline_model/simcse"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda:1" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

# 加载 Contriever 结果
def load_contriever_results(file_path, top_k=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for query_id, doc_scores in data.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[query_id] = [doc_id for doc_id, _ in sorted_docs]
    return results

# 加载 Corpus
def load_corpus(file_path):
    corpus = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = item["title"] + " " + item["text"]
    return corpus

# 加载 Query
def load_queries(file_path):
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
    return queries

# 获取 SimCSE 嵌入
def get_embeddings(texts, tokenizer, model, batch_size=16):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded_input)
            batch_embeddings = outputs.pooler_output  # 使用 SimCSE 的 pooler_output
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# 计算余弦相似度
def cosine_similarity(query_embedding, doc_embeddings):
    """
    计算查询嵌入和文档嵌入之间的余弦相似度。

    Args:
        query_embedding (torch.Tensor): 查询的嵌入，形状为 (D,)
        doc_embeddings (torch.Tensor): 文档的嵌入，形状为 (N, D)

    Returns:
        torch.Tensor: 查询和文档之间的相似度，形状为 (N,)
    """
    # 确保 query_embedding 是二维的 (1, D)
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding

    # 对查询和文档嵌入进行归一化
    query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    doc_norms = doc_embeddings / doc_embeddings.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    return torch.mm(doc_norms, query_norm.t()).squeeze(1)

# 主流程
def main():
    contriever_file = "./Datasets/nq/nq-contriever.json"
    corpus_file = "./Datasets/nq/corpus.jsonl"
    query_file = "./Datasets/nq/test_queries.jsonl"
    output_file = "./Datasets/nq/nq-simcse.json"
    
    # 加载数据
    print("加载 Contriever 结果...")
    contriever_results = load_contriever_results(contriever_file, top_k=100)
    print("加载文档数据...")
    corpus = load_corpus(corpus_file)
    print("加载查询数据...")
    queries = load_queries(query_file)
    
    print("加载 SimCSE 模型...")
    tokenizer, model = load_simcse_model()
    
    final_results = {}
    
    print("开始对每个查询进行处理...")
    for query_id, doc_ids in tqdm(contriever_results.items(), desc="处理查询"):
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"警告: 查询 {query_id} 未找到对应的文本。")
            continue
        
        # 获取文档文本
        doc_texts = [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus]
        if not doc_texts:
            print(f"警告: 查询 {query_id} 没有找到有效的文档。")
            continue
        
        # 获取嵌入
        all_texts = [query_text] + doc_texts
        embeddings = get_embeddings(all_texts, tokenizer, model)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        # 重新排序
        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        sorted_doc_scores = {doc_ids[i]: similarities[i].item() for i in sorted_indices}
        
        # 保存结果
        final_results[query_id] = sorted_doc_scores
    
    # 保存结果到文件
    print("保存最终的排序结果...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main()