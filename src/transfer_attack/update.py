import json
import openai
import numpy as np
from tqdm import tqdm
import pdb

# set the openai api key
openai.api_key = "sk-proj-"  

def get_embeddings(texts, model="text-embedding-ada-002"):
    """
    get the embeddings of the texts.
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

def load_queries(query_file):
    """
    read query.jsonl and return {query_id: query_text} dict.    
    """
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
    return queries

def load_corpus(corpus_file):
    """
    read corpus.jsonl and return {doc_id: doc_text} dict.
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = item["text"]
    return corpus

def load_contriever_results(contriever_file, top_k=60):
    """
    read nq-contriever.json and extract top 60 docs for each query.
    """
    with open(contriever_file, 'r', encoding='utf-8') as f:
        contriever_data = json.load(f)
    
    top_docs = {}
    for query_id, docs in contriever_data.items():
        sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
        top_docs[query_id] = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    return top_docs

def rank_docs_with_openai(query_id, query_text, doc_ids, corpus, model="text-embedding-ada-002"):
    """
    rank the documents by the similarity between the query and the documents.
    """
    
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    texts = [query_text] + doc_texts
    embeddings = get_embeddings(texts, model=model)
    
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]
    
    similarities = np.dot(query_embedding, np.array(doc_embeddings).T)
    similarities = similarities.flatten()
    
    ranked_docs = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
    
    return ranked_docs

def rerank_and_save(contriever_file, corpus_file, query_file, output_file):
    """
    rerank the documents and save the results to the output file.
    """
    # 1. load data
    queries = load_queries(query_file)
    corpus = load_corpus(corpus_file)
    contriever_results = load_contriever_results(contriever_file)
    
    # 2. rerank the documents
    reranked_results = {}
    for query_id, doc_ids in contriever_results.items():
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"Warning: No text found for query {query_id}.")
            continue
        
        reranked_docs = rank_docs_with_openai(query_id, query_text, doc_ids, corpus)
        
        reranked_results[query_id] = {doc_id: score for doc_id, score in reranked_docs}
    
    # 3. save the reranked results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reranked_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    rerank_and_save('./Datasets/nq/nq-contriever.json', './Datasets/nq/corpus.jsonl', './Datasets/nq/test_queries.jsonl', './Datasets/nq/nq-openai.json')
