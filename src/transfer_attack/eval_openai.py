import json
import openai
import numpy as np
from tqdm import tqdm
import pdb
# set the openai api key
openai.api_key = "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"  

def get_embeddings(texts, model="text-embedding-3-large"):
    """
    get the embeddings of the texts.
    """
    embeddings = []
    batch_size = 1000  
    for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入"):
        batch = texts[i:i+batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [item['embedding'] for item in response['data']]
        embeddings.extend(batch_embeddings)
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

def load_contriever_results(contriever_file, top_k=50):
    """
    read nq-contriever.json and extract top 50 docs for each query.
    """
    with open(contriever_file, 'r', encoding='utf-8') as f:
        contriever_data = json.load(f)
    
    top_docs = {}
    for query_id, docs in contriever_data.items():
        sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
        top_docs[query_id] = [doc_id for doc_id, score in sorted_docs[:top_k]]
    return top_docs

def cosine_similarity(a, b):
    """
    calculate the cosine similarity between two vectors.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    return np.dot(b_norm, a_norm)

def main():
    contriever_file = "./Datasets/nq/nq-contriever.json"
    query_file = "./Datasets/nq/test_openai.jsonl"
    corpus_file = "./Datasets/nq/corpus.jsonl"
    output_file = "./Datasets/nq/nq-openai-large.json"
    
    print("Loading contriever results...")
    contriever_results = load_contriever_results(contriever_file, top_k=50)
    
    print("Loading queries...")
    queries = load_queries(query_file)
    
    print("Loading corpus...")
    corpus = load_corpus(corpus_file)
    
    final_results = {}
    
    print("Calculating similarities and ranking documents...")
    for query_id, doc_ids in tqdm(contriever_results.items(), desc="process queries"):
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"Warning: Query {query_id} not found")
            continue
        
        docs_text = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            doc_text = corpus.get(doc_id, "")
            if doc_text:
                docs_text.append(doc_text)
                valid_doc_ids.append(doc_id)
            else:
                print(f"Warning: Document {doc_id} not found")
        
        if not docs_text:
            print(f"Warning: Query {query_id} has no valid documents")
            continue
        texts = [query_text] + docs_text
        embeddings = get_embeddings(texts)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        sorted_indices = np.argsort(-similarities)  
        sorted_doc_ids = [valid_doc_ids[idx] for idx in sorted_indices]
        
        final_results[query_id] = sorted_doc_ids
    
    print("Saving the final results to file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()
