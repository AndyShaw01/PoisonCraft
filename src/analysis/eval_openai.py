import json
from tqdm import tqdm
import torch
import openai
import time

# Set OpenAI API key    
openai.api_key = 'sk-proj-'  

# Load Contriever results
def load_contriever_results(file_path, top_k=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for query_id, doc_scores in data.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[query_id] = [doc_id for doc_id, _ in sorted_docs]
    return results

# Load corpus
def load_corpus(file_path):
    corpus = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = item["title"] + " " + item["text"]
    return corpus

# Load queries
def load_queries(file_path):
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
    return queries

# Get embeddings
def get_embeddings(texts, model="text-embedding-3-large", batch_size=64, retry=3, delay=5):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="获取嵌入"):
        batch_texts = texts[i:i + batch_size]
        for attempt in range(retry):
            try:
                response = openai.Embedding.create(
                    input=batch_texts,
                    model=model
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                break  
            except openai.error.RateLimitError:
                print("Rare limit exceeded. Waiting for a while before retrying...")
                time.sleep(delay)
            except openai.error.APIError as e:
                print(f"OpenAI API error: {e}. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            except Exception as e:
                print(f"Other error: {e}. Skipping the current batch...")   
                embeddings.extend([None] * len(batch_texts))  
                break
        else:
            print("Multiple attempts failed. Skipping the current batch...")
            embeddings.extend([None] * len(batch_texts))
    # transform embeddings to tensor
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    return torch.tensor(valid_embeddings)

def cosine_similarity(query_embedding, doc_embeddings):
    """
    Calculate the cosine similarity between the query embedding and document embeddings.

    Args:
        query_embedding (torch.Tensor): The embedding of the query, with shape (D,)
        doc_embeddings (torch.Tensor): The embeddings of the documents, with shape (N, D)

    Returns:
        torch.Tensor: The similarity between the query and documents, with shape (N,)
    """

    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding

    query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    doc_norms = doc_embeddings / doc_embeddings.norm(dim=-1, keepdim=True)

    return torch.mm(doc_norms, query_norm.t()).squeeze(1)

def main():
    contriever_file = "./Datasets/nq/nq-contriever-msmarco.json"
    corpus_file = "./Datasets/nq/corpus.jsonl"
    query_file = "./Datasets/nq/test_queries.jsonl"
    output_file = "./Datasets/nq/nq-openai_3-large.json"
    
    # Load data
    print("Loading Contriever results...")
    contriever_results = load_contriever_results(contriever_file, top_k=100)
    print("Loading corpus...")
    corpus = load_corpus(corpus_file)
    print("Loading queries...")
    queries = load_queries(query_file)
    
    final_results = {}
    
    for query_id, doc_ids in tqdm(contriever_results.items(), desc="处理查询"):
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"Waring: Query {query_id} not found.")
            continue
        
        doc_texts = [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus]
        if not doc_texts:
            print(f"Waring: Query {query_id} has no valid documents.")
            continue
        
        all_texts = [query_text] + doc_texts
        embeddings = get_embeddings(all_texts)
        
        if len(embeddings) != len(all_texts):
            print(f"Warning: The number of embeddings for query {query_id} does not match the number of texts. Skipping this query.")
            continue
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        query_embedding = query_embedding.float()
        doc_embeddings = doc_embeddings.float()
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        sorted_doc_scores = {doc_ids[i]: similarities[i].item() for i in sorted_indices}
        
        final_results[query_id] = sorted_doc_scores
    
    print("Saving the final ranking results...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()