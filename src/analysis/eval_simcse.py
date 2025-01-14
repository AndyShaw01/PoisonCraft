import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def load_simcse_model(model_path="/data1/shaoyangguang/offline_model/simcse"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to("cuda:2" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def load_contriever_results(file_path, top_k=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for query_id, doc_scores in data.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[query_id] = [doc_id for doc_id, _ in sorted_docs]
    return results

def load_corpus(file_path):
    corpus = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = item["title"] + " " + item["text"]
    return corpus

def load_queries(file_path):
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]
    return queries

def get_embeddings(texts, tokenizer, model, batch_size=64):
    """
    Get the embeddings of the texts.

    Args:
        texts (List[str]): list of texts
        tokenizer (transformers.PreTrainedTokenizer): tokenizer
        model (transformers.PreTrainedModel): model
        batch_size (int): batch size

    Returns:
        torch.Tensor: embeddings, shape (N, D)
    """
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded_input)
            batch_embeddings = outputs.pooler_output  
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

def cosine_similarity(query_embedding, doc_embeddings):
    """
    Calculate the cosine similarity between the query embedding and document embeddings.

    Args:
        query_embedding (torch.Tensor): The embedding of the query, with shape (D,)
        doc_embeddings (torch.Tensor): The embeddings of the documents, with shape (N, D)
    
    Return:
        torch.Tensor: The similarity between the query and the documents, with shape (N,)
    """

    # Ensure query_embedding is 2D (1, D)
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding

    # Normalize the query and document embeddings
    query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    doc_norms = doc_embeddings / doc_embeddings.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity
    return torch.mm(doc_norms, query_norm.t()).squeeze(1)

def main():
    contriever_file = "./Datasets/hotpotqa/hotpotqa-contriever.json"
    corpus_file = "./Datasets/hotpotqa/corpus.jsonl"
    query_file = "./Datasets/hotpotqa/test_queries.jsonl"
    output_file = "./Datasets/hotpotqa/hotpotqa-simcse.json"
    
    print("Loading Contriever results...")
    contriever_results = load_contriever_results(contriever_file, top_k=100)
    print("Loading corpus...")
    corpus = load_corpus(corpus_file)
    print("Loading queries...")
    queries = load_queries(query_file)
    
    print("Loading SimCSE model...")
    tokenizer, model = load_simcse_model()
    
    final_results = {}
    
    print("Ranking documents for each query...")
    for query_id, doc_ids in tqdm(contriever_results.items(), desc="Processing queries"):
        query_text = queries.get(query_id, "")
        if not query_text:
            print(f"Warning: Query {query_id} not found.")
            continue
        
        doc_texts = [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus]
        if not doc_texts:
            print(f"Warning: Query {query_id} has no valid documents.")
            continue
        
        # Get embeddings for the query and documents
        all_texts = [query_text] + doc_texts
        embeddings = get_embeddings(all_texts, tokenizer, model)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        # Sort the similarities and save the results
        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        sorted_doc_scores = {doc_ids[i]: similarities[i].item() for i in sorted_indices}
        
        final_results[query_id] = sorted_doc_scores
    
    print("Saving the final ranking results...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    main()