import json
import csv
import re
import torch
import pandas as pd
from src.models.gpt import GPTModel

# Predefined prompt template
RAG_PROMPT_TEMPLATE = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. "
    "\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:"
)

# Mapping of model codes to model paths
MODEL_CODE_TO_MODEL_NAME = {
    "contriever": "/data1/shaoyangguang/offline_model/contriever",
    "contriever-msmarco": "/data1/shaoyangguang/offline_model/contriever-msmarco",
    "dpr-single": "/data1/shaoyangguang/offline_model/dpr-question_encoder-single-nq-base",
    "dpr-multi": "/data1/shaoyangguang/offline_model/dpr-question_encoder-multiset-base",
    "ance": "/data1/shaoyangguang/offline_model/ance",
    "simcse": "/data1/shaoyangguang/offline_model/simcse",
    "bge-small": "/data1/shaoyangguang/offline_model/bge-small-en-v1.5",
    "bge-unsp": "/data1/shaoyangguang/offline_model/bge-m3-unsupervised",
}

# MODEL_CODE_TO_CMODEL_NAME = MODEL_CODE_TO_QMODEL_NAME.copy()

# File processing functions
def load_json(file_path):
    """
    Load and parse a JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def load_jsonl_to_json(jsonl_file):
    """
    Load a JSONL file and convert it to a list of JSON objects.
    """
    with open(jsonl_file, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def load_jsonl_to_dict(jsonl_file, key_field='id'):
    """
    Load a JSONL file and convert it to a dictionary using a specified key field.
    """
    json_data = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            json_data[json_obj[key_field]] = json_obj
    return json_data

def load_tsv_to_dict(tsv_file, key_field='id'):
    """
    Load a TSV file and convert it to a dictionary using a specified key field.
    """
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return {row[key_field]: row for row in reader}

# Predictor base class
class Predictor:
    """
    Base class for implementing predictors.
    """
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor must implement the predict method.")

class MatchPredictor(Predictor):
    """
    Predictor that checks if a sequence matches a specified target.
    """
    def __init__(self, match_target):
        super().__init__(path=None)
        self.match_target = match_target

    def predict(self, sequences):
        match_regex = re.compile(re.escape(self.match_target), re.IGNORECASE)
        results = int(bool(match_regex.search(sequences)))
        return results

# Prompt wrapping functions
def wrap_prompt(question, context, prompt_id=1) -> str:
    """
    Wrap a question and context into a prompt string.
    """
    if prompt_id == 4:
        assert isinstance(context, list), "Context must be a list for prompt_id=4."
        context_str = "\n".join(context)
    else:
        context_str = context
    return RAG_PROMPT_TEMPLATE.replace('[question]', question).replace('[context]', context_str)

def wrap_prompt_url(question, context, url) -> str:
    """
    Wrap a question, context, and URL into a prompt string.
    """
    assert isinstance(context, list), "Context must be a list."
    context_str = "\n".join(context)
    input_prompt = RAG_PROMPT_TEMPLATE.replace('[question]', question).replace('[context]', context_str)
    # 'asdasd' placeholder was replaced with the URL in the original code
    return input_prompt.replace('asdasd', url)

# Model creation functions
def create_model(model_path):
    """
    Factory method to create a LLM instance
    """
    api_key = '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'
    model = GPTModel(api_key=api_key, model_path=model_path)
    return model

# Model utility functions
def cosine_similarity(query_embeddings, doc_embeddings):
    """ 
    Calculate the cosine similarity between multiple queries and multiple documents.

    Args:
        query_embeddings (torch.Tensor): Query embeddings, shape (x, d)
        doc_embeddings (torch.Tensor): Document embeddings, shape (n, d)

    Returns:
        torch.Tensor: Cosine similarity between queries and documents, shape (x, n)
    """
    # Normalize the query and document embeddings
    query_norm = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)  # (x, d)
    doc_norm = doc_embeddings / doc_embeddings.norm(dim=-1, keepdim=True)       # (n, d)

    # Calculate the cosine similarity
    # query_norm: (x, d)
    # doc_norm.t(): (d, n)
    # similarity: (x, n)
    similarity = torch.mm(query_norm, doc_norm.t())
    return similarity

def save_top_words_to_json(frequency, result_file, top_n=10):
    """
    Save the top N frequent words to a JSON file in the desired format.
    """
    top_words = [word for word, _ in frequency.most_common(top_n)]
    result = {"frequent_words": top_words}

    with open(result_file, 'w') as file:
        json.dump(result, file, indent=4)

def load_frequent_words_as_str(result_file):
    """
    Load the frequent words from a JSON file and convert them to a single space-separated string.
    """
    with open(result_file, 'r') as file:
        data = json.load(file)
    frequent_words = data.get("frequent_words", [])
    return " ".join(frequent_words)

def get_poisoned_info_for_baseline_pj(dataset):
    """
    Load injected queries for prompt injection baseline.

    Args:
        dataset (str): The dataset name.

    Returns:
        list: List of injected queries.
    """
    file_path = f'./Main_Results/baseline/prompt_injection/{dataset}/shadow_queries.csv'
    df = pd.read_csv(file_path)
    return df['injected_query'].tolist()

def get_poisoned_info_for_baseline_pr(file_path):
    """
    Load and merge contexts for poisonedRAG baseline.

    Args:
        file_path (str): Path to the CSV file containing queries and contexts.

    Returns:
        list: List of merged query and context strings.
    """
    df = pd.read_csv(file_path)
    suffix_db = []
    for i in range(len(df)):
        query = df['query'][i]
        context1 = df['context1'][i]
        context2 = df['context2'][i]
        context3 = df['context3'][i]
        context4 = df['context4'][i]
        context5 = df['context5'][i]
        suffix_db.append(query + ' ' + context1)
        suffix_db.append(query + ' ' + context2)
        suffix_db.append(query + ' ' + context3)
        suffix_db.append(query + ' ' + context4)
        suffix_db.append(query + ' ' + context5)
    return suffix_db

def get_poisoned_info_for_main_result(domain_list, control_str_len_list, attack_info, retriever, dataset):
    """
    Load injected queries for the main results.

    Args:
        domain_list (list): List of domain names.
        control_str_len_list (list): List of control string lengths.
        attack_info (str): Attack information.
        retriever (str): The retriever type.
        dataset (str): The dataset name.

    Returns:
        dict, list: Dictionary of suffixes for each control string length, and a list of all suffixes.
    """
    suffix_all = {}
    all_list = []
    if retriever == 'contriever' and dataset == 'nq':
        exp_list = ['improve_gcg_test']
    else:
        exp_list = ['batch-4-stage1', 'batch-4-stage2'] 
    for domain in domain_list:
        for control_str_len in control_str_len_list:
            for exp in exp_list:
                if retriever == 'contriever':
                    if dataset == 'nq':
                        candidate_file = f'./results_from_A800/part_results/Results/{exp}/batch-4/category_{domain}/results_{control_str_len}.csv'
                    elif dataset == 'hotpotqa':
                        candidate_file = f'./Main_Results/contriever/{dataset}_1126/{exp}/domain_{domain}/combined_results_{control_str_len}.csv'
                else:
                    candidate_file = f'./Main_Results/{retriever}/{dataset}/{exp}/domain_{domain}/combined_results_{control_str_len}.csv'
                try:
                    df = pd.read_csv(candidate_file)
                except:
                    continue
                attack_suffix = [attack_info + ' ' + x for x in df['control_suffix'].tolist()]
                suffix_all[control_str_len] = attack_suffix
                all_list += attack_suffix
    return all_list

def batch_process_embeddings(embedding_model, texts, batch_size=32):
    """
    Process embeddings in batches.

    Args:
        embedding_model (SentenceEmbeddingModel): The sentence embedding model.
        texts (list): List of texts to embed.
        batch_size (int): Batch size for processing.

    Returns:
        torch.Tensor: Concatenated embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model(batch)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

def load_beir_data(args):
    """
    Load queries, corpus, and qrels based on the dataset.

    Args:
        args (argparse.Namespace): The arguments for the script.

    Returns:
        tuple: Loaded queries, corpus, and qrels.
    """
    queries = load_jsonl_to_json(args.queries_folder + f"/domain_{args.target_category}.jsonl")
    corpus = load_jsonl_to_dict(f"./Datasets/{args.eval_dataset}/corpus.jsonl", key_field="_id")
    if args.eval_dataset == 'msmarco':
        qrels = load_tsv_to_dict(f"./Datasets/{args.eval_dataset}/qrels/dev.tsv", key_field="query-id")
    elif args.eval_dataset == 'nq':
        qrels = load_tsv_to_dict(f"./Datasets/{args.eval_dataset}/qrels/ground_truth.tsv", key_field="query-id")
    elif args.eval_dataset == 'hotpotqa':
        qrels = load_tsv_to_dict(f"./Datasets/{args.eval_dataset}/qrels/test.tsv", key_field="query-id")
    else:
        raise ValueError(f"Invalid eval_dataset: {args.eval_dataset}")
    return queries, corpus, qrels
