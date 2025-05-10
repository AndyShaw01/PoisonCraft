import os
import json
import csv
import subprocess
import re
import torch
import pandas as pd
from src.models.gpt import GPTModel,DeepSeekModel

# Predefined prompt template
RAG_PROMPT_TEMPLATE = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. "
    "\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:"
)

# Mapping of model codes to model paths
MODEL_CODE_TO_MODEL_NAME = {
    "contriever": "facebook/contriever",
    "simcse": "princeton-nlp/unsup-simcse-bert-base-uncased",
}

MODEL_PATH_TO_MODEL_CONFIG = {
    "gpt4o-mini": "./src/models/gpt4o_mini_config.json",
    "deepseek-reasoner": "./src/models/deepseek_r1_config.json"
}

TEST_DATASETS = {"nq":2762, "hotpotqa":5924}

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
    return input_prompt.replace('asdasd', url)

# Model creation functions
def create_model(model_path):
    """
    Factory method to create a LLM instance
    """
    # map the model_path to the model path
    config_path = MODEL_PATH_TO_MODEL_CONFIG[model_path]
    config = load_json(config_path)
    if model_path == "gpt4o-mini":
        model = GPTModel(config)
    elif model_path == "deepseek-reasoner":
        model = DeepSeekModel(config)
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
    query_norm = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True) 
    doc_norm = doc_embeddings / doc_embeddings.norm(dim=-1, keepdim=True)       

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

def load_frequent_words_as_str(result_file, model_code=None):
    """
    Load the frequent words from a JSON file and convert them to a single space-separated string.
    If the file does not exist, generate it using the provided model code.

    Args:
        result_file (str): Path to the JSON file containing frequent words.
        model_code (str): Code to specify the model used to generate frequent words.

    Returns:
        str: A space-separated string of frequent words.
    """
    # Check if the result file exists
    if not os.path.exists(result_file):
        if model_code is None:
            raise ValueError("The file does not exist and model_code is not provided to generate it.")
        
        # Prepare the command to generate the file
        cmd = f"python ./experiments/get_frequent_words.py --model_code {model_code}"
        print(f"File not found. Running command to generate it:\n{cmd}")
        
        # Execute the command
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(f"Failed to generate the file using command: {cmd}")
    
    # Load the frequent words from the JSON file
    try:
        with open(result_file, 'r') as file:
            data = json.load(file)
    except Exception as e:
        raise RuntimeError(f"Failed to load the JSON file: {e}")
    
    # Convert frequent words to a space-separated string
    frequent_words = data.get("frequent_words", [])
    return " ".join(frequent_words)

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
    exp_list = ['batch-4-stage1', 'batch-4-stage2'] 
    for domain in domain_list:
        for control_str_len in control_str_len_list:
            for exp in exp_list:
                candidate_file = f'./results/{retriever}/{dataset}/{exp}/domain_{domain}/combined_results_{control_str_len}.csv'
                try:
                    df = pd.read_csv(candidate_file)
                except:
                    continue
                attack_suffix = [attack_info + ' ' + x for x in df['adv_suffix'].tolist()]
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
        batch_embeddings = embedding_model(texts[i:i+batch_size])   
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
    corpus = load_jsonl_to_dict(f"./datasets/{args.eval_dataset}/corpus.jsonl", key_field="_id")
    if args.eval_dataset == 'msmarco':
        qrels = load_tsv_to_dict(f"./datasets/{args.eval_dataset}/qrels/dev.tsv", key_field="query-id")
    elif args.eval_dataset == 'nq':
        qrels = load_tsv_to_dict(f"./datasets/{args.eval_dataset}/qrels/ground_truth.tsv", key_field="query-id")
    elif args.eval_dataset == 'hotpotqa':
        qrels = load_tsv_to_dict(f"./datasets/{args.eval_dataset}/qrels/test.tsv", key_field="query-id")
    else:
        raise ValueError(f"Invalid eval_dataset: {args.eval_dataset}")
    return queries, corpus, qrels
