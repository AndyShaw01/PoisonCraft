import sys, os
import json
import random
import torch
import csv
import re
import numpy as np
import pandas as pd

from transformers import MPNetModel, T5EncoderModel, T5Tokenizer, MPNetTokenizer, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import defaultdict

from .llm import OpenAILLM

model_code_to_qmodel_name = {
    "contriever": "/data1/shaoyangguang/offline_model/contriever",
    "contriever-msmarco": "/data1/shaoyangguang/offline_model/contriever-msmarco",
    "dpr-single": "/data1/shaoyangguang/offline_model/dpr-question_encoder-single-nq-base",
    "dpr-multi": "/data1/shaoyangguang/offline_model/dpr-question_encoder-multiset-base",
    "ance": "/data1/shaoyangguang/offline_model/ance"
}

model_code_to_cmodel_name = {
    "contriever": "/data1/shaoyangguang/offline_model/contriever",
    "contriever-msmarco": "/data1/shaoyangguang/offline_model/contriever-msmarco",
    "dpr-single": "/data1/shaoyangguang/offline_model/dpr-question_encoder-single-nq-base",
    "dpr-multi": "/data1/shaoyangguang/offline_model/dpr-question_encoder-multiset-base",
    "ance": "/data1/shaoyangguang/offline_model/ance"
}


def contriever_get_emb(model, input):
    return model(**input)

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = AutoModel.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

def load_beir_datasets(dataset_name, split):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)    

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores

def create_model(model_path):
    """
    Factory method to create a LLM instance
    """
    api_key = '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'
    model = OpenAILLM(api_key=api_key, model_path=model_path)
    return model

def get_suffix_db(category_list, threshold_list, attack_info):
    suffix_db = {}
    suffix_all = {}
    all_list = []
    for category in category_list:
        suffix_db[category] = {}
        for threshold in threshold_list:
            candidate_file = f'./Results/improve_exp_0912/batch-4/category_{category}/results_top_{threshold}.csv'
            try:
                df = pd.read_csv(candidate_file)
            except:
                continue
            attack_suffix = [attack_info + ' ' + x for x in df['control_suffix'].tolist()]
            suffix_db[category][threshold] = attack_suffix
            suffix_all[category] = attack_suffix
            all_list += attack_suffix

    return suffix_db, suffix_all, all_list

def load_jsonl_to_json(jsonl_file):
    # Read JSONL file
    with open(jsonl_file, 'r') as f:
        jsonl_data = f.readlines()

    # Parse JSONL data and store as list
    json_data = []
    for line in jsonl_data:
        json_obj = json.loads(line.strip())
        json_data.append(json_obj)

    # Return the parsed JSON object (list of dictionaries)
    return json_data

def load_jsonl_to_dict(jsonl_file, key_field='id'):
    # Initialize an empty dictionary
    json_data = {}

    # Read JSONL file line by line
    with open(jsonl_file, 'r') as f:
        # Use map for direct parsing of each line
        for line in f:
            json_obj = json.loads(line.strip())
            
            # Directly populate the dictionary using key_field
            json_data[json_obj[key_field]] = json_obj

    # Return the dictionary
    return json_data

def load_tsv_to_dict(tsv_file, key_field='id'):
    # Initialize an empty dictionary
    tsv_dict = {}

    # Open the TSV file
    with open(tsv_file, 'r') as f:
        # Use csv.DictReader to parse the TSV file
        reader = csv.DictReader(f, delimiter='\t')

        # Use dictionary comprehension and direct assignment
        tsv_dict = {row[key_field]: row for row in reader}

    # Return the resulting dictionary
    return tsv_dict

class Predictor:
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor must implement predict method.")

class MatchPredictor(Predictor):
    def __init__(self, match_target):
        self.match_target = match_target

    def predict(self, sequences):
        MATCH_REGEX = re.compile(re.escape(self.match_target), re.IGNORECASE)
        # for sequence in sequences:
            # check if the sequence contains the match target
        results = int(bool(MATCH_REGEX.search(sequences)))
        return results
