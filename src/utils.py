import json
import csv
import re

MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

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

# file processing
def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

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

def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

def wrap_prompt_url(question, context, url) -> str:
    assert type(context) == list
    context_str = "\n".join(context)
    input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str).replace('asdasd', url)
    return input_prompt
