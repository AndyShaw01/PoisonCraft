import os
import sys
import json
import argparse
import pdb
from collections import Counter

sys.path.append(os.path.abspath('../RAGInjection/'))
from src.embedding.sentence_embedding import SentenceEmbeddingModel
from src.utils import MODEL_CODE_TO_MODEL_NAME
from src.utils import save_top_words_to_json

def load_shadow_dataset(shadow_dataset_path):
    """
    Load a shadow dataset from the specified file path.
    Assumes the dataset is in JSONL format.
    """
    shadow_data = []
    if os.path.exists(shadow_dataset_path):
        with open(shadow_dataset_path, 'r') as file:
            for line in file:
                shadow_data.append(json.loads(line.strip()))
    else:
        raise FileNotFoundError(f"The file {shadow_dataset_path} does not exist.")
    return shadow_data

def tokenize_datasets(datasets, tokenizer):
    """
    Tokenize all text in the datasets using the specified tokenizer.
    """
    tokenized_data = []
    for data in datasets:
        if 'text' in data:
            tokenized_text = tokenizer.tokenize(data['text'])
            tokenized_data.extend(tokenized_text)
    return tokenized_data

def filter_stopwords(tokens, stopwords):
    """
    Filter out stopwords from the token list.
    """
    return [token for token in tokens if token.lower() not in stopwords]

def main(args):
    # Load the target tokenizer of the target model
    model_name = MODEL_CODE_TO_MODEL_NAME[args.model_code]
    tokenizer = SentenceEmbeddingModel(model_name).tokenizer

    # Load the shadow datasets
    shadow_data = load_shadow_dataset(args.shadow_dataset)
    
    # Transform the shadow datasets to the tokenized format
    tokenized_data = tokenize_datasets(shadow_data, tokenizer)

    # Calculate the frequency of the original query in the shadow datasets
    token_frequency = Counter(tokenized_data)

    # Set the filter stopwords
    stopwords = set(["the", "is", "and", "in", "to", "a", "of", "for", "on", "with", "by", "an", "it", "this", "s", "'", "did"])  # Add more stopwords as needed
    filtered_tokens = filter_stopwords(token_frequency.keys(), stopwords)
    filtered_frequency = Counter({token: token_frequency[token] for token in filtered_tokens})

    # Save the result to the result file
    # save_frequency_to_file(filtered_frequency, args.result_file)
    save_top_words_to_json(filtered_frequency, args.result_file, top_n=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", choices=['nq', 'hotpotqa', 'msmarco'], default='nq', help='The target dataset')
    parser.add_argument('--shadow_dataset', type=str, default='./datasets/nq/shadow_queries.jsonl', help='The folder of shadow datasets')
    parser.add_argument('--result_file', type=str, default='./datasets/nq/frequent_words.json', help='The result file of frequency of the original query in the shadow datasets')
    parser.add_argument('--model_code', choices=['contriever', 'simcse', 'ance', 'bge-small'], default='contriever', help='The target model code')

    args = parser.parse_args()

    args.shawdow_dataset = f'./datasets/{args.target_dataset}/shadow_queries.jsonl'
    args.result_file = f'./datasets/{args.target_dataset}/frequent_words_{args.model_code}.json'

    main(args)