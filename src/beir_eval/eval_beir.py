import argparse
import json
import logging
import os
from typing import Dict

import torch
import transformers
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.models import DPR
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel, AutoTokenizer
from contriever_src.contriever import Contriever
from contriever_src.beir_utils import DenseEncoderModel

# Mapping model codes to model paths
MODEL_CODE_TO_MODEL_NAME = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "Luyu/ANCE-MSMarco",
    "simcse": "princeton-nlp/unsup-simcse-bert-base-uncased",
    "bge-small": "BAAI/bge-small-en",
    "bge-unsp": "BAAI/bge-base-en-uncased"
}

import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch

class CustomDEModel:
    def __init__(self, model_path=None, **kwargs):
        """
        Initialize your custom dense retrieval model.
        Args:
            model_path (str): Path to your pretrained model.
            **kwargs: Additional parameters for customization.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.batch_size = kwargs.get("batch_size", 128)
        self.max_length = kwargs.get("max_length", 128)
        self.normalize_embeddings = kwargs.get("normalize_embeddings", True)
        self.sentence_embedding_mode = kwargs.get("sentence_embedding_mode", "cls") # cls, mean, max

    def encode_queries(self, queries: List[str], batch_size: int = None, **kwargs) -> np.ndarray:
        """
        Encode queries into dense embeddings.
        Args:
            queries (List[str]): List of queries to encode.
            batch_size (int): Batch size for encoding.
        Returns:
            np.ndarray: Query embeddings.
        """
        batch_size = batch_size or self.batch_size
        self.model.eval()
        query_embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_queries, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                if self.sentence_embedding_mode == "cls":
                    embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
                elif self.sentence_embedding_mode == "mean":
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif self.sentence_embedding_mode == "max":
                    embeddings = outputs.last_hidden_state.max(dim=1).values
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                query_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(query_embeddings)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = None, **kwargs) -> np.ndarray:
        """
        Encode documents into dense embeddings.
        Args:
            corpus (List[Dict[str, str]]): List of documents (each a dictionary with "title" and "text").
            batch_size (int): Batch size for encoding.
        Returns:
            np.ndarray: Document embeddings.
        """
        batch_size = batch_size or self.batch_size
        self.model.eval()
        doc_embeddings = []

        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                batch_corpus = corpus[i:i + batch_size]
                texts = [
                    (doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"]
                    for doc in batch_corpus
                ]
                inputs = self.tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embeddings
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                doc_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(doc_embeddings)

def setup() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BEIR Dense Retrieval Evaluation")
    parser.add_argument('--model_code', type=str, default="bge-small", help="Model code to use for retrieval")
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="nq", help="BEIR dataset to evaluate")
    parser.add_argument('--split', type=str, default='test', help="Dataset split (e.g., test, dev, train)")
    parser.add_argument('--result_output', type=str, default="results/beir_results/nq-simcse-cos.json", help="Output file for results")
    parser.add_argument('--gpu_id', type=int, default=2, help="GPU ID to use")
    parser.add_argument('--per_gpu_batch_size', type=int, default=128, help="Batch size per GPU/CPU for indexing")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length")

    args = parser.parse_args()
    args.result_output = f"./results/beir_results/{args.dataset}-{args.model_code}-{args.score_function}.json"
    if not os.path.exists("results/beir_results"):
        os.makedirs("results/beir_results")

    setup_logging()
    
    return args

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()]
    )

def compress_results(results: Dict) -> Dict:
    """Compress retrieval results to top-2000."""
    k_old = len(next(iter(results.values())))
    compressed_results = {}

    for query_id, scores in results.items():
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        compressed_results[query_id] = dict(sorted_scores[:2000])

    k_new = len(next(iter(compressed_results.values())))
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return compressed_results

def load_model(args) -> DRES:
    """Load the specified retrieval model."""
    logging.info("Loading model...")
    model_code = args.model_code

    if 'contriever' in model_code:
        encoder = Contriever.from_pretrained(MODEL_CODE_TO_MODEL_NAME[model_code]).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(MODEL_CODE_TO_MODEL_NAME[model_code])
        model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size)
    elif 'dpr' in model_code:
        model = DRES(
            DPR((MODEL_CODE_TO_MODEL_NAME[model_code], MODEL_CODE_TO_MODEL_NAME[model_code])),
            batch_size=args.per_gpu_batch_size,
            corpus_chunk_size=1000
        )
    elif 'ance' in model_code:
        model = DRES(models.SentenceBERT(MODEL_CODE_TO_MODEL_NAME[model_code]), batch_size=args.per_gpu_batch_size, max_length=128, normalize_embeddings=True)
    elif 'simcse' in model_code:
        model_path = MODEL_CODE_TO_MODEL_NAME[args.model_code]
        model = DRES(CustomDEModel(model_path, sentence_embedding_mode='mean'), batch_size=args.per_gpu_batch_size)
    elif 'bge' in model_code:
        model_path = MODEL_CODE_TO_MODEL_NAME[args.model_code]
        model = DRES(CustomDEModel(model_path, sentence_embedding_mode='cls'), batch_size=args.per_gpu_batch_size)
    else:
        raise NotImplementedError(f"Model code '{model_code}' is not implemented.")
    
    logging.info(f"Model loaded: {model.model}")
    return model

def main():
    """Main function for running retrieval and evaluation."""
    args = setup()
    
    logging.info(args)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = args.dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    logging.info(f"Dataset path: {data_path}")

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

    # Load model
    model = load_model(args)
    retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[args.top_k])

    # Perform retrieval
    results = retriever.retrieve(corpus, queries)

    # Compress and save results
    logging.info(f"Saving results to {args.result_output}")
    compressed_results = compress_results(results)
    with open(args.result_output, "w") as f:
        json.dump(compressed_results, f)

if __name__ == "__main__":
    main()