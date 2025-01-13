# PoisonCraft

This repository provides the official implementation of PosionCraft: Practical Poisoning of Retrieval-Augmented Generation for Large Language Models.

# Overview

POISONCRAFT aims to demonstrate how a malicious actor can plant “poisoned” content into the corpus used by Retrieval-Augmented Generation (RAG) pipelines, thereby misleading a Large Language Model into hallucinating or referencing malicious content. This codebase provides scripts to:

- Prepare and preprocess standard datasets (e.g., Natural Questions, MS MARCO, HotpotQA).

- Inject adversarial suffixes (i.e., “poisons”) into query-like text in order to degrade or manipulate retrieval results.

- Evaluate the poisoning effectiveness under different retrievers (e.g., Contriever, SimCSE, and BGE) and measure transferability.

![Framework](./images/framework.png)

# Quick Start
## Environment Setup

Below is a high-level guide to configuring the environment. The exact requirements are listed in requirements.txt. We recommend creating a new virtual environment to avoid conflicts:

```bash
# Example using conda
conda create -n poisoncraft python=3.10
conda activate poisoncraft

# Install dependencies
pip install -r requirements.txt
```

## How to Run the Code

We provide various bash scripts under the scripts folder for different stages of the pipeline. Below is a step-by-step reference:

### Preparing the Datasets

1.	Download & Preprocess

Run scripts/run_prepare_dataset.sh to download and unzip the datasets, and to create training/test splits:

```bash
bash scripts/run_prepare_dataset.sh
```

This script internally calls:
- experiments/prepare_datasets.py
- experiments/classify_queries_by_domain.py (for domain classification, if needed)

You can find the 

2. Further Preprocessing

If you want to process data (for instance, calculating ground-truth top-k similarity scores for each domain), run:

```bash
bash scripts/run_process_data.sh
```
This script calls experiments/process_data.py.

### Poisoning Attack Training

To generate adversarial suffixes (i.e., “poisons”), you can use:
```bash
bash scripts/run_multi_poisoncraft.sh
```

- This script will launch multiple runs of run_poisoncraft.sh in parallel, each exploring different adversarial suffix lengths and domain indexes.
- The core logic calls experiments/poisoncraft_exp.py 

You may customize parameters such as:
- DOMAIN_LIST (the domain indexes of the queries you want to poison)
- ADV_LENGTH_LIST (the adversarial suffix lengths to test)

### Running the Retrieval Attack Evaluation

After generating adversarial suffixes, you can evaluate how many queries get “poisoned” references by running:
```bash
bash scripts/run_retrieval_attack.sh
```

This script:
- Loads the pre-computed adversarial suffixes.
- Runs an attack test using experiments/retrieval_attack.py.
- Logs the results of how many queries ended up retrieving the poisoned text.

### Running the Targeted Generation Attack Evaluation

Alternatively, for LLM-level evaluation (i.e., final text generation with top-k retrieval results included), you can use:
```bash
bash scripts/run_target_attack.sh
```

# Acknowledgement

- The model part of our code is from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
- Our code used the implementation of [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)
- Our code used [beir](https://github.com/beir-cellar/beir) benchmark.
- Our code used [contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG).
- Our code used the implementation of [llms-attacks](https://github.com/llm-attacks/llm-attacks)

<!-- # Citation -->
