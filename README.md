# PoisonCraft

This repository provides the official implementation of **POISONCRAFT: Practical Poisoning of Retrieval-Augmented Generation for Large Language Models.**

# Overview

![Framework](./images/framework.png)

**POISONCRAFT** aims to demonstrate how a malicious actor can plant “poisoned” content into the corpus used by Retrieval-Augmented Generation (RAG) pipelines, thereby misleading a Large Language Model into hallucinating or referencing malicious content. This codebase provides scripts to:

- Prepare and preprocess standard datasets (e.g., Natural Questions, MS MARCO, HotpotQA).

- Inject adversarial suffixes (i.e., “poisons”) into query-like text in order to degrade or manipulate retrieval results.

- Evaluate the poisoning effectiveness under different retrievers (e.g., Contriever, SimCSE, and BGE) and measure transferability.



# Quick Usage
## Environment Setup

Below is a high-level guide to configuring the environment. The exact requirements are listed in requirements.txt. We recommend creating a new virtual environment to avoid conflicts:

```bash
# Example using conda
conda create -n PoisonCraft python=3.10
conda activate PoisonCraft

# Install dependencies
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Set API Key and Model Configuration

Certain parts of this project, such as domain classification and query categorization, rely on API calls (e.g., OpenAI GPT models). You need to configure the API key and model settings as follows:

1. Navigate to the src/models_configs/ directory.
2. Locate the appropriate configuration file (e.g., gpt3.5_config.py).
3. Update the api_keys field with your API key:
    ```json
    "api_key_info": {
        "api_keys": ["your_openai_api_key"],
        "api_key_use": 0
    }
    ```
Note: This project supports multiple LLM providers (e.g., OpenAI, Llama, and PaLM2). Ensure you select the appropriate configuration file for your model.
## How to Run the Code

We provide various bash scripts under the scripts folder for different stages of the pipeline. Below is a step-by-step reference:

### Preparing the Datasets

**1. Download and Prepare Datasets**

Run scripts/run_prepare_dataset.sh to download and unzip the datasets, and to create training/test splits:
```bash
bash scripts/run_prepare_dataset.sh
```

This script internally calls:
- experiments/prepare_datasets.py
- experiments/classify_queries_by_domain.py (for domain classification, if needed)

**2. Preprocess Datasets**

To calculate top-k ground truth similarity scores for each domain, run:
```bash
bash scripts/run_process_data.sh
```

**3. [Optional]Evaluating BEIR for Ground Truth**

Execute this step only if you want to evaluate retriever model, and it will take a while. You can follow these steps:

1. Run the BEIR evaluation script to compute retrieval results:
    ```bash
    python src/beir_eval/eval_beir.py --model_code bge-small --dataset nq --top_k 100
    ```
2. Use the retrieved results to generate ground truth scores:
    ```bash
    bash scripts/run_process_data.sh
    ```

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
### Results and Logging
- Logs are stored in the logs/ directory.
- Results (e.g., poisoned queries and attack success rates) are saved in the results/ directory.

# Acknowledgement

- [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection): The foundational model code is based on this repository.
- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG): Some implementations were adapted from this project.
- [Beir Benchmark](https://github.com/beir-cellar/beir): Used as a benchmark framework for evaluation.
- [Contriever](https://github.com/facebookresearch/contriever): Utilized for Retrieval-Augmented Generation (RAG) capabilities.
- [llms-attacks](https://github.com/llm-attacks/llm-attacks): Certain implementations for adversarial training were derived from this repository.

<!-- # Citation -->
