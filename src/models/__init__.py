from .palm import PaLM2Model
from .gpt import GPTModel
from .llama import LlamaModel
from .qwen import QwenModel
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        model = PaLM2Model(config)
    elif provider == 'gpt':
        model = GPTModel(config)
    elif provider == 'llama':
        model = LlamaModel(config)
    elif provider == 'qwen':
        model = QwenModel(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model