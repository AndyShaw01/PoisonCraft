import random
import os
import torch
import numpy as np


class BaseModel:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.seed = int(config["params"]["seed"])
        self.temperature = float(config["params"]["temperature"])
        self.gpus = [str(gpu) for gpu in config["params"]["gpus"]]
        self.initialize_seed()
        if self.gpus:
            self.initialize_gpus()

    def initialize_seed(self):
        """Set the random seed for reproducibility."""
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if len(self.gpus) > 1:
            torch.cuda.manual_seed_all(self.seed)

    def initialize_gpus(self):
        """Set CUDA visible devices."""
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpus)

    def print_model_info(self):
        """Print the model information."""
        separator = '-' * len(f"| Model name: {self.name}")
        print(f"{separator}\n| Provider: {self.provider}\n| Model name: {self.name}\n{separator}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Method 'set_API_key' is not implemented")

    def query(self, msg):
        raise NotImplementedError("ERROR: Method 'query' is not implemented")