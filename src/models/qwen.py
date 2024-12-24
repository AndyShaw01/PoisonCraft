import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel


class QwenModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.device = config["params"]["device"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        # Load Hugging Face token and model
        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, use_auth_token=hf_token, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.float16, use_auth_token=hf_token, trust_remote_code=True
        ).to(self.device)

    def query(self, msg):
        """
        Generate a response for the given message.

        Args:
            msg (str): The input message.

        Returns:
            str: The generated response.
        """
        # Tokenize input
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)

        # Generate output
        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True
        )
        
        # Decode and return the generated response
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg):]  # Remove input message from output
        return result