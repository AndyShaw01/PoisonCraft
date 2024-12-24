import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from .base_model import BaseModel


class LlamaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.device = config["params"]["device"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        self.tokenizer = LlamaTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        self.model = LlamaForCausalLM.from_pretrained(
            self.name, torch_dtype=torch.float16, use_auth_token=hf_token
        ).to(self.device)

    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True
        )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return out[len(msg):]