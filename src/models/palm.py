import google.generativeai as palm
import time
from .base_model import BaseModel


class PaLM2Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        self.api_key = None if api_pos == -1 else self.api_keys[api_pos]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.key_id = 0
        if self.api_key:
            self.set_API_key()

    def set_API_key(self):
        palm.configure(api_key=self.api_key)

    def query(self, msg):
        if not self.api_key:
            palm.configure(api_key=self.api_keys[self.key_id % len(self.api_keys)])
            self.key_id += 1

        try:
            completion = palm.generate_text(
                model=self.name,
                prompt=msg,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens
            )
            response = completion.result
        except Exception as e:
            print(f"Error querying PaLM2: {e}")
            response = "Input may contain harmful content and was blocked by PaLM."
        return response