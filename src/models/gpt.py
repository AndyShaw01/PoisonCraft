from openai import OpenAI
from .base_model import BaseModel


class GPTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please provide a valid API key"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos])

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"Error querying GPT: {e}")
            response = ""
        return response