from openai import OpenAI
import logging
import time


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")
    

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 temperature=0.1,
                 max_output_tokens=512
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
    
    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_path,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response