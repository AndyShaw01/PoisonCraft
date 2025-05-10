from openai import OpenAI

class DeepSeekModel:
    def __init__(self, config):
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please provide a valid API key"

        self.api_key = api_keys[api_pos]
        self.model_name = config["model_info"]["name"]
        self.max_tokens = int(config["params"].get("max_tokens", 4096))  # 官方默认 4K
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            message = completion.choices[0].message
            reasoning = getattr(message, "reasoning_content", "")
            content = message.content
            response = f"[Reasoning]: {reasoning}\n[Answer]: {content}"
        except Exception as e:
            print(f"Error querying DeepSeek: {e}")
            response = ""
        return response