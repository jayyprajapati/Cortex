from ollama import Client
from app.llm.base import BaseLLM


class OllamaCloudLLM(BaseLLM):
    def __init__(self, api_key, model="gpt-oss:120b"):
        self.client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]