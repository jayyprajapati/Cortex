from openai import OpenAI
from app.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You answer using only provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content