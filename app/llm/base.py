class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError