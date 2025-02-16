import os
from openai import OpenAI
from groq import Groq
from openrouter import OpenRouter

class AIProvider:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_text(self, prompt, max_tokens=100):
        raise NotImplementedError("Subclasses must implement generate_text method")

class OpenAIProvider(AIProvider):
    def __init__(self):
        super().__init__(os.getenv("OPENAI_API_KEY"))
        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, prompt, max_tokens=100):
        response = self.client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()

class GroqProvider(AIProvider):
    def __init__(self):
        super().__init__(os.getenv("GROQ_API_KEY"))
        self.client = Groq(api_key=self.api_key)

    def generate_text(self, prompt, max_tokens=100):
        response = self.client.chat.completions.create(
            model="llama2-70b-4096",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

class OpenRouterProvider(AIProvider):
    def __init__(self):
        super().__init__(os.getenv("OPENROUTER_API_KEY"))
        self.client = OpenRouter(api_key=self.api_key)

    def generate_text(self, prompt, max_tokens=100):
        response = self.client.completions.create(
            model="openai/gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()

