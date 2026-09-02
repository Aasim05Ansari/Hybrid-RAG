from abc import ABC, abstractmethod

from openai import OpenAI

from app.config.settings import settings


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAILLMProvider(LLMProvider):

    def __init__(self, model: str | None = None):

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required "
                "for OpenAI generation"
            )

        self.model = model or settings.llm_model

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def generate(self, prompt: str) -> str:

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty"
            )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text
