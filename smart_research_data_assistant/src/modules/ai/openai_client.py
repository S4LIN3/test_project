from __future__ import annotations

from openai import OpenAI


class OpenAIService:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self.client is not None

    def summarize(self, text: str, max_words: int = 180) -> str:
        if not self.client:
            return "OpenAI API key is not configured."

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize technical documents clearly and concisely.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following in less than {max_words} words:\n\n{text[:12000]}"
                    ),
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No summary generated."

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            return "OpenAI API key is not configured."

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or "No response generated."
