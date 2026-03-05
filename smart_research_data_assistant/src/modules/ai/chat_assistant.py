from __future__ import annotations

from typing import Any

from src.modules.ai.openai_client import OpenAIService


class ChatAssistant:
    def __init__(self, api_key: str | None):
        self.openai_service = OpenAIService(api_key=api_key)

    def answer(
        self,
        user_query: str,
        dataset_summary: dict[str, Any] | None = None,
        document_summary: str | None = None,
        rag_context: list[str] | None = None,
    ) -> str:
        context_parts: list[str] = []

        if dataset_summary:
            context_parts.append(
                "Dataset Summary:\n"
                f"Rows: {dataset_summary.get('rows')}\n"
                f"Columns: {dataset_summary.get('columns')}\n"
                f"Numeric columns: {dataset_summary.get('numeric_columns')}\n"
                f"Categorical columns: {dataset_summary.get('categorical_columns')}"
            )

        if document_summary:
            context_parts.append(f"Document Summary:\n{document_summary}")

        if rag_context:
            context_parts.append("Retrieved Context:\n" + "\n\n".join(rag_context[:3]))

        combined_context = "\n\n".join(context_parts)

        if self.openai_service.is_available():
            prompt = (
                f"User Question: {user_query}\n\n"
                f"Context:\n{combined_context if combined_context else 'No context provided.'}"
            )
            return self.openai_service.chat(
                system_prompt=(
                    "You are an AI assistant for data science and research workflows. "
                    "Use the provided context when available and be explicit about uncertainty."
                ),
                user_prompt=prompt,
            )

        return (
            "OpenAI API key is not configured.\n\n"
            "You can still use analytics, ML, visualizations, PDF extraction, and local retrieval."
        )
