from __future__ import annotations

from pathlib import Path

from src.modules.docs.pdf_processor import join_context


class DocumentRAG:
    def __init__(self, persist_dir: Path, openai_api_key: str | None = None):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key
        self.vectorstore = None
        self.raw_chunks: list[str] = []

    def _embedding_model(self):
        if self.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(api_key=self.openai_api_key)

        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def ingest(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("No chunks available for ingestion.")

        self.raw_chunks = chunks

        try:
            from langchain_community.vectorstores import FAISS

            embeddings = self._embedding_model()
            self.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            self.vectorstore.save_local(str(self.persist_dir / "faiss_index"))
        except Exception:
            self.vectorstore = None

    def load(self) -> bool:
        index_dir = self.persist_dir / "faiss_index"
        if not index_dir.exists():
            return False

        try:
            from langchain_community.vectorstores import FAISS

            embeddings = self._embedding_model()
            self.vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            self.vectorstore = None
            return False

    def _keyword_retrieve(self, question: str, k: int = 4) -> list[str]:
        if not self.raw_chunks:
            return []

        terms = [term.lower() for term in question.split() if len(term) > 2]
        scored: list[tuple[int, str]] = []
        for chunk in self.raw_chunks:
            text = chunk.lower()
            score = sum(text.count(term) for term in terms)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored[:k] if score > 0]
        if not top_chunks:
            return self.raw_chunks[:k]
        return top_chunks

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        if self.vectorstore is not None:
            docs = self.vectorstore.similarity_search(question, k=k)
            return [doc.page_content for doc in docs]

        return self._keyword_retrieve(question=question, k=k)

    def answer(self, question: str) -> tuple[str, list[str]]:
        context_chunks = self.retrieve(question=question, k=4)
        context = join_context(context_chunks, max_chars=4500)

        if not context:
            return ("No document context is indexed yet.", [])

        if self.openai_api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer strictly from provided context. If unknown, say so.",
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}\n\nContext:\n{context}",
                        },
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content or "No answer generated."
                return (answer, context_chunks)
            except Exception:
                pass

        fallback = (
            "OpenAI API key is missing or unavailable. Returning retrieval context only.\n\n"
            f"Relevant context:\n{context[:1200]}"
        )
        return (fallback, context_chunks)

