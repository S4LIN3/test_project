from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except Exception:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end
        return chunks


def summarize_text_simple(text: str, max_sentences: int = 6) -> str:
    if not text.strip():
        return "No text was extracted from the document."

    sentence_candidates = [s.strip() for s in text.replace("\n", " ").split(".")]
    sentence_candidates = [s for s in sentence_candidates if len(s) > 30]

    if not sentence_candidates:
        return text[:600]

    selected = sentence_candidates[:max_sentences]
    return ". ".join(selected) + "."


def extract_nlp_insights(text: str, top_k: int = 10) -> dict[str, object]:
    if not text.strip():
        return {"word_count": 0, "top_keywords": [], "named_entities": []}

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "you",
        "your",
        "but",
        "can",
        "into",
        "will",
        "about",
        "data",
    }

    filtered = [token for token in tokens if token not in stopwords and len(token) > 2]
    keyword_counts = Counter(filtered).most_common(top_k)

    entities: list[dict[str, str]] = []
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:120000])
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents[:30]]
    except Exception:
        pass

    return {
        "word_count": len(tokens),
        "top_keywords": keyword_counts,
        "named_entities": entities,
    }


def join_context(chunks: Iterable[str], max_chars: int = 4500) -> str:
    collected = []
    total = 0
    for chunk in chunks:
        chunk_len = len(chunk)
        if total + chunk_len > max_chars:
            break
        collected.append(chunk)
        total += chunk_len
    return "\n\n".join(collected)
