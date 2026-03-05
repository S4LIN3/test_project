from __future__ import annotations

import traceback
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.modules.cv.image_analyzer import analyze_image
from src.modules.data.analysis import clean_dataset, detect_patterns, summarize_dataset
from src.modules.data.visualization import build_visualizations
from src.modules.docs.pdf_processor import chunk_text, extract_nlp_insights, summarize_text_simple
from src.modules.docs.rag import DocumentRAG
from src.modules.ml.pipeline import train_models


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    clean_df = pd.DataFrame()

    try:
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4, 5, 6],
                "feature_b": [6, 5, 4, 3, 2, 1],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        clean_df, _ = clean_dataset(df)
        _ = summarize_dataset(clean_df)
        _ = detect_patterns(clean_df)
        figures = build_visualizations(clean_df)
        checks.append(("data_pipeline", True, f"figures={list(figures.keys())}"))
    except Exception as exc:
        checks.append(("data_pipeline", False, str(exc)))

    try:
        result = train_models(clean_df, target_column="target", test_size=0.33)
        checks.append(("ml_pipeline", True, result.best_model_name))
    except Exception as exc:
        checks.append(("ml_pipeline", False, str(exc)))

    try:
        sample_text = (
            "This is a test document for cloud smoke checks. "
            "It includes repeated terms for retrieval retrieval retrieval."
        )
        chunks = chunk_text(sample_text)
        summary = summarize_text_simple(sample_text)
        nlp = extract_nlp_insights(sample_text)
        rag = DocumentRAG(persist_dir=Path("data/vectorstore"))
        rag.ingest(chunks)
        answer, _ = rag.answer("What does this document discuss?")
        checks.append(
            (
                "docs_rag",
                True,
                f"summary_len={len(summary)}, answer_len={len(answer)}, nlp={bool(nlp)}",
            )
        )
    except Exception as exc:
        checks.append(("docs_rag", False, str(exc)))

    try:
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        vision = analyze_image(img)
        checks.append(("vision", True, vision.classification.get("label", "unknown")))
    except Exception as exc:
        checks.append(("vision", False, str(exc)))

    all_passed = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        all_passed = all_passed and ok

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

