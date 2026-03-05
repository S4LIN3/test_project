from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from src.config.settings import AppSettings, DB_PATH, UPLOAD_DIR, VECTORSTORE_DIR, ensure_directories
from src.db.database import DatabaseManager
from src.modules.ai.chat_assistant import ChatAssistant
from src.modules.ai.openai_client import OpenAIService
from src.modules.cv.image_analyzer import analyze_image
from src.modules.data.analysis import clean_dataset, detect_patterns, load_dataset, summarize_dataset
from src.modules.data.visualization import build_visualizations
from src.modules.docs.pdf_processor import (
    chunk_text,
    extract_nlp_insights,
    extract_text_from_pdf,
    summarize_text_simple,
)
from src.modules.docs.rag import DocumentRAG
from src.modules.ml.pipeline import train_models, train_with_pycaret
from src.utils.io import save_uploaded_file


def initialize_state() -> None:
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "dataset_summary": None,
        "dataset_patterns": None,
        "figures": {},
        "ml_result": None,
        "doc_text": "",
        "doc_summary": "",
        "doc_nlp": None,
        "rag": None,
        "last_rag_chunks": [],
        "vision_result": None,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_sidebar(settings: AppSettings, openai_enabled: bool) -> None:
    st.sidebar.title("Smart Research Assistant")
    st.sidebar.caption("Python + Streamlit AI platform")

    st.sidebar.markdown("### Runtime")
    st.sidebar.write(f"OpenAI available: {'Yes' if openai_enabled else 'No'}")
    st.sidebar.write(f"Vector DB: {settings.vector_db}")

    st.sidebar.markdown("### Modules")
    st.sidebar.write("1. Dataset Analysis")
    st.sidebar.write("2. Auto Visualization")
    st.sidebar.write("3. ML Prediction")
    st.sidebar.write("4. Document AI + RAG")
    st.sidebar.write("5. Image Analysis")
    st.sidebar.write("6. AI Chat")


def dataset_tab(db: DatabaseManager) -> None:
    st.subheader("1) Dataset Analysis")
    data_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="dataset_uploader",
    )

    if data_file is not None:
        file_path = save_uploaded_file(data_file, UPLOAD_DIR)
        raw_df = load_dataset(str(file_path))
        clean_df, clean_report = clean_dataset(raw_df)

        st.session_state["raw_df"] = raw_df
        st.session_state["clean_df"] = clean_df
        st.session_state["dataset_summary"] = summarize_dataset(clean_df)
        st.session_state["dataset_patterns"] = detect_patterns(clean_df)
        st.session_state["figures"] = build_visualizations(clean_df)

        db.insert_run(
            "dataset_analysis",
            (
                f"Raw shape={clean_report.raw_shape}, Clean shape={clean_report.clean_shape}, "
                f"Duplicates dropped={clean_report.dropped_duplicates}, "
                f"Missing filled={clean_report.missing_values_filled}"
            ),
        )

        st.success("Dataset loaded and analyzed.")

    if st.session_state["clean_df"] is None:
        st.info("Upload a dataset to start analysis.")
        return

    clean_df = st.session_state["clean_df"]
    dataset_summary = st.session_state["dataset_summary"]
    patterns = st.session_state["dataset_patterns"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", dataset_summary["rows"])
    col2.metric("Columns", dataset_summary["columns"])
    col3.metric("Missing Values", int(clean_df.isna().sum().sum()))

    st.markdown("#### Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)

    st.markdown("#### Pattern Detection")
    strong_corr = patterns.get("strong_correlations", [])
    skew_cols = patterns.get("high_skew_columns", [])

    if strong_corr:
        st.write("Strong correlations (|r| >= 0.6)")
        st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    else:
        st.write("No strong correlations detected.")

    if skew_cols:
        st.write("High skew columns (|skew| > 1.0)")
        st.dataframe(pd.DataFrame(skew_cols), use_container_width=True)


def visualization_tab() -> None:
    st.subheader("2) Automatic Visualization")
    figures = st.session_state.get("figures", {})

    if not figures:
        st.info("Run dataset analysis first.")
        return

    for chart_name, figure in figures.items():
        st.plotly_chart(figure, use_container_width=True, key=f"viz_{chart_name}")


def ml_tab(db: DatabaseManager) -> None:
    st.subheader("3) Machine Learning Prediction")

    if st.session_state["clean_df"] is None:
        st.info("Upload and analyze a dataset first.")
        return

    df = st.session_state["clean_df"]
    target = st.selectbox("Select target variable", df.columns.tolist())
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    col1, col2 = st.columns(2)
    train_button = col1.button("Train models", type="primary")
    pycaret_button = col2.button("Train with PyCaret (optional)")

    if train_button:
        result = train_models(df=df, target_column=target, test_size=test_size)
        st.session_state["ml_result"] = result
        db.insert_run(
            "ml_training",
            f"Task={result.task_type}, Best model={result.best_model_name}",
        )
        st.success(f"Training complete. Best model: {result.best_model_name}")

    if pycaret_button:
        pycaret_result = train_with_pycaret(df, target)
        if pycaret_result:
            st.info(f"PyCaret best model generated: {pycaret_result[0]}")
        else:
            st.warning("PyCaret is not available or failed to run in this environment.")

    result = st.session_state.get("ml_result")
    if result is not None:
        st.markdown(f"Task type: **{result.task_type}**")
        st.dataframe(result.leaderboard, use_container_width=True)


def document_tab(db: DatabaseManager, openai_service: OpenAIService) -> None:
    st.subheader("4) Document AI and RAG")

    pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_uploader")

    if st.session_state["rag"] is None:
        rag = DocumentRAG(
            persist_dir=VECTORSTORE_DIR,
            openai_api_key=openai_service.api_key,
        )
        rag.load()
        st.session_state["rag"] = rag

    if pdf_file is not None and st.button("Process document", type="primary"):
        file_path = save_uploaded_file(pdf_file, UPLOAD_DIR)
        text = extract_text_from_pdf(str(file_path))
        chunks = chunk_text(text)

        st.session_state["doc_text"] = text
        st.session_state["doc_nlp"] = extract_nlp_insights(text)
        st.session_state["rag"].ingest(chunks)

        if openai_service.is_available() and text.strip():
            doc_summary = openai_service.summarize(text)
        else:
            doc_summary = summarize_text_simple(text)

        st.session_state["doc_summary"] = doc_summary

        db.insert_run(
            "document_ai",
            f"Document processed with {len(chunks)} chunks and summary generated.",
        )
        st.success("Document indexed for Q&A.")

    if st.session_state["doc_summary"]:
        st.markdown("#### Document Summary")
        st.write(st.session_state["doc_summary"])

    doc_nlp = st.session_state.get("doc_nlp")
    if doc_nlp:
        st.markdown("#### NLP Insights")
        st.write(
            {
                "word_count": doc_nlp.get("word_count"),
                "top_keywords": doc_nlp.get("top_keywords", [])[:10],
            }
        )
        if doc_nlp.get("named_entities"):
            st.dataframe(pd.DataFrame(doc_nlp.get("named_entities")[:15]), use_container_width=True)

    question = st.text_input("Ask a question about the document")
    if st.button("Ask document", key="ask_document") and question:
        answer, chunks = st.session_state["rag"].answer(question)
        st.session_state["last_rag_chunks"] = chunks

        st.markdown("#### Answer")
        st.write(answer)

        if chunks:
            st.markdown("#### Retrieved Context Chunks")
            for idx, chunk in enumerate(chunks, start=1):
                st.caption(f"Chunk {idx}")
                st.write(chunk[:1000])


def image_tab(db: DatabaseManager) -> None:
    st.subheader("5) Image Analysis")

    image_file = st.file_uploader(
        "Upload an image for object detection and classification",
        type=["png", "jpg", "jpeg"],
        key="image_uploader",
    )

    if image_file is not None and st.button("Analyze image", type="primary"):
        from PIL import Image

        image = Image.open(image_file)
        result = analyze_image(image)
        st.session_state["vision_result"] = result

        db.insert_run(
            "image_analysis",
            f"Objects detected: {len(result.objects)}, Classification: {result.classification.get('label')}",
        )

    result = st.session_state.get("vision_result")
    if result is None:
        st.info("Upload an image and click Analyze image.")
        return

    st.markdown("#### Annotated Image")
    image_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, use_container_width=True)

    st.markdown("#### Object Detection")
    st.dataframe(pd.DataFrame(result.objects), use_container_width=True)

    st.markdown("#### Image Classification")
    st.json(result.classification)


def chat_tab(db: DatabaseManager, assistant: ChatAssistant) -> None:
    st.subheader("6) AI Chat Assistant")

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask about datasets, documents, and analysis results")
    if not user_query:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": user_query})
    db.insert_chat("user", user_query)

    rag_context = []
    if st.session_state.get("rag") is not None:
        rag_context = st.session_state["rag"].retrieve(user_query, k=3)

    response = assistant.answer(
        user_query=user_query,
        dataset_summary=st.session_state.get("dataset_summary"),
        document_summary=st.session_state.get("doc_summary"),
        rag_context=rag_context,
    )

    st.session_state["chat_messages"].append({"role": "assistant", "content": response})
    db.insert_chat("assistant", response)

    with st.chat_message("assistant"):
        st.markdown(response)


def dashboard_tab(db: DatabaseManager) -> None:
    st.subheader("7) Interactive Dashboard")

    summary = st.session_state.get("dataset_summary")
    result = st.session_state.get("ml_result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset loaded", "Yes" if summary else "No")
    c2.metric("ML trained", "Yes" if result else "No")
    c3.metric("Document indexed", "Yes" if st.session_state.get("doc_summary") else "No")

    if summary:
        st.markdown("#### Data Snapshot")
        st.write(
            {
                "rows": summary.get("rows"),
                "columns": summary.get("columns"),
                "numeric_columns": len(summary.get("numeric_columns", [])),
                "categorical_columns": len(summary.get("categorical_columns", [])),
            }
        )

    if st.session_state.get("figures"):
        st.markdown("#### Key Visualization")
        first_name, first_fig = next(iter(st.session_state["figures"].items()))
        st.plotly_chart(
            first_fig,
            use_container_width=True,
            key=f"dashboard_primary_{first_name}",
        )

    if result is not None:
        st.markdown("#### Model Leaderboard")
        st.dataframe(result.leaderboard, use_container_width=True)

    st.markdown("#### Recent Chat Logs")
    recent_logs = db.get_recent_chat(limit=10)
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs), use_container_width=True)
    else:
        st.caption("No chat history yet.")


def main() -> None:
    ensure_directories()

    settings = AppSettings()
    db = DatabaseManager(DB_PATH)
    db.initialize()

    assistant = ChatAssistant(api_key=settings.openai_api_key)
    openai_service = OpenAIService(api_key=settings.openai_api_key)

    st.set_page_config(
        page_title=settings.app_name,
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()
    build_sidebar(settings=settings, openai_enabled=openai_service.is_available())

    st.title("Smart Research and Data Analysis Assistant")
    st.caption(
        "Analyze datasets, build ML models, process documents with RAG, inspect images, and chat with AI from one Streamlit app."
    )

    tabs = st.tabs(
        [
            "Dataset",
            "Visualizations",
            "ML",
            "Documents + RAG",
            "Image Analysis",
            "AI Chat",
            "Dashboard",
        ]
    )

    with tabs[0]:
        dataset_tab(db)
    with tabs[1]:
        visualization_tab()
    with tabs[2]:
        ml_tab(db)
    with tabs[3]:
        document_tab(db, openai_service)
    with tabs[4]:
        image_tab(db)
    with tabs[5]:
        chat_tab(db, assistant)
    with tabs[6]:
        dashboard_tab(db)


if __name__ == "__main__":
    main()
