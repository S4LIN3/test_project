# Smart Research and Data Analysis Assistant

A complete Streamlit-based AI application for:
- Dataset profiling, cleaning, pattern detection
- Automatic visualization
- Auto ML model training and prediction
- PDF extraction, summarization, RAG Q and A
- Image object detection and classification
- AI chat across all generated context

No JavaScript frontend framework is used.

## 1) System Architecture

### High-level flow
1. Streamlit UI receives user uploads and user prompts.
2. Backend service modules process data/documents/images.
3. ML module trains models on selected target variables.
4. Document module extracts text and builds FAISS index for retrieval.
5. Chat module combines dataset summary + document context + retrieval chunks.
6. OpenAI API generates summaries, explanations, and chat responses when configured.
7. SQLite stores chat history and run logs.

### Components
- UI Layer: `app.py` (Streamlit tabs and interactions)
- Data Layer: `src/modules/data/*`
- ML Layer: `src/modules/ml/pipeline.py`
- Document AI + RAG Layer: `src/modules/docs/*`
- Computer Vision Layer: `src/modules/cv/image_analyzer.py`
- GenAI Layer: `src/modules/ai/*`
- Persistence Layer: `src/db/database.py`
- Config Layer: `src/config/settings.py`

## 2) Folder Structure

```text
smart_research_data_assistant/
  app.py
  requirements.txt
  .env.example
  .streamlit/
    config.toml
  data/
    uploads/
    vectorstore/
  models/
  src/
    config/
      settings.py
    core/
      logger.py
    db/
      database.py
    modules/
      data/
        analysis.py
        visualization.py
      ml/
        pipeline.py
      docs/
        pdf_processor.py
        rag.py
      cv/
        image_analyzer.py
      ai/
        openai_client.py
        chat_assistant.py
    utils/
      io.py
```

## 3) Streamlit Application Layout

Tabs in `app.py`:
1. Dataset: upload + clean + summarize + pattern detection
2. Visualizations: auto line/bar/histogram/scatter plots
3. ML: target selection + model training + leaderboard
4. Documents + RAG: PDF processing + Q and A
5. Image Analysis: object detection + classification
6. AI Chat: context-aware assistant
7. Dashboard: consolidated metrics and outputs

## 4) Backend Logic

- File uploads are persisted into `data/uploads`.
- Processing outputs are stored in `st.session_state` for cross-tab reuse.
- Logs and chat history are persisted in SQLite (`data/app.db`).
- Modules are isolated so each capability can be tested and scaled independently.

## 5) Machine Learning Pipeline

Implemented in `src/modules/ml/pipeline.py`:
- Detects task type (classification or regression)
- Builds preprocessing with imputers + scaling + one-hot encoding
- Trains and compares multiple models
- Returns leaderboard and best pipeline
- Includes optional PyCaret helper (`train_with_pycaret`)

## 6) RAG Implementation

Implemented in `src/modules/docs/rag.py`:
- PDF text chunking from `src/modules/docs/pdf_processor.py`
- Embeddings:
  - OpenAI embeddings if API key exists
  - Local sentence-transformers embeddings otherwise
- Vector store: FAISS
- Retrieval: top-k similar chunks
- Answer generation:
  - OpenAI grounded response when API key exists
  - Retrieval-only fallback when key is missing

## 7) Computer Vision Module

Implemented in `src/modules/cv/image_analyzer.py`:
- Object detection:
  - YOLO (ultralytics) if available
  - OpenCV Haar face-detection fallback
- Image classification:
  - ResNet18 if torch/torchvision available
  - Brightness-based fallback when unavailable

## 8) OpenAI API Integration

Implemented in `src/modules/ai/openai_client.py`:
- Document summarization
- Context-aware chat completions
- Used by:
  - Document tab
  - Chat tab

Set the key in `.env`:

```bash
OPENAI_API_KEY=your_key
```

## 9) Example Execution

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py

# full feature profile (heavier)
pip install -r requirements-full.txt
streamlit run app.py
```

Open browser:
- http://localhost:8501

## 10) Deployment Instructions

### Streamlit Cloud
1. Push this folder to GitHub.
2. In Streamlit Cloud, create app with `app.py` as entry point.
3. Add secret `OPENAI_API_KEY` in app settings.
4. Deploy.

### Render
1. Create a new Web Service from repo.
2. Build command:
   - `pip install -r requirements.txt`
3. Start command:
   - `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add env variables (`OPENAI_API_KEY`, optional `VECTOR_DB`).

## Production Notes
- Use Postgres by replacing `DatabaseManager` connection layer.
- Add authentication for multi-user environments.
- Add object storage (S3/GCS) for uploaded files.
- Add model registry for trained models.
- Add tests for each module in a `tests/` package.

## 11) Cloud Deployment Profile (Added)

This repository now includes deployment-specific files for cloud builds:
- `requirements.txt` (lean dependency set used by Streamlit Cloud by default)
- `requirements-full.txt` (full/heavy local profile)
- `requirements-cloud.txt` (alias of lean profile for clarity)
- `render.yaml` (Render service definition)
- `Procfile` (generic process entry)
- `runtime.txt` (Python version pin)
- `scripts/cloud_smoke_test.py` (quick runtime checks)

### A) Test cloud profile locally

```bash
pip install -r requirements.txt
python scripts/cloud_smoke_test.py
streamlit run app.py
```

### B) Deploy on Streamlit Community Cloud

1. Push repo to GitHub.
2. In Streamlit Cloud, set entrypoint to `app.py`.
3. In app settings, set Python to 3.11 and add secret:

```toml
OPENAI_API_KEY="your_key"
```

4. Streamlit Cloud installs `requirements.txt` automatically (already set to cloud-safe profile).

### C) Deploy on Render

Render will auto-detect `render.yaml`.

Manual settings (if needed):
- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`

### D) Feature notes for cloud profile

The cloud profile intentionally excludes heavy/optional packages (`tensorflow`, `torch`, `pycaret`, `ultralytics`, `chromadb`, `sentence-transformers`) to avoid build failures.
Fallback behavior in app:
- Image analysis falls back to lightweight OpenCV logic.
- Document RAG falls back to keyword retrieval when embedding backends are unavailable.
- ML still works via scikit-learn pipeline.

If you need full feature parity, deploy with `requirements-full.txt` on a larger instance.


