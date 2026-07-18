# NovaAI — Chat with PDF using RAG

NovaAI is a Retrieval-Augmented Generation (RAG) app with a clean, dark,
minimal chat interface. Upload a PDF, get an instant summary on demand, and
ask questions that are answered only from the uploaded document.

**Live demo:** deploy your own on [Streamlit Community Cloud](https://share.streamlit.io/) — see below.

## Features

- Dark, minimal two-panel UI (document panel + chat panel), styled after
  ChatGPT/Claude rather than a marketing landing page
- Robot avatar and NovaAI branding throughout
- On-demand document summary — generated only when you click **Generate summary**,
  not automatically on every upload
- Fixed-position chat input with an internally scrolling message list, so long
  answers never push the input box out of view
- Answers are grounded only in the uploaded PDF; no outside knowledge is used

## How it works

```
PDF Upload → Text Extraction → Cleaning → Chunking → Embeddings → ChromaDB → Retriever → LLM → Grounded Answer
```

- `app.py` — Streamlit UI, session state, and end-to-end workflow
- `src/config.py` — settings loaded from `.env` (`get_settings`)
- `src/pdf_loader.py` — PDF validation and text extraction (`validate_pdf_upload`, `save_uploaded_pdf`, `extract_pdf_pages`)
- `src/chunker.py` — text splitting with page metadata (`chunk_pages`)
- `src/vector_store.py` — persistent ChromaDB storage (`get_vector_store`, `add_chunks_to_vector_store`, `collection_name_for_hash`)
- `src/llm.py` — Groq / Ollama LLM access (`get_llm`, `MissingApiKeyError`)
- `src/rag_chain.py` — prompt building and grounded-answer formatting (`answer_question`)
- `src/utils.py` — helpers (`generate_document_id`, `ensure_directories`)
- `src/logger.py` — app-wide logging (`get_logger`)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env              # then edit .env with your keys
streamlit run app.py
```

Set in `.env` for Groq (recommended, works everywhere including the cloud):

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_real_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
LLM_MAX_TOKENS=1000
```

Or for local Ollama:

```
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### Theme

NovaAI's dark theme lives in `.streamlit/config.toml`. Restart the Streamlit
server (not just a browser refresh) after any theme change for it to apply:

```toml
[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0A0E17"
secondaryBackgroundColor = "#10141F"
textColor = "#E8ECF4"
```

## Deploy to Streamlit Community Cloud

Ollama needs a local server, so it will not work on Streamlit Cloud — use
`LLM_PROVIDER=groq` for the deployed app.

1. Push this folder (including `.streamlit/config.toml`) to a **public or private GitHub repo**.
2. Do **not** commit your `.env` file — it's already git-ignored.
3. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
4. Click **New app**, pick your repo/branch, and set the main file path to `app.py`.
5. Under **Advanced settings → Secrets**, paste:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your_real_groq_key_here"
   GROQ_MODEL = "llama-3.1-8b-instant"
   LLM_MAX_TOKENS = "1000"

   EMBEDDING_PROVIDER = "sentence-transformers"
   SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   EMBEDDING_BATCH_SIZE = "8"

   CHROMA_PERSIST_DIR = "vector_db"
   TOP_K = "8"

   MAX_PDF_MB = "25"
   CHUNK_SIZE = "900"
   CHUNK_OVERLAP = "180"
   ```
6. Click **Deploy**. First boot will be slower while the embedding model downloads.

Note: Streamlit Community Cloud's filesystem is ephemeral — the ChromaDB index
resets whenever the app restarts/redeploys, so users re-upload their PDF each
fresh session. That's expected for this kind of demo app.

## Using NovaAI

1. Upload a PDF in the **Document** panel.
2. Once indexing finishes, the document card shows page and chunk counts.
3. Click **Generate summary** if you want a quick overview — it's not created
   automatically, so indexing stays fast.
4. Ask questions in the **Chat** panel. Answers come only from the uploaded PDF.
5. Use **Upload a different PDF** or the sidebar's **Reset document** to start over.

## Common errors

- `ModuleNotFoundError` → activate your venv, `pip install -r requirements.txt`.
- `GROQ_API_KEY missing` → set it in `.env` or Streamlit Cloud secrets.
- `No extractable text` → the PDF is scanned; run OCR first.
- Theme looks off / white buttons → make sure `.streamlit/config.toml` has
  `base = "dark"` and restart the server (a browser refresh isn't enough).
- Slow first run → the embedding model downloads on first use.

## Tech stack

Python · Streamlit · PyMuPDF · LangChain text splitters · ChromaDB ·
SentenceTransformers / Ollama embeddings · Groq / Ollama LLMs · python-dotenv
