# 📊 DocRAG · Finance

A **Retrieval-Augmented Generation (RAG)** app for financial documents, powered by **Groq** (`llama-3.3-70b-versatile`) and **FAISS** vector search.

## Pipeline

```
PDF Upload → Extract (PyMuPDF) → Chunk (500 tok / 50 overlap)
         → Embed (MiniLM-L6)  → FAISS cosine search
         → Inject top-5 into Groq prompt → Cited answer
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 3. Run
streamlit run app.py
```

## Features

| Feature | Detail |
|---|---|
| PDF parsing | PyMuPDF (any finance PDF) |
| Chunking | 500 tokens, 50 token overlap |
| Embedding | `all-MiniLM-L6-v2` (local, free) |
| Vector DB | FAISS (in-memory, fast) |
| LLM | `llama-3.3-70b-versatile` via Groq |
| Citations | Page + Section + direct quote |
| Confidence | 0–100% per answer |

## Output Format

```
Answer:     [grounded answer]
Source:     [Page X / Section Y]
Quote:      "[exact text from document]"
Confidence: XX%
```

## Notes

- Model never answers from general knowledge — only the uploaded PDF.
- If an answer isn't in the document it says so explicitly.
- API key: get one free at [console.groq.com](https://console.groq.com)
