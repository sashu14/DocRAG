# 🤖 DOCRAG — Document Based RAG Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sashu14-docrag-app-i3gf29.streamlit.app/)

> **Live Demo → [sashu14-docrag-app-i3gf29.streamlit.app](https://sashu14-docrag-app-i3gf29.streamlit.app/)**

A **Retrieval-Augmented Generation (RAG)** app for financial documents — answers questions **exclusively from your uploaded PDF** with citations, quotes, and confidence scores. Powered by **Groq** (`llama-3.3-70b-versatile`) and **FAISS** vector search.

---

## ⚙️ Pipeline

```
PDF Upload → Extract (PyMuPDF) → Chunk (500 tok / 50 overlap)
          → Embed (MiniLM-L6)  → FAISS cosine search
          → Inject top-5 chunks → Groq LLM → Cited answer
```

## 📋 Output Format

```
Answer:     [grounded answer — only from the document]
Source:     [Page X / Section Y]
Quote:      "[exact text from document]"
Confidence: XX%
```

## 🚀 Run Locally

```bash
git clone https://github.com/sashu14/DocRAG.git
cd DocRAG
pip install -r requirements.txt

# Add your Groq API key
echo 'GROQ_API_KEY=gsk_...' > .env

streamlit run app.py
```

## ✨ Features

| Feature | Detail |
|---|---|
| PDF parsing | PyMuPDF (any finance PDF) |
| Chunking | 500 tokens, 50 token overlap |
| Embedding | `all-MiniLM-L6-v2` (local, free) |
| Vector DB | FAISS (in-memory, fast) |
| LLM | `llama-3.3-70b-versatile` via Groq |
| Citations | Page + Section + direct quote |
| Confidence | 0–100% per answer |

## 🔑 API Key

Get a **free** Groq API key at [console.groq.com](https://console.groq.com)
