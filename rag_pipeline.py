"""
Finance RAG Pipeline
PDF → Extract → Chunk → Embed → FAISS Index → Retrieve → Groq Answer
"""

import re
import math
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dataclasses import dataclass
from typing import Optional

# ─── Constants ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # tokens (approx 4 chars per token → ~2000 chars)
CHUNK_OVERLAP = 50    # token overlap
CHARS_PER_TOK = 4
TOP_K         = 5     # number of chunks to retrieve

GROQ_MODEL    = "llama-3.3-70b-versatile"
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024

EMBED_MODEL   = "all-MiniLM-L6-v2"   # small, fast, free


# ─── Data Class ─────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    page: int
    section: str
    chunk_id: int


# ─── PDF Extraction ──────────────────────────────────────────────────────────
def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Return list of {page_num, text, sections}."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def detect_section(text: str) -> str:
    """Very simple heuristic: first ALL-CAPS or Title-Case short line = section."""
    for line in text.split("\n")[:5]:
        line = line.strip()
        if len(line) > 3 and (line.isupper() or (line.istitle() and len(line.split()) <= 8)):
            return line
    return "Body"


# ─── Chunking ────────────────────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[Chunk]:
    """Chunk each page's text by approximate token count with overlap."""
    char_size    = CHUNK_SIZE    * CHARS_PER_TOK
    char_overlap = CHUNK_OVERLAP * CHARS_PER_TOK

    chunks: list[Chunk] = []
    cid = 0

    for pg in pages:
        text    = pg["text"]
        page_no = pg["page"]
        start   = 0

        while start < len(text):
            end         = min(start + char_size, len(text))
            chunk_text  = text[start:end].strip()

            if chunk_text:
                section = detect_section(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    page=page_no,
                    section=section,
                    chunk_id=cid
                ))
                cid += 1

            # Move forward with overlap
            start += char_size - char_overlap
            if start >= len(text):
                break

    return chunks


# ─── Embedding & FAISS Index ─────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def build_index(chunks: list[Chunk]) -> tuple[faiss.Index, np.ndarray]:
    """Embed all chunks and build a FAISS flat L2 index."""
    embedder = get_embedder()
    texts    = [c.text for c in chunks]
    vecs     = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    dim      = vecs.shape[1]
    index    = faiss.IndexFlatIP(dim)   # Inner Product on normalised vecs = cosine sim
    index.add(vecs.astype(np.float32))
    return index, vecs


def retrieve(query: str, chunks: list[Chunk], index: faiss.Index, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-k chunks most similar to the query."""
    embedder  = get_embedder()
    q_vec     = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        c = chunks[idx]
        results.append({
            "chunk_id": c.chunk_id,
            "page":     c.page,
            "section":  c.section,
            "text":     c.text,
            "score":    float(score),   # cosine similarity 0-1
        })
    return results


# ─── Groq Generation ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are DocRAG — a document-grounded financial assistant.

Rules:
- Answer EXCLUSIVELY from the retrieved document chunks provided below.
- NEVER use general knowledge.
- If the answer is not in the chunks, say exactly:
  "This information was not found in the uploaded document."
- Always cite the source like: [Page 4, Section: Risk Factors]
- Quote key phrases directly from the document when relevant.
- Give a confidence score (0–100%) based on how clearly the chunk answers.
- If multiple chunks are relevant, synthesize them and note all sources.

Output format (strict):
Answer: [your grounded answer]
Source: [Page X / Section Y]
Quote: "[exact text from document]"
Confidence: XX%
"""


def build_user_prompt(question: str, retrieved: list[dict]) -> str:
    chunk_block = ""
    for i, r in enumerate(retrieved, 1):
        chunk_block += (
            f"Chunk {i} [Page {r['page']}, Section: {r['section']}]:\n"
            f"\"{r['text']}\"\n\n"
        )
    return f"RETRIEVED CHUNKS:\n---\n{chunk_block}---\n\nUSER QUESTION:\n{question}"


def ask_groq(question: str, retrieved: list[dict], api_key: str) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(question, retrieved)},
        ],
    )
    return response.choices[0].message.content


# ─── Full Pipeline ───────────────────────────────────────────────────────────
def process_pdf(pdf_bytes: bytes) -> tuple[list[Chunk], faiss.Index]:
    pages  = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages)
    index, _ = build_index(chunks)
    return chunks, index


def query(question: str, chunks: list[Chunk], index: faiss.Index, api_key: str) -> dict:
    retrieved = retrieve(question, chunks, index)
    answer    = ask_groq(question, retrieved, api_key)
    return {
        "answer":    answer,
        "retrieved": retrieved,
    }
