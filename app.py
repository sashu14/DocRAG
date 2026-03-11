"""
DocRAG — Finance Document QA powered by Groq
Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
import rag_pipeline as rag

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOCRAG — Document Based RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark base */
    .stApp { background: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Header gradient */
    .hero {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 50%, #58a6ff 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(31,111,235,0.25);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #fff; }
    .hero p  { margin: 0.4rem 0 0; color: rgba(255,255,255,0.82); font-size: 1rem; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        flex: 1; background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1rem 1.2rem;
    }
    .metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }

    /* Answer box */
    .answer-box {
        background: #161b22; border: 1px solid #1f6feb;
        border-left: 4px solid #1f6feb;
        border-radius: 10px; padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        white-space: pre-wrap;
        line-height: 1.8;
        font-size: 0.97rem;
    }

    /* Chunk pills */
    .chunk-pill {
        display: inline-block;
        background: #21262d;
        border: 1px solid #388bfd44;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: #58a6ff;
        margin-right: 0.4rem;
    }

    /* Input styling override */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Expander */
    details { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 0.5rem 1rem; }

    /* Score bar */
    .score-bar-wrap { background: #21262d; border-radius: 20px; height: 8px; margin-top: 4px; }
    .score-bar      { height: 8px; border-radius: 20px;
                      background: linear-gradient(90deg, #1f6feb, #58a6ff); }

    hr { border-color: #21262d; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <h1>🤖 DOCRAG</h1>
  <p style="font-size:1.1rem; font-weight:600; color:rgba(255,255,255,0.95)">Document Based RAG Agent</p>
  <p style="margin-top:0.3rem">Ask questions grounded exclusively in your uploaded financial document.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
# Load API key — Streamlit Cloud secrets first, then local .env, then UI input
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # API Key — show input field if not already set from env/secrets
    if not api_key:
        st.markdown("### 🔑 Groq API Key")
        api_key = st.text_input(
            "Enter your Groq API key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
            help="Get a free key at https://console.groq.com",
        )
        if not api_key:
            st.warning("⚠️ Paste your Groq API key above to enable Q&A.")
        st.markdown("---")

    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        """
**Pipeline**
```
PDF → Extract (PyMuPDF)
    → Chunk (500 tok, 50 overlap)
    → Embed (MiniLM-L6)
    → FAISS cosine search
    → Groq llama-3.3-70b
    → Cited answer
```
"""
    )

    if st.button("🗑️ Clear Session"):
        for k in ["chunks", "faiss_index", "history"]:
            st.session_state.pop(k, None)
        st.rerun()

# ─── Session State Init ───────────────────────────────────────────────────────
for k, v in [("chunks", None), ("faiss_index", None), ("history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── PDF Processing ───────────────────────────────────────────────────────────
if uploaded and (st.session_state.chunks is None):
    with st.spinner("🔍 Processing PDF — extracting, chunking, embedding…"):
        try:
            pdf_bytes = uploaded.read()
            chunks, index = rag.process_pdf(pdf_bytes)
            st.session_state.chunks     = chunks
            st.session_state.faiss_index = index
            st.success(f"✅ Ready! **{len(chunks)}** chunks indexed from **{uploaded.name}**")
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")

# ─── Metrics Row ─────────────────────────────────────────────────────────────
chunks = st.session_state.chunks
if chunks:
    pages = len(set(c.page for c in chunks))
    st.markdown(
        f"""
<div class="metric-row">
  <div class="metric-card"><div class="label">Chunks Indexed</div><div class="value">{len(chunks)}</div></div>
  <div class="metric-card"><div class="label">Pages Processed</div><div class="value">{pages}</div></div>
  <div class="metric-card"><div class="label">Top-K Retrieved</div><div class="value">{rag.TOP_K}</div></div>
  <div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:0.85rem;padding-top:6px">llama-3.3-70b</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ─── Chat History ─────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-box">{entry["answer"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        with st.expander("📎 Retrieved Source Chunks"):
            for i, r in enumerate(entry["retrieved"], 1):
                sim_pct = int(r["score"] * 100)
                st.markdown(
                    f'<span class="chunk-pill">Page {r["page"]}</span>'
                    f'<span class="chunk-pill">{r["section"]}</span>'
                    f'<span class="chunk-pill">Similarity {sim_pct}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="score-bar-wrap"><div class="score-bar" style="width:{sim_pct}%"></div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(r["text"][:400] + ("…" if len(r["text"]) > 400 else ""))
                if i < len(entry["retrieved"]):
                    st.divider()

# ─── Query Input ──────────────────────────────────────────────────────────────
st.markdown("---")
question = st.chat_input(
    "Ask a question about your document…",
    disabled=(not chunks),
)

if not chunks and not uploaded:
    st.info("👈 Upload a Finance PDF in the sidebar to get started.")

if question:
    if not chunks:
        st.error("Please upload a PDF first.")
    elif not api_key:
        st.error("❌ Please enter your Groq API key in the sidebar to get answers.")
    elif api_key:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Retrieving and generating answer…"):
                try:
                    result = rag.query(
                        question,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        api_key,
                    )
                    answer    = result["answer"]
                    retrieved = result["retrieved"]

                    st.markdown(f'<div class="answer-box">{answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                    with st.expander("📎 Retrieved Source Chunks"):
                        for i, r in enumerate(retrieved, 1):
                            sim_pct = int(r["score"] * 100)
                            st.markdown(
                                f'<span class="chunk-pill">Page {r["page"]}</span>'
                                f'<span class="chunk-pill">{r["section"]}</span>'
                                f'<span class="chunk-pill">Similarity {sim_pct}%</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="score-bar-wrap"><div class="score-bar" style="width:{sim_pct}%"></div></div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(r["text"][:400] + ("…" if len(r["text"]) > 400 else ""))
                            if i < len(retrieved):
                                st.divider()

                    st.session_state.history.append(
                        {"question": question, "answer": answer, "retrieved": retrieved}
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
