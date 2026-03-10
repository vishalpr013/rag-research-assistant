# 📚 Research RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you chat with AI research papers. Built with hybrid retrieval (FAISS + BM25), CrossEncoder reranking, and a Groq LLM — all wrapped in a clean Streamlit UI.

---

## 🚀 What This Project Does

- Loads research papers (PDFs) and splits them into chunks stored in a FAISS vector index
- At query time: runs **hybrid retrieval** (vector search + BM25), fuses results with **RRF**, reranks with a **CrossEncoder**, then passes top-5 docs to an LLM for a grounded answer with citations
- Evaluated with real retrieval metrics: **MRR** and **documents found** over 50 QA pairs

---

## 🏗️ Architecture

```
                        ┌─────────────────────┐
                        │   Research Papers   │
                        │       (PDFs)        │
                        └────────┬────────────┘
                                 │
                                 ▼
                        ┌─────────────────────┐
                        │   PyMuPDFLoader     │
                        └────────┬────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │RecursiveCharacterTextSplitter│
                  │  chunk_size=600, overlap=120 │ 
                  └──────────┬───────────────┬───┘
                             │               │
                             ▼               ▼
                   ┌──────────────┐  ┌──────────────────┐
                   │  BM25 Index  │  │   FAISS Index    │
                   │  (Keyword)   │  │ (bge-large-en)   │
                   └──────┬───────┘  └────────┬─────────┘
                          │                   │
                          │    User Query     │
                          │        │          │
                          ▼        ▼          ▼
                   ┌──────────────────────────────┐
                   │   Reciprocal Rank Fusion     │
                   │          (RRF)               │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │     CrossEncoder Reranker    │
                   │    (bge-reranker-large)      │
                   │       → Top-5 Docs           │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │        Groq LLM              │
                   │   (llama-3.1-8b-instant)     │
                   │   LangChain LCEL Chain       │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Answer + Paper Citations    │
                   └──────────────────────────────┘
```

---

## 🗂️ Project Structure

```
research-rag-chatbot/
├── app.py               # Streamlit chatbot UI
├── build_index.py       # One-time script: load PDFs → chunk → BM25 + FAISS
├── evaluate_rag.py      # Evaluation pipeline (MRR, Recall@5, Semantic Similarity)
├── eval_questions.json  # 50 hand-crafted QA pairs for evaluation
├── chunks.json          # Text chunks saved after indexing
├── faiss_index/         # FAISS vector index (index.faiss + index.pkl)
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Groq API key
- Sign up free at [console.groq.com](https://console.groq.com)
- Create a `.env` file in this folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. FAISS Index
The `faiss_index/` folder is already included — **no need to rebuild it.** Just make sure `faiss_index/` and `chunks.json` are present in the project folder and you're good to go.

### 4. Run the chatbot
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 5. Run evaluation *(optional)*
```bash
python evaluate_rag.py
```
Results are saved to `evaluation_results.json`.

---

## 🧠 Tech Stack

| Layer | Tool / Model |
|---|---|
| PDF Loading | `PyMuPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` — chunk 600, overlap 120 |
| Embedding Model | `BAAI/bge-large-en-v1.5` (HuggingFace, GPU) |
| Vector Store | `FAISS` |
| Sparse Retrieval | `BM25Okapi` (rank-bm25) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `BAAI/bge-reranker-large` (CrossEncoder) |
| LLM | `llama-3.1-8b-instant` via **Groq API** |
| Framework | LangChain LCEL + Streamlit |
| Evaluation | `all-MiniLM-L6-v2` (Semantic Similarity) |

---

## 📊 Evaluation

Evaluated on **50 NLP/ML questions** using **MRR (Mean Reciprocal Rank)** and **documents found** to measure how well the retrieval pipeline surfaces relevant content.

---

## 🔑 API Keys Required

| Service | Where to get it |
|---|---|
| **Groq** (LLM) | [console.groq.com](https://console.groq.com) — free tier available |

> All embedding and reranking models run **locally** on your GPU (CUDA) or CPU — no additional API keys needed.
