import streamlit as st
import torch
import os
import re
import json

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# ENV
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD VECTOR DB
# -----------------------------
@st.cache_resource
def load_vector_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# -----------------------------
# LOAD BM25
# -----------------------------
@st.cache_resource
def load_bm25():

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [c["content"] for c in chunks_data]

    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    bm25_corpus = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(bm25_corpus)

    return bm25, chunks_data


# -----------------------------
# LOAD RERANKER
# -----------------------------
@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-large", device=device)


# -----------------------------
# LOAD LLM
# -----------------------------
@st.cache_resource
def load_llm():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    prompt = PromptTemplate.from_template("""
You are a research assistant.

Answer the question using ONLY the context below.
If the answer cannot be found in the context, say "Not found in provided papers."

Context:
{context}

Question:
{question}

Return:

Answer:
<explanation>

Sources:
- Paper: <paper title> | Page: <page number>
""")

    chain = prompt | llm | StrOutputParser()
    return chain


# -----------------------------
# LOAD EVERYTHING
# -----------------------------
db = load_vector_db()
bm25, chunks_data = load_bm25()
reranker = load_reranker()
chain = load_llm()


# -----------------------------
# RETRIEVAL
# -----------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


def hybrid_retrieval(query, k_vector=80, k_bm25=80, rrf_k=30, top_n=40):
    """
    Hybrid retrieval using Vector + BM25 merged with Reciprocal Rank Fusion (RRF).
    """
    # Vector search
    vector_results = db.similarity_search(query, k=k_vector)

    # BM25 search
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k_bm25]

    # Reconstruct BM25 docs as plain text wrapped in a simple object
    class SimpleDoc:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata

    bm25_results = [
        SimpleDoc(chunks_data[i]["content"], chunks_data[i].get("metadata", {}))
        for i in top_bm25_indices
    ]

    # RRF fusion
    def doc_key(doc):
        md = getattr(doc, "metadata", {}) or {}
        key = md.get("id") or md.get("source") or md.get("title")
        if key:
            return f"{key}|{md.get('page', '')}"
        return str(hash(doc.page_content))

    scores = {}
    doc_map = {}

    for i, doc in enumerate(vector_results):
        k = doc_key(doc)
        doc_map[k] = doc
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + (i + 1))

    for i, doc in enumerate(bm25_results):
        k = doc_key(doc)
        if k not in doc_map:
            doc_map[k] = doc
        scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + (i + 1))

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    ranked_docs = [doc_map[k] for k in ranked_keys][:top_n]

    return ranked_docs


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📚 AI Research RAG Assistant")

query = st.text_input("Ask a question about AI research papers")

if query:

    with st.spinner("Searching..."):

        docs = hybrid_retrieval(query)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:5]]

        context = "\n\n".join([
            f"Paper: {doc.metadata.get('title')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in top_docs
        ])

        answer = chain.invoke({
            "context": context,
            "question": query
        })

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Top Retrieved Chunks")
        for i, doc in enumerate(top_docs):
            st.markdown(f"**Chunk {i+1}** — {doc.metadata.get('title', 'Unknown')} | Page {doc.metadata.get('page', '?')}")
            st.write(doc.page_content[:500])
