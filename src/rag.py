"""RAG pipeline with retrieval, grounded answer generation, and refusal gate.
Includes answer_no_rag for baseline comparison in evaluation.
"""
import sys
import re
import json
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

from src.config import INDEX_DIR, EMBED_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K, MIN_SCORE

# Global state (lazy loaded)
_embedder = None
_index = None
_chunks = None
_client = None

REFUSAL_ANSWER = "I don't know based on the PM-USP CSSS guidelines I have."

SYSTEM_PROMPT = """You are an assistant that answers questions about the PM-USP CSSS scholarship using ONLY the provided context from the official guidelines.

Rules:
- Answer ONLY from the provided context
- If the context does not contain the answer, say EXACTLY: "I don't know based on the PM-USP CSSS guidelines I have."
- Keep answers short and clear
- Quote amounts and percentages exactly as written
- Cite section numbers in the form [Section N]
- NEVER invent dates, deadlines, links, or amounts
- NEVER use outside knowledge about other scholarships or schemes
- DO NOT make assumptions beyond what the context explicitly states"""


def load_index():
    """Load FAISS index, chunks, and embedding model (cached)."""
    global _embedder, _index, _chunks

    if _index is not None:
        return

    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found. Run 'python -m src.ingest' first.\n"
            f"Expected: {index_path} and {chunks_path}"
        )

    _index = faiss.read_index(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    _embedder = SentenceTransformer(EMBED_MODEL)


def get_client():
    """Get OpenAI client (cached)."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def retrieve(question, k=TOP_K):
    """Retrieve top-k chunks for a question.

    Returns:
        List of dicts with keys: chunk, score
        Sorted by score descending
    """
    load_index()

    # For bge models: add query prefix
    query_text = f"Represent this sentence for searching relevant passages: {question}"

    query_embedding = _embedder.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = _index.search(query_embedding.astype('float32'), k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for invalid indices
            continue
        chunk = _chunks[idx]
        results.append({
            "chunk": chunk,
            "score": float(score)
        })

    return results


def call_llm(messages, model=None, max_tokens=1500):
    """Call LLM, retry once with more tokens if the reply is empty or truncated,
    and strip reasoning blocks.

    Returns:
        dict with keys: answer (str), model_used (str)
    """
    client = get_client()
    model = model or LLM_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
    except Exception as e:
        # Check if router is unreachable
        if "Connection" in str(e) or "connect" in str(e).lower():
            raise RuntimeError(
                "Could not connect to LLM router. Please start omniroute:\n"
                "  omniroute start\n"
                f"Expected URL: {LLM_BASE_URL}"
            ) from e
        raise

    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason

    # Retry once with higher max_tokens if truncated
    if (not content or finish_reason == "length") and max_tokens < 3000:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=3000,
            temperature=0
        )
        content = response.choices[0].message.content or ""

    # Strip reasoning blocks (e.g., <think>...</think>)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()

    model_used = getattr(response, 'model', model)

    return {
        "answer": content,
        "model_used": model_used
    }


def is_model_refusal(answer_text):
    """True if the model's own answer is an "I don't know" refusal.

    Normalizes curly apostrophes (don’t) to straight ones (don't) first.
    """
    text = answer_text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return "i don't know" in text or "i do not know" in text


def answer(question, model=None):
    """Answer a question using RAG.

    Returns:
        dict with keys:
            - answer (str)
            - sources (list of dicts with section, clause, score, snippet)
            - grounded (bool): False if the score gate fired or the model refused
            - top_score (float)
            - model_used (str or None; None means the score gate refused
              without calling the LLM)
    """
    results = retrieve(question)

    if not results or results[0]["score"] < MIN_SCORE:
        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "grounded": False,
            "top_score": results[0]["score"] if results else 0.0,
            "model_used": None
        }

    # Build context (send full chunks to LLM)
    context_parts = []
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        context_parts.append(f"[{i}] {chunk['text']}")

    context = "\n\n".join(context_parts)

    user_message = f"""Context from PM-USP CSSS guidelines:

{context}

Question: {question}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    llm_response = call_llm(messages, model=model)

    # Build sources (the snippet is truncated for display only)
    sources = []
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        sources.append({
            "section": chunk["section_number"],
            "clause": chunk.get("clause"),
            "score": result["score"],
            "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        })

    answer_text = llm_response["answer"]

    return {
        "answer": answer_text,
        "sources": sources,
        "grounded": not is_model_refusal(answer_text),
        "top_score": results[0]["score"],
        "model_used": llm_response["model_used"]
    }


def answer_no_rag(question, model=None):
    """Answer without RAG (baseline for evaluation).

    Returns:
        dict with keys: answer (str), model_used (str)
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question."},
        {"role": "user", "content": question}
    ]

    return call_llm(messages, model=model, max_tokens=1500)