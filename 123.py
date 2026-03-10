#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bioasqvs_adapted.py
A hardened BioASQ runner (v3.2) that loads pre-built FAISS indexes
from 01_fetch_corpus.py + 02_build_index.py instead of downloading
papers at runtime.

CRITICAL: EMBED_MODEL_NAME MUST match the model used in 02_build_index.py.

Usage:
    python bioasqvs_adapted.py <bioasq_dataset.json> [max_questions]
"""

from __future__ import annotations

import json, re, sys, time, uuid, hashlib, logging, pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL    = "gemma-3-27b-it"
LLM_API_KEY  = "EMPTY"

# ⚠ MUST match the model used when 02_build_index.py was run
# If you ran 02_build_index.py with intfloat/e5-base-v2 → keep this
# If you used all-MiniLM-L6-v2 → change to that and rebuild index
EMBED_MODEL_NAME = "intfloat/e5-base-v2"

DENSE_TOP_K         = 12
DENSE_FETCH_K       = 80
FINAL_TOP_K         = 8
HYDE_TRIGGER_CHARS  = 80
MIN_SNIPPETS_FOR_DIRECT = 3

MAX_RETRIEVAL_LOOPS = 4
MAX_REWRITE_LOOPS   = 2
MAX_DECOMPOSE_LOOPS = 2
MAX_REFLECT_LOOPS   = 2

# Pre-built index paths (output of 02_build_index.py)
INDEX_DIR     = Path("./index")
DATA_DIR      = Path("./data")

MEMORY_FAISS_DIR       = Path("./memory_faiss")
ENABLE_LONGTERM_MEMORY = False
STRICT_ID_ONLY         = False
MEMORY_FETCH_K         = 6
MEMORY_TOP_K           = 3

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Shared lazy objects
embeddings:          Optional[HuggingFaceEmbeddings] = None
cross_encoder                                        = None
papers_vs:           Optional[FAISS]                 = None
snippets_vs:         Optional[FAISS]                 = None
papers_bm25_index                                    = None
snippets_bm25_index                                  = None
DOCID_TO_CHUNKS:     Dict[str, List[Document]]       = {}
paper_metadata:      Dict[str, Dict[str, Any]]       = {}
memory_vs:           Optional[FAISS]                 = None


# ═══════════════════════════════════════════════════════════════════
# LAZY LOADERS
# ═══════════════════════════════════════════════════════════════════
def _get_embeddings() -> HuggingFaceEmbeddings:
    global embeddings
    if embeddings is None:
        log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return embeddings


def _get_cross_encoder():
    global cross_encoder
    if cross_encoder is None and CROSS_ENCODER_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        except Exception as e:
            log.warning("Cross-encoder unavailable: %s", e)
    return cross_encoder


def _make_llm(temperature: float = 0.1, max_tokens: int = 1024,
              disable_tools: bool = True) -> ChatOpenAI:
    extra_body = {"tool_choice": "none"} if disable_tools else {}
    return ChatOpenAI(
        base_url    = LLM_BASE_URL,
        api_key     = LLM_API_KEY,
        model       = LLM_MODEL,
        temperature = temperature,
        max_tokens  = max_tokens,
        extra_body  = extra_body,
    )


llm_answer = _make_llm(temperature=0.2, max_tokens=2048, disable_tools=True)
llm_grader = _make_llm(temperature=0.0, max_tokens=512,  disable_tools=True)
llm_query  = _make_llm(temperature=0.3, max_tokens=512,  disable_tools=True)


# ═══════════════════════════════════════════════════════════════════
# INDEX LOADING  (replaces build_all_indices + download_papers)
# ═══════════════════════════════════════════════════════════════════
def _chunks_pkl_to_docs(chunks: List[dict]) -> List[Document]:
    docs = []
    for i, c in enumerate(chunks):
        pmid = c.get("pmid", "")
        docs.append(Document(
            page_content=c.get("text", ""),
            metadata={
                "_id"         : pmid,
                "chunk_index" : i,
                "total_chunks": len(chunks),
                "title"       : c.get("title", "")[:100],
                "url"         : f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                "section"     : c.get("section", ""),
                "source"      : c.get("source", ""),
            },
        ))
    return docs


def _rebuild_faiss_from_raw(faiss_path: str, docs: List[Document]) -> "FAISS":
    """
    Reconstruct a LangChain FAISS from a raw faiss index file + Documents.

    02_build_index.py saves:
      index/papers_hnsw.faiss   <- raw faiss index (faiss.write_index format)
      index/papers_chunks.pkl   <- list of chunk dicts  (NOT LangChain docstore)

    LangChain's FAISS.load_local() expects a companion .pkl in its OWN docstore
    format, which does NOT exist here. So we reconstruct manually.
    """
    import faiss as faiss_lib
    from langchain_community.docstore.in_memory import InMemoryDocstore

    raw_index            = faiss_lib.read_index(faiss_path)
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    docstore             = InMemoryDocstore({str(i): d for i, d in enumerate(docs)})

    vs = FAISS.__new__(FAISS)
    vs.embedding_function   = _get_embeddings().embed_query
    vs.index                = raw_index
    vs.docstore             = docstore
    vs.index_to_docstore_id = index_to_docstore_id
    # HNSW inner-product (embeddings are L2-normalised in 02_build_index.py)
    try:
        from langchain_community.vectorstores.faiss import dependable_faiss_import
        _faiss = dependable_faiss_import()
        vs.distance_strategy = _faiss.METRIC_INNER_PRODUCT
    except Exception:
        vs.distance_strategy = "EUCLIDEAN_DISTANCE"
    return vs


def load_prebuilt_indices() -> None:
    global papers_vs, snippets_vs, papers_bm25_index, snippets_bm25_index
    global DOCID_TO_CHUNKS, paper_metadata, memory_vs

    # ── Papers ────────────────────────────────────────────────────
    log.info("Loading papers FAISS index...")
    papers_chunks_path = INDEX_DIR / "papers_chunks.pkl"
    papers_faiss_path  = INDEX_DIR / "papers_hnsw.faiss"
    with open(papers_chunks_path, "rb") as f:
        raw_paper_chunks = pickle.load(f)
    paper_docs = _chunks_pkl_to_docs(raw_paper_chunks)
    papers_vs  = _rebuild_faiss_from_raw(str(papers_faiss_path), paper_docs)
    log.info("  Papers: %d chunks loaded", len(paper_docs))

    # ── Snippets ──────────────────────────────────────────────────
    log.info("Loading snippets FAISS index...")
    snippets_chunks_path = INDEX_DIR / "snippets_chunks.pkl"
    snippets_faiss_path  = INDEX_DIR / "snippets_hnsw.faiss"
    with open(snippets_chunks_path, "rb") as f:
        raw_snippet_chunks = pickle.load(f)
    snippet_docs = _chunks_pkl_to_docs(raw_snippet_chunks)
    snippets_vs  = _rebuild_faiss_from_raw(str(snippets_faiss_path), snippet_docs)
    log.info("  Snippets: %d chunks loaded", len(snippet_docs))

    # BM25
    bm25_papers_path   = INDEX_DIR / "papers_bm25.pkl"
    bm25_snippets_path = INDEX_DIR / "snippets_bm25.pkl"
    if bm25_papers_path.exists():
        with open(bm25_papers_path, "rb") as f:
            papers_bm25_index = pickle.load(f)
    if bm25_snippets_path.exists():
        with open(bm25_snippets_path, "rb") as f:
            snippets_bm25_index = pickle.load(f)
    log.info("  BM25 indexes loaded")

    # DOCID_TO_CHUNKS
    all_docs: List[Document] = paper_docs + snippet_docs
    tmp: Dict[str, List[Document]] = {}
    for d in all_docs:
        pmid = (d.metadata or {}).get("_id", "")
        if pmid:
            tmp.setdefault(pmid, []).append(d)
    for pmid in tmp:
        tmp[pmid].sort(key=lambda d: int((d.metadata or {}).get("chunk_index", 0)))
    DOCID_TO_CHUNKS = tmp
    log.info("  DOCID_TO_CHUNKS: %d PMIDs", len(DOCID_TO_CHUNKS))

    # paper_metadata from papers.jsonl
    papers_jsonl = DATA_DIR / "papers.jsonl"
    if papers_jsonl.exists():
        with open(papers_jsonl) as f:
            for line in f:
                try:
                    p    = json.loads(line)
                    pmid = p.get("pmid", "")
                    if pmid:
                        paper_metadata[pmid] = {
                            "title"   : p.get("title", ""),
                            "abstract": (p.get("abstract") or "")[:500],
                            "url"     : f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}",
                        }
                except Exception:
                    pass
        log.info("  paper_metadata: %d entries", len(paper_metadata))

    # Long-term memory
    if ENABLE_LONGTERM_MEMORY and MEMORY_FAISS_DIR.exists():
        try:
            memory_vs = FAISS.load_local(
                str(MEMORY_FAISS_DIR), emb,
                allow_dangerous_deserialization=True,
            )
            log.info("Long-term memory loaded.")
        except Exception:
            memory_vs = None

    log.info("All indices loaded successfully.")


def load_bioasq_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        if isinstance(data, list):
            return data
        raise ValueError("Unexpected JSON structure")
    elif p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported file type: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL HELPERS
# ═══════════════════════════════════════════════════════════════════
CODE_RE = re.compile(r"~P\b[A-Z][0,5]\d{4,}(?:[A-Za-z]\d*)*\b")


def extract_codes(text: str) -> List[str]:
    return CODE_RE.findall(text or "")


def _parse_subquestions(blob: str) -> List[str]:
    if "SUBQUESTIONS:" in blob:
        qs = [l.strip().lstrip("- ") for l in blob.split("\n")
              if l.strip() and l.strip() != "SUBQUESTIONS:"]
        return qs if qs else [blob.strip()]
    return [blob.strip()]


def hyde_passage(question: str) -> str:
    try:
        resp = llm_query.invoke([{"role": "user", "content":
            f"Write a short factual paragraph that directly answers "
            f"this biomedical question: {question}"}])
        return (resp.content or "").strip()
    except Exception:
        return ""


def _unified_similarity_search(query: str, top_k: int,
                                fetch_k: int) -> List[Document]:
    """Search papers + snippets indexes in parallel and merge."""
    def _search(vs, q, k, fk):
        if vs is None:
            return []
        try:
            return vs.similarity_search(q, k=k, fetch_k=fk)
        except Exception:
            try:
                return vs.similarity_search(q, k=k)
            except Exception:
                return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fp = pool.submit(_search, papers_vs,   query, top_k, fetch_k)
        fs = pool.submit(_search, snippets_vs, query, top_k, fetch_k)
        return fp.result() + fs.result()


def _expand_neighbors(docs: List[Document], window: int = 1,
                      max_seed: int = 25) -> List[Document]:
    expanded = list(docs)
    for d in docs[:max_seed]:
        meta  = d.metadata or {}
        docid = meta.get("_id")
        idx   = meta.get("chunk_index")
        if docid is None or idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        chunks = DOCID_TO_CHUNKS.get(str(docid), [])
        if not chunks:
            continue
        lo = max(0, idx - window)
        hi = min(len(chunks), idx + window + 1)
        expanded.extend(chunks[lo:hi])
    return expanded


def _dedup_chunks(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.md5((d.page_content or "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def _rerank_single(query: str, docs: List[Document],
                   top_n: int = FINAL_TOP_K) -> List[Document]:
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs[:top_n]
    try:
        pairs  = [(query, d.page_content) for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


def _rerank_multi(queries: List[str], docs: List[Document],
                  top_n: int = FINAL_TOP_K) -> List[Document]:
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs[:top_n]
    try:
        score_acc = np.zeros(len(docs))
        for q in queries:
            pairs      = [(q, d.page_content) for d in docs]
            score_acc += np.array(ce.predict(pairs))
        score_acc /= max(len(queries), 1)
        ranked = sorted(zip(docs, score_acc), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


def _format_snippets(docs: List[Document], query: str,
                     limit: int = FINAL_TOP_K) -> str:
    blocks = []
    for i, d in enumerate(docs[:limit], start=1):
        meta      = d.metadata or {}
        pmid      = meta.get("_id", "")
        title     = (meta.get("title", "") or "")[:80]
        chunk_idx = meta.get("chunk_index", "?")
        total     = meta.get("total_chunks", "?")
        header    = f"[{i}] PMID={pmid} chunk={chunk_idx}/{total}"
        if title:
            header += f" | {title}"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_payload(docs: List[Document], query: str,
                     limit: int = FINAL_TOP_K) -> dict:
    snippets_text = _format_snippets(docs, query, limit)
    chunks = []
    for d in docs[:limit]:
        meta = d.metadata or {}
        chunks.append({
            "text"       : d.page_content,
            "pmid"       : meta.get("_id", ""),
            "chunk_index": meta.get("chunk_index", ""),
            "title"      : meta.get("title", ""),
        })
    return {"snippets_text": snippets_text, "chunks": chunks}


# ═══════════════════════════════════════════════════════════════════
# MAIN RETRIEVAL TOOL
# ═══════════════════════════════════════════════════════════════════
@tool("smart_retrieve_jsonl")
def smart_retrieve_jsonl(query: str) -> str:
    """
    Retrieve evidence snippets for a query from pre-built FAISS indexes.
    Returns a JSON string with keys:
      - snippets_text: formatted text blocks with [i] citations
      - chunks: list of {text, pmid, chunk_index, title}
    """
    subqs      = [q for q in _parse_subquestions(query) if q.strip()]
    full_query = " ".join(subqs)
    all_candidates: List[Document] = []

    for q in subqs:
        candidates: List[Document] = []
        codes = extract_codes(q)

        # 1) Code / ID pinpointing
        if codes:
            for pmid, chunks in DOCID_TO_CHUNKS.items():
                for chunk in chunks:
                    if all(c in (chunk.page_content or "") for c in codes):
                        candidates.append(chunk)
            if STRICT_ID_ONLY and candidates:
                all_candidates.extend(_dedup_chunks(candidates))
                continue

        # 2) Unified dense retrieval (papers + snippets in parallel)
        try:
            candidates.extend(
                _unified_similarity_search(q, top_k=DENSE_TOP_K,
                                           fetch_k=DENSE_FETCH_K)
            )
        except Exception as e:
            log.warning("Dense retrieval failed: %s", e)

        # 3) HyDE for short questions
        if len(q) <= HYDE_TRIGGER_CHARS and not codes:
            hyp = hyde_passage(q)
            if hyp:
                try:
                    candidates.extend(
                        _unified_similarity_search(hyp,
                            top_k=DENSE_TOP_K // 2,
                            fetch_k=DENSE_FETCH_K // 2)
                    )
                except Exception:
                    pass

        # 4) Neighbor expansion
        candidates = _expand_neighbors(_dedup_chunks(candidates), window=1)
        all_candidates.extend(_dedup_chunks(candidates))

    all_candidates = _dedup_chunks(all_candidates)
    log.info("  %d candidates for: %s", len(all_candidates), full_query[:80])

    # 5) Cross-encoder rerank
    if len(subqs) == 1:
        reranked = _rerank_single(subqs[0], all_candidates)
    else:
        reranked = _rerank_multi(subqs, all_candidates, top_n=FINAL_TOP_K)

    # 6) Prioritise exact-code hits
    codes = extract_codes(query)
    if codes:
        hits = [d for d in reranked
                if all(c in (d.page_content or "") for c in codes)]
        if hits:
            reranked = (hits if STRICT_ID_ONLY
                        else hits + [d for d in reranked if d not in hits])

    payload = _docs_to_payload(reranked, query=query, limit=FINAL_TOP_K)
    return json.dumps(payload, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# MEMORY STUBS
# ═══════════════════════════════════════════════════════════════════
def retrieve_longterm_memory(query: str, thread_id: str) -> str:
    if not (ENABLE_LONGTERM_MEMORY and memory_vs):
        return ""
    try:
        docs = memory_vs.similarity_search(query, k=MEMORY_FETCH_K)
    except Exception:
        return ""
    docs   = [d for d in docs if (d.metadata or {}).get("thread_id") == thread_id]
    blocks = [f"[M{i}] {(d.page_content or '')[:600]}"
              for i, d in enumerate(docs[:MEMORY_TOP_K], 1)]
    return "\n\n---\n\n".join(blocks)


def write_longterm_memory(question: str, answer: str, thread_id: str):
    global memory_vs
    if not ENABLE_LONGTERM_MEMORY:
        return
    mem_doc = Document(
        page_content=f"Q: {question}\nA: {answer}",
        metadata={"thread_id": thread_id, "ts": int(time.time())},
    )
    try:
        if memory_vs is None:
            memory_vs = FAISS.from_documents([mem_doc], _get_embeddings())
        else:
            memory_vs.add_documents([mem_doc])
        memory_vs.save_local(str(MEMORY_FAISS_DIR))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# AGENT HELPERS & PROMPTS  (identical to your original)
# ═══════════════════════════════════════════════════════════════════
AgentState = dict


def _latest_user_question(msgs: list) -> str:
    for m in reversed(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            return str(getattr(m, "content", "") or "")
    return ""


def _looks_multihop(q: str) -> bool:
    ql   = (q or "").lower(
