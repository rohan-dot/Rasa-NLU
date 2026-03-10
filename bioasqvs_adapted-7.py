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

MAX_RETRIEVAL_LOOPS = 3
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
                str(MEMORY_FAISS_DIR), _get_embeddings(),
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
                                fetch_k: int = 0) -> List[Document]:
    """Search papers + snippets indexes in parallel and merge.
    fetch_k param kept for signature compatibility but ignored —
    fetch_k is MMR-only and throws TypeError on plain similarity_search.
    """
    def _search(vs, name, q, k):
        if vs is None:
            log.warning("  %s index is None — skipping", name)
            return []
        try:
            results = vs.similarity_search(q, k=k)
            log.info("  %s search returned %d docs", name, len(results))
            return results
        except Exception as e:
            log.error("  %s similarity_search failed: %s: %s",
                      name, type(e).__name__, e)
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fp = pool.submit(_search, papers_vs,   "papers",   query, top_k)
        fs = pool.submit(_search, snippets_vs, "snippets", query, top_k)
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
    ql   = (q or "").lower()
    cues = [" and ", " vs ", " compare ", " both ", " either ", " steps ",
            " between ", " then ", " after ", " relationship ", " mechanism "]
    return any(c in ql for c in cues) or q.count(",") >= 2 or len(q.split()) >= 18


def _looks_like_followup(q: str) -> bool:
    ql       = (q or "").strip().lower()
    pronouns = ("it", "they", "that", "those", "these", "he", "she", "this", "there")
    starts   = ("and ", "also ", "what about", "how about", "then ", "so ")
    short    = len(ql.split()) <= 10
    return ql.startswith(starts) or (short and any(p in ql.split()[:3] for p in pronouns))


def _history_to_str(msgs: list, max_turns: int = 8) -> str:
    out, turns = [], 0
    for m in reversed(msgs[:-1]):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            out.append(f"User: {m.content}")
            turns += 1
        elif role in ("ai", "assistant"):
            out.append(f"Assistant: {m.content}")
        if turns >= max_turns:
            break
    return "\n".join(reversed(out))


def _clean_model_text(t: str) -> str:
    return (t or "").replace("<|eot_id|>", "").strip()


def _context_signals(ctx: str) -> Dict[str, Any]:
    parts  = [p for p in re.split(r"\n\s*---\s*\n", ctx or "") if p.strip()]
    docids = set(re.findall(r"PMID=([^\s]+)", ctx or ""))
    return {"snippet_count": len(parts), "unique_docids": len(docids)}


CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the conversation and a follow up question, rewrite the follow up question "
    "to be a standalone question. Preserve any IDs/codes exactly.\n\n"
    "Conversation:\n{chat_history}\n\nFollow up question: {question}\n\nStandalone question:"
)

PLAN_PROMPT = PromptTemplate.from_template(
    "Decompose the question into 2-4 independent sub-questions that, when answered and combined, "
    "solve the original. Keep IDs/codes EXACT. Return JSON with key subquestions.\n\nQuestion: {q}"
)

GRADE_PROMPT = PromptTemplate.from_template(
    "You are a retrieval supervisor for a biomedical RAG system.\n"
    "Given the user question and retrieved snippets.\n\n"
    "Actions:\n"
    "- ANSWER: snippets are clearly relevant and contain the needed details.\n"
    "- DECOMPOSE: question is multi-hop / multiple parts and snippets cover only partial pieces.\n"
    "- REWRITE: snippets are off-topic OR the query is phrased poorly.\n\n"
    "Return ONLY valid JSON with keys: relevance, coverage, conflict, action.\n\n"
    "Question:\n{q}\n\n"
    "Signals: snippet_count={snippet_count} unique_docids={unique_docids}\n\n"
    "Snippets:\n{ctx}"
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "Rewrite the question to be clearer and retrieval-friendly WITHOUT changing intent. "
    "Preserve any IDs/codes exactly. Output ONLY the rewritten question.\n\nOriginal: {q}"
)

DECOMP_PROMPT = PromptTemplate.from_template(
    "Decompose the question into 2-4 independent sub-questions that, when answered and combined, "
    "solve the original. Keep IDs/codes EXACT. Return JSON with key subquestions.\n\nOriginal: {q}"
)

GENERATE_PROMPT = PromptTemplate.from_template(
    "You are a biomedical question-answering expert. Answer using ONLY the retrieved PubMed/PMC snippets below.\n\n"
    "Rules:\n"
    "- Only use information explicitly stated in the snippets.\n"
    "- If the answer is not in the snippets, say 'Not found in the provided evidence.'\n"
    "- Cite evidence as [1], [2], etc. matching snippet numbers.\n"
    "- Answer in your own words, do not paste long quotes.\n"
    "- For factoid questions: give the specific answer entity first, then evidence.\n"
    "- For list questions: provide all items found, cite each.\n"
    "- For yes/no questions: answer yes/no first, then explain.\n"
    "- For summary questions: synthesize a concise paragraph from all relevant snippets.\n\n"
    "Question: {question}\n\n"
    "Conversation memory (may help resolve pronouns; not authoritative):\n{memory}\n\n"
    "Retrieved snippets:\n{context}\n\n"
    "Your answer:"
)

REFLECT_PROMPT = PromptTemplate.from_template(
    "You are a QA controller for a biomedical RAG system.\n"
    "Given the question, retrieved snippets, and a draft answer:\n"
    "- Judge if the draft is grounded in the snippets and complete.\n"
    "- If incomplete, propose up to 3 improved retrieval queries.\n"
    "- If info must come from the user, propose ONE clarifying question.\n"
    "Return ONLY valid JSON with keys: grounded, complete, missing, suggested_queries, action.\n"
    "Actions: FINALIZE, RETRIEVE_MORE, ASK_USER\n\n"
    "Question:\n{q}\n\nSnippets:\n{ctx}\n\nDraft:\n{a}"
)


class Plan(BaseModel):
    subquestions: List[str] = Field(min_length=1, max_length=5)

class RetrievalGrade(BaseModel):
    relevance: float = Field(ge=0, le=1)
    coverage:  float = Field(ge=0, le=1)
    conflict:  bool  = False
    action:    Literal["ANSWER", "DECOMPOSE", "REWRITE"]

class AnswerReflection(BaseModel):
    grounded:          float     = Field(ge=0, le=1)
    complete:          float     = Field(ge=0, le=1)
    missing:           List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list, max_length=4)
    action:            Literal["FINALIZE", "RETRIEVE_MORE", "ASK_USER"]


# ═══════════════════════════════════════════════════════════════════
# GRAPH NODES  (same as your original)
# ═══════════════════════════════════════════════════════════════════
def prepare_question(state: dict, config: Optional[dict] = None) -> dict:
    msgs      = state.get("messages", [])
    q         = _latest_user_question(msgs)
    config    = config or {}
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    history   = _history_to_str(msgs)
    standalone = q
    if history and _looks_like_followup(q):
        try:
            standalone = llm_query.invoke([{"role": "user", "content":
                CONDENSE_PROMPT.format(chat_history=history, question=q)}]
            ).content.strip()
        except Exception:
            standalone = q
    mem = retrieve_longterm_memory(standalone, thread_id=thread_id)
    return {
        "question"            : q,
        "standalone_question" : standalone,
        "memory_context"      : mem,
        "plan"                : [],
        "context"             : "",
        "draft_answer"        : "",
        "reflection"          : "",
        "reflect_action"      : "FINALIZE",
        "reflect_queries"     : [],
        "retrieved_chunks"    : [],
        "_retrieval_loops"    : 0,
        "_rewrite_loops"      : 0,
        "_decompose_loops"    : 0,
        "_reflect_loops"      : 0,
    }


def plan_question(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    if extract_codes(q) or not _looks_multihop(q):
        return {"plan": []}
    try:
        plan = llm_query.with_structured_output(Plan).invoke(
            [{"role": "user", "content": PLAN_PROMPT.format(q=q)}]
        )
        subs = [s.strip() for s in plan.subquestions if s.strip()]
        return {"plan": subs if len(subs) >= 2 else []}
    except Exception:
        return {"plan": []}


def call_retriever(state: dict) -> dict:
    q    = state.get("standalone_question") or state.get("question") or ""
    plan = state.get("plan") or []
    if plan and len(plan) >= 2:
        query_blob = "SUBQUESTIONS:\n" + "\n".join(f"- {s}" for s in plan)
    else:
        query_blob = q
    tool_id = uuid.uuid4().hex
    msg     = AIMessage(content="", tool_calls=[{
        "name": "smart_retrieve_jsonl",
        "args": {"query": query_blob},
        "id"  : tool_id,
    }])
    loops = state.get("_retrieval_loops", 0) + 1
    return {"messages": [msg], "_retrieval_loops": loops}


def capture_context(state: dict) -> dict:
    msgs     = state.get("messages", [])
    ctx      = ""
    chunks: List[Dict[str, Any]] = []
    tool_msg = None
    for m in reversed(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role == "tool":
            tool_msg = m
            break
    if tool_msg is not None:
        raw = str(getattr(tool_msg, "content", "") or "")
        try:
            obj    = json.loads(raw)
            ctx    = obj.get("snippets_text", raw)
            chunks = obj.get("chunks", [])
        except Exception:
            ctx = raw
    return {"context": ctx, "retrieved_chunks": chunks}


def grade_retrieval(state: dict) -> Literal["generate_draft", "rewrite_question", "decompose_question"]:
    q   = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    sig = _context_signals(ctx)

    log.info("grade_retrieval: loops=%d snippet_count=%d unique_docids=%d",
             state.get("_retrieval_loops", 0), sig["snippet_count"], sig["unique_docids"])

    # Hard stop on too many loops — always generate with whatever we have
    if state.get("_retrieval_loops", 0) >= MAX_RETRIEVAL_LOOPS:
        log.info("grade: hit MAX_RETRIEVAL_LOOPS → generate_draft")
        return "generate_draft"

    # Good retrieval: enough snippets → go generate immediately, no LLM call needed
    if sig["snippet_count"] >= MIN_SNIPPETS_FOR_DIRECT:
        log.info("grade: %d snippets → generate_draft", sig["snippet_count"])
        return "generate_draft"

    # Poor retrieval: ask LLM grader whether to rewrite or decompose
    try:
        resp = llm_grader.with_structured_output(RetrievalGrade).invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(q=q, ctx=ctx, **sig)}]
        )
        log.info("grade: LLM says action=%s relevance=%.2f coverage=%.2f",
                 resp.action, resp.relevance, resp.coverage)
    except Exception as e:
        log.warning("grade: LLM grader failed (%s) → generate_draft", e)
        return "generate_draft"

    if resp.action == "DECOMPOSE" and state.get("_decompose_loops", 0) < MAX_DECOMPOSE_LOOPS:
        return "decompose_question"
    if resp.action == "REWRITE" and state.get("_rewrite_loops", 0) < MAX_REWRITE_LOOPS:
        return "rewrite_question"
    return "generate_draft"


def rewrite_question(state: dict) -> dict:
    q             = state.get("standalone_question") or state.get("question") or ""
    rewrite_loops = state.get("_rewrite_loops", 0) + 1
    codes         = extract_codes(q)
    if codes:
        rewritten = f"{'  '.join(codes)} biomedical mechanism function role"
        return {"standalone_question": rewritten, "plan": [], "_rewrite_loops": rewrite_loops}
    try:
        rewritten = llm_query.invoke([{"role": "user", "content":
            REWRITE_PROMPT.format(q=q)}]).content.strip().strip('"')
        return {"standalone_question": rewritten, "plan": [], "_rewrite_loops": rewrite_loops}
    except Exception:
        return {"standalone_question": q, "plan": [], "_rewrite_loops": rewrite_loops}


def decompose_question(state: dict) -> dict:
    q               = state.get("standalone_question") or state.get("question") or ""
    decompose_loops = state.get("_decompose_loops", 0) + 1
    if extract_codes(q):
        return {"plan": [f"What is the status/decision for {c}?" for c in extract_codes(q)],
                "_decompose_loops": decompose_loops}
    try:
        plan = llm_query.with_structured_output(Plan).invoke(
            [{"role": "user", "content": DECOMP_PROMPT.format(q=q)}]
        )
        subs = [s.strip() for s in plan.subquestions if s.strip()]
        return {"plan": subs if len(subs) >= 2 else [q], "_decompose_loops": decompose_loops}
    except Exception:
        return {"plan": [q], "_decompose_loops": decompose_loops}


def generate_draft(state: dict) -> dict:
    q   = state.get("question") or ""
    ctx = state.get("context") or ""
    mem = state.get("memory_context") or ""
    sys_msg = SystemMessage(content="You are a biomedical expert. Ground every claim in the provided snippets.")
    prompt  = GENERATE_PROMPT.format(question=q, context=ctx, memory=mem)
    try:
        resp  = llm_answer.invoke([sys_msg, HumanMessage(content=prompt)])
        draft = (getattr(resp, "content", "") or "").strip()
    except Exception as e:
        draft = f"LLM error: {type(e).__name__}: {e}"
    if not draft:
        sig   = _context_signals(ctx)
        draft = (
            "LLM returned empty content.\n"
            f"Debug: snippet_count={sig['snippet_count']} unique_docids={sig['unique_docids']}\n"
            "Fix: run vLLM without tool auto-choice/template, "
            "and keep tool_choice='none' in this script."
        )
    return {"draft_answer": draft}


def reflect(state: dict) -> dict:
    q             = state.get("standalone_question") or state.get("question") or ""
    ctx           = state.get("context") or ""
    draft         = state.get("draft_answer") or ""
    reflect_loops = state.get("_reflect_loops", 0) + 1
    if reflect_loops > MAX_REFLECT_LOOPS:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Max reflection loops.", "_reflect_loops": reflect_loops}
    try:
        r = llm_grader.with_structured_output(AnswerReflection).invoke(
            [{"role": "user", "content": REFLECT_PROMPT.format(q=q, ctx=ctx, a=draft)}]
        )
        return {
            "reflect_action" : r.action,
            "reflect_queries": r.suggested_queries,
            "reflection"     : f"grounded={r.grounded} complete={r.complete} missing={r.missing}",
            "_reflect_loops" : reflect_loops,
        }
    except Exception:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Reflection failed.", "_reflect_loops": reflect_loops}


def route_after_reflect(state: dict) -> Literal["finalize_answer", "call_retriever", "ask_user"]:
    action = state.get("reflect_action", "FINALIZE")
    if action == "RETRIEVE_MORE" and (state.get("reflect_queries") or []):
        if state.get("_retrieval_loops", 0) < MAX_RETRIEVAL_LOOPS:
            return "call_retriever"
    if action == "ASK_USER":
        return "ask_user"
    return "finalize_answer"


def apply_reflection_queries(state: dict) -> dict:
    queries = state.get("reflect_queries") or []
    if not queries:
        return {}
    if len(queries) >= 2:
        return {"plan": queries}
    return {"standalone_question": queries[0], "plan": []}


def finalize_answer(state: dict) -> dict:
    a   = _clean_model_text(state.get("draft_answer") or "")
    ctx = state.get("context") or ""
    sig = _context_signals(ctx)
    log.info("Finalizing: answer_len=%d snippet_count=%d unique_docids=%d",
             len(a), sig["snippet_count"], sig["unique_docids"])
    return {"messages": [AIMessage(content=a)]}


def ask_user(state: dict) -> dict:
    qs = state.get("reflect_queries") or []
    q  = qs[0] if qs else "Could you clarify your question?"
    return {"messages": [AIMessage(content=q)]}


def write_memory_node(state: dict, config: Optional[dict] = None) -> dict:
    config    = config or {}
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    write_longterm_memory(
        state.get("question") or "",
        state.get("draft_answer") or "",
        thread_id=thread_id,
    )
    return {}


# ═══════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════
def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("prepare_question",         prepare_question)
    workflow.add_node("plan_question",            plan_question)
    workflow.add_node("call_retriever",           call_retriever)
    workflow.add_node("retrieve",                 ToolNode([smart_retrieve_jsonl]))
    workflow.add_node("capture_context",          capture_context)
    workflow.add_node("rewrite_question",         rewrite_question)
    workflow.add_node("decompose_question",       decompose_question)
    workflow.add_node("generate_draft",           generate_draft)
    workflow.add_node("reflect",                  reflect)
    workflow.add_node("apply_reflection_queries", apply_reflection_queries)
    workflow.add_node("finalize_answer",          finalize_answer)
    workflow.add_node("ask_user",                 ask_user)
    workflow.add_node("write_memory",             write_memory_node)

    def route_to_planner(state: dict) -> Literal["plan_question", "call_retriever"]:
        q = state.get("standalone_question") or state.get("question") or ""
        return ("plan_question"
                if (not extract_codes(q) and _looks_multihop(q))
                else "call_retriever")

    workflow.add_edge(START, "prepare_question")
    workflow.add_conditional_edges("prepare_question", route_to_planner,
        {"plan_question": "plan_question", "call_retriever": "call_retriever"})
    workflow.add_edge("plan_question",  "call_retriever")
    workflow.add_edge("call_retriever", "retrieve")
    workflow.add_edge("retrieve",       "capture_context")
    workflow.add_conditional_edges("capture_context", grade_retrieval,
        {"generate_draft"    : "generate_draft",
         "rewrite_question"  : "rewrite_question",
         "decompose_question": "decompose_question"})
    workflow.add_edge("rewrite_question",   "call_retriever")
    workflow.add_edge("decompose_question", "call_retriever")
    workflow.add_edge("generate_draft",     "reflect")
    workflow.add_conditional_edges("reflect", route_after_reflect,
        {"finalize_answer": "finalize_answer",
         "call_retriever" : "apply_reflection_queries",
         "ask_user"       : "ask_user"})
    workflow.add_edge("apply_reflection_queries", "call_retriever")
    workflow.add_edge("finalize_answer", "write_memory")
    workflow.add_edge("write_memory",    END)
    workflow.add_edge("ask_user",        END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_final_answer_from_state(out: dict) -> str:
    msgs = out.get("messages") or []
    for m in reversed(msgs):
        role    = getattr(m, "type", getattr(m, "role", ""))
        content = (getattr(m, "content", "") or "").strip()
        if role in ("ai", "assistant") and content:
            return content
    return out.get("draft_answer") or ""


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════
def evaluate_bioasq(dataset_path: str, max_questions: int = None,
                    output_path: str = "bioasq_results.json"):
    questions = load_bioasq_dataset(dataset_path)
    log.info("Loaded %d questions from %s", len(questions), dataset_path)

    load_prebuilt_indices()   # <-- replaces build_all_indices()

    graph      = build_graph()
    total      = min(len(questions), max_questions) if max_questions else len(questions)
    results    = []

    # Debug single test
    try:
        log.info("DEBUG: single test invoke")
        test_out = graph.invoke(
            {"messages": [HumanMessage(content="Describe RankMHC")]},
            config={"recursion_limit": 60,
                    "configurable": {"thread_id": "debug-rankmhc"}},
        )
        log.info("DEBUG out keys: %s", list(test_out.keys()))
        log.info("DEBUG draft_answer len: %d", len(test_out.get("draft_answer") or ""))
        log.info("DEBUG messages types: %s",
                 [getattr(m, "type", getattr(m, "role", "")) for m in (test_out.get("messages") or [])])
        log.info("DEBUG extracted answer: %r", extract_final_answer_from_state(test_out))
    except Exception as e:
        log.warning("DEBUG invoke failed: %s", e)

    for i, q_data in enumerate(questions[:total]):
        q_body = q_data.get("body", "")
        q_type = q_data.get("type", "factoid")
        q_id   = q_data.get("id", f"q_{i}")
        ideal  = q_data.get("ideal_answer", [""])
        exact  = q_data.get("exact_answer", "")

        log.info("\n[%d/%d] Q(%s): %s", i + 1, total, q_type, q_body[:100])

        try:
            out    = graph.invoke(
                {"messages": [HumanMessage(content=q_body)]},
                config={"recursion_limit": 60,
                        "configurable": {"thread_id": f"bioasq-eval-{q_id}"}},
            )
            answer = _clean_model_text(extract_final_answer_from_state(out))
            if not answer.strip():
                msgs = out.get("messages") or []
                tail = []
                for m in msgs[-8:]:
                    role    = getattr(m, "type", getattr(m, "role", ""))
                    content = (getattr(m, "content", "") or "")
                    tail.append(f"{role}: {repr(content)[:300]}")
                answer = (
                    "EMPTY_PREDICTED_ANSWER\n"
                    f"out_keys={list(out.keys())}\n"
                    f"draft_len={len(out.get('draft_answer') or '')}\n"
                    f"context_len={len(out.get('context') or '')}\n"
                    f"reflect_action={out.get('reflect_action')}\n"
                    f"tail_messages={json.dumps(tail)}\n"
                )
        except Exception as e:
            log.error("Error on question %s: %s", q_id, e)
            answer = f"ERROR: {e}"

        log.info("  -> %s", answer[:200])
        results.append({
            "id"              : q_id,
            "type"            : q_type,
            "question"        : q_body,
            "predicted_answer": answer,
            "ideal_answer"    : ideal,
            "exact_answer"    : exact,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("\nSaved %d results to %s", len(results), output_path)
    return results


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        dataset_path = sys.argv[1]
        max_q        = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate_bioasq(dataset_path, max_questions=max_q)
    else:
        print("=" * 60)
        print("BioASQ Agentic QA System  v3.2 (pre-built indexes)")
        print("=" * 60)
        print("\nUsage:")
        print("  python bioasqvs_adapted.py <bioasq_dataset.json> [max_questions]")
        print("\nRequired index layout:")
        print("  index/papers_hnsw.faiss    index/papers_chunks.pkl")
        print("  index/snippets_hnsw.faiss  index/snippets_chunks.pkl")
        print("  data/papers.jsonl")
        print("\n  EMBED_MODEL_NAME must match model used in 02_build_index.py")
