"""
BioASQ 13b — Step 3 (Agentic): LangGraph RAG Pipeline
======================================================
Adapts your existing agentic QA pipeline to run over the
pre-built BioASQ FAISS + BM25 indexes from 02_build_index.py.

Features:
  - MMR dense retrieval (FAISS)
  - BM25 retrieval
  - EnsembleRetriever (dense + BM25)
  - MultiQueryRetriever (LLM generates n rephrasings)
  - HyDE (Hypothetical Document Embedding for short questions)
  - Cross-encoder reranker (multi-subquestion aware)
  - Neighbor chunk expansion (avoids partial answers from chunking)
  - Code/ID pinpointing
  - Retrieval grading → ANSWER / REWRITE / DECOMPOSE
  - Reflection loop (grounded + complete check)
  - Long-term memory FAISS store (per thread_id)
  - BioASQ question-type-aware answer formatting
  - Batch mode (full test set) + interactive single-query mode

Usage — interactive:
  python 03_agentic_pipeline.py --mode interactive --index index/

Usage — batch over test set:
  python 03_agentic_pipeline.py --mode batch \
    --test BioASQ-13b-testset.json \
    --index index/ \
    --out submissions/submission.json

Environment variables (all optional, have defaults):
  OPENAI_BASE_URL   http://127.0.0.1:8000/v1
  OPENAI_MODEL      gemma-3-27b-it
  OPENAI_API_KEY    (any string for local servers)
  EMBED_DEVICE      cuda:0
  EMBED_BATCH       8
  RERANK_MODEL      cross-encoder/qnli-electra-base
  RERANK_DEVICE     cpu
  ENABLE_LONGTERM_MEMORY  1
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, json, re, uuid, time, pickle, argparse
from pathlib import Path
from collections import defaultdict
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

# LangChain
from langchain.schema import Document
from langchain.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Pydantic
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════
LLM_BASE_URL  = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL     = os.environ.get("OPENAI_MODEL",    "gemma-3-27b-it")
EMBED_MODEL   = os.environ.get("EMBED_MODEL",     "intfloat/e5-base-v2")
EMBED_DEVICE  = os.environ.get("EMBED_DEVICE",    "cuda:0")
EMBED_BATCH   = int(os.environ.get("EMBED_BATCH", "8"))
RERANK_MODEL  = os.environ.get("RERANK_MODEL",    "cross-encoder/qnli-electra-base")
RERANK_DEVICE = os.environ.get("RERANK_DEVICE",   "cpu")

ENABLE_LONGTERM_MEMORY = os.environ.get("ENABLE_LONGTERM_MEMORY", "1").strip() not in ("0", "false", "False")
MEMORY_TOP_K  = int(os.environ.get("MEMORY_TOP_K",   "4"))
MEMORY_FETCH_K = int(os.environ.get("MEMORY_FETCH_K", "20"))

# Retrieval knobs
DENSE_TOP_K        = 20     # MMR candidates from FAISS
DENSE_FETCH_K      = 60     # over-fetch for MMR diversity
MMR_LAMBDA         = 0.5    # 0 = max diversity, 1 = max relevance
BM25_TOP_K         = 20
ENSEMBLE_WEIGHTS   = [0.6, 0.4]   # [dense, bm25]
NUM_MQR            = 3      # extra queries from MultiQueryRetriever
HYDE_TRIGGER_CHARS = 120    # use HyDE for questions shorter than this
RERANK_CANDIDATES  = 40     # candidates fed to cross-encoder
FINAL_TOP_K        = 8      # chunks in LLM context
SNIPPET_CHARS      = 600    # max chars per snippet in prompt
MAX_RETRIEVAL_LOOPS = 3

MIN_SNIPPETS_FOR_DIRECT    = 2
MIN_UNIQUE_DOCIDS_DIRECT   = 1
STRICT_ID_ONLY             = False   # if True, code-ID questions skip general retrieval

# Memory dirs
MEMORY_FAISS_DIR = "memory_faiss"

CODE_RE = re.compile(r"~[0-9A-Za-z]{10,}")


# ═══════════════════════════════════════════════════════════════════
# 2. LLM + EMBEDDING CLIENTS
# ═══════════════════════════════════════════════════════════════════
print("Loading LLM clients...")
_llm_kwargs = dict(
    base_url   = LLM_BASE_URL,
    api_key    = os.environ.get("OPENAI_API_KEY", "sk-local"),
    model      = LLM_MODEL,
    timeout    = 60,
    max_retries= 2,
    streaming  = True,
)
llm_answer  = ChatOpenAI(temperature=0.1, **_llm_kwargs)
llm_grader  = ChatOpenAI(temperature=0.0, **_llm_kwargs)
llm_query   = ChatOpenAI(temperature=0.0, **_llm_kwargs)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name   = EMBED_MODEL,
    model_kwargs = {"device": EMBED_DEVICE},
    encode_kwargs= {
        "normalize_embeddings": True,
        "batch_size"          : EMBED_BATCH,
    },
)


# ═══════════════════════════════════════════════════════════════════
# 3. INDEX LOADING  (called once at startup)
# ═══════════════════════════════════════════════════════════════════
def load_indexes(index_dir: str):
    """
    Load the FAISS + BM25 indexes built by 02_build_index.py.
    Returns LangChain-compatible retriever objects.
    """
    idx = Path(index_dir)
    print(f"Loading indexes from {idx} ...")

    # Load raw chunks
    with open(idx / "papers_chunks.pkl",   "rb") as f: paper_chunks   = pickle.load(f)
    with open(idx / "snippets_chunks.pkl", "rb") as f: snippet_chunks = pickle.load(f)

    # Convert to LangChain Documents
    def to_docs(chunks):
        return [
            Document(
                page_content=c["text"],
                metadata={
                    "_id"             : c["pmid"],
                    "chunk_id"        : f"{c['pmid']}::{i}",
                    "chunk_index"     : i,
                    "section"         : c.get("section", ""),
                    "section_priority": c.get("section_priority", 2),
                    "source"          : c.get("source", ""),
                    "title"           : c.get("title", ""),
                    "year"            : c.get("year", 0),
                    "mesh_terms"      : c.get("mesh_terms", []),
                },
            )
            for i, c in enumerate(chunks)
        ]

    paper_docs   = to_docs(paper_chunks)
    snippet_docs = to_docs(snippet_chunks)
    all_docs     = paper_docs + snippet_docs

    print(f"  paper docs: {len(paper_docs)} | snippet docs: {len(snippet_docs)}")

    # ── FAISS vector store (MMR) ──────────────────────────────────
    # Load the raw FAISS index and wrap with LangChain
    raw_paper_idx   = faiss.read_index(str(idx / "papers_hnsw.faiss"))
    raw_snippet_idx = faiss.read_index(str(idx / "snippets_hnsw.faiss"))

    # Merge into one LangFAISS for simplicity
    # (LangFAISS can load from_documents but that re-embeds; we load raw + attach docs)
    faiss_vs = LangFAISS.from_documents([], embeddings)   # empty shell
    faiss_vs.index            = faiss.read_index(str(idx / "papers_hnsw.faiss"))
    faiss_vs.index_to_docstore_id = {i: str(i) for i in range(len(paper_docs))}
    from langchain_community.docstore.in_memory import InMemoryDocstore
    faiss_vs.docstore = InMemoryDocstore({str(i): d for i, d in enumerate(paper_docs)})

    # Snippet FAISS (separate, searched in parallel)
    faiss_snip = LangFAISS.from_documents([], embeddings)
    faiss_snip.index = faiss.read_index(str(idx / "snippets_hnsw.faiss"))
    faiss_snip.index_to_docstore_id = {i: str(i) for i in range(len(snippet_docs))}
    faiss_snip.docstore = InMemoryDocstore({str(i): d for i, d in enumerate(snippet_docs)})

    # ── BM25 retrievers ───────────────────────────────────────────
    bm25_papers   = BM25Retriever.from_documents(paper_docs)
    bm25_papers.k = BM25_TOP_K

    bm25_snippets = BM25Retriever.from_documents(snippet_docs)
    bm25_snippets.k = BM25_TOP_K

    # ── Cross-encoder reranker ────────────────────────────────────
    ce_model = HuggingFaceCrossEncoder(
        model_name   = RERANK_MODEL,
        model_kwargs = {"device": RERANK_DEVICE},
    )
    reranker = CrossEncoderReranker(model=ce_model, top_n=FINAL_TOP_K)

    # ── Helper lookups ────────────────────────────────────────────
    # DOCID → all its chunks (for parent/neighbor expansion)
    DOCID_TO_CHUNKS: Dict[str, List[Document]] = defaultdict(list)
    CODE_TO_CHUNKS:  Dict[str, List[Document]] = defaultdict(list)

    for doc in all_docs:
        docid = doc.metadata.get("_id", "")
        if docid:
            DOCID_TO_CHUNKS[docid].append(doc)
        for code in extract_codes(doc.page_content):
            CODE_TO_CHUNKS[code].append(doc)

    # Sort each docid's chunks by chunk_index
    for docid in DOCID_TO_CHUNKS:
        DOCID_TO_CHUNKS[docid] = sorted(
            DOCID_TO_CHUNKS[docid],
            key=lambda d: int((d.metadata or {}).get("chunk_index", 0)),
        )

    print("Indexes loaded.")
    return (
        faiss_vs, faiss_snip,
        bm25_papers, bm25_snippets,
        reranker,
        DOCID_TO_CHUNKS, CODE_TO_CHUNKS,
        all_docs,
    )


# ═══════════════════════════════════════════════════════════════════
# 4. RETRIEVAL HELPERS
# ═══════════════════════════════════════════════════════════════════
def extract_codes(text: str) -> List[str]:
    return sorted(set(CODE_RE.findall(text or "")))


def _dedup_chunks(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        cid = (d.metadata or {}).get("chunk_id") or uuid.uuid4().hex
        if cid not in seen:
            out.append(d)
            seen.add(cid)
    return out


def _expand_neighbors(
    docs: List[Document],
    DOCID_TO_CHUNKS: Dict,
    window: int = 1,
    max_seed: int = 25,
) -> List[Document]:
    """Bring in adjacent chunks for context completeness."""
    expanded = list(docs)
    for d in docs[:max_seed]:
        meta  = d.metadata or {}
        docid = meta.get("_id", "")
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


def _code_candidates(q: str, CODE_TO_CHUNKS: Dict) -> List[Document]:
    codes = extract_codes(q)
    if not codes:
        return []
    lists = [CODE_TO_CHUNKS.get(c, []) for c in codes]
    if not lists or any(len(x) == 0 for x in lists):
        return []
    sets = []
    for lst in lists:
        s = {((d.metadata or {}).get("chunk_id") or ""): d for d in lst}
        sets.append(s)
    common = set(sets[0].keys())
    for s in sets[1:]:
        common &= set(s.keys())
    return [sets[0][cid] for cid in common if cid]


def _extract_relevant_span(text: str, query: str, max_chars: int = SNIPPET_CHARS) -> str:
    if not text:
        return ""
    codes  = extract_codes(query)
    lowered = text.lower()
    # Code-first match
    for c in codes:
        pos = text.find(c)
        if pos != -1:
            start = max(0, pos - max_chars // 2)
            end   = min(len(text), start + max_chars)
            span  = text[start:end]
            if start > 0:   span = "…" + span
            if end < len(text): span = span + "…"
            return span
    # Keyword window
    terms = [
        t for t in re.findall(r"[A-Za-z0-9_-]{4,}", query or "")
        if t.lower() not in {"what", "when", "where", "which", "that", "this",
                              "from", "with", "have", "does", "then", "than"}
    ]
    for t in terms:
        pos = lowered.find(t.lower())
        if pos != -1:
            start = max(0, pos - max_chars // 2)
            end   = min(len(text), start + max_chars)
            span  = text[start:end]
            if start > 0:   span = "…" + span
            if end < len(text): span = span + "…"
            return span
    span = text[:max_chars]
    if len(text) > max_chars:
        span += "…"
    return span


def _format_snippets(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> str:
    query_codes = extract_codes(query)
    blocks      = []
    for i, d in enumerate(docs[:limit], start=1):
        meta     = d.metadata or {}
        docid    = meta.get("_id", "")
        chunk_id = meta.get("chunk_id", "")
        text     = d.page_content or ""
        span     = _extract_relevant_span(text, query=query, max_chars=SNIPPET_CHARS)
        is_hit   = bool(query_codes) and all(c in text for c in query_codes)
        hit_flag = "HIT" if is_hit else "REL"
        sec      = meta.get("section", "")
        src      = meta.get("source", "")
        header   = f"[{i}] {hit_flag} docid={docid} chunk_id={chunk_id} sec={sec} src={src}"
        blocks.append(f"{header}\n{span}")
    return "\n\n---\n\n".join(blocks)


def _context_signals(ctx: str) -> Dict[str, Any]:
    parts         = [p for p in re.split(r"\n\s*---\s*\n", ctx or "") if p.strip()]
    snippet_count = len(parts)
    docids        = set(re.findall(r"docid=([^\s]+)", ctx or ""))
    return {"snippet_count": snippet_count, "unique_docids": len(docids)}


def _looks_multihop(q: str) -> bool:
    ql   = (q or "").lower()
    cues = [" and ", " vs ", " compare ", " both ", " either ",
            " steps ", " between ", " then ", " after "]
    return any(c in ql for c in cues) or q.count(",") >= 2 or len(q.split()) >= 18


def _looks_like_followup(q: str) -> bool:
    ql       = (q or "").strip().lower()
    pronouns = ("it", "they", "that", "those", "these", "he", "she", "this", "there")
    starts   = ("and ", "also ", "what about", "how about", "then ", "so ")
    short    = len(ql.split()) <= 10
    return ql.startswith(starts) or (short and any(p in ql.split()[:3] for p in pronouns))


def _history_to_str(msgs: List[BaseMessage], max_turns: int = 8) -> str:
    out    = []
    turns  = 0
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


def _retrieval_calls_this_turn(msgs: List[BaseMessage]) -> int:
    n = 0
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", ""))
        name = getattr(m, "name", "") or getattr(m, "tool", "")
        if role == "tool" and name == "smart_retrieve_bioasq":
            n += 1
    return n


def _parse_subquestions(blob: str) -> List[str]:
    if not (blob or "").strip().upper().startswith("SUBQUESTIONS:"):
        return [blob.strip()]
    qs: List[str] = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SUBQUESTIONS"):
            continue
        line = line.lstrip("-•0123456789. ").strip()
        if line:
            qs.append(line)
    return qs or [blob.strip()]


# ═══════════════════════════════════════════════════════════════════
# 5. MULTI-QUERY + HYDE PROMPTS
# ═══════════════════════════════════════════════════════════════════
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Generate {n} semantically different search queries for the same biomedical intent.\n"
        "Rules:\n"
        "- One query per line\n"
        "- No numbering, no bullets, no extra text\n"
        "- Preserve any IDs/codes exactly\n\n"
        "Question: {question}"
    ).format(n=NUM_MQR, question="{question}"),
)

HYDE_PROMPT = PromptTemplate.from_template(
    "Write a detailed, factual biomedical paragraph that would answer the question. "
    "Use terms likely to appear in PubMed abstracts or full-text papers.\n\nQuestion: {q}"
)


def hyde_passage(q: str) -> str:
    try:
        return llm_query.invoke(
            [{"role": "user", "content": HYDE_PROMPT.format(q=q)}]
        ).content
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════
# 6. CORE SMART RETRIEVER  (called as LangGraph ToolNode)
# ═══════════════════════════════════════════════════════════════════
# Globals populated by load_indexes()
_faiss_vs          = None
_faiss_snip        = None
_bm25_papers       = None
_bm25_snippets     = None
_reranker          = None
_DOCID_TO_CHUNKS   = None
_CODE_TO_CHUNKS    = None
_mqr               = None   # MultiQueryRetriever (lazy-init)
_hybrid            = None   # EnsembleRetriever

def _init_retrievers():
    global _mqr, _hybrid
    dense_mmr = _faiss_vs.as_retriever(
        search_type  = "mmr",
        search_kwargs= {
            "k"           : DENSE_TOP_K,
            "fetch_k"     : DENSE_FETCH_K,
            "lambda_mult" : MMR_LAMBDA,
        },
    )
    _hybrid = EnsembleRetriever(
        retrievers = [dense_mmr, _bm25_papers],
        weights    = list(ENSEMBLE_WEIGHTS),
    )
    _mqr = MultiQueryRetriever.from_llm(
        retriever        = _hybrid,
        llm              = llm_query,
        prompt           = MULTI_QUERY_PROMPT,
        include_original = True,
    )


def _invoke_retriever(retriever, query: str) -> List[Document]:
    try:
        return retriever.invoke(query)
    except Exception:
        return retriever.get_relevant_documents(query)


def smart_retrieve_bioasq(query: str) -> str:
    """
    Full hybrid retrieval pipeline:
      1. Code/ID pinpointing
      2. Multi-query + MMR (papers + snippets in parallel)
      3. HyDE for short questions
      4. Neighbor expansion
      5. Cross-encoder rerank (multi-subquestion aware)
      Returns JSON string with snippets_text + chunks list.
    """
    subqs          = _parse_subquestions(query)
    all_candidates : List[Document] = []

    for q in subqs:
        candidates: List[Document] = []

        # 1) Code / ID pinpointing
        codes = extract_codes(q)
        if codes:
            exact = _code_candidates(q, _CODE_TO_CHUNKS)
            if exact:
                candidates.extend(exact)
                docids = {(d.metadata or {}).get("_id", "") for d in exact}
                for docid in docids:
                    candidates.extend(_DOCID_TO_CHUNKS.get(docid, []))
                if STRICT_ID_ONLY:
                    all_candidates.extend(_dedup_chunks(candidates))
                    continue

        # 2) Multi-query + MMR (papers)
        try:
            candidates.extend(_invoke_retriever(_mqr, q))
        except Exception:
            candidates.extend(_invoke_retriever(_hybrid, q))

        # 2b) Snippet index (parallel)
        snip_dense = _faiss_snip.as_retriever(
            search_type  = "mmr",
            search_kwargs= {"k": DENSE_TOP_K, "fetch_k": DENSE_FETCH_K, "lambda_mult": MMR_LAMBDA},
        )
        snip_ensemble = EnsembleRetriever(
            retrievers = [snip_dense, _bm25_snippets],
            weights    = list(ENSEMBLE_WEIGHTS),
        )
        try:
            candidates.extend(_invoke_retriever(snip_ensemble, q))
        except Exception:
            pass

        # 3) HyDE for short questions
        if len(q) <= HYDE_TRIGGER_CHARS and not codes:
            hyp = hyde_passage(q)
            if hyp:
                candidates.extend(_invoke_retriever(_hybrid, hyp))

        # 4) Neighbor expansion
        candidates = _expand_neighbors(
            _dedup_chunks(candidates),
            _DOCID_TO_CHUNKS,
            window=1,
        )
        all_candidates.extend(_dedup_chunks(candidates))

    all_candidates = _dedup_chunks(all_candidates)

    # 5) Cross-encoder rerank (multi-subquestion aware)
    if len(subqs) == 1:
        reranked = _rerank_single(subqs[0], all_candidates)
    else:
        reranked = _rerank_multi(subqs, all_candidates, top_n=FINAL_TOP_K)

    # 6) If codes present, prioritise exact-code hits at top
    codes = extract_codes(query)
    if codes:
        hits = [d for d in reranked if all(c in (d.page_content or "") for c in codes)]
        if hits:
            reranked = hits if STRICT_ID_ONLY else hits + [d for d in reranked if d not in hits]

    payload = _docs_to_payload(reranked, query=query, limit=FINAL_TOP_K)
    return json.dumps(payload, ensure_ascii=False)


def _rerank_single(query: str, docs: List[Document]) -> List[Document]:
    docs = docs[:max(RERANK_CANDIDATES, FINAL_TOP_K)]
    try:
        return _reranker.compress_documents(documents=docs, query=query)
    except Exception:
        return docs[:FINAL_TOP_K]


def _rerank_multi(queries: List[str], docs: List[Document], top_n: int = FINAL_TOP_K) -> List[Document]:
    if not docs:
        return []
    pairs: List[Tuple[str, str]] = [(q, d.page_content or "") for q in queries for d in docs]
    try:
        scores = list(_reranker.model.score(pairs))
    except Exception:
        return _rerank_single(" | ".join(queries), docs)

    per_doc_max: List[float] = [float("-inf")] * len(docs)
    idx = 0
    for _q in queries:
        for di in range(len(docs)):
            try:
                s = float(scores[idx])
            except Exception:
                s = float("-inf")
            if s > per_doc_max[di]:
                per_doc_max[di] = s
            idx += 1

    ranked = [d for d, _s in sorted(zip(docs, per_doc_max), key=lambda x: x[1], reverse=True)]
    return ranked[:top_n]


def _docs_to_payload(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> Dict[str, Any]:
    items = []
    for i, d in enumerate(docs[:limit], start=1):
        meta     = d.metadata or {}
        docid    = meta.get("_id", "")
        chunk_id = meta.get("chunk_id", "")
        text     = d.page_content or ""
        span     = _extract_relevant_span(text, query=query, max_chars=SNIPPET_CHARS)
        items.append({
            "rank"       : i,
            "docid"      : docid,
            "chunk_id"   : chunk_id,
            "chunk_index": meta.get("chunk_index"),
            "snippet"    : span,
            "full_text"  : text,
            "metadata"   : meta,
        })
    return {
        "query"        : query,
        "snippets_text": _format_snippets(docs, query=query, limit=limit),
        "chunks"       : items,
    }


# ═══════════════════════════════════════════════════════════════════
# 7. LONG-TERM MEMORY
# ═══════════════════════════════════════════════════════════════════
memory_vs: Optional[LangFAISS] = None

if ENABLE_LONGTERM_MEMORY and Path(MEMORY_FAISS_DIR).exists():
    try:
        memory_vs = LangFAISS.load_local(
            MEMORY_FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("Long-term memory loaded.")
    except Exception:
        memory_vs = None


def retrieve_longterm_memory(query: str, thread_id: str) -> str:
    if not (ENABLE_LONGTERM_MEMORY and memory_vs):
        return ""
    try:
        docs = memory_vs.similarity_search(query, k=MEMORY_FETCH_K)
    except Exception:
        return ""
    docs = [d for d in docs if (d.metadata or {}).get("thread_id") == thread_id]
    return _format_memory(docs[:MEMORY_TOP_K])


def write_longterm_memory(question: str, answer: str, thread_id: str):
    global memory_vs
    if not ENABLE_LONGTERM_MEMORY:
        return
    mem_doc = Document(
        page_content = f"Q: {question}\nA: {answer}",
        metadata     = {"thread_id": thread_id, "ts": int(time.time())},
    )
    try:
        if memory_vs is None:
            memory_vs = LangFAISS.from_documents([mem_doc], embeddings)
        else:
            memory_vs.add_documents([mem_doc])
        memory_vs.save_local(MEMORY_FAISS_DIR)
    except Exception:
        pass


def _format_memory(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        ts = (d.metadata or {}).get("ts", "")
        blocks.append(f"[M{i}] ts={ts}\n{(d.page_content or '')[:600]}")
    return "\n\n---\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════
# 8. AGENT STATE
# ═══════════════════════════════════════════════════════════════════
class AgentState(dict, total=False):
    messages         : Annotated[List[BaseMessage], add_messages]
    question         : str
    standalone_question: str
    plan             : List[str]
    context          : str
    memory_context   : str
    retrieved_chunks : List[Dict[str, Any]]
    draft_answer     : str
    reflection       : str
    reflect_action   : Literal["FINALIZE", "RETRIEVE_MORE", "ASK_USER"]
    reflect_queries  : List[str]
    question_type    : str   # yesno | factoid | list | summary


# ═══════════════════════════════════════════════════════════════════
# 9. GRAPH NODE HELPERS
# ═══════════════════════════════════════════════════════════════════
def _latest_user_question(msgs: List[BaseMessage]) -> str:
    for m in reversed(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            return str(getattr(m, "content", "") or "")
    return ""


def _msgs_since_last_user(msgs: List[BaseMessage]) -> List[BaseMessage]:
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        role = getattr(msgs[i], "type", getattr(msgs[i], "role", ""))
        if role in ("human", "user"):
            last_user_idx = i
            break
    return msgs[last_user_idx:] if last_user_idx is not None else msgs


# ── Prompts ───────────────────────────────────────────────────────
CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the conversation and a follow-up question, rewrite the follow-up question "
    "to be a standalone question. Preserve any IDs/codes exactly.\n\n"
    "Conversation:\n{chat_history}\n\nFollow-up question: {question}\n\nStandalone question:"
)

PLAN_PROMPT = PromptTemplate.from_template(
    "Decompose the biomedical question into 2-4 independent sub-questions that, "
    "when answered and combined, solve the original. "
    "Keep IDs/codes EXACT. Return JSON with key subquestions.\n\nQuestion: {q}"
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "Rewrite this biomedical question to be clearer and retrieval-friendly "
    "WITHOUT changing intent. Preserve any IDs/codes exactly. "
    "Output ONLY the rewritten question.\n\nOriginal: {q}"
)

DECOMP_PROMPT = PromptTemplate.from_template(
    "Decompose the question into 2-4 independent sub-questions that, "
    "when answered and combined, solve the original. "
    "Keep IDs/codes EXACT. Return JSON with key subquestions.\n\nOriginal: {q}"
)

GRADE_PROMPT = PromptTemplate.from_template(
    "You are a retrieval supervisor for a BioASQ RAG system.\n"
    "Decide what to do next given the user question and retrieved snippets.\n\n"
    "Actions:\n"
    "- ANSWER: snippets are clearly relevant and contain the needed details.\n"
    "- DECOMPOSE: question is multi-hop / multiple parts and snippets cover only partial pieces.\n"
    "- REWRITE: snippets are off-topic OR the query is phrased poorly.\n"
    "Return ONLY valid JSON with keys: relevance, coverage, conflict, action.\n\n"
    "Important rule for ID/code questions:\n"
    "If the question contains a code like ~2001101... and at least one snippet is labeled HIT, "
    "prefer ANSWER even if coverage is partial.\n\n"
    "Question:\n{q}\n\n"
    "Signals: snippet_count={snippet_count} unique_docids={unique_docids}\n\n"
    "Snippets:\n{ctx}"
)

# BioASQ type-specific generation prompts
GENERATE_PROMPTS = {
    "yesno": PromptTemplate.from_template(
        "You are answering BioASQ yes/no questions using retrieved PubMed passages.\n"
        "Core grounding rules:\n"
        "- Only use information explicitly stated in the snippets.\n"
        "- Answer 'yes' or 'no' first, then explain with evidence.\n"
        "- Cite snippets like [1], [2].\n"
        "- If answer is not stated, say 'Not stated in provided snippets'.\n\n"
        "Output format:\n"
        "1) Answer: yes/no\n"
        "2) Evidence: bullet list of key claims with citations\n"
        "3) Not stated / Missing (if anything)\n"
        "4) References: list each cited snippet with docid and chunk_id\n\n"
        "User question: {question}\n\n"
        "Conversation memory (not authoritative):\n{memory}\n\n"
        "Snippets:\n{context}"
    ),
    "factoid": PromptTemplate.from_template(
        "You are answering BioASQ factoid questions using retrieved PubMed passages.\n"
        "Core grounding rules:\n"
        "- Only use information explicitly stated in the snippets.\n"
        "- Provide the single most specific answer entity.\n"
        "- Cite snippets like [1], [2].\n"
        "- If answer is not stated, say so explicitly.\n\n"
        "Output format:\n"
        "1) Answer: <exact entity>\n"
        "2) Synonyms/variants: <list if any>\n"
        "3) Evidence: bullet list with citations\n"
        "4) References: list each cited snippet\n\n"
        "User question: {question}\n\n"
        "Conversation memory:\n{memory}\n\n"
        "Snippets:\n{context}"
    ),
    "list": PromptTemplate.from_template(
        "You are answering BioASQ list questions using retrieved PubMed passages.\n"
        "Core grounding rules:\n"
        "- Only use information explicitly stated in the snippets.\n"
        "- Provide a complete list of all relevant entities.\n"
        "- Cite snippets like [1], [2] for each item.\n"
        "- If the list is incomplete, say so.\n\n"
        "Output format:\n"
        "1) List: bullet list of entities\n"
        "2) Evidence per item with citations\n"
        "3) Missing: what might be absent from corpus\n"
        "4) References\n\n"
        "User question: {question}\n\n"
        "Conversation memory:\n{memory}\n\n"
        "Snippets:\n{context}"
    ),
    "summary": PromptTemplate.from_template(
        "You are answering BioASQ summary questions using retrieved PubMed passages.\n"
        "Core grounding rules:\n"
        "- Only use information explicitly stated in the snippets.\n"
        "- Do not guess. If the answer is not stated, say 'Not stated in provided snippets'.\n"
        "- Do not paste long quotes. Answer in your own words.\n"
        "- For multi-hop questions: answer each sub-question in detail, then combine.\n"
        "- Cite snippets like [1], [2] for each key claim.\n"
        "- Ambiguity rule: if vague, ask ONE clarifying question.\n\n"
        "Output format:\n"
        "1) Answer: a direct response (3-5 sentences)\n"
        "2) Evidence: bullet list of key claims with citations\n"
        "3) Not stated / Missing\n"
        "4) References: list each cited snippet with docid and chunk_id\n\n"
        "User question: {question}\n\n"
        "Conversation memory (may help resolve pronouns; not authoritative):\n{memory}\n\n"
        "Snippets:\n{context}"
    ),
}

REFLECT_PROMPT = PromptTemplate.from_template(
    "You are a QA controller for a BioASQ RAG system.\n"
    "Given the question, retrieved snippets, and a draft answer:\n"
    "- Judge if the draft is grounded in the snippets and complete for the question.\n"
    "- If not grounded or incomplete, propose up to 3 improved retrieval queries "
    "that would likely find the missing info.\n"
    "- If the missing info is not in the corpus and must be provided by the user, "
    "propose ONE clarifying question.\n"
    "Return ONLY valid JSON with keys: grounded, complete, missing, suggested_queries, action.\n\n"
    "Question: {q}\n\nSnippets:\n{ctx}\n\nDraft answer:\n{a}"
)


# ═══════════════════════════════════════════════════════════════════
# 10. GRAPH NODES
# ═══════════════════════════════════════════════════════════════════

# ── Pydantic schemas for structured output ────────────────────────
class Plan(BaseModel):
    subquestions: List[str] = Field(min_length=1, max_length=5)

class RetrievalGrade(BaseModel):
    relevance : float = Field(ge=0, le=1)
    coverage  : float = Field(ge=0, le=1)
    conflict  : bool  = False
    action    : Literal["ANSWER", "DECOMPOSE", "REWRITE"]

class AnswerReflection(BaseModel):
    grounded         : float = Field(ge=0, le=1)
    complete         : float = Field(ge=0, le=1)
    missing          : List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list, max_length=4)
    action           : Literal["FINALIZE", "RETRIEVE_MORE", "ASK_USER"]


def prepare_question(state: AgentState, config) -> AgentState:
    msgs       = state.get("messages", [])
    q          = _latest_user_question(msgs)
    history    = _history_to_str(msgs)
    thread_id  = (config.get("configurable", {}) or {}).get("thread_id", "default")
    standalone = q
    if history and _looks_like_followup(q):
        try:
            standalone = llm_query.invoke(
                [{"role": "user", "content": CONDENSE_PROMPT.format(chat_history=history, question=q)}]
            ).content.strip()
        except Exception:
            standalone = q
    mem = retrieve_longterm_memory(standalone, thread_id=thread_id)
    # Infer question type from BioASQ test data if available, else default summary
    q_type = state.get("question_type", "summary")
    return {
        "question"          : q,
        "standalone_question": standalone,
        "memory_context"    : mem,
        "plan"              : [],
        "context"           : "",
        "draft_answer"      : "",
        "reflection"        : "",
        "reflect_action"    : "FINALIZE",
        "reflect_queries"   : [],
        "question_type"     : q_type,
    }


def plan_question(state: AgentState) -> AgentState:
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


def call_retriever(state: AgentState) -> AgentState:
    q    = state.get("standalone_question") or state.get("question") or ""
    plan = state.get("plan") or []
    if plan and len(plan) >= 2:
        query_blob = "SUBQUESTIONS:\n" + "\n".join(f"- {s}" for s in plan)
    else:
        query_blob = q
    tool_id = uuid.uuid4().hex
    msg = AIMessage(
        content    = "",
        tool_calls = [{"name": "smart_retrieve_bioasq", "args": {"query": query_blob}, "id": tool_id}],
    )
    return {"messages": [msg]}


def retrieve(state: AgentState) -> AgentState:
    """Execute the tool call produced by call_retriever."""
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if not last:
        return {}
    tool_calls = getattr(last, "tool_calls", [])
    results    = []
    for tc in tool_calls:
        if tc.get("name") == "smart_retrieve_bioasq":
            raw = smart_retrieve_bioasq(tc["args"]["query"])
            from langchain.schema.messages import ToolMessage
            results.append(ToolMessage(content=raw, tool_call_id=tc["id"], name="smart_retrieve_bioasq"))
    return {"messages": results} if results else {}


def capture_context(state: AgentState) -> AgentState:
    msgs   = state.get("messages", [])
    ctx    = ""
    chunks : List[Dict[str, Any]] = []
    if msgs:
        last = msgs[-1]
        role = getattr(last, "type", getattr(last, "role", ""))
        if role == "tool":
            raw = str(getattr(last, "content", "") or "")
            try:
                obj    = json.loads(raw)
                ctx    = obj.get("snippets_text", raw)
                chunks = obj.get("chunks", [])
            except Exception:
                ctx = raw
    return {"context": ctx, "retrieved_chunks": chunks}


def grade_retrieval(state: AgentState) -> Literal["generate_draft", "rewrite_question", "decompose_question"]:
    q     = state.get("standalone_question") or state.get("question") or ""
    ctx   = state.get("context") or ""
    sig   = _context_signals(ctx)
    loops = _retrieval_calls_this_turn(state.get("messages", []))
    multi = _looks_multihop(q)
    codes = extract_codes(q)

    if loops >= MAX_RETRIEVAL_LOOPS:
        return "generate_draft"
    if codes and "HIT" in (ctx or ""):
        return "generate_draft"
    if sig["snippet_count"] < MIN_SNIPPETS_FOR_DIRECT or sig["unique_docids"] < MIN_UNIQUE_DOCIDS_DIRECT:
        return "decompose_question" if multi else "rewrite_question"
    try:
        resp = llm_grader.with_structured_output(RetrievalGrade).invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(q=q, ctx=ctx, **sig)}]
        )
    except Exception:
        return "generate_draft" if sig["snippet_count"] >= 1 else "rewrite_question"

    if resp.action == "ANSWER":      return "generate_draft"
    if resp.action == "DECOMPOSE":   return "decompose_question"
    return "rewrite_question"


def rewrite_question(state: AgentState) -> AgentState:
    q     = state.get("standalone_question") or state.get("question") or ""
    codes = extract_codes(q)
    if codes:
        rewritten = f"{' '.join(codes)} status approval waiver ETAR quiet hours crew rest"
        return {"standalone_question": rewritten, "plan": []}
    try:
        rewritten = llm_query.invoke(
            [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}]
        ).content.strip().strip('"').strip()
        return {"standalone_question": rewritten, "plan": []}
    except Exception:
        return {"standalone_question": q, "plan": []}


def decompose_question(state: AgentState) -> AgentState:
    q = state.get("standalone_question") or state.get("question") or ""
    if extract_codes(q):
        return {"plan": [f"What is the status/decision for {c}?" for c in extract_codes(q)]}
    try:
        plan = llm_query.with_structured_output(Plan).invoke(
            [{"role": "user", "content": DECOMP_PROMPT.format(q=q)}]
        )
        subs = [s.strip() for s in plan.subquestions if s.strip()]
        return {"plan": subs if len(subs) >= 2 else [q]}
    except Exception:
        return {"plan": [q]}


def generate_draft(state: AgentState) -> AgentState:
    q      = state.get("question") or ""
    ctx    = state.get("context") or ""
    mem    = state.get("memory_context") or ""
    q_type = state.get("question_type", "summary")
    prompt = GENERATE_PROMPTS.get(q_type, GENERATE_PROMPTS["summary"])
    sys    = SystemMessage(content="You are a precise biomedical expert. Ground every claim in the provided snippets.")
    user   = HumanMessage(content=prompt.format(question=q, context=ctx, memory=mem))
    try:
        draft = llm_answer.invoke([sys, user]).content
    except Exception as e:
        draft = (
            "I couldn't reach the LLM server to generate an answer. "
            f"Error: {type(e).__name__}: {e}"
        )
    return {"draft_answer": draft}


def reflect(state: AgentState) -> AgentState:
    q     = state.get("standalone_question") or state.get("question") or ""
    ctx   = state.get("context") or ""
    draft = state.get("draft_answer") or ""
    loops = _retrieval_calls_this_turn(state.get("messages", []))
    if loops >= MAX_RETRIEVAL_LOOPS:
        return {"reflect_action": "FINALIZE", "reflect_queries": [], "reflection": "Max loops reached."}
    try:
        r = llm_grader.with_structured_output(AnswerReflection).invoke(
            [{"role": "user", "content": REFLECT_PROMPT.format(q=q, ctx=ctx, a=draft)}]
        )
        return {
            "reflect_action" : r.action,
            "reflect_queries": r.suggested_queries,
            "reflection"     : f"grounded={r.grounded:.2f} complete={r.complete:.2f} missing={r.missing}",
        }
    except Exception:
        return {"reflect_action": "FINALIZE", "reflect_queries": [], "reflection": "Reflection failed."}


def route_after_reflect(state: AgentState) -> Literal["finalize_answer", "call_retriever", "ask_user"]:
    action = state.get("reflect_action", "FINALIZE")
    if action == "RETRIEVE_MORE" and (state.get("reflect_queries") or []):
        return "call_retriever"
    if action == "ASK_USER":
        return "ask_user"
    return "finalize_answer"


def apply_reflection_queries(state: AgentState) -> AgentState:
    queries = state.get("reflect_queries") or []
    if not queries:
        return {}
    if len(queries) >= 2:
        return {"plan": queries}
    return {"standalone_question": queries[0], "plan": []}


def finalize_answer(state: AgentState) -> AgentState:
    return {"messages": [AIMessage(content=state.get("draft_answer") or "")]}


def ask_user(state: AgentState) -> AgentState:
    qs = state.get("reflect_queries") or []
    q  = qs[0] if qs else "Can you clarify what specific information you need?"
    return {"messages": [AIMessage(content=q)]}


def write_memory_node(state: AgentState, config) -> AgentState:
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    write_longterm_memory(
        state.get("question") or "",
        state.get("draft_answer") or "",
        thread_id=thread_id,
    )
    return {}


def route_to_planner(state: AgentState) -> Literal["plan_question", "call_retriever"]:
    q = state.get("standalone_question") or state.get("question") or ""
    return "plan_question" if (not extract_codes(q) and _looks_multihop(q)) else "call_retriever"


def _clean_model_text(t: str) -> str:
    return (t or "").replace("<|eot_id|>", "").strip()


# ═══════════════════════════════════════════════════════════════════
# 11. BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("prepare_question",       prepare_question)
    workflow.add_node("plan_question",          plan_question)
    workflow.add_node("call_retriever",         call_retriever)
    workflow.add_node("retrieve",               retrieve)
    workflow.add_node("capture_context",        capture_context)
    workflow.add_node("rewrite_question",       rewrite_question)
    workflow.add_node("decompose_question",     decompose_question)
    workflow.add_node("generate_draft",         generate_draft)
    workflow.add_node("reflect",                reflect)
    workflow.add_node("apply_reflection_queries", apply_reflection_queries)
    workflow.add_node("finalize_answer",        finalize_answer)
    workflow.add_node("ask_user",               ask_user)
    workflow.add_node("write_memory",           write_memory_node)

    workflow.add_edge(START, "prepare_question")
    workflow.add_conditional_edges(
        "prepare_question",
        route_to_planner,
        {"plan_question": "plan_question", "call_retriever": "call_retriever"},
    )
    workflow.add_edge("plan_question",  "call_retriever")
    workflow.add_edge("call_retriever", "retrieve")
    workflow.add_edge("retrieve",       "capture_context")
    workflow.add_conditional_edges(
        "capture_context",
        grade_retrieval,
        {
            "generate_draft"     : "generate_draft",
            "rewrite_question"   : "rewrite_question",
            "decompose_question" : "decompose_question",
        },
    )
    workflow.add_edge("rewrite_question",   "call_retriever")
    workflow.add_edge("decompose_question", "call_retriever")
    workflow.add_edge("generate_draft",     "reflect")
    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "finalize_answer"          : "finalize_answer",
            "call_retriever"           : "apply_reflection_queries",
            "ask_user"                 : "ask_user",
        },
    )
    workflow.add_edge("apply_reflection_queries", "call_retriever")
    workflow.add_edge("finalize_answer", "write_memory")
    workflow.add_edge("write_memory",    END)
    workflow.add_edge("ask_user",        END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════════
# 12. BIOASQ FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════
def _extract_exact_answer(draft: str, q_type: str):
    """Best-effort extraction of exact answer from draft text."""
    if q_type == "yesno":
        lower = draft.lower()
        for line in lower.splitlines():
            if "answer:" in line:
                if "yes" in line: return "yes"
                if "no"  in line: return "no"
        return "yes" if "yes" in lower[:200] else "no"
    elif q_type == "factoid":
        for line in draft.splitlines():
            if "answer:" in line.lower():
                ans = re.sub(r".*answer:\s*", "", line, flags=re.IGNORECASE).strip()
                return [[ans]] if ans else [["unknown"]]
        return [["unknown"]]
    elif q_type == "list":
        items = []
        for line in draft.splitlines():
            line = line.strip().lstrip("-•*0123456789. ").strip()
            if line and len(line) > 2:
                items.append([line])
        return items[:20] if items else [["unknown"]]
    return None


def format_bioasq_answer(q: dict, draft: str, chunks: List[Dict]) -> dict:
    q_type = q.get("type", "summary")
    doc_urls = list({
        f"http://www.ncbi.nlm.nih.gov/pubmed/{c['docid']}"
        for c in chunks if c.get("docid")
    })
    answer = {
        "id"           : q["id"],
        "type"         : q_type,
        "ideal_answer" : [draft],
        "documents"    : doc_urls[:10],
        "snippets"     : [
            {
                "text"               : c.get("snippet", c.get("full_text", ""))[:500],
                "document"           : f"http://www.ncbi.nlm.nih.gov/pubmed/{c['docid']}",
                "offsetInBeginSection": 0,
                "offsetInEndSection" : 500,
                "beginSection"       : c.get("metadata", {}).get("section", "abstract"),
                "endSection"         : c.get("metadata", {}).get("section", "abstract"),
            }
            for c in chunks[:5] if c.get("docid")
        ],
    }
    exact = _extract_exact_answer(draft, q_type)
    if exact is not None:
        answer["exact_answer"] = exact
    return answer


# ═══════════════════════════════════════════════════════════════════
# 13. RUN MODES
# ═══════════════════════════════════════════════════════════════════
def run_single_query(graph, question: str, q_type: str = "summary", thread_id: str = "interactive") -> str:
    """Run a single question and return the answer text."""
    RUN_CONFIG = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}
    out = graph.invoke(
        {"messages": [HumanMessage(content=question)], "question_type": q_type},
        config=RUN_CONFIG,
    )
    answer = _clean_model_text(out["messages"][-1].content)
    chunks = out.get("retrieved_chunks", [])
    print(f"\n{'='*60}")
    print(f"Q ({q_type}): {question}")
    print(f"{'='*60}")
    print(f"A: {answer}")
    print(f"\nUsed {len(chunks)} chunks from {len({c.get('docid') for c in chunks})} PMIDs")
    return answer


def run_batch(graph, test_path: str, out_path: str):
    """Run full BioASQ test set and write submission JSON."""
    with open(test_path) as f:
        test_data = json.load(f)
    questions = test_data["questions"]
    answers   = []
    errors    = []

    for q in tqdm(questions, desc="BioASQ batch"):
        thread_id = f"bioasq-{q['id']}"
        RUN_CONFIG = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}
        try:
            out = graph.invoke(
                {
                    "messages"     : [HumanMessage(content=q["body"])],
                    "question_type": q.get("type", "summary"),
                },
                config=RUN_CONFIG,
            )
            draft  = _clean_model_text(out["messages"][-1].content)
            chunks = out.get("retrieved_chunks", [])
            answers.append(format_bioasq_answer(q, draft, chunks))
        except Exception as e:
            errors.append({"id": q["id"], "error": str(e)})
            answers.append({
                "id"          : q["id"],
                "type"        : q.get("type", "summary"),
                "ideal_answer": ["Unable to retrieve answer."],
                "documents"   : [],
                "snippets"    : [],
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"questions": answers}, f, indent=2)
    print(f"\nSubmission saved → {out_path}")
    print(f"Answered: {len(answers)} | Errors: {len(errors)}")
    if errors:
        err_path = Path(out_path).parent / "errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)


def run_interactive(graph):
    """Interactive REPL for single queries."""
    print("\n" + "="*60)
    print("BioASQ Interactive Mode  (type 'quit' to exit)")
    print("="*60)
    thread_id = f"interactive-{uuid.uuid4().hex[:6]}"
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        q_type = input("Type [yesno/factoid/list/summary, default=summary]: ").strip() or "summary"
        run_single_query(graph, q, q_type=q_type, thread_id=thread_id)


# ═══════════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="BioASQ 13b Agentic Pipeline")
    parser.add_argument("--mode",  default="interactive", choices=["interactive", "batch", "single"])
    parser.add_argument("--index", default="index",  help="Path to index/ directory from 02_build_index.py")
    parser.add_argument("--test",  default="",       help="[batch] BioASQ test JSON path")
    parser.add_argument("--out",   default="submissions/submission.json", help="[batch] Output path")
    parser.add_argument("--query", default="",       help="[single] Question text")
    parser.add_argument("--qtype", default="summary",help="[single] Question type")
    args = parser.parse_args()

    # Load indexes
    global _faiss_vs, _faiss_snip, _bm25_papers, _bm25_snippets, _reranker, _DOCID_TO_CHUNKS, _CODE_TO_CHUNKS
    (
        _faiss_vs, _faiss_snip,
        _bm25_papers, _bm25_snippets,
        _reranker,
        _DOCID_TO_CHUNKS, _CODE_TO_CHUNKS,
        _,
    ) = load_indexes(args.index)
    _init_retrievers()

    # Build graph
    print("Building LangGraph agent...")
    graph = build_graph()

    if args.mode == "batch":
        if not args.test:
            raise ValueError("--test is required for batch mode")
        run_batch(graph, args.test, args.out)
    elif args.mode == "single":
        if not args.query:
            raise ValueError("--query is required for single mode")
        run_single_query(graph, args.query, q_type=args.qtype)
    else:
        run_interactive(graph)


if __name__ == "__main__":
    main()
