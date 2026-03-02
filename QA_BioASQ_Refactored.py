#!/usr/bin/env python3
"""
BioASQ Agentic Retrieval System — Refactored
=============================================
Key improvements over original QA_2_Try.py:
  1. Per-paper FAISS indices with a lightweight master router
  2. Intelligent document-selection agent (abstract/title similarity)
  3. Robust graph traversal with hard loop caps & stuck-state guards
  4. Improved prompts tuned for Gemma-3-27b-it
  5. Parallel multi-paper retrieval
  6. Better evidence synthesis and context management

Requires:
  pip install langchain langchain-community langchain-openai faiss-cpu
          sentence-transformers rank_bm25 pydantic
  vLLM serving gemma-3-27b-it on http://127.0.0.1:8000/v1
"""

# ──────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────
from __future__ import annotations

import json, os, re, time, uuid, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "gemma-3-27b-it"
LLM_API_KEY = "EMPTY"

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval knobs
DENSE_TOP_K = 12
DENSE_FETCH_K = 80
BM25_TOP_K = 28
ENSEMBLE_WEIGHTS = (0.65, 0.35)  # (dense, bm25)
FINAL_TOP_K = 8
SNIPPET_CHARS = 1200
HYDE_TRIGGER_CHARS = 80

# Router knobs — how many papers to select per question
ROUTER_TOP_K_PAPERS = 5
ROUTER_ABSTRACT_CHARS = 500

# Heuristic thresholds
MIN_SNIPPETS_FOR_DIRECT = 4
MIN_UNIQUE_DOCIDS_DIRECT = 2

# Loop controls — hard caps prevent infinite cycles
MAX_RETRIEVAL_LOOPS = 4
MAX_REWRITE_LOOPS = 2
MAX_DECOMPOSE_LOOPS = 2
MAX_REFLECT_LOOPS = 2

# Paths
PAPER_INDEX_DIR = Path("./bioasq_paper_indices")
MASTER_INDEX_DIR = Path("./bioasq_master_index")
MEMORY_FAISS_DIR = Path("./memory_faiss")
ENABLE_LONGTERM_MEMORY = False

# Cross-encoder (optional — set to None to skip reranking)
CROSS_ENCODER_MODEL: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STRICT_ID_ONLY = False

# ──────────────────────────────────────────────
#  Shared objects (initialised lazily)
# ──────────────────────────────────────────────
embeddings: HuggingFaceEmbeddings = None          # type: ignore
cross_encoder = None
paper_indices: Dict[str, FAISS] = {}               # paper_id -> FAISS
paper_metadata: Dict[str, Dict[str, Any]] = {}     # paper_id -> {title, abstract, url, ...}
master_index: Optional[FAISS] = None               # lightweight abstract index for routing
DOCID_TO_CHUNKS: Dict[str, List[Document]] = {}    # docid -> ordered chunks

memory_vs: Optional[FAISS] = None
MEMORY_FETCH_K = 6
MEMORY_TOP_K = 3


def _get_embeddings() -> HuggingFaceEmbeddings:
    global embeddings
    if embeddings is None:
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


# ──────────────────────────────────────────────
#  LLM instances
# ──────────────────────────────────────────────
def _make_llm(temperature: float = 0.1, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


llm_answer = _make_llm(temperature=0.2, max_tokens=2048)
llm_grader = _make_llm(temperature=0.0, max_tokens=512)
llm_query  = _make_llm(temperature=0.3, max_tokens=512)


# ══════════════════════════════════════════════
#  SECTION 1 — BioASQ Data Loading & Multi-Index Creation
# ══════════════════════════════════════════════

def _paper_id_from_url(url: str) -> str:
    """Extract a stable paper id from a PubMed URL."""
    # e.g. http://www.ncbi.nlm.nih.gov/pubmed/12345678 → 12345678
    m = re.search(r"/pubmed/(\d+)", url)
    if m:
        return m.group(1)
    # fallback — hash the URL
    return hashlib.md5(url.encode()).hexdigest()[:12]


def load_bioasq_dataset(path: str) -> List[Dict[str, Any]]:
    """Load BioASQ JSON (single file with 'questions' key or JSONL)."""
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
        questions = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def build_paper_indices(questions: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
    """
    Build one FAISS index per unique paper found in the BioASQ dataset.
    Also build a lightweight master index over paper abstracts for routing.
    """
    global paper_indices, paper_metadata, master_index, DOCID_TO_CHUNKS

    emb = _get_embeddings()

    # ── Collect all snippets grouped by paper ──
    paper_snippets: Dict[str, List[Dict[str, Any]]] = {}   # paper_id -> [snippet dicts]

    for q in questions:
        for snippet in q.get("snippets", []):
            doc_url = snippet.get("document", "")
            if not doc_url:
                continue
            pid = _paper_id_from_url(doc_url)
            paper_snippets.setdefault(pid, [])
            paper_snippets[pid].append({
                "text": snippet.get("text", ""),
                "section": snippet.get("beginSection", ""),
                "document": doc_url,
                "question_id": q.get("id", ""),
            })
            # Store metadata once
            if pid not in paper_metadata:
                paper_metadata[pid] = {
                    "url": doc_url,
                    "title": "",        # BioASQ doesn't always give titles directly
                    "abstract": "",
                }

    log.info("Found %d unique papers across %d questions", len(paper_snippets), len(questions))

    # ── Build per-paper FAISS indices ──
    PAPER_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for pid, snippets in paper_snippets.items():
        idx_path = PAPER_INDEX_DIR / pid
        if not force_rebuild and idx_path.exists():
            try:
                paper_indices[pid] = FAISS.load_local(
                    str(idx_path), emb, allow_dangerous_deserialization=True,
                )
                # Rebuild DOCID_TO_CHUNKS from loaded index
                _rebuild_docid_chunks(pid, paper_indices[pid])
                continue
            except Exception:
                pass  # rebuild

        # Deduplicate snippets by text
        seen_texts = set()
        docs: List[Document] = []
        for i, sn in enumerate(snippets):
            txt = sn["text"].strip()
            if not txt or txt in seen_texts:
                continue
            seen_texts.add(txt)
            doc = Document(
                page_content=txt,
                metadata={
                    "_id": pid,
                    "chunk_index": i,
                    "section": sn.get("section", ""),
                    "document": sn.get("document", ""),
                    "question_id": sn.get("question_id", ""),
                },
            )
            docs.append(doc)

        if not docs:
            continue

        # Store ordered chunks for neighbour expansion
        DOCID_TO_CHUNKS[pid] = docs

        # Build & save FAISS
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(str(idx_path))
        paper_indices[pid] = vs
        log.info("  Built index for paper %s  (%d chunks)", pid, len(docs))

    # ── Collect abstract text for each paper (use first snippet as proxy) ──
    abstract_docs: List[Document] = []
    for pid, snippets in paper_snippets.items():
        # Combine first few snippets as a pseudo-abstract for routing
        combined = " ".join(s["text"] for s in snippets[:3])[:ROUTER_ABSTRACT_CHARS]
        paper_metadata[pid]["abstract"] = combined
        abstract_docs.append(Document(
            page_content=combined,
            metadata={"paper_id": pid, "url": paper_metadata[pid]["url"]},
        ))

    # ── Build master routing index ──
    if abstract_docs:
        MASTER_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        master_index = FAISS.from_documents(abstract_docs, emb)
        master_index.save_local(str(MASTER_INDEX_DIR))
        log.info("Built master routing index over %d papers", len(abstract_docs))


def _rebuild_docid_chunks(pid: str, vs: FAISS) -> None:
    """Rebuild DOCID_TO_CHUNKS from an existing FAISS index."""
    try:
        docs = list(vs.docstore._dict.values())
        docs.sort(key=lambda d: (d.metadata or {}).get("chunk_index", 0))
        DOCID_TO_CHUNKS[pid] = docs
    except Exception:
        pass


# ══════════════════════════════════════════════
#  SECTION 2 — Intelligent Paper Router
# ══════════════════════════════════════════════

def select_relevant_papers(question: str, top_k: int = ROUTER_TOP_K_PAPERS) -> List[str]:
    """
    Given a question, return the paper_ids most likely to contain the answer.
    Uses the lightweight master abstract index for fast similarity search.
    """
    if master_index is None:
        # Fallback: return all papers
        return list(paper_indices.keys())[:top_k]

    try:
        results = master_index.similarity_search_with_score(question, k=top_k)
        paper_ids = []
        for doc, score in results:
            pid = (doc.metadata or {}).get("paper_id", "")
            if pid and pid in paper_indices:
                paper_ids.append(pid)
        return paper_ids if paper_ids else list(paper_indices.keys())[:top_k]
    except Exception as e:
        log.warning("Paper selection failed: %s", e)
        return list(paper_indices.keys())[:top_k]


# ══════════════════════════════════════════════
#  SECTION 3 — Retrieval Functions
# ══════════════════════════════════════════════

CODE_RE = re.compile(r"~?\b[A-Z]{0,5}\d{4,}(?:[A-Za-z]\d*)*\b")


def extract_codes(text: str) -> List[str]:
    """Pull ID-like codes from text (e.g. ~2001101, ABC123)."""
    return CODE_RE.findall(text or "")


def _parse_subquestions(blob: str) -> List[str]:
    """Split a SUBQUESTIONS:\\n block or return as single query."""
    if "SUBQUESTIONS:" in blob:
        qs = [l.strip().lstrip("- ") for l in blob.split("\n") if l.strip() and l.strip() != "SUBQUESTIONS:"]
        return qs if qs else [blob.strip()]
    return [blob.strip()]


def hyde_passage(question: str) -> str:
    """Generate a hypothetical answer passage for HyDE retrieval."""
    try:
        resp = llm_query.invoke(
            [{"role": "user", "content":
              f"Write a short factual paragraph that directly answers: {question}"}]
        )
        return resp.content.strip()
    except Exception:
        return ""


def _invoke_retriever_on_paper(paper_id: str, query: str, top_k: int = DENSE_TOP_K) -> List[Document]:
    """Retrieve from a single paper's FAISS index."""
    vs = paper_indices.get(paper_id)
    if vs is None:
        return []
    try:
        docs = vs.similarity_search(query, k=top_k, fetch_k=DENSE_FETCH_K)
        return docs
    except Exception as e:
        log.warning("Retrieval from paper %s failed: %s", paper_id, e)
        return []


def _retrieve_from_papers(paper_ids: List[str], query: str) -> List[Document]:
    """
    Retrieve from multiple paper indices in parallel.
    Returns a deduplicated, merged candidate list.
    """
    all_docs: List[Document] = []
    seen_texts = set()

    with ThreadPoolExecutor(max_workers=min(len(paper_ids), 8)) as pool:
        futures = {
            pool.submit(_invoke_retriever_on_paper, pid, query): pid
            for pid in paper_ids
        }
        for future in as_completed(futures):
            try:
                docs = future.result()
                for d in docs:
                    txt_hash = hashlib.md5(d.page_content.encode()).hexdigest()
                    if txt_hash not in seen_texts:
                        seen_texts.add(txt_hash)
                        all_docs.append(d)
            except Exception:
                pass

    return all_docs


def _expand_neighbors(docs: List[Document], window: int = 1, max_seed: int = 25) -> List[Document]:
    """Bring in adjacent chunks for context completeness."""
    expanded = list(docs)
    for d in docs[:max_seed]:
        meta = d.metadata or {}
        docid = meta.get("_id")
        idx = meta.get("chunk_index")
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
    """Deduplicate documents by page_content."""
    seen = set()
    out = []
    for d in docs:
        h = hashlib.md5(d.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def _rerank_single(query: str, docs: List[Document], top_n: int = FINAL_TOP_K) -> List[Document]:
    """Rerank with cross-encoder for a single query."""
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs[:top_n]
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = ce.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


def _rerank_multi(queries: List[str], docs: List[Document], top_n: int = FINAL_TOP_K) -> List[Document]:
    """Rerank with cross-encoder averaging scores across multiple sub-queries."""
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs[:top_n]
    try:
        score_acc = np.zeros(len(docs))
        for q in queries:
            pairs = [(q, d.page_content) for d in docs]
            scores = ce.predict(pairs)
            score_acc += np.array(scores)
        score_acc /= len(queries)
        ranked = sorted(zip(docs, score_acc), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


def _extract_relevant_span(text: str, query: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Return a window around the first strong match (code first, then keyword)."""
    if not text:
        return ""
    codes = extract_codes(query)
    lowered = text.lower()

    # Code-first
    for c in codes:
        pos = text.find(c)
        if pos != -1:
            start = max(0, pos - max_chars // 2)
            end = min(len(text), start + max_chars)
            span = text[start:end]
            if start > 0:
                span = "…" + span
            if end < len(text):
                span = span + "…"
            return span

    # Keyword window (light lexical anchor)
    stopwords = {"what", "when", "where", "which", "that", "this", "from",
                 "with", "have", "does", "then", "than", "the", "and", "for",
                 "are", "was", "were", "been", "being", "how", "why", "who"}
    terms = [
        t for t in re.findall(r"[A-Za-z0-9_]{4,}", query or "")
        if t.lower() not in stopwords
    ]
    for t in terms:
        pos = lowered.find(t.lower())
        if pos != -1:
            start = max(0, pos - max_chars // 2)
            end = min(len(text), start + max_chars)
            span = text[start:end]
            if start > 0:
                span = "…" + span
            if end < len(text):
                span = span + "…"
            return span

    # Fallback
    span = text[:max_chars]
    if len(text) > max_chars:
        span = span + "…"
    return span


def _format_snippets(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> str:
    """Stable snippet format; ensures code-bearing matches appear in the span."""
    query_codes = extract_codes(query)
    blocks = []
    for i, d in enumerate(docs[:limit], start=1):
        meta = d.metadata or {}
        docid = meta.get("_id") or ""
        section = meta.get("section", "")
        span = _extract_relevant_span(d.page_content, query)
        header = f"[{i}] docid={docid}"
        if section:
            header += f" section={section}"
        blocks.append(f"{header}\n{span}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_payload(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> dict:
    """Return a JSON-serialisable payload with both formatted text and raw chunks."""
    snippets_text = _format_snippets(docs, query, limit)
    chunks = []
    for d in docs[:limit]:
        meta = d.metadata or {}
        chunks.append({
            "text": d.page_content,
            "docid": meta.get("_id", ""),
            "chunk_index": meta.get("chunk_index", ""),
            "section": meta.get("section", ""),
            "snippet": _extract_relevant_span(d.page_content, query),
        })
    return {"snippets_text": snippets_text, "chunks": chunks}


# ── The main retrieval tool (called by the graph) ──

@tool("smart_retrieve_jsonl")
def smart_retrieve_jsonl(query: str) -> str:
    """
    Hybrid retriever with:
      - Intelligent paper selection (multi-index routing)
      - code/ID pinpointing + parent expansion
      - multi-query + HyDE recall boost
      - cross-encoder rerank (multi-subquestion aware)
      - better snippets (window around match)
    """
    subqs = [q for q in _parse_subquestions(query) if q.strip()]
    all_candidates: List[Document] = []

    # Select relevant papers
    full_query = " ".join(subqs)
    selected_papers = select_relevant_papers(full_query)
    log.info("Selected %d papers for query: %s", len(selected_papers), full_query[:80])

    for q in subqs:
        candidates: List[Document] = []

        # 1) Pinpoint for code queries
        codes = extract_codes(q)
        if codes:
            exact = _code_candidates_for_query(q, selected_papers)
            if exact:
                candidates.extend(exact)
                docids = {((d.metadata or {}).get("_id") or "") for d in exact}
                for docid in docids:
                    candidates.extend(DOCID_TO_CHUNKS.get(docid, []))
                if STRICT_ID_ONLY:
                    all_candidates.extend(_dedup_chunks(candidates))
                    continue

        # 2) High recall retrieval from selected papers
        try:
            candidates.extend(_retrieve_from_papers(selected_papers, q))
        except Exception:
            # Fallback to all papers
            candidates.extend(_retrieve_from_papers(list(paper_indices.keys())[:10], q))

        # 3) HyDE for short questions (non-code)
        if len(q) <= HYDE_TRIGGER_CHARS and not codes:
            hyp = hyde_passage(q)
            if hyp:
                candidates.extend(_retrieve_from_papers(selected_papers, hyp))

        # 4) Neighbor expansion
        candidates = _expand_neighbors(_dedup_chunks(candidates), window=1)
        all_candidates.extend(_dedup_chunks(candidates))

    all_candidates = _dedup_chunks(all_candidates)

    # Rerank
    if len(subqs) == 1:
        reranked = _rerank_single(subqs[0], all_candidates)
    else:
        reranked = _rerank_multi(subqs, all_candidates, top_n=FINAL_TOP_K)

    # If we have codes, prioritize exact-code hits
    codes = extract_codes(query)
    if codes:
        hits = [d for d in reranked if all(c in (d.page_content or "") for c in codes)]
        if hits:
            reranked = hits if STRICT_ID_ONLY else hits + [d for d in reranked if d not in hits]

    payload = _docs_to_payload(reranked, query=query, limit=FINAL_TOP_K)
    return json.dumps(payload, ensure_ascii=False)


def _code_candidates_for_query(query: str, paper_ids: List[str]) -> List[Document]:
    """Find chunks containing the exact codes in the query."""
    codes = extract_codes(query)
    if not codes:
        return []
    results = []
    for pid in paper_ids:
        for chunk in DOCID_TO_CHUNKS.get(pid, []):
            if all(c in chunk.page_content for c in codes):
                results.append(chunk)
    return results


# ══════════════════════════════════════════════
#  SECTION 4 — Long-term Memory (optional)
# ══════════════════════════════════════════════

def _load_memory():
    global memory_vs
    if ENABLE_LONGTERM_MEMORY and MEMORY_FAISS_DIR.exists():
        try:
            memory_vs = FAISS.load_local(
                str(MEMORY_FAISS_DIR), _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception:
            memory_vs = None


def _format_memory(docs: List[Document], limit: int = 3) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        ts = (d.metadata or {}).get("ts", "")
        blocks.append(f"[M{i}] ts={ts}\n{(d.page_content or '')[:600]}")
    return "\n\n---\n\n".join(blocks)


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


# ══════════════════════════════════════════════
#  SECTION 5 — Agent State & Helpers
# ══════════════════════════════════════════════

class AgentState(Dict):  # using plain dict for broad langgraph compat
    """
    Keys:
      messages, question, standalone_question, plan, context,
      memory_context, retrieved_chunks, draft_answer, reflection,
      reflect_action, reflect_queries,
      _retrieval_loops, _rewrite_loops, _decompose_loops, _reflect_loops
    """
    pass


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


def _retrieval_calls_this_turn(msgs: List[BaseMessage]) -> int:
    n = 0
    for m in _msgs_since_last_user(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        name = getattr(m, "name", "") or getattr(m, "tool", "")
        if role == "tool" and name == "smart_retrieve_jsonl":
            n += 1
    return n


def _context_signals(ctx: str) -> Dict[str, Any]:
    parts = [p for p in re.split(r"\n\s*---\s*\n", ctx or "") if p.strip()]
    snippet_count = len(parts)
    docids = set(re.findall(r"docid=([^\s]+)", ctx or ""))
    return {"snippet_count": snippet_count, "unique_docids": len(docids)}


def _looks_multihop(q: str) -> bool:
    ql = (q or "").lower()
    cues = [" and ", " vs ", " compare ", " both ", " either ", " steps ",
            " between ", " then ", " after ", " relationship ", " mechanism "]
    return any(c in ql for c in cues) or q.count(",") >= 2 or len(q.split()) >= 18


def _looks_like_followup(q: str) -> bool:
    ql = (q or "").strip().lower()
    pronouns = {"it", "they", "that", "those", "these", "he", "she", "this", "there"}
    starts = ("and ", "also ", "what about ", "how about ", "then ", "so ")
    short = len(ql.split()) <= 10
    return ql.startswith(starts) or (short and any(p in ql.split()[:3] for p in pronouns))


def _history_to_str(msgs: List[BaseMessage], max_turns: int = 8) -> str:
    out = []
    turns = 0
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


# ══════════════════════════════════════════════
#  SECTION 6 — Prompts (tuned for Gemma-3-27b-it)
# ══════════════════════════════════════════════

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
    "Decide what to do next given the user question and retrieved snippets.\n\n"
    "Actions:\n"
    "- ANSWER: snippets are clearly relevant and contain the needed details.\n"
    "- DECOMPOSE: question is multi-hop / multiple parts and snippets cover only partial pieces.\n"
    "- REWRITE: snippets are off-topic OR the query is phrased poorly.\n\n"
    "Return ONLY valid JSON with keys: relevance, coverage, conflict, action.\n\n"
    "Important rule for code/ID questions:\n"
    "If the question contains a code like ~2001101... and at least one snippet is labeled HIT, "
    "prefer ANSWER even if coverage is partial (answer may be 'not stated').\n\n"
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
    "You are answering biomedical questions using retrieved PubMed snippets.\n\n"
    "Core grounding rules:\n"
    "- Only use information that is explicitly stated in the snippets.\n"
    "- Do not guess. If the answer is not stated, say 'Not stated in the provided snippets' and specify what would confirm it.\n"
    "- Do NOT paste long quotes. Answer in your own words.\n\n"
    "ID/code pinpoint rule:\n"
    "- If the question contains one or more codes like ~ABC123..., you MUST base the answer ONLY on snippets that contain the exact code(s).\n"
    "- If no snippet contains the exact code(s), say so explicitly and ask for more context or a different identifier.\n\n"
    "Ambiguity rule:\n"
    "- If the question is vague/underspecified and the snippets do not uniquely determine an answer, ask ONE clarifying question.\n"
    "- Do not list multiple speculative 'versions' of an answer.\n\n"
    "Answer the what and why and how of the question as well, if anything related to a mechanism is asked, also state more about the activities around the mechanism.\n\n"
    "Output format:\n"
    "1) Answer: a direct response.\n"
    "2) Evidence: bullet list of key claims with citations like [1], [2].\n"
    "3) Not stated / Missing: what is missing (if anything).\n"
    "4) References: list each cited snippet with its docid.\n\n"
    "User question: {question}\n\n"
    "Conversation memory (may help resolve pronouns; not authoritative):\n{memory}\n\n"
    "Snippets:\n{context}"
)

# Separate prompt for BioASQ-specific question types
GENERATE_PROMPT_BIOASQ = PromptTemplate.from_template(
    "You are a biomedical question-answering expert using retrieved PubMed snippets.\n\n"
    "Question type: {question_type}\n"
    "Question: {question}\n\n"
    "Instructions based on question type:\n"
    "- factoid: Provide a single, specific answer entity (name, number, date, etc.)\n"
    "- list: Provide a comprehensive list of all relevant items found in snippets.\n"
    "- yesno: Answer 'yes' or 'no' first, then provide the supporting evidence.\n"
    "- summary: Provide a concise summary synthesizing information from all relevant snippets.\n\n"
    "Grounding rules:\n"
    "- ONLY use information explicitly stated in the snippets below.\n"
    "- Cite evidence as [1], [2], etc. matching snippet numbers.\n"
    "- If the answer is not in the snippets, say so clearly.\n\n"
    "Snippets:\n{context}\n\n"
    "Provide your answer:"
)

REFLECT_PROMPT = PromptTemplate.from_template(
    "You are a QA controller for a biomedical RAG system.\n"
    "Given the question, retrieved snippets, and a draft answer:\n"
    "- Judge if the draft is grounded in the snippets and complete for the question.\n"
    "- If not grounded or incomplete, propose up to 3 improved retrieval queries that would likely find the missing info.\n"
    "- If the missing info is not in the corpus and must be provided by the user, propose ONE clarifying question.\n"
    "Return ONLY valid JSON with keys grounded, complete, missing, suggested_queries, action.\n\n"
    "Actions: FINALIZE, RETRIEVE_MORE, ASK_USER\n\n"
    "Question:\n{q}\n\nSnippets:\n{ctx}\n\nDraft answer:\n{a}"
)


# ══════════════════════════════════════════════
#  SECTION 7 — Pydantic models for structured output
# ══════════════════════════════════════════════

class Plan(BaseModel):
    subquestions: List[str] = Field(min_length=1, max_length=5)


class RetrievalGrade(BaseModel):
    relevance: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    conflict: bool = False
    action: Literal["ANSWER", "DECOMPOSE", "REWRITE"]


class AnswerReflection(BaseModel):
    grounded: float = Field(ge=0, le=1)
    complete: float = Field(ge=0, le=1)
    missing: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list, max_length=4)
    action: Literal["FINALIZE", "RETRIEVE_MORE", "ASK_USER"]


# ══════════════════════════════════════════════
#  SECTION 8 — Graph Node Functions
# ══════════════════════════════════════════════

def prepare_question(state: AgentState, config: Optional[dict] = None) -> dict:
    msgs = state.get("messages", [])
    q = _latest_user_question(msgs)

    config = config or {}
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    history = _history_to_str(msgs)

    standalone = q
    if history and _looks_like_followup(q):
        try:
            standalone = llm_query.invoke(
                [{"role": "user", "content": CONDENSE_PROMPT.format(chat_history=history, question=q)}]
            ).content.strip()
        except Exception:
            standalone = q

    mem = retrieve_longterm_memory(standalone, thread_id=thread_id)

    return {
        "question": q,
        "standalone_question": standalone,
        "memory_context": mem,
        "plan": [],
        "context": "",
        "draft_answer": "",
        "reflection": "",
        "reflect_action": "FINALIZE",
        "reflect_queries": [],
        "retrieved_chunks": [],
        # Loop counters — reset each turn
        "_retrieval_loops": 0,
        "_rewrite_loops": 0,
        "_decompose_loops": 0,
        "_reflect_loops": 0,
    }


def plan_question(state: AgentState) -> dict:
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


def call_retriever(state: AgentState) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    plan = state.get("plan") or []

    if plan and len(plan) >= 2:
        query_blob = "SUBQUESTIONS:\n" + "\n".join(f"- {s}" for s in plan)
    else:
        query_blob = q

    tool_id = uuid.uuid4().hex
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "smart_retrieve_jsonl", "args": {"query": query_blob}, "id": tool_id}],
    )
    # Increment retrieval loop counter
    loops = state.get("_retrieval_loops", 0) + 1
    return {"messages": [msg], "_retrieval_loops": loops}


def capture_context(state: AgentState) -> dict:
    msgs = state.get("messages", [])
    ctx = ""
    chunks: List[Dict[str, Any]] = []

    if msgs:
        last = msgs[-1]
        role = getattr(last, "type", getattr(last, "role", ""))
        if role == "tool":
            raw = str(getattr(last, "content", "") or "")
            try:
                obj = json.loads(raw)
                ctx = obj.get("snippets_text", raw)
                chunks = obj.get("chunks", [])
            except Exception:
                ctx = raw

    return {"context": ctx, "retrieved_chunks": chunks}


def grade_retrieval(state: AgentState) -> Literal["generate_draft", "rewrite_question", "decompose_question"]:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    sig = _context_signals(ctx)

    loops = state.get("_retrieval_loops", 0)
    multihop = _looks_multihop(q)
    codes = extract_codes(q)

    # Hard cap: if we've looped too many times, just generate
    if loops >= MAX_RETRIEVAL_LOOPS:
        return "generate_draft"

    # Code hit shortcut
    if codes and "HIT" in (ctx or ""):
        return "generate_draft"

    # Too few snippets and room to retry
    if sig["snippet_count"] < MIN_SNIPPETS_FOR_DIRECT or sig["unique_docids"] < MIN_UNIQUE_DOCIDS_DIRECT:
        # Prefer decompose for multihop, rewrite otherwise
        if multihop and state.get("_decompose_loops", 0) < MAX_DECOMPOSE_LOOPS:
            return "decompose_question"
        if state.get("_rewrite_loops", 0) < MAX_REWRITE_LOOPS:
            return "rewrite_question"
        return "generate_draft"

    # Use LLM grader for nuanced decision
    try:
        resp = llm_grader.with_structured_output(RetrievalGrade).invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(q=q, ctx=ctx, **sig)}]
        )
    except Exception:
        return "generate_draft" if sig["snippet_count"] >= 1 else "rewrite_question"

    if resp.action == "ANSWER":
        return "generate_draft"
    if resp.action == "DECOMPOSE":
        if state.get("_decompose_loops", 0) < MAX_DECOMPOSE_LOOPS:
            return "decompose_question"
        return "generate_draft"
    # REWRITE
    if state.get("_rewrite_loops", 0) < MAX_REWRITE_LOOPS:
        return "rewrite_question"
    return "generate_draft"


def rewrite_question(state: AgentState) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    codes = extract_codes(q)
    rewrite_loops = state.get("_rewrite_loops", 0) + 1

    if codes:
        rewritten = f"{'  '.join(codes)} status approval waiver ETAR quiet hours crew rest"
        return {"standalone_question": rewritten, "plan": [], "_rewrite_loops": rewrite_loops}

    try:
        rewritten = llm_query.invoke(
            [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}]
        ).content
        return {"standalone_question": rewritten.strip().strip('"').strip(), "plan": [], "_rewrite_loops": rewrite_loops}
    except Exception:
        return {"standalone_question": q, "plan": [], "_rewrite_loops": rewrite_loops}


def decompose_question(state: AgentState) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
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


def generate_draft(state: AgentState) -> dict:
    q = state.get("question") or ""
    ctx = state.get("context") or ""
    mem = state.get("memory_context") or ""

    sys = SystemMessage(content="You are helpful and precise. Ground every claim in the provided snippets.")
    prompt = GENERATE_PROMPT.format(question=q, context=ctx, memory=mem)

    try:
        draft = llm_answer.invoke([sys, HumanMessage(content=prompt)]).content
    except Exception as e:
        draft = (
            "I couldn't reach the LLM server to generate an answer. "
            "Check that your OpenAI-compatible endpoint is running and base_url is correct.\n\n"
            f"Error: {type(e).__name__}: {e}"
        )
    return {"draft_answer": draft}


def reflect(state: AgentState) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    draft = state.get("draft_answer") or ""
    reflect_loops = state.get("_reflect_loops", 0) + 1

    # Hard cap
    if reflect_loops > MAX_REFLECT_LOOPS:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Max reflection loops reached.", "_reflect_loops": reflect_loops}

    try:
        r = llm_grader.with_structured_output(AnswerReflection).invoke(
            [{"role": "user", "content": REFLECT_PROMPT.format(q=q, ctx=ctx, a=draft)}]
        )
        return {
            "reflect_action": r.action,
            "reflect_queries": r.suggested_queries,
            "reflection": f"grounded={r.grounded} complete={r.complete} missing={r.missing}",
            "_reflect_loops": reflect_loops,
        }
    except Exception:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Reflection failed.", "_reflect_loops": reflect_loops}


def route_after_reflect(state: AgentState) -> Literal["finalize_answer", "call_retriever", "ask_user"]:
    action = state.get("reflect_action", "FINALIZE")
    if action == "RETRIEVE_MORE" and (state.get("reflect_queries") or []):
        # Only allow more retrieval if we haven't hit the cap
        if state.get("_retrieval_loops", 0) < MAX_RETRIEVAL_LOOPS:
            return "call_retriever"
        return "finalize_answer"
    if action == "ASK_USER":
        return "ask_user"
    return "finalize_answer"


def apply_reflection_queries(state: AgentState) -> dict:
    queries = state.get("reflect_queries") or []
    if not queries:
        return {}
    if len(queries) >= 2:
        return {"plan": queries}
    return {"standalone_question": queries[0], "plan": []}


def finalize_answer(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=_clean_model_text(state.get("draft_answer") or ""))]}


def ask_user(state: AgentState) -> dict:
    qs = state.get("reflect_queries") or []
    q = qs[0] if qs else "Can you clarify what exactly you want me to verify (e.g., which specific aspect, status field, or time window)?"
    return {"messages": [AIMessage(content=q)]}


def write_memory_node(state: AgentState, config: Optional[dict] = None) -> dict:
    config = config or {}
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    write_longterm_memory(state.get("question") or "", state.get("draft_answer") or "", thread_id=thread_id)
    return {}


# ══════════════════════════════════════════════
#  SECTION 9 — Build the StateGraph
# ══════════════════════════════════════════════

def build_graph():
    workflow = StateGraph(dict)   # using plain dict state for compat

    # Add nodes
    workflow.add_node("prepare_question", prepare_question)
    workflow.add_node("plan_question", plan_question)
    workflow.add_node("call_retriever", call_retriever)
    workflow.add_node("retrieve", ToolNode([smart_retrieve_jsonl]))
    workflow.add_node("capture_context", capture_context)

    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("decompose_question", decompose_question)

    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("reflect", reflect)
    workflow.add_node("apply_reflection_queries", apply_reflection_queries)

    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("write_memory", write_memory_node)

    # ── Edges ──

    # Entry
    workflow.add_edge(START, "prepare_question")

    # After prepare → decide: plan or retrieve directly
    def route_to_planner(state) -> Literal["plan_question", "call_retriever"]:
        q = state.get("standalone_question") or state.get("question") or ""
        return "plan_question" if (not extract_codes(q) and _looks_multihop(q)) else "call_retriever"

    workflow.add_conditional_edges(
        "prepare_question",
        route_to_planner,
        {"plan_question": "plan_question", "call_retriever": "call_retriever"},
    )

    # plan → retriever
    workflow.add_edge("plan_question", "call_retriever")

    # retriever → tool → capture
    workflow.add_edge("call_retriever", "retrieve")
    workflow.add_edge("retrieve", "capture_context")

    # capture → grade → {generate | rewrite | decompose}
    workflow.add_conditional_edges(
        "capture_context",
        grade_retrieval,
        {
            "generate_draft": "generate_draft",
            "rewrite_question": "rewrite_question",
            "decompose_question": "decompose_question",
        },
    )

    # rewrite / decompose → back to retriever
    workflow.add_edge("rewrite_question", "call_retriever")
    workflow.add_edge("decompose_question", "call_retriever")

    # generate → reflect
    workflow.add_edge("generate_draft", "reflect")

    # reflect → {finalize | call_retriever (via apply_reflection_queries) | ask_user}
    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "finalize_answer": "finalize_answer",
            "call_retriever": "apply_reflection_queries",
            "ask_user": "ask_user",
        },
    )

    # apply_reflection_queries → call_retriever
    workflow.add_edge("apply_reflection_queries", "call_retriever")

    # finalize → memory → END
    workflow.add_edge("finalize_answer", "write_memory")
    workflow.add_edge("write_memory", END)

    # ask_user → END (user will follow up in next turn)
    workflow.add_edge("ask_user", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


# ══════════════════════════════════════════════
#  SECTION 10 — BioASQ Evaluation Harness
# ══════════════════════════════════════════════

def evaluate_bioasq(dataset_path: str, max_questions: int = None, output_path: str = "bioasq_results.json"):
    """
    Run the pipeline on a BioASQ dataset and collect results.
    """
    questions = load_bioasq_dataset(dataset_path)
    log.info("Loaded %d questions from %s", len(questions), dataset_path)

    # Build indices
    build_paper_indices(questions)

    # Build graph
    graph = build_graph()

    results = []
    run_config = {"recursion_limit": 30, "configurable": {"thread_id": "bioasq-eval"}}

    for i, q_data in enumerate(questions):
        if max_questions and i >= max_questions:
            break

        q_body = q_data.get("body", "")
        q_type = q_data.get("type", "factoid")
        q_id = q_data.get("id", f"q_{i}")
        ideal = q_data.get("ideal_answer", [""])
        exact = q_data.get("exact_answer", "")

        log.info("\n[%d/%d] Q: %s", i + 1, len(questions), q_body[:100])

        try:
            out = graph.invoke(
                {"messages": [HumanMessage(content=q_body)]},
                config=run_config,
            )
            answer = _clean_model_text(out["messages"][-1].content) if out.get("messages") else ""
        except Exception as e:
            log.error("Error on question %s: %s", q_id, e)
            answer = f"ERROR: {e}"

        results.append({
            "id": q_id,
            "type": q_type,
            "question": q_body,
            "predicted_answer": answer,
            "ideal_answer": ideal,
            "exact_answer": exact,
        })

        log.info("  A: %s", answer[:200])

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved %d results to %s", len(results), output_path)
    return results


# ══════════════════════════════════════════════
#  SECTION 11 — Interactive / Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── If a dataset path is provided, run evaluation ──
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        dataset_path = sys.argv[1]
        max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate_bioasq(dataset_path, max_questions=max_q)
        sys.exit(0)

    # ── Otherwise, run interactive demo ──
    print("=== BioASQ Agentic QA System (Refactored) ===")
    print("Usage: python QA_BioASQ_Refactored.py <bioasq_dataset.json> [max_questions]")
    print("\nNo dataset provided. Running interactive mode.\n")
    print("Type your biomedical question (or 'quit' to exit):\n")

    graph = build_graph()
    THREAD_ID = "interactive-1"
    RUN_CONFIG = {"recursion_limit": 30, "configurable": {"thread_id": THREAD_ID}}

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        try:
            out = graph.invoke(
                {"messages": [HumanMessage(content=q)]},
                config=RUN_CONFIG,
            )
            answer = _clean_model_text(out["messages"][-1].content) if out.get("messages") else "(no answer)"
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
