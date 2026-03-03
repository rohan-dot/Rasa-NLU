#!/usr/bin/env python3
"""
BioASQ Agentic Retrieval System — Refactored v2
=================================================
Fixes from v1:
  - "Selected 0 papers" → unified single FAISS + per-paper metadata filtering
  - Recursion limit → hard global step counter, simplified graph
  - Blank answers → guaranteed fallback generation
  - Config parameter → uses RunnableConfig properly
  - Thread ID reuse → unique per question in eval

Requires:
  pip install langchain langchain-community langchain-openai faiss-cpu
          sentence-transformers pydantic langgraph
  vLLM serving gemma-3-27b-it on http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import json, os, re, time, uuid, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "gemma-3-27b-it"
LLM_API_KEY = "EMPTY"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
DENSE_TOP_K = 15
DENSE_FETCH_K = 80
FINAL_TOP_K = 8
SNIPPET_CHARS = 1200
HYDE_TRIGGER_CHARS = 80

# Loop caps (these are HARD — graph will never exceed them)
MAX_RETRIEVAL_LOOPS = 3
MAX_TOTAL_STEPS = 12  # absolute cap on graph node visits

# Heuristic thresholds
MIN_SNIPPETS_FOR_DIRECT = 2
MIN_UNIQUE_DOCIDS_DIRECT = 1

# Paths
INDEX_DIR = Path("./bioasq_faiss_index")
MEMORY_FAISS_DIR = Path("./memory_faiss")
ENABLE_LONGTERM_MEMORY = False

# Cross-encoder (set to None to skip)
CROSS_ENCODER_MODEL: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ──────────────────────────────────────────────
#  Shared objects
# ──────────────────────────────────────────────
embeddings: HuggingFaceEmbeddings = None  # type: ignore
cross_encoder = None
unified_index: Optional[FAISS] = None  # single FAISS with ALL snippets
DOCID_TO_CHUNKS: Dict[str, List[Document]] = {}  # paper_id -> ordered chunks
ALL_PAPER_IDS: set = set()

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
#  SECTION 1 — BioASQ Data Loading & Index Building
# ══════════════════════════════════════════════

def _paper_id_from_url(url: str) -> str:
    """Extract a stable paper id from a PubMed URL."""
    m = re.search(r"/pubmed/(\d+)", url)
    if m:
        return m.group(1)
    return hashlib.md5(url.encode()).hexdigest()[:12]


def load_bioasq_dataset(path: str) -> List[Dict[str, Any]]:
    """Load BioASQ JSON (single file with 'questions' key, list, or JSONL)."""
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


def build_index(questions: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
    """
    Build a SINGLE unified FAISS index over ALL snippets from all papers.
    Each document carries paper_id in metadata for filtering.
    This avoids the "0 papers selected" problem entirely.
    """
    global unified_index, DOCID_TO_CHUNKS, ALL_PAPER_IDS

    emb = _get_embeddings()

    # Try to load existing index
    if not force_rebuild and INDEX_DIR.exists():
        try:
            unified_index = FAISS.load_local(
                str(INDEX_DIR), emb, allow_dangerous_deserialization=True,
            )
            # Rebuild DOCID_TO_CHUNKS
            _rebuild_all_chunks(unified_index)
            log.info("Loaded existing index with %d papers", len(ALL_PAPER_IDS))
            return
        except Exception as e:
            log.warning("Failed to load index, rebuilding: %s", e)

    # ── Collect ALL snippets from all questions ──
    all_docs: List[Document] = []
    seen_texts: set = set()
    paper_chunk_counter: Dict[str, int] = {}

    for q in questions:
        for snippet in q.get("snippets", []):
            doc_url = snippet.get("document", "")
            if not doc_url:
                continue
            pid = _paper_id_from_url(doc_url)
            ALL_PAPER_IDS.add(pid)

            txt = snippet.get("text", "").strip()
            if not txt:
                continue

            # Deduplicate by (paper_id, text)
            dedup_key = f"{pid}:{hashlib.md5(txt.encode()).hexdigest()}"
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            chunk_idx = paper_chunk_counter.get(pid, 0)
            paper_chunk_counter[pid] = chunk_idx + 1

            doc = Document(
                page_content=txt,
                metadata={
                    "paper_id": pid,
                    "chunk_index": chunk_idx,
                    "section": snippet.get("beginSection", ""),
                    "document_url": doc_url,
                    "question_id": q.get("id", ""),
                },
            )
            all_docs.append(doc)
            DOCID_TO_CHUNKS.setdefault(pid, []).append(doc)

    log.info("Collected %d unique snippets from %d papers across %d questions",
             len(all_docs), len(ALL_PAPER_IDS), len(questions))

    if not all_docs:
        log.error("No snippets found in dataset!")
        return

    # Build unified FAISS index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    unified_index = FAISS.from_documents(all_docs, emb)
    unified_index.save_local(str(INDEX_DIR))
    log.info("Built unified FAISS index with %d documents", len(all_docs))


def _rebuild_all_chunks(vs: FAISS) -> None:
    """Rebuild DOCID_TO_CHUNKS and ALL_PAPER_IDS from loaded index."""
    global DOCID_TO_CHUNKS, ALL_PAPER_IDS
    try:
        for doc in vs.docstore._dict.values():
            pid = (doc.metadata or {}).get("paper_id", "")
            if pid:
                ALL_PAPER_IDS.add(pid)
                DOCID_TO_CHUNKS.setdefault(pid, []).append(doc)
        # Sort chunks by index
        for pid in DOCID_TO_CHUNKS:
            DOCID_TO_CHUNKS[pid].sort(
                key=lambda d: (d.metadata or {}).get("chunk_index", 0)
            )
    except Exception as e:
        log.warning("Chunk rebuild failed: %s", e)


# ══════════════════════════════════════════════
#  SECTION 2 — Retrieval Functions
# ══════════════════════════════════════════════

CODE_RE = re.compile(r"~?\b[A-Z]{0,5}\d{4,}(?:[A-Za-z]\d*)*\b")


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
        resp = llm_query.invoke(
            [{"role": "user", "content":
              f"Write a short factual paragraph that directly answers this biomedical question: {question}"}]
        )
        return resp.content.strip()
    except Exception:
        return ""


def retrieve_from_index(query: str, top_k: int = DENSE_TOP_K) -> List[Document]:
    """Retrieve from the unified FAISS index."""
    if unified_index is None:
        log.warning("No index available!")
        return []
    try:
        docs = unified_index.similarity_search(query, k=top_k, fetch_k=DENSE_FETCH_K)
        return docs
    except Exception as e:
        log.warning("Retrieval failed: %s", e)
        return []


def _expand_neighbors(docs: List[Document], window: int = 1, max_seed: int = 25) -> List[Document]:
    """Bring in adjacent chunks for context completeness."""
    expanded = list(docs)
    for d in docs[:max_seed]:
        meta = d.metadata or {}
        pid = meta.get("paper_id")
        idx = meta.get("chunk_index")
        if pid is None or idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        chunks = DOCID_TO_CHUNKS.get(str(pid), [])
        if not chunks:
            continue
        lo = max(0, idx - window)
        hi = min(len(chunks), idx + window + 1)
        expanded.extend(chunks[lo:hi])
    return expanded


def _dedup_chunks(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        h = hashlib.md5(d.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def _rerank_single(query: str, docs: List[Document], top_n: int = FINAL_TOP_K) -> List[Document]:
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

    # Keyword window
    stopwords = {"what", "when", "where", "which", "that", "this", "from",
                 "with", "have", "does", "then", "than", "the", "and", "for",
                 "are", "was", "were", "been", "being", "how", "why", "who"}
    terms = [t for t in re.findall(r"[A-Za-z0-9_]{4,}", query or "")
             if t.lower() not in stopwords]
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

    # Fallback: return beginning
    span = text[:max_chars]
    if len(text) > max_chars:
        span = span + "…"
    return span


def _format_snippets(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> str:
    blocks = []
    for i, d in enumerate(docs[:limit], start=1):
        meta = d.metadata or {}
        pid = meta.get("paper_id") or ""
        section = meta.get("section", "")
        span = _extract_relevant_span(d.page_content, query)
        header = f"[{i}] paper={pid}"
        if section:
            header += f" section={section}"
        blocks.append(f"{header}\n{span}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_payload(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> dict:
    snippets_text = _format_snippets(docs, query, limit)
    chunks = []
    for d in docs[:limit]:
        meta = d.metadata or {}
        chunks.append({
            "text": d.page_content,
            "paper_id": meta.get("paper_id", ""),
            "chunk_index": meta.get("chunk_index", ""),
            "section": meta.get("section", ""),
            "snippet": _extract_relevant_span(d.page_content, query),
        })
    return {"snippets_text": snippets_text, "chunks": chunks}


# ── Main retrieval tool ──

@tool("smart_retrieve_jsonl")
def smart_retrieve_jsonl(query: str) -> str:
    """
    Hybrid retriever:
      - Unified FAISS over all BioASQ snippets
      - Multi-query support (SUBQUESTIONS block)
      - HyDE for short questions
      - Neighbor expansion + cross-encoder reranking
    """
    subqs = [q for q in _parse_subquestions(query) if q.strip()]
    all_candidates: List[Document] = []

    log.info("Retrieving for %d sub-queries from unified index", len(subqs))

    for q in subqs:
        candidates: List[Document] = []

        # 1) Direct retrieval
        candidates.extend(retrieve_from_index(q))

        # 2) HyDE for short non-code questions
        codes = extract_codes(q)
        if len(q) <= HYDE_TRIGGER_CHARS and not codes:
            hyp = hyde_passage(q)
            if hyp:
                candidates.extend(retrieve_from_index(hyp))

        # 3) Neighbor expansion
        candidates = _expand_neighbors(_dedup_chunks(candidates), window=1)
        all_candidates.extend(candidates)

    all_candidates = _dedup_chunks(all_candidates)
    log.info("Total candidates after dedup: %d", len(all_candidates))

    # Rerank
    if len(subqs) == 1:
        reranked = _rerank_single(subqs[0], all_candidates)
    else:
        reranked = _rerank_multi(subqs, all_candidates, top_n=FINAL_TOP_K)

    # Prioritize exact code hits
    codes = extract_codes(query)
    if codes:
        hits = [d for d in reranked if all(c in (d.page_content or "") for c in codes)]
        if hits:
            reranked = hits + [d for d in reranked if d not in hits]

    payload = _docs_to_payload(reranked, query=query, limit=FINAL_TOP_K)
    return json.dumps(payload, ensure_ascii=False)


# ══════════════════════════════════════════════
#  SECTION 3 — Memory (optional)
# ══════════════════════════════════════════════

def retrieve_longterm_memory(query: str, thread_id: str) -> str:
    if not (ENABLE_LONGTERM_MEMORY and memory_vs):
        return ""
    try:
        docs = memory_vs.similarity_search(query, k=MEMORY_FETCH_K)
    except Exception:
        return ""
    docs = [d for d in docs if (d.metadata or {}).get("thread_id") == thread_id]
    blocks = []
    for i, d in enumerate(docs[:MEMORY_TOP_K], 1):
        ts = (d.metadata or {}).get("ts", "")
        blocks.append(f"[M{i}] ts={ts}\n{(d.page_content or '')[:600]}")
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
        emb = _get_embeddings()
        if memory_vs is None:
            memory_vs = FAISS.from_documents([mem_doc], emb)
        else:
            memory_vs.add_documents([mem_doc])
        memory_vs.save_local(str(MEMORY_FAISS_DIR))
    except Exception:
        pass


# ══════════════════════════════════════════════
#  SECTION 4 — Helpers
# ══════════════════════════════════════════════

def _latest_user_question(msgs: list) -> str:
    for m in reversed(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            return str(getattr(m, "content", "") or "")
    return ""


def _context_signals(ctx: str) -> Dict[str, Any]:
    parts = [p for p in re.split(r"\n\s*---\s*\n", ctx or "") if p.strip()]
    snippet_count = len(parts)
    docids = set(re.findall(r"paper=([^\s]+)", ctx or ""))
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


def _history_to_str(msgs: list, max_turns: int = 8) -> str:
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
    return (t or "").replace("<|eot_id|>", "").replace("<eos>", "").strip()


# ══════════════════════════════════════════════
#  SECTION 5 — Prompts
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
    "- DECOMPOSE: question is multi-hop and snippets only partially cover it.\n"
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
    "You are answering biomedical questions using retrieved PubMed snippets.\n\n"
    "Core grounding rules:\n"
    "- Only use information that is explicitly stated in the snippets.\n"
    "- Do not guess. If the answer is not stated, say 'Not stated in the provided snippets'.\n"
    "- Do NOT paste long quotes. Answer in your own words.\n\n"
    "Answer requirements:\n"
    "1) For multi-hop questions: answer each sub-question in detail, then provide a final combined answer.\n"
    "2) Cite snippets like [1], [2] for each key claim.\n"
    "3) If something is NOT STATED, say exactly what information would confirm it.\n"
    "4) End with References: list each cited snippet with its paper ID.\n\n"
    "User question: {question}\n\n"
    "Conversation memory (may help resolve pronouns; not authoritative):\n{memory}\n\n"
    "Snippets:\n{context}"
)

REFLECT_PROMPT = PromptTemplate.from_template(
    "You are a QA controller for a biomedical RAG system.\n"
    "Given the question, retrieved snippets, and a draft answer:\n"
    "- Judge if the draft is grounded in the snippets and complete for the question.\n"
    "- If not grounded or incomplete, propose up to 3 improved retrieval queries.\n"
    "- If the missing info must be provided by the user, propose ONE clarifying question.\n"
    "Return ONLY valid JSON with keys grounded, complete, missing, suggested_queries, action.\n\n"
    "Actions: FINALIZE, RETRIEVE_MORE, ASK_USER\n\n"
    "Question:\n{q}\n\nSnippets:\n{ctx}\n\nDraft answer:\n{a}"
)


# ══════════════════════════════════════════════
#  SECTION 6 — Pydantic models
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
#  SECTION 7 — Graph Node Functions
# ══════════════════════════════════════════════

def prepare_question(state: dict, config: RunnableConfig = None) -> dict:
    config = config or {}
    msgs = state.get("messages", [])
    q = _latest_user_question(msgs)

    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    elif hasattr(config, "get"):
        configurable = config.get("configurable", {}) or {}
    thread_id = configurable.get("thread_id", "default")

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
        # Step counter — incremented by every node, hard cap in routing
        "_step_count": 0,
        "_retrieval_count": 0,
    }


def plan_question(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    step = state.get("_step_count", 0) + 1

    if extract_codes(q) or not _looks_multihop(q):
        return {"plan": [], "_step_count": step}

    try:
        plan = llm_query.with_structured_output(Plan).invoke(
            [{"role": "user", "content": PLAN_PROMPT.format(q=q)}]
        )
        subs = [s.strip() for s in plan.subquestions if s.strip()]
        return {"plan": subs if len(subs) >= 2 else [], "_step_count": step}
    except Exception:
        return {"plan": [], "_step_count": step}


def call_retriever(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    plan = state.get("plan") or []
    step = state.get("_step_count", 0) + 1
    ret_count = state.get("_retrieval_count", 0) + 1

    if plan and len(plan) >= 2:
        query_blob = "SUBQUESTIONS:\n" + "\n".join(f"- {s}" for s in plan)
    else:
        query_blob = q

    tool_id = uuid.uuid4().hex
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "smart_retrieve_jsonl", "args": {"query": query_blob}, "id": tool_id}],
    )
    return {"messages": [msg], "_step_count": step, "_retrieval_count": ret_count}


def capture_context(state: dict) -> dict:
    msgs = state.get("messages", [])
    ctx = ""
    chunks: List[Dict[str, Any]] = []
    step = state.get("_step_count", 0) + 1

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

    return {"context": ctx, "retrieved_chunks": chunks, "_step_count": step}


def grade_retrieval(state: dict) -> Literal["generate_draft", "rewrite_question", "decompose_question"]:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    sig = _context_signals(ctx)
    step = state.get("_step_count", 0)
    ret_count = state.get("_retrieval_count", 0)

    # ── HARD CAPS: always generate if we've used up our budget ──
    if step >= MAX_TOTAL_STEPS:
        log.info("Hit max total steps (%d), forcing generate", MAX_TOTAL_STEPS)
        return "generate_draft"

    if ret_count >= MAX_RETRIEVAL_LOOPS:
        log.info("Hit max retrieval loops (%d), forcing generate", MAX_RETRIEVAL_LOOPS)
        return "generate_draft"

    # If we have some context, lean towards answering
    if sig["snippet_count"] >= MIN_SNIPPETS_FOR_DIRECT:
        return "generate_draft"

    # If we have at least 1 snippet, just answer (don't loop for marginal gains)
    if sig["snippet_count"] >= 1 and ret_count >= 2:
        return "generate_draft"

    # No context at all — try ONE rewrite
    if sig["snippet_count"] == 0 and ret_count < 2:
        if _looks_multihop(q):
            return "decompose_question"
        return "rewrite_question"

    # Default: just answer with what we have
    return "generate_draft"


def rewrite_question(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    step = state.get("_step_count", 0) + 1

    codes = extract_codes(q)
    if codes:
        rewritten = f"{'  '.join(codes)} biomedical clinical treatment mechanism"
        return {"standalone_question": rewritten, "plan": [], "_step_count": step}

    try:
        rewritten = llm_query.invoke(
            [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}]
        ).content
        return {"standalone_question": rewritten.strip().strip('"').strip(), "plan": [], "_step_count": step}
    except Exception:
        return {"standalone_question": q, "plan": [], "_step_count": step}


def decompose_question(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    step = state.get("_step_count", 0) + 1

    if extract_codes(q):
        return {"plan": [f"What is the status/decision for {c}?" for c in extract_codes(q)],
                "_step_count": step}
    try:
        plan = llm_query.with_structured_output(Plan).invoke(
            [{"role": "user", "content": DECOMP_PROMPT.format(q=q)}]
        )
        subs = [s.strip() for s in plan.subquestions if s.strip()]
        return {"plan": subs if len(subs) >= 2 else [q], "_step_count": step}
    except Exception:
        return {"plan": [q], "_step_count": step}


def generate_draft(state: dict) -> dict:
    q = state.get("question") or ""
    ctx = state.get("context") or ""
    mem = state.get("memory_context") or ""
    step = state.get("_step_count", 0) + 1

    sys_msg = SystemMessage(content="You are helpful and precise. Ground every claim in the provided snippets.")
    prompt = GENERATE_PROMPT.format(question=q, context=ctx, memory=mem)

    try:
        draft = llm_answer.invoke([sys_msg, HumanMessage(content=prompt)]).content
    except Exception as e:
        draft = (
            "I couldn't reach the LLM server to generate an answer. "
            "Check that your OpenAI-compatible endpoint is running and base_url is correct.\n\n"
            f"Error: {type(e).__name__}: {e}"
        )

    return {"draft_answer": draft, "_step_count": step}


def reflect(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    draft = state.get("draft_answer") or ""
    step = state.get("_step_count", 0) + 1
    ret_count = state.get("_retrieval_count", 0)

    # If we've already done retrieval multiple times, just finalize
    if ret_count >= MAX_RETRIEVAL_LOOPS or step >= MAX_TOTAL_STEPS:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Budget exhausted, finalizing.", "_step_count": step}

    try:
        r = llm_grader.with_structured_output(AnswerReflection).invoke(
            [{"role": "user", "content": REFLECT_PROMPT.format(q=q, ctx=ctx, a=draft)}]
        )
        return {
            "reflect_action": r.action,
            "reflect_queries": r.suggested_queries,
            "reflection": f"grounded={r.grounded} complete={r.complete} missing={r.missing}",
            "_step_count": step,
        }
    except Exception:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Reflection failed.", "_step_count": step}


def route_after_reflect(state: dict) -> Literal["finalize_answer", "call_retriever", "ask_user"]:
    action = state.get("reflect_action", "FINALIZE")
    ret_count = state.get("_retrieval_count", 0)
    step = state.get("_step_count", 0)

    # Hard cap check
    if ret_count >= MAX_RETRIEVAL_LOOPS or step >= MAX_TOTAL_STEPS:
        return "finalize_answer"

    if action == "RETRIEVE_MORE" and (state.get("reflect_queries") or []):
        return "call_retriever"
    if action == "ASK_USER":
        return "ask_user"
    return "finalize_answer"


def apply_reflection_queries(state: dict) -> dict:
    queries = state.get("reflect_queries") or []
    step = state.get("_step_count", 0) + 1
    if not queries:
        return {"_step_count": step}
    if len(queries) >= 2:
        return {"plan": queries, "_step_count": step}
    return {"standalone_question": queries[0], "plan": [], "_step_count": step}


def finalize_answer(state: dict) -> dict:
    draft = _clean_model_text(state.get("draft_answer") or "")
    if not draft:
        draft = "I was unable to find relevant information in the available snippets to answer this question."
    return {"messages": [AIMessage(content=draft)]}


def ask_user(state: dict) -> dict:
    qs = state.get("reflect_queries") or []
    q = qs[0] if qs else "Can you clarify what exactly you want me to verify?"
    return {"messages": [AIMessage(content=q)]}


def write_memory_node(state: dict, config: RunnableConfig = None) -> dict:
    config = config or {}
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    elif hasattr(config, "get"):
        configurable = config.get("configurable", {}) or {}
    thread_id = configurable.get("thread_id", "default")
    write_longterm_memory(state.get("question") or "", state.get("draft_answer") or "", thread_id=thread_id)
    return {}


# ══════════════════════════════════════════════
#  SECTION 8 — Build the StateGraph
# ══════════════════════════════════════════════

def build_graph():
    workflow = StateGraph(dict)

    # Nodes
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

    workflow.add_edge(START, "prepare_question")

    def route_to_planner(state) -> Literal["plan_question", "call_retriever"]:
        q = state.get("standalone_question") or state.get("question") or ""
        return "plan_question" if (not extract_codes(q) and _looks_multihop(q)) else "call_retriever"

    workflow.add_conditional_edges(
        "prepare_question",
        route_to_planner,
        {"plan_question": "plan_question", "call_retriever": "call_retriever"},
    )

    workflow.add_edge("plan_question", "call_retriever")
    workflow.add_edge("call_retriever", "retrieve")
    workflow.add_edge("retrieve", "capture_context")

    workflow.add_conditional_edges(
        "capture_context",
        grade_retrieval,
        {
            "generate_draft": "generate_draft",
            "rewrite_question": "rewrite_question",
            "decompose_question": "decompose_question",
        },
    )

    workflow.add_edge("rewrite_question", "call_retriever")
    workflow.add_edge("decompose_question", "call_retriever")
    workflow.add_edge("generate_draft", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "finalize_answer": "finalize_answer",
            "call_retriever": "apply_reflection_queries",
            "ask_user": "ask_user",
        },
    )

    workflow.add_edge("apply_reflection_queries", "call_retriever")
    workflow.add_edge("finalize_answer", "write_memory")
    workflow.add_edge("write_memory", END)
    workflow.add_edge("ask_user", END)

    # Compile with higher recursion limit (our own caps prevent infinite loops)
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


# ══════════════════════════════════════════════
#  SECTION 9 — Evaluation
# ══════════════════════════════════════════════

def evaluate_bioasq(dataset_path: str, max_questions: int = None, output_path: str = "bioasq_results.json"):
    """Run pipeline on BioASQ dataset and collect results."""
    questions = load_bioasq_dataset(dataset_path)
    log.info("Loaded %d questions from %s", len(questions), dataset_path)

    # Build unified index
    build_index(questions)

    # Build graph
    graph = build_graph()

    results = []

    for i, q_data in enumerate(questions):
        if max_questions and i >= max_questions:
            break

        q_body = q_data.get("body", "")
        q_type = q_data.get("type", "factoid")
        q_id = q_data.get("id", f"q_{i}")
        ideal = q_data.get("ideal_answer", [""])
        exact = q_data.get("exact_answer", "")

        log.info("\n[%d/%d] Q: %s", i + 1, len(questions), q_body[:100])

        # IMPORTANT: unique thread_id per question to avoid state leaking
        run_config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": f"eval-{q_id}-{i}"},
        }

        try:
            out = graph.invoke(
                {"messages": [HumanMessage(content=q_body)]},
                config=run_config,
            )
            answer = ""
            if out.get("messages"):
                last_msg = out["messages"][-1]
                answer = _clean_model_text(getattr(last_msg, "content", "") or "")
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
#  SECTION 10 — Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        dataset_path = sys.argv[1]
        max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate_bioasq(dataset_path, max_questions=max_q)
        sys.exit(0)

    print("=== BioASQ Agentic QA System (Refactored v2) ===")
    print("Usage: python bioasq.py <bioasq_dataset.json> [max_questions]")
    print("\nNo dataset provided. Running interactive mode.\n")
    print("Type your biomedical question (or 'quit' to exit):\n")

    graph = build_graph()
    THREAD_ID = "interactive-1"
    RUN_CONFIG = {"recursion_limit": 50, "configurable": {"thread_id": THREAD_ID}}

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
            answer = ""
            if out.get("messages"):
                answer = _clean_model_text(out["messages"][-1].content)
            print(f"\nAssistant: {answer or '(no answer)'}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
