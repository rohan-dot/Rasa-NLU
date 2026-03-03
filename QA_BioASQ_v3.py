#!/usr/bin/env python3
"""
BioASQ Agentic Retrieval System — v3 (Full Paper Download)
============================================================
KEY FIX: Actually downloads full paper abstracts from PubMed via Entrez API,
stores each paper in its own subfolder, builds FAISS indices over REAL content
(not just the tiny BioASQ snippets).

Architecture:
  Phase 1: Parse BioASQ JSON → extract all PubMed IDs
  Phase 2: Download full abstracts from PubMed E-utilities (batch)
  Phase 3: Store each paper in ./papers/<pmid>/paper.json
  Phase 4: Chunk each paper's abstract → build per-paper FAISS index
  Phase 5: Build master routing index over titles+abstracts
  Phase 6: Run agentic RAG pipeline with multi-paper retrieval

Requires:
  pip install langchain langchain-community langchain-openai faiss-cpu \
              sentence-transformers pydantic langgraph requests lxml
  vLLM serving gemma-3-27b-it on http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import json, os, re, sys, time, uuid, hashlib, logging, textwrap
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════╗
# ║  CONFIGURATION                               ║
# ╚══════════════════════════════════════════════╝

LLM_BASE_URL   = "http://127.0.0.1:8000/v1"
LLM_MODEL      = "gemma-3-27b-it"
LLM_API_KEY    = "EMPTY"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval knobs ---
DENSE_TOP_K      = 12
DENSE_FETCH_K    = 80
FINAL_TOP_K      = 8
SNIPPET_CHARS    = 1200
HYDE_TRIGGER_CHARS = 80

# --- Paper router ---
ROUTER_TOP_K_PAPERS = 5

# --- Heuristic thresholds ---
MIN_SNIPPETS_FOR_DIRECT   = 3
MIN_UNIQUE_DOCIDS_DIRECT  = 1

# --- Loop caps (prevent stuck states) ---
MAX_RETRIEVAL_LOOPS  = 4
MAX_REWRITE_LOOPS    = 2
MAX_DECOMPOSE_LOOPS  = 2
MAX_REFLECT_LOOPS    = 2

# --- Paths ---
PAPERS_DIR       = Path("./papers")            # each paper in ./papers/<pmid>/
PAPER_INDEX_DIR  = Path("./paper_faiss_indices")
MASTER_INDEX_DIR = Path("./master_routing_index")
MEMORY_FAISS_DIR = Path("./memory_faiss")

ENABLE_LONGTERM_MEMORY = False
STRICT_ID_ONLY = False

# --- PubMed download ---
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_BATCH_SIZE = 100      # PubMed allows up to 200 per request
PUBMED_DELAY      = 0.35     # seconds between batches (NCBI rate limit)
NCBI_EMAIL        = "bioasq_research@example.com"   # required by NCBI policy
NCBI_API_KEY      = None     # set this for 10 req/sec instead of 3

# --- Chunking ---
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 100

# --- Cross-encoder (optional) ---
CROSS_ENCODER_MODEL: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ╔══════════════════════════════════════════════╗
# ║  SHARED OBJECTS (lazy init)                  ║
# ╚══════════════════════════════════════════════╝

embeddings: HuggingFaceEmbeddings = None          # type: ignore
cross_encoder = None
paper_indices:   Dict[str, FAISS] = {}            # pmid → FAISS index
paper_metadata:  Dict[str, Dict[str, Any]] = {}   # pmid → {title, abstract, ...}
master_index:    Optional[FAISS] = None
DOCID_TO_CHUNKS: Dict[str, List[Document]] = {}   # pmid → ordered chunks
memory_vs:       Optional[FAISS] = None

MEMORY_FETCH_K = 6
MEMORY_TOP_K   = 3

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


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


# ╔══════════════════════════════════════════════╗
# ║  LLM INSTANCES                               ║
# ╚══════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 + 2: PARSE BIOASQ → DOWNLOAD FULL PAPERS FROM PUBMED  ║
# ╚══════════════════════════════════════════════════════════════════╝

def _pmid_from_url(url: str) -> Optional[str]:
    """Extract PubMed ID from URL like http://www.ncbi.nlm.nih.gov/pubmed/12345678"""
    m = re.search(r"(?:pubmed|pmc)[/:](\d+)", url)
    return m.group(1) if m else None


def load_bioasq_dataset(path: str) -> List[Dict[str, Any]]:
    """Load BioASQ JSON (single file with 'questions' key, plain list, or JSONL)."""
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
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def extract_all_pmids(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Walk the BioASQ dataset and collect every unique PubMed ID,
    along with any snippets the dataset already provides for it.
    Returns: {pmid: {"url": ..., "snippets": [text, ...]}}
    """
    pmid_info: Dict[str, Dict[str, Any]] = {}

    for q in questions:
        # From documents list
        for doc_url in q.get("documents", []):
            pmid = _pmid_from_url(doc_url)
            if pmid and pmid not in pmid_info:
                pmid_info[pmid] = {"url": doc_url, "snippets": []}

        # From snippets
        for snip in q.get("snippets", []):
            doc_url = snip.get("document", "")
            pmid = _pmid_from_url(doc_url)
            if not pmid:
                continue
            if pmid not in pmid_info:
                pmid_info[pmid] = {"url": doc_url, "snippets": []}
            text = snip.get("text", "").strip()
            if text and text not in pmid_info[pmid]["snippets"]:
                pmid_info[pmid]["snippets"].append(text)

    return pmid_info


def _parse_pubmed_xml(xml_text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse PubMed eFetch XML and extract title + abstract for each article.
    Returns {pmid: {"title": ..., "abstract": ...}}
    """
    results = {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            # PMID
            pmid_el = article.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()

            # Title
            title_el = article.find(".//ArticleTitle")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""

            # Abstract — may have multiple AbstractText elements (structured abstract)
            abstract_parts = []
            for abs_el in article.findall(".//AbstractText"):
                label = abs_el.get("Label", "")
                # Get all text including tail text of sub-elements
                text = "".join(abs_el.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)

            abstract = "\n\n".join(abstract_parts)

            # MeSH terms for extra context
            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text.strip())

            # Keywords
            keywords = []
            for kw in article.findall(".//Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

            results[pmid] = {
                "title": title,
                "abstract": abstract,
                "mesh_terms": "; ".join(mesh_terms[:15]),
                "keywords": "; ".join(keywords[:15]),
            }
    except Exception as e:
        log.error("XML parse error: %s", e)

    return results


def download_papers_from_pubmed(pmid_info: Dict[str, Dict[str, Any]],
                                 force_redownload: bool = False) -> int:
    """
    Download full abstracts from PubMed for all PMIDs.
    Stores each paper in ./papers/<pmid>/paper.json
    Returns count of successfully downloaded papers.
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure out which PMIDs we still need
    pmids_to_fetch = []
    for pmid in pmid_info:
        paper_file = PAPERS_DIR / pmid / "paper.json"
        if not force_redownload and paper_file.exists():
            continue
        pmids_to_fetch.append(pmid)

    already_have = len(pmid_info) - len(pmids_to_fetch)
    if already_have > 0:
        log.info("Already have %d papers cached, need to fetch %d more",
                 already_have, len(pmids_to_fetch))

    if not pmids_to_fetch:
        log.info("All %d papers already downloaded.", len(pmid_info))
        return len(pmid_info)

    log.info("Downloading %d papers from PubMed...", len(pmids_to_fetch))

    downloaded = 0
    # Batch fetch
    for batch_start in range(0, len(pmids_to_fetch), PUBMED_BATCH_SIZE):
        batch = pmids_to_fetch[batch_start:batch_start + PUBMED_BATCH_SIZE]
        ids_str = ",".join(batch)

        params = {
            "db": "pubmed",
            "id": ids_str,
            "rettype": "xml",
            "retmode": "xml",
            "email": NCBI_EMAIL,
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        try:
            resp = requests.get(PUBMED_EFETCH_URL, params=params, timeout=30)
            resp.raise_for_status()

            parsed = _parse_pubmed_xml(resp.text)
            log.info("  Batch %d-%d: fetched %d/%d from PubMed",
                     batch_start, batch_start + len(batch), len(parsed), len(batch))

            for pmid in batch:
                paper_dir = PAPERS_DIR / pmid
                paper_dir.mkdir(parents=True, exist_ok=True)
                paper_file = paper_dir / "paper.json"

                if pmid in parsed:
                    paper_data = {
                        "pmid": pmid,
                        "url": pmid_info.get(pmid, {}).get("url", ""),
                        "title": parsed[pmid]["title"],
                        "abstract": parsed[pmid]["abstract"],
                        "mesh_terms": parsed[pmid].get("mesh_terms", ""),
                        "keywords": parsed[pmid].get("keywords", ""),
                        "bioasq_snippets": pmid_info.get(pmid, {}).get("snippets", []),
                    }
                else:
                    # PubMed didn't return this one — use BioASQ snippets as fallback
                    snippets = pmid_info.get(pmid, {}).get("snippets", [])
                    paper_data = {
                        "pmid": pmid,
                        "url": pmid_info.get(pmid, {}).get("url", ""),
                        "title": f"PubMed article {pmid}",
                        "abstract": "\n\n".join(snippets) if snippets else "",
                        "mesh_terms": "",
                        "keywords": "",
                        "bioasq_snippets": snippets,
                        "_note": "Abstract not available from PubMed; using BioASQ snippets",
                    }

                with open(paper_file, "w") as f:
                    json.dump(paper_data, f, indent=2, ensure_ascii=False)
                downloaded += 1

        except requests.RequestException as e:
            log.error("  PubMed batch fetch failed: %s — falling back to snippets for this batch", e)
            # Fallback: save what we have from BioASQ
            for pmid in batch:
                paper_dir = PAPERS_DIR / pmid
                paper_dir.mkdir(parents=True, exist_ok=True)
                paper_file = paper_dir / "paper.json"
                if paper_file.exists():
                    continue
                snippets = pmid_info.get(pmid, {}).get("snippets", [])
                paper_data = {
                    "pmid": pmid,
                    "url": pmid_info.get(pmid, {}).get("url", ""),
                    "title": f"PubMed article {pmid}",
                    "abstract": "\n\n".join(snippets) if snippets else "",
                    "mesh_terms": "",
                    "keywords": "",
                    "bioasq_snippets": snippets,
                    "_note": "PubMed fetch failed; using BioASQ snippets only",
                }
                with open(paper_file, "w") as f:
                    json.dump(paper_data, f, indent=2, ensure_ascii=False)
                downloaded += 1

        # Rate limiting
        time.sleep(PUBMED_DELAY)

    total = already_have + downloaded
    log.info("Paper download complete: %d total papers available", total)
    return total


# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3 + 4: BUILD PER-PAPER FAISS INDICES                ║
# ╚══════════════════════════════════════════════════════════════╝

def _load_paper(pmid: str) -> Optional[Dict[str, Any]]:
    """Load a paper's JSON from its subfolder."""
    paper_file = PAPERS_DIR / pmid / "paper.json"
    if not paper_file.exists():
        return None
    with open(paper_file) as f:
        return json.load(f)


def _build_paper_text(paper: Dict[str, Any]) -> str:
    """
    Combine all available text for a paper into a single string for chunking.
    Order: title → abstract → mesh terms → keywords → BioASQ snippets (deduped).
    """
    parts = []

    title = paper.get("title", "").strip()
    if title:
        parts.append(f"Title: {title}")

    abstract = paper.get("abstract", "").strip()
    if abstract:
        parts.append(f"\n{abstract}")

    mesh = paper.get("mesh_terms", "").strip()
    if mesh:
        parts.append(f"\nMeSH Terms: {mesh}")

    keywords = paper.get("keywords", "").strip()
    if keywords:
        parts.append(f"\nKeywords: {keywords}")

    # Add BioASQ snippets that aren't already in the abstract
    bioasq_snippets = paper.get("bioasq_snippets", [])
    if bioasq_snippets and abstract:
        abstract_lower = abstract.lower()
        for snip in bioasq_snippets:
            # Only add if it's not a substring of the abstract
            if snip.strip().lower() not in abstract_lower:
                parts.append(f"\nAdditional evidence: {snip.strip()}")
    elif bioasq_snippets and not abstract:
        for snip in bioasq_snippets:
            parts.append(snip.strip())

    return "\n".join(parts)


def build_all_indices(questions: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
    """
    Master function: download papers, build per-paper FAISS, build routing index.
    """
    global paper_indices, paper_metadata, master_index, DOCID_TO_CHUNKS

    emb = _get_embeddings()

    # --- Step 1: Extract all PMIDs ---
    pmid_info = extract_all_pmids(questions)
    log.info("Found %d unique papers referenced in %d questions", len(pmid_info), len(questions))

    # --- Step 2: Download from PubMed ---
    download_papers_from_pubmed(pmid_info, force_redownload=force_rebuild)

    # --- Step 3: Build per-paper FAISS indices ---
    PAPER_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    abstract_docs_for_router: List[Document] = []
    papers_with_content = 0
    papers_empty = 0

    for pmid in pmid_info:
        # Try loading cached FAISS first
        idx_path = PAPER_INDEX_DIR / pmid
        if not force_rebuild and idx_path.exists():
            try:
                paper_indices[pmid] = FAISS.load_local(
                    str(idx_path), emb, allow_dangerous_deserialization=True,
                )
                _rebuild_docid_chunks(pmid, paper_indices[pmid])
                # Load metadata
                paper = _load_paper(pmid)
                if paper:
                    paper_metadata[pmid] = {
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", "")[:500],
                        "url": paper.get("url", ""),
                    }
                    abstract_docs_for_router.append(Document(
                        page_content=f"{paper.get('title', '')} {paper.get('abstract', '')[:400]}",
                        metadata={"paper_id": pmid},
                    ))
                papers_with_content += 1
                continue
            except Exception:
                pass

        # Load the paper
        paper = _load_paper(pmid)
        if not paper:
            papers_empty += 1
            continue

        full_text = _build_paper_text(paper)
        if not full_text.strip():
            papers_empty += 1
            log.warning("  Paper %s has no content — skipping", pmid)
            continue

        # Chunk the paper
        chunks = text_splitter.split_text(full_text)
        if not chunks:
            papers_empty += 1
            continue

        docs: List[Document] = []
        for i, chunk_text in enumerate(chunks):
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "_id": pmid,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "title": paper.get("title", "")[:100],
                    "url": paper.get("url", ""),
                },
            ))

        # Store for neighbor expansion
        DOCID_TO_CHUNKS[pmid] = docs

        # Build FAISS index
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(str(idx_path))
        paper_indices[pmid] = vs

        # Metadata for routing
        paper_metadata[pmid] = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "")[:500],
            "url": paper.get("url", ""),
        }
        abstract_docs_for_router.append(Document(
            page_content=f"{paper.get('title', '')} {paper.get('abstract', '')[:400]}",
            metadata={"paper_id": pmid},
        ))

        papers_with_content += 1

    log.info("Built FAISS indices: %d papers with content, %d empty/skipped",
             papers_with_content, papers_empty)

    # --- Step 4: Build master routing index ---
    if abstract_docs_for_router:
        MASTER_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        master_index = FAISS.from_documents(abstract_docs_for_router, emb)
        master_index.save_local(str(MASTER_INDEX_DIR))
        log.info("Built master routing index over %d papers", len(abstract_docs_for_router))
    else:
        log.warning("No papers to build routing index from!")


def _rebuild_docid_chunks(pmid: str, vs: FAISS) -> None:
    try:
        docs = list(vs.docstore._dict.values())
        docs.sort(key=lambda d: (d.metadata or {}).get("chunk_index", 0))
        DOCID_TO_CHUNKS[pmid] = docs
    except Exception:
        pass


# ╔══════════════════════════════════════════════╗
# ║  PAPER ROUTER                                ║
# ╚══════════════════════════════════════════════╝

def select_relevant_papers(question: str, top_k: int = ROUTER_TOP_K_PAPERS) -> List[str]:
    """Select the most relevant paper indices for a given question."""
    if master_index is None:
        return list(paper_indices.keys())[:top_k]
    try:
        results = master_index.similarity_search_with_score(question, k=min(top_k, len(paper_indices)))
        paper_ids = []
        for doc, score in results:
            pid = (doc.metadata or {}).get("paper_id", "")
            if pid and pid in paper_indices:
                paper_ids.append(pid)
        return paper_ids if paper_ids else list(paper_indices.keys())[:top_k]
    except Exception as e:
        log.warning("Paper selection failed: %s", e)
        return list(paper_indices.keys())[:top_k]


# ╔══════════════════════════════════════════════╗
# ║  RETRIEVAL FUNCTIONS                         ║
# ╚══════════════════════════════════════════════╝

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


def _invoke_retriever_on_paper(paper_id: str, query: str, top_k: int = DENSE_TOP_K) -> List[Document]:
    vs = paper_indices.get(paper_id)
    if vs is None:
        return []
    try:
        return vs.similarity_search(query, k=top_k, fetch_k=DENSE_FETCH_K)
    except Exception:
        return []


def _retrieve_from_papers(paper_ids: List[str], query: str) -> List[Document]:
    """Retrieve from multiple paper indices in parallel, deduplicated."""
    all_docs: List[Document] = []
    seen = set()

    with ThreadPoolExecutor(max_workers=min(len(paper_ids), 8)) as pool:
        futures = {pool.submit(_invoke_retriever_on_paper, pid, query): pid
                   for pid in paper_ids}
        for future in as_completed(futures):
            try:
                for d in future.result():
                    h = hashlib.md5(d.page_content.encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        all_docs.append(d)
            except Exception:
                pass
    return all_docs


def _expand_neighbors(docs: List[Document], window: int = 1, max_seed: int = 25) -> List[Document]:
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
    try:
        pairs = [(query, d.page_content) for d in docs]
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
            score_acc += np.array(ce.predict(pairs))
        score_acc /= len(queries)
        ranked = sorted(zip(docs, score_acc), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]
    except Exception:
        return docs[:top_n]


def _format_snippets(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> str:
    blocks = []
    for i, d in enumerate(docs[:limit], start=1):
        meta = d.metadata or {}
        pmid = meta.get("_id", "")
        title = meta.get("title", "")[:80]
        chunk_idx = meta.get("chunk_index", "?")
        total = meta.get("total_chunks", "?")
        header = f"[{i}] PMID={pmid} chunk={chunk_idx}/{total}"
        if title:
            header += f" | {title}"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_payload(docs: List[Document], query: str, limit: int = FINAL_TOP_K) -> dict:
    snippets_text = _format_snippets(docs, query, limit)
    chunks = []
    for d in docs[:limit]:
        meta = d.metadata or {}
        chunks.append({
            "text": d.page_content,
            "pmid": meta.get("_id", ""),
            "chunk_index": meta.get("chunk_index", ""),
            "title": meta.get("title", ""),
        })
    return {"snippets_text": snippets_text, "chunks": chunks}


# ── Main retrieval tool ──

@tool("smart_retrieve_jsonl")
def smart_retrieve_jsonl(query: str) -> str:
    """
    Hybrid retriever: paper selection → parallel multi-paper retrieval
    → neighbor expansion → cross-encoder rerank → format snippets.
    """
    subqs = [q for q in _parse_subquestions(query) if q.strip()]
    all_candidates: List[Document] = []

    full_query = " ".join(subqs)
    selected_papers = select_relevant_papers(full_query)
    log.info("Router selected %d papers for: %s", len(selected_papers), full_query[:80])

    for q in subqs:
        candidates: List[Document] = []

        # 1) Code pinpointing
        codes = extract_codes(q)
        if codes:
            for pid in selected_papers:
                for chunk in DOCID_TO_CHUNKS.get(pid, []):
                    if all(c in chunk.page_content for c in codes):
                        candidates.append(chunk)
            if STRICT_ID_ONLY and candidates:
                all_candidates.extend(_dedup_chunks(candidates))
                continue

        # 2) Dense retrieval from selected papers
        try:
            candidates.extend(_retrieve_from_papers(selected_papers, q))
        except Exception:
            candidates.extend(_retrieve_from_papers(list(paper_indices.keys())[:10], q))

        # 3) HyDE boost for short questions
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

    # Code prioritization
    codes = extract_codes(query)
    if codes:
        hits = [d for d in reranked if all(c in (d.page_content or "") for c in codes)]
        if hits:
            reranked = hits if STRICT_ID_ONLY else hits + [d for d in reranked if d not in hits]

    payload = _docs_to_payload(reranked, query=query, limit=FINAL_TOP_K)
    return json.dumps(payload, ensure_ascii=False)


# ╔══════════════════════════════════════════════╗
# ║  LONG-TERM MEMORY (optional)                 ║
# ╚══════════════════════════════════════════════╝

def _load_memory():
    global memory_vs
    if ENABLE_LONGTERM_MEMORY and MEMORY_FAISS_DIR.exists():
        try:
            memory_vs = FAISS.load_local(
                str(MEMORY_FAISS_DIR), _get_embeddings(),
                allow_dangerous_deserialization=True)
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
    blocks = [f"[M{i}] {(d.page_content or '')[:600]}" for i, d in enumerate(docs[:MEMORY_TOP_K], 1)]
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


# ╔══════════════════════════════════════════════╗
# ║  AGENT STATE & HELPERS                       ║
# ╚══════════════════════════════════════════════╝

AgentState = dict   # plain dict state for langgraph compatibility


def _latest_user_question(msgs: list) -> str:
    for m in reversed(msgs):
        role = getattr(m, "type", getattr(m, "role", ""))
        if role in ("human", "user"):
            return str(getattr(m, "content", "") or "")
    return ""

def _msgs_since_last_user(msgs: list) -> list:
    for i in range(len(msgs) - 1, -1, -1):
        role = getattr(msgs[i], "type", getattr(msgs[i], "role", ""))
        if role in ("human", "user"):
            return msgs[i:]
    return msgs

def _context_signals(ctx: str) -> Dict[str, Any]:
    parts = [p for p in re.split(r"\n\s*---\s*\n", ctx or "") if p.strip()]
    docids = set(re.findall(r"PMID=([^\s]+)", ctx or ""))
    return {"snippet_count": len(parts), "unique_docids": len(docids)}

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


# ╔══════════════════════════════════════════════╗
# ║  PROMPTS (tuned for Gemma-3-27b-it)          ║
# ╚══════════════════════════════════════════════╝

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
    "You are a biomedical question-answering expert. Answer using ONLY the retrieved PubMed snippets below.\n\n"
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


# ╔══════════════════════════════════════════════╗
# ║  PYDANTIC MODELS                             ║
# ╚══════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════╗
# ║  GRAPH NODE FUNCTIONS                        ║
# ╚══════════════════════════════════════════════╝

def prepare_question(state: dict, config: Optional[dict] = None) -> dict:
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
        "_retrieval_loops": 0,
        "_rewrite_loops": 0,
        "_decompose_loops": 0,
        "_reflect_loops": 0,
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
    loops = state.get("_retrieval_loops", 0) + 1
    return {"messages": [msg], "_retrieval_loops": loops}


def capture_context(state: dict) -> dict:
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


def grade_retrieval(state: dict) -> Literal["generate_draft", "rewrite_question", "decompose_question"]:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    sig = _context_signals(ctx)

    loops = state.get("_retrieval_loops", 0)

    # Hard cap
    if loops >= MAX_RETRIEVAL_LOOPS:
        return "generate_draft"

    # If we have reasonable content, generate
    if sig["snippet_count"] >= MIN_SNIPPETS_FOR_DIRECT:
        return "generate_draft"

    # Too few snippets
    if sig["snippet_count"] < MIN_SNIPPETS_FOR_DIRECT:
        if _looks_multihop(q) and state.get("_decompose_loops", 0) < MAX_DECOMPOSE_LOOPS:
            return "decompose_question"
        if state.get("_rewrite_loops", 0) < MAX_REWRITE_LOOPS:
            return "rewrite_question"
        return "generate_draft"

    # LLM-based grading
    try:
        resp = llm_grader.with_structured_output(RetrievalGrade).invoke(
            [{"role": "user", "content": GRADE_PROMPT.format(q=q, ctx=ctx, **sig)}]
        )
    except Exception:
        return "generate_draft"

    if resp.action == "ANSWER":
        return "generate_draft"
    if resp.action == "DECOMPOSE" and state.get("_decompose_loops", 0) < MAX_DECOMPOSE_LOOPS:
        return "decompose_question"
    if resp.action == "REWRITE" and state.get("_rewrite_loops", 0) < MAX_REWRITE_LOOPS:
        return "rewrite_question"
    return "generate_draft"


def rewrite_question(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    rewrite_loops = state.get("_rewrite_loops", 0) + 1
    codes = extract_codes(q)
    if codes:
        rewritten = f"{'  '.join(codes)} biomedical mechanism function role"
        return {"standalone_question": rewritten, "plan": [], "_rewrite_loops": rewrite_loops}
    try:
        rewritten = llm_query.invoke(
            [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}]
        ).content.strip().strip('"')
        return {"standalone_question": rewritten, "plan": [], "_rewrite_loops": rewrite_loops}
    except Exception:
        return {"standalone_question": q, "plan": [], "_rewrite_loops": rewrite_loops}


def decompose_question(state: dict) -> dict:
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


def generate_draft(state: dict) -> dict:
    q = state.get("question") or ""
    ctx = state.get("context") or ""
    mem = state.get("memory_context") or ""

    sys_msg = SystemMessage(content="You are a biomedical expert. Ground every claim in the provided snippets.")
    prompt = GENERATE_PROMPT.format(question=q, context=ctx, memory=mem)

    try:
        draft = llm_answer.invoke([sys_msg, HumanMessage(content=prompt)]).content
    except Exception as e:
        draft = f"LLM error: {type(e).__name__}: {e}"
    return {"draft_answer": draft}


def reflect(state: dict) -> dict:
    q = state.get("standalone_question") or state.get("question") or ""
    ctx = state.get("context") or ""
    draft = state.get("draft_answer") or ""
    reflect_loops = state.get("_reflect_loops", 0) + 1

    if reflect_loops > MAX_REFLECT_LOOPS:
        return {"reflect_action": "FINALIZE", "reflect_queries": [],
                "reflection": "Max reflection loops.", "_reflect_loops": reflect_loops}
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


def route_after_reflect(state: dict) -> Literal["finalize_answer", "call_retriever", "ask_user"]:
    action = state.get("reflect_action", "FINALIZE")
    if action == "RETRIEVE_MORE" and (state.get("reflect_queries") or []):
        if state.get("_retrieval_loops", 0) < MAX_RETRIEVAL_LOOPS:
            return "call_retriever"
        return "finalize_answer"
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
    return {"messages": [AIMessage(content=_clean_model_text(state.get("draft_answer") or ""))]}


def ask_user(state: dict) -> dict:
    qs = state.get("reflect_queries") or []
    q = qs[0] if qs else "Could you clarify your question?"
    return {"messages": [AIMessage(content=q)]}


def write_memory_node(state: dict, config: Optional[dict] = None) -> dict:
    config = config or {}
    thread_id = (config.get("configurable", {}) or {}).get("thread_id", "default")
    write_longterm_memory(state.get("question") or "", state.get("draft_answer") or "", thread_id=thread_id)
    return {}


# ╔══════════════════════════════════════════════╗
# ║  BUILD GRAPH                                 ║
# ╚══════════════════════════════════════════════╝

def build_graph():
    workflow = StateGraph(dict)

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

    # Entry
    workflow.add_edge(START, "prepare_question")

    # Routing after prepare
    def route_to_planner(state) -> Literal["plan_question", "call_retriever"]:
        q = state.get("standalone_question") or state.get("question") or ""
        return "plan_question" if (not extract_codes(q) and _looks_multihop(q)) else "call_retriever"

    workflow.add_conditional_edges("prepare_question", route_to_planner,
        {"plan_question": "plan_question", "call_retriever": "call_retriever"})

    workflow.add_edge("plan_question", "call_retriever")
    workflow.add_edge("call_retriever", "retrieve")
    workflow.add_edge("retrieve", "capture_context")

    workflow.add_conditional_edges("capture_context", grade_retrieval, {
        "generate_draft": "generate_draft",
        "rewrite_question": "rewrite_question",
        "decompose_question": "decompose_question",
    })

    workflow.add_edge("rewrite_question", "call_retriever")
    workflow.add_edge("decompose_question", "call_retriever")
    workflow.add_edge("generate_draft", "reflect")

    workflow.add_conditional_edges("reflect", route_after_reflect, {
        "finalize_answer": "finalize_answer",
        "call_retriever": "apply_reflection_queries",
        "ask_user": "ask_user",
    })

    workflow.add_edge("apply_reflection_queries", "call_retriever")
    workflow.add_edge("finalize_answer", "write_memory")
    workflow.add_edge("write_memory", END)
    workflow.add_edge("ask_user", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ╔══════════════════════════════════════════════╗
# ║  EVALUATION HARNESS                          ║
# ╚══════════════════════════════════════════════╝

def evaluate_bioasq(dataset_path: str, max_questions: int = None,
                    output_path: str = "bioasq_results.json"):
    """Load dataset → download papers → build indices → run all questions."""
    questions = load_bioasq_dataset(dataset_path)
    log.info("Loaded %d questions from %s", len(questions), dataset_path)

    # Phase 1-4: Download papers & build indices
    build_all_indices(questions)

    # Phase 5: Run pipeline
    graph = build_graph()
    run_config = {"recursion_limit": 30, "configurable": {"thread_id": "bioasq-eval"}}

    results = []
    total = min(len(questions), max_questions) if max_questions else len(questions)

    for i, q_data in enumerate(questions[:total]):
        q_body = q_data.get("body", "")
        q_type = q_data.get("type", "factoid")
        q_id   = q_data.get("id", f"q_{i}")
        ideal  = q_data.get("ideal_answer", [""])
        exact  = q_data.get("exact_answer", "")

        log.info("\n[%d/%d] Q(%s): %s", i + 1, total, q_type, q_body[:100])

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
        log.info("  → %s", answer[:200])

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("\nSaved %d results to %s", len(results), output_path)
    return results


# ╔══════════════════════════════════════════════╗
# ║  MAIN                                        ║
# ╚══════════════════════════════════════════════╝

if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        dataset_path = sys.argv[1]
        max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate_bioasq(dataset_path, max_questions=max_q)
    else:
        print("=" * 60)
        print("BioASQ Agentic QA System — v3 (Full Paper Download)")
        print("=" * 60)
        print("\nUsage:")
        print("  python QA_BioASQ_v3.py <bioasq_dataset.json> [max_questions]")
        print("\nThis will:")
        print("  1. Parse the BioASQ dataset for all PubMed IDs")
        print("  2. Download full abstracts from PubMed (cached in ./papers/)")
        print("  3. Chunk & build per-paper FAISS indices")
        print("  4. Build a master routing index")
        print("  5. Run each question through the agentic pipeline")
        print("  6. Save results to bioasq_results.json")
