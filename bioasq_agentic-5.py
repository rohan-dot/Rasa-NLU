#!/usr/bin/env python3
"""
BioASQ Task 14b Phase B — Agentic RAG with On-the-Fly Indexing
================================================================

PubMed is ONLY a data pipe. The model does its own retrieval:

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  1. LLM generates search terms                              │
  │  2. PubMed E-utilities FETCHES articles (data source only)  │
  │  3. We CHUNK abstracts into passages                        │
  │  4. We EMBED chunks with PubMedBERT (local model)          │
  │  5. We BUILD a FAISS vector index on the fly                │
  │  6. LLM generates a QUERY EMBEDDING                        │
  │  7. FAISS retrieves top-k passages by semantic similarity   │
  │  8. LLM EVALUATES: enough evidence?                         │
  │     └── NO → generate new terms → fetch more → re-index    │
  │  9. LLM ANSWERS from retrieved passages                     │
  │  10. LLM VERIFIES answer against evidence                   │
  │  11. CONSENSUS across multiple passes                       │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

Start vLLM server first:
    vllm serve google/gemma-3-27b-it --port 8000 --max-model-len 8192 \
        --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --dtype bfloat16

Then run:
    pip install requests sentence-transformers faiss-cpu numpy
    python bioasq_agentic.py \
        --test-input  BioASQ-task14bPhaseB-testset1.json \
        --training    training13b.json \
        -o submission_phaseB.json
"""

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class Passage:
    text: str
    pmid: str
    doc_url: str
    section: str = "abstract"
    offset_begin: int = 0
    offset_end: int = 0
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0


# =====================================================================
# PubMed Downloader — downloads articles, NEVER ranks them
# =====================================================================
# PubMed is a DATA SOURCE only. It gives us raw articles.
# ALL retrieval/ranking is done by OUR models (FAISS + BM25).
#
# Three ways to download articles:
#   1. get_seed_pmids()   — initial broad fetch to populate corpus
#   2. get_related()      — elink API: "articles similar to this one"
#   3. get_citing()       — elink API: "articles that cite this one"
#
# None of these rank results. They just hand us PMIDs to download.
# =====================================================================

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

class PubMedDownloader:
    """Downloads articles from PubMed. Does NOT search or rank."""

    def __init__(self, api_key: str | None = None):
        self.session = requests.Session()
        self.api_key = api_key
        self._last_request = 0.0
        self._min_interval = 0.2 if api_key else 1.0
        self._backoff = 1.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        wait = max(self._min_interval, self._backoff) - elapsed
        if wait > 0: time.sleep(wait)
        self._last_request = time.time()

    def _request(self, url, params, max_retries=3):
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    self._backoff = min(self._backoff * 2, 10.0)
                    time.sleep(self._backoff + attempt * 2)
                    continue
                resp.raise_for_status()
                self._backoff = max(self._backoff * 0.8, self._min_interval)
                return resp
            except Exception as e:
                log.warning("    Request failed (attempt %d): %s", attempt+1, e)
                time.sleep(2 * (attempt + 1))
        return None

    # ---- Download methods (get articles into our corpus) ----

    def get_seed_pmids(self, terms: str, max_results: int = 50) -> list[str]:
        """
        Broad fetch of PMIDs matching terms. This is NOT ranked retrieval —
        it's just finding articles that exist in PubMed about a topic.
        We fetch MORE than needed (50) because OUR index does the ranking.
        """
        params = {"db": "pubmed", "term": terms, "retmax": max_results,
                  "retmode": "json"}
        if self.api_key: params["api_key"] = self.api_key
        resp = self._request(ESEARCH_URL, params)
        if not resp: return []
        try: return resp.json().get("esearchresult", {}).get("idlist", [])
        except: return []

    def get_related(self, pmid: str, max_results: int = 20) -> list[str]:
        """
        Get PMIDs of articles RELATED to a known-good article.
        Uses NCBI's elink API — this finds articles by citation network
        and content similarity, NOT by search ranking.
        This is how you expand your corpus from a seed.
        """
        params = {
            "dbfrom": "pubmed", "db": "pubmed", "id": pmid,
            "cmd": "neighbor_score", "retmode": "json",
        }
        if self.api_key: params["api_key"] = self.api_key
        resp = self._request(ELINK_URL, params)
        if not resp: return []
        try:
            data = resp.json()
            link_sets = data.get("linksets", [])
            if not link_sets: return []
            links = link_sets[0].get("linksetdbs", [])
            for ls in links:
                if ls.get("linkname") == "pubmed_pubmed":
                    return [str(l["id"]) for l in ls.get("links", [])][:max_results]
            return []
        except Exception as e:
            log.warning("elink parse failed: %s", e)
            return []

    def get_citing(self, pmid: str, max_results: int = 15) -> list[str]:
        """
        Get PMIDs of articles that CITE a known-good article.
        These are likely to contain related findings or updates.
        """
        params = {
            "dbfrom": "pubmed", "db": "pubmed", "id": pmid,
            "cmd": "neighbor", "linkname": "pubmed_pubmed_citedin",
            "retmode": "json",
        }
        if self.api_key: params["api_key"] = self.api_key
        resp = self._request(ELINK_URL, params)
        if not resp: return []
        try:
            data = resp.json()
            link_sets = data.get("linksets", [])
            if not link_sets: return []
            links = link_sets[0].get("linksetdbs", [])
            for ls in links:
                if "citedin" in ls.get("linkname", ""):
                    return [str(l["id"]) for l in ls.get("links", [])][:max_results]
            return []
        except:
            return []

    def fetch_articles(self, pmids: list[str]) -> list[dict]:
        """Download full article data (title + abstract) by PMID."""
        if not pmids: return []
        # Batch in groups of 50 to avoid URL length limits
        all_articles = []
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]
            params = {"db": "pubmed", "id": ",".join(batch),
                      "rettype": "xml", "retmode": "xml"}
            if self.api_key: params["api_key"] = self.api_key
            resp = self._request(EFETCH_URL, params)
            if resp:
                all_articles.extend(self._parse_xml(resp.text))
        return all_articles

    def _parse_xml(self, xml_text):
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for elem in root.findall(".//PubmedArticle"):
                pmid_el = elem.find(".//PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
                title_el = elem.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""
                abstract_parts = []
                for abs_el in elem.findall(".//AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label and text: abstract_parts.append(f"{label}: {text}")
                    elif text: abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
                if pmid and (title or abstract):
                    articles.append({"pmid": pmid, "title": title, "abstract": abstract,
                                     "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"})
        except ET.ParseError as e:
            log.warning("XML parse error: %s", e)
        return articles


# =====================================================================
# Embedding Engine — PubMedBERT for semantic encoding
# =====================================================================

class EmbeddingEngine:
    def __init__(self, model_name="pritamdeka/S-PubMedBert-MS-MARCO"):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("Embedding dim: %d", self.dim)

    def encode(self, texts, batch_size=64):
        return np.array(self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True), dtype=np.float32)

    def encode_one(self, text):
        return self.encode([text])[0]


# =====================================================================
# On-the-Fly Hybrid Index — FAISS (semantic) + BM25 (keyword)
# =====================================================================

class LiveIndex:
    """
    Dual-mode retrieval index built on the fly per question:
      - FAISS: semantic similarity via PubMedBERT embeddings
      - BM25:  exact keyword matching (catches drug names, genes, numbers)
      - Hybrid: Reciprocal Rank Fusion merges both rankings

    Why hybrid wins:
      BM25 alone:  finds "pasireotide" but misses "somatostatin analog"
      FAISS alone: finds "somatostatin analog" but might rank "pasireotide" lower
      Hybrid:      if either method finds it, it surfaces
    """

    def __init__(self, embedding_engine):
        import faiss
        from rank_bm25 import BM25Okapi
        self.faiss_mod = faiss
        self.embedder = embedding_engine
        self.dim = embedding_engine.dim
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.passages: list[Passage] = []
        self.seen_texts: set[str] = set()
        self._bm25 = None
        self._bm25_dirty = True  # rebuild BM25 when passages change

    def _tokenize(self, text):
        """Simple whitespace + lowercase tokenizer for BM25."""
        return re.findall(r'[a-z0-9\-]+', text.lower())

    def _rebuild_bm25(self):
        """Rebuild BM25 index from current passages."""
        from rank_bm25 import BM25Okapi
        if not self.passages:
            self._bm25 = None
            return
        corpus = [self._tokenize(p.text) for p in self.passages]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_dirty = False

    def add_article(self, article, question=""):
        """Multi-granularity indexing: title + full abstract + overlapping chunks."""
        pmid, url = article["pmid"], article["url"]
        new_passages = []

        title = article.get("title", "").strip()
        if title and title not in self.seen_texts:
            self.seen_texts.add(title)
            new_passages.append(Passage(text=title, pmid=pmid, doc_url=url,
                                        section="title", offset_end=len(title)))

        abstract = article.get("abstract", "").strip()
        if abstract:
            # Full abstract
            if abstract not in self.seen_texts and len(abstract) > 50:
                self.seen_texts.add(abstract)
                new_passages.append(Passage(
                    text=abstract, pmid=pmid, doc_url=url,
                    section="abstract", offset_begin=0,
                    offset_end=len(abstract)))

            # Overlapping 3-sentence chunks, stride 1
            sentences = re.split(r'(?<=[.!?])\s+', abstract)
            if len(sentences) > 3:
                for i in range(0, len(sentences) - 2):
                    chunk = " ".join(s.strip() for s in sentences[i:i+3]).strip()
                    if len(chunk) < 50 or chunk in self.seen_texts:
                        continue
                    begin = abstract.find(sentences[i])
                    if begin == -1: begin = 0
                    self.seen_texts.add(chunk)
                    new_passages.append(Passage(
                        text=chunk, pmid=pmid, doc_url=url,
                        section="abstract",
                        offset_begin=begin, offset_end=begin+len(chunk)))

        if not new_passages:
            return
        texts = [p.text for p in new_passages]
        embeddings = self.embedder.encode(texts)
        for p, emb in zip(new_passages, embeddings):
            p.embedding = emb
        self.faiss_index.add(embeddings)
        self.passages.extend(new_passages)
        self._bm25_dirty = True

    def add_raw_passages(self, passages):
        if not passages: return
        texts = [p.text for p in passages]
        embs = self.embedder.encode(texts)
        for p, e in zip(passages, embs): p.embedding = e
        self.faiss_index.add(embs)
        self.passages.extend(passages)
        self._bm25_dirty = True

    def search_semantic(self, query, top_k=20):
        """Pure FAISS semantic search."""
        if self.faiss_index.ntotal == 0: return []
        qe = self.embedder.encode_one(query).reshape(1, -1)
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(qe, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.passages):
                results.append((idx, float(score)))
        return results

    def search_bm25(self, query, top_k=20):
        """Pure BM25 keyword search."""
        if self._bm25_dirty:
            self._rebuild_bm25()
        if not self._bm25: return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]

    def search_hybrid(self, query, top_k=15, semantic_weight=0.6, bm25_weight=0.4):
        """
        Reciprocal Rank Fusion (RRF) of semantic + BM25 results.

        RRF score = sum( 1 / (k + rank) ) across both methods
        This is the standard approach used by Elasticsearch, Pinecone, etc.
        It's better than raw score combination because BM25 and cosine
        similarity are on different scales.
        """
        k_rrf = 60  # standard RRF constant

        sem_results = self.search_semantic(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # Build RRF scores
        rrf_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(sem_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + semantic_weight / (k_rrf + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (k_rrf + rank + 1)

        # Sort by RRF score and deduplicate
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        used_sigs = set()

        for idx, score in ranked:
            if idx >= len(self.passages): continue
            p = self.passages[idx]
            sig = p.text[:80].lower()
            if sig in used_sigs: continue
            used_sigs.add(sig)
            p.similarity_score = score
            results.append(p)
            if len(results) >= top_k: break

        return results

    # Keep backward compat
    def search(self, query, top_k=15):
        return self.search_hybrid(query, top_k)

    @property
    def total_passages(self): return len(self.passages)
    @property
    def total_articles(self): return len({p.pmid for p in self.passages})


# =====================================================================
# LLM Client — vLLM at localhost:8000
# =====================================================================

class LLMEngine:
    def __init__(self, base_url="http://localhost:8000", model="google/gemma-3-27b-it"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
            log.info("vLLM: %s — models: %s", self.base_url, available)
            if available and self.model not in available:
                self.model = available[0]
        except Exception as e:
            log.error("Cannot reach vLLM at %s: %s", self.base_url, e); sys.exit(1)

    def generate(self, prompt, max_tokens=1024, temperature=0.3):
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95}
        for attempt in range(3):
            try:
                resp = self.session.post(f"{self.base_url}/v1/chat/completions",
                                         json=payload, timeout=180)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("LLM failed (attempt %d): %s", attempt+1, e)
                time.sleep(2*(attempt+1))
        return ""


# =====================================================================
# Few-Shot Bank
# =====================================================================

class FewShotBank:
    def __init__(self):
        self.examples = {"factoid": [], "list": [], "yesno": [], "summary": []}

    def load(self, path, max_per_type=40):
        with open(path) as f: data = json.load(f)
        for q in data.get("questions", []):
            qtype = q.get("type", "").lower()
            if qtype not in self.examples: continue
            snippets = q.get("snippets", [])
            exact, ideal = q.get("exact_answer"), q.get("ideal_answer")
            if not snippets or not ideal: continue
            if qtype != "summary" and not exact: continue
            self.examples[qtype].append({
                "body": q["body"], "snippets": [s.get("text","") for s in snippets[:5]],
                "exact_answer": exact,
                "ideal_answer": ideal if isinstance(ideal, str) else ideal[0] if isinstance(ideal, list) and ideal else "",
            })
        for qt in self.examples:
            self.examples[qt] = sorted(self.examples[qt], key=lambda x: len(x["body"]))[:max_per_type]
            log.info("  Few-shot %s: %d", qt, len(self.examples[qt]))

    def get(self, qtype, n=2):
        exs = self.examples.get(qtype, [])
        short = [e for e in exs if len(" ".join(e["snippets"])) < 1500]
        return (short or exs)[:n]


# =====================================================================
# Prompts
# =====================================================================

def prompt_generate_queries(question, previous=None):
    p = ("You are a biomedical retrieval expert. Generate 3 short PubMed search queries (3-6 words each) "
         "to find articles answering this question. No boolean operators, no field tags.\n\n"
         f"QUESTION: {question}\n\n")
    if previous:
        p += "Previous queries found insufficient evidence. Try DIFFERENT terms:\n"
        p += "\n".join(f"  - {q}" for q in previous[-3:]) + "\n\n"
    p += "Return exactly 3 queries, numbered 1-3.\n\nQUERIES:\n"
    return p

def prompt_evaluate(question, passages):
    block = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages[:10]))
    return (f"Can this question be answered with these passages?\n\nQUESTION: {question}\n\n"
            f"PASSAGES:\n{block}\n\nReply SUFFICIENT or INSUFFICIENT.\n\nVERDICT:")

def _sblock(texts, max_chars=4000):
    b = ""
    for i, s in enumerate(texts, 1):
        line = f"[{i}] {s.strip()}\n"
        if len(b) + len(line) > max_chars: break
        b += line
    return b.strip()

def _fmt_exact(exact):
    if isinstance(exact, str): return exact
    if isinstance(exact, list):
        if exact and isinstance(exact[0], list):
            return "; ".join([", ".join(x) if isinstance(x, list) else str(x) for x in exact])
        return ", ".join(str(x) for x in exact)
    return str(exact)

def prompt_answer(question, qtype, passages, few_shot):
    inst = {"factoid": "Provide EXACT_ANSWER: a short phrase. Then IDEAL_ANSWER: 2-4 sentences.",
            "yesno": "Provide EXACT_ANSWER: exactly 'yes' or 'no'. Then IDEAL_ANSWER: 2-4 sentences.",
            "list": "Provide EXACT_ANSWER: items prefixed with '- '. Then IDEAL_ANSWER: 2-4 sentences.",
            "summary": "Provide IDEAL_ANSWER: a comprehensive 3-6 sentence paragraph."}
    p = f"You are an expert biomedical QA system. {inst.get(qtype, inst['summary'])}\n\n"
    for ex in few_shot[:2]:
        p += f"---\nQUESTION: {ex['body']}\nEVIDENCE:\n{_sblock(ex['snippets'], 800)}\n"
        if qtype != "summary": p += f"EXACT_ANSWER: {_fmt_exact(ex['exact_answer'])}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQUESTION: {question}\nEVIDENCE:\n{_sblock(passages)}\n"
    p += "EXACT_ANSWER:" if qtype != "summary" else "IDEAL_ANSWER:"
    return p

def prompt_verify(question, passages, candidate):
    return (f"Check this answer against evidence. Fix errors.\n\nQUESTION: {question}\n"
            f"EVIDENCE:\n{_sblock(passages, 3000)}\n\nCANDIDATE:\n{candidate}\n\nCORRECTED_ANSWER:")

def _simple_kw(question):
    STOP = set("what is are the a an of in on to for with by from and or not how does do can which who whom where when why this that these those it its has have had been being please list describe common main types most often typically usually associated caused present".split())
    tokens = re.findall(r"[A-Za-z0-9\-']+", question)
    return " ".join(t for t in tokens if t.lower() not in STOP and len(t) > 2)[:80]


# =====================================================================
# Parsers & Consensus
# =====================================================================

def parse_factoid(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    e = parts[0].strip().strip('"\'').rstrip(".")
    for pfx in ["The answer is","The exact answer is","Answer:"]:
        if e.lower().startswith(pfx.lower()): e = e[len(pfx):].strip()
    ideal = parts[1].strip() if len(parts)==2 else ""
    return [e] if e else ["unknown"], ideal or e

def parse_yesno(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    e = parts[0].strip().lower().strip('"\'').rstrip(".")
    ideal = parts[1].strip() if len(parts)==2 else ""
    if "yes" in e[:20]: exact="yes"
    elif "no" in e[:20]: exact="no"
    else: exact = "yes" if ideal and "yes" in ideal[:50].lower() else "no"
    return exact, ideal or f"The answer is {exact}."

def parse_list(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    lr = parts[0].strip(); ideal = parts[1].strip() if len(parts)==2 else ""
    items = []
    for line in lr.split("\n"):
        line = re.sub(r'^[\-\*•]\s*', '', line.strip())
        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip().strip('"\'').rstrip(".")
        if line and len(line) > 1: items.append([line])
    return items or [["unknown"]], ideal

def parse_summary(r):
    return re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', r, flags=re.IGNORECASE).strip() or "No answer."

def con_factoid(c):
    flat = [x[0].lower().strip() for x in c if x]
    if not flat: return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip() == best: return x
    return [best]
def con_yesno(c): return Counter(c).most_common(1)[0][0]
def con_list(c):
    items={}
    for cand in c:
        for s in cand:
            k=s[0].lower().strip() if s else ""
            if k: items.setdefault(k,s)
    return list(items.values()) or [["unknown"]]
def con_ideal(c):
    if not c: return ""
    s=sorted(c,key=len); return s[len(s)//2]


# =====================================================================
# AGENT TOOLS — each is a callable the LLM can invoke
# =====================================================================

class AgentTools:
    """
    Tool registry. PubMed tools DOWNLOAD data. Index tools RETRIEVE.
    The LLM never sees PubMed's ranking — only our own.
    """

    def __init__(self, pubmed: PubMedDownloader, index: LiveIndex,
                 llm: 'LLMEngine'):
        self.pubmed = pubmed
        self.index = index
        self.llm = llm
        self.seen_pmids: set[str] = set()
        self.queries_used: list[str] = []

    # ---- DOWNLOAD tools (populate corpus) ----

    def download_seed(self, terms: str, max_articles: int = 30) -> int:
        """Broad download of articles about a topic into local index."""
        self.queries_used.append(terms)
        pmids = self.pubmed.get_seed_pmids(terms, max_articles)
        new_pmids = [p for p in pmids if p not in self.seen_pmids]
        self.seen_pmids.update(new_pmids)
        if not new_pmids: return 0
        articles = self.pubmed.fetch_articles(new_pmids)
        for art in articles:
            self.index.add_article(art)
        return len(articles)

    def expand_related(self, pmid: str, max_articles: int = 15) -> int:
        """Download articles related to a known-good one via citation network."""
        related_pmids = self.pubmed.get_related(pmid, max_articles)
        new_pmids = [p for p in related_pmids if p not in self.seen_pmids]
        self.seen_pmids.update(new_pmids)
        if not new_pmids: return 0
        articles = self.pubmed.fetch_articles(new_pmids)
        for art in articles:
            self.index.add_article(art)
        return len(articles)

    def expand_citing(self, pmid: str, max_articles: int = 10) -> int:
        """Download articles that cite a known-good one."""
        citing_pmids = self.pubmed.get_citing(pmid, max_articles)
        new_pmids = [p for p in citing_pmids if p not in self.seen_pmids]
        self.seen_pmids.update(new_pmids)
        if not new_pmids: return 0
        articles = self.pubmed.fetch_articles(new_pmids)
        for art in articles:
            self.index.add_article(art)
        return len(articles)

    # ---- RETRIEVAL tools (search YOUR index, not PubMed) ----

    def search_hybrid(self, query: str, top_k: int = 15) -> list[Passage]:
        """OUR retrieval: hybrid semantic + BM25 with rank fusion."""
        return self.index.search_hybrid(query, top_k)

    def evaluate(self, question: str) -> str:
        """LLM judges if current evidence is sufficient."""
        top = self.search_hybrid(question, top_k=10)
        if not top: return "INSUFFICIENT"
        texts = [p.text for p in top]
        resp = self.llm.generate(
            prompt_evaluate(question, texts), max_tokens=20, temperature=0.1)
        return "SUFFICIENT" if "SUFFICIENT" in resp.upper() else "INSUFFICIENT"

    def rerank(self, question: str, passages: list[Passage],
               top_k: int = 10) -> list[Passage]:
        """LLM re-scores passage relevance for final precision."""
        if not passages: return []
        texts = [p.text for p in passages[:20]]
        block = "\n".join(f"[{i+1}] {t[:200]}" for i, t in enumerate(texts))
        prompt = (f"Rate each passage's relevance to answering this question.\n"
                  f"QUESTION: {question}\n\nPASSAGES:\n{block}\n\n"
                  f"Return scores 0-2 per line: [N] SCORE\nSCORES:\n")
        resp = self.llm.generate(prompt, max_tokens=256, temperature=0.1)
        scores = {}
        for line in resp.strip().split("\n"):
            m = re.match(r'\[(\d+)\]\s*(\d)', line)
            if m: scores[int(m.group(1))-1] = int(m.group(2))
        for i, p in enumerate(passages[:20]):
            if i in scores:
                p.similarity_score = p.similarity_score * 0.4 + (scores[i] / 2.0) * 0.6
        return sorted(passages, key=lambda p: p.similarity_score, reverse=True)[:top_k]


# =====================================================================
# THE AGENT
# =====================================================================

class BioASQAgent:
    """
    Agentic RAG pipeline:
      Phase 1 — BUILD CORPUS: download articles into local FAISS+BM25 index
      Phase 2 — RETRIEVE: search YOUR index (not PubMed's)
      Phase 3 — EXPAND: find best articles → download related → re-search
      Phase 4 — ANSWER: multi-pass generation + verification + consensus
    """

    def __init__(self, llm, pubmed, embedder, bank, num_passes=3, max_iters=3, max_articles=30):
        self.llm = llm
        self.pubmed = pubmed
        self.embedder = embedder
        self.bank = bank
        self.num_passes = num_passes
        self.max_iters = max_iters
        self.max_articles = max_articles

    def solve(self, question):
        qid, body = question["id"], question["body"]
        qtype = question.get("type", "summary").lower()
        log.info("━━━ %s [%s]: %.70s", qid, qtype, body)

        index = LiveIndex(self.embedder)
        tools = AgentTools(self.pubmed, index, self.llm)

        self._retrieve(body, qtype, question, index, tools)
        result = self._answer(body, qtype, index, tools)
        result["id"] = qid
        log.info("  ✓ exact=%s", str(result.get("exact_answer","N/A"))[:60])
        return result

    def _retrieve(self, question, qtype, raw_q, index, tools):

        # ── Step 0: Ingest gold data ──
        gold_passages = []
        for gs in raw_q.get("snippets", []):
            text = gs.get("text","").strip()
            if not text: continue
            doc_url = gs.get("document","")
            pm = re.search(r'/pubmed/(\d+)', doc_url)
            gold_passages.append(Passage(
                text=text, pmid=pm.group(1) if pm else "gold", doc_url=doc_url,
                section=gs.get("beginSection","abstract"),
                offset_begin=gs.get("offsetInBeginSection",0),
                offset_end=gs.get("offsetInEndSection",0)))
        if gold_passages:
            index.add_raw_passages(gold_passages)

        gold_pmids = []
        for du in raw_q.get("documents", []):
            m = re.search(r'/pubmed/(\d+)', du)
            if m and m.group(1) not in tools.seen_pmids:
                gold_pmids.append(m.group(1))
                tools.seen_pmids.add(m.group(1))
        if gold_pmids:
            for art in self.pubmed.fetch_articles(gold_pmids):
                index.add_article(art)

        log.info("  Gold: %d passages indexed", index.total_passages)

        # ── Step 1: SEED the corpus — broad download ──
        kw = _simple_kw(question)
        if kw:
            n = tools.download_seed(kw, self.max_articles)
            log.info("  Tool DOWNLOAD_SEED('%s'): +%d articles → %d passages",
                     kw[:40], n, index.total_passages)

        # ── Step 2: Agentic loop ──
        for it in range(self.max_iters):
            log.info("  Iteration %d/%d", it+1, self.max_iters)

            # 2a. LLM generates download terms
            qp = prompt_generate_queries(question, tools.queries_used or None)
            qr = self.llm.generate(qp, max_tokens=256, temperature=0.3)
            new_queries = []
            for line in qr.strip().split("\n"):
                line = re.sub(r'^\d+[\.\)]\s*', '', line.strip()).strip('"')
                if line and 5 < len(line) < 200: new_queries.append(line)
            new_queries = new_queries[:3]
            if not new_queries:
                words = _simple_kw(question).split()
                if len(words)>=3: new_queries=[" ".join(words[:3])]

            # 2b. Download articles for each query
            for q in new_queries:
                if q in tools.queries_used: continue
                n = tools.download_seed(q, self.max_articles)
                log.info("    Tool DOWNLOAD_SEED('%s'): +%d articles", q[:50], n)

            # 2c. SEARCH OUR INDEX (this is YOUR retrieval, not PubMed's)
            top = tools.search_hybrid(question, top_k=10)
            log.info("    Tool SEARCH_HYBRID: top %d passages from OUR index",
                     len(top))
            if top:
                log.info("      Best (%.4f): %.60s...",
                         top[0].similarity_score, top[0].text)

            # 2d. EXPAND: take best-matching articles → download their
            #     related articles and citations into our corpus
            if top and it < self.max_iters - 1:
                best_pmids = list(dict.fromkeys(p.pmid for p in top[:3]
                                                 if p.pmid != "gold"))
                for bp in best_pmids[:2]:
                    n = tools.expand_related(bp, max_articles=10)
                    log.info("    Tool EXPAND_RELATED(PMID:%s): +%d articles", bp, n)
                    if top[0].pmid != "gold":
                        n2 = tools.expand_citing(top[0].pmid, max_articles=5)
                        log.info("    Tool EXPAND_CITING(PMID:%s): +%d articles",
                                 top[0].pmid, n2)
                        break  # one citing expansion is enough

            log.info("    Index: %d articles, %d passages",
                     index.total_articles, index.total_passages)

            # 2e. EVALUATE: does our index have enough evidence?
            if index.total_passages >= 15:
                verdict = tools.evaluate(question)
                log.info("    Tool EVALUATE: %s", verdict)
                if verdict == "SUFFICIENT":
                    break

        # ── Step 3: Final rerank ──
        if index.total_passages > 0:
            top = tools.search_hybrid(question, top_k=20)
            reranked = tools.rerank(question, top, top_k=15)
            log.info("  Tool RERANK: top (%.3f): %.60s...",
                     reranked[0].similarity_score if reranked else 0,
                     reranked[0].text[:60] if reranked else "none")

        log.info("  Final: %d articles, %d passages, %d downloads",
                 index.total_articles, index.total_passages,
                 len(tools.queries_used))

    def _answer(self, question, qtype, index, tools):
        few_shot = self.bank.get(qtype, n=2)

        # Hybrid retrieval for final passages
        top = tools.search_hybrid(question, top_k=15)
        ptexts = [p.text for p in top] or [question]

        log.info("  Tool SEARCH_HYBRID: %d passages for answering", len(ptexts))
        if top:
            log.info("    Best: (%.3f) %.60s...",
                     top[0].similarity_score, top[0].text)

        ec, ic = [], []
        for pi in range(self.num_passes):
            temp = 0.15 + pi*0.15
            log.info("  Answer pass %d/%d (temp=%.2f)", pi+1, self.num_passes, temp)
            raw = self.llm.generate(prompt_answer(question, qtype, ptexts, few_shot),
                                    max_tokens=768, temperature=temp)
            if qtype=="factoid": e,i=parse_factoid(raw); ec.append(e)
            elif qtype=="yesno": e,i=parse_yesno(raw); ec.append(e)
            elif qtype=="list": e,i=parse_list(raw); ec.append(e)
            else: i=parse_summary(raw)
            ic.append(i)

            if pi==0:
                corr = self.llm.generate(prompt_verify(question, ptexts, raw),
                                         max_tokens=768, temperature=0.2)
                if qtype=="factoid": re_,ri=parse_factoid(corr); ec.append(re_); ic.append(ri)
                elif qtype=="yesno": re_,ri=parse_yesno(corr); ec.append(re_); ic.append(ri)
                elif qtype=="list": re_,ri=parse_list(corr); ec.append(re_); ic.append(ri)
                else: ic.append(parse_summary(corr))

        result = {}
        if qtype=="factoid": result["exact_answer"]=con_factoid(ec)
        elif qtype=="yesno": result["exact_answer"]=con_yesno(ec)
        elif qtype=="list": result["exact_answer"]=con_list(ec)
        result["ideal_answer"] = con_ideal(ic)

        seen=set(); result["documents"]=[]
        for p in top[:10]:
            if p.doc_url and p.doc_url not in seen: seen.add(p.doc_url); result["documents"].append(p.doc_url)
        result["snippets"] = [{"text":p.text,"document":p.doc_url,"beginSection":p.section,
            "endSection":p.section,"offsetInBeginSection":p.offset_begin,
            "offsetInEndSection":p.offset_end} for p in top[:10]]
        return result


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="BioASQ 14b — Agentic RAG with On-the-Fly Indexing")
    parser.add_argument("--test-input", "-t", required=True)
    parser.add_argument("--training", "-tr", default=None)
    parser.add_argument("--output", "-o", default="submission_phaseB.json")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", "-m", default="google/gemma-3-27b-it")
    parser.add_argument("--embedding-model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--retrieval-iterations", type=int, default=3)
    parser.add_argument("--articles-per-query", type=int, default=10)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--question-ids", nargs="*", default=None)
    args = parser.parse_args()

    bank = FewShotBank()
    if args.training: bank.load(args.training)

    with open(args.test_input) as f: questions = json.load(f).get("questions", [])
    log.info("Loaded %d questions", len(questions))
    if args.question_ids: questions = [q for q in questions if q["id"] in set(args.question_ids)]

    llm = LLMEngine(base_url=args.vllm_url, model=args.model)
    pubmed = PubMedDownloader(api_key=args.api_key)
    embedder = EmbeddingEngine(model_name=args.embedding_model)

    agent = BioASQAgent(llm=llm, pubmed=pubmed, embedder=embedder, bank=bank,
                        num_passes=args.passes, max_iters=args.retrieval_iterations,
                        max_articles=args.articles_per_query)

    results = []
    for i, q in enumerate(questions):
        log.info("═══ Question %d / %d ═══", i+1, len(questions))
        try: results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s", q.get("id"), e, exc_info=True)
            results.append({"id": q["id"], "ideal_answer": "Unable to generate."})

    for r in results:
        if r.get("exact_answer") is None: r.pop("exact_answer", None)

    with open(args.output, "w") as f:
        json.dump({"questions": results}, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)} questions → {args.output}")
    print(f"Types: {dict(Counter(q.get('type','?') for q in questions))}")

if __name__ == "__main__":
    main()
