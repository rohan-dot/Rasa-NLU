#!/usr/bin/env python3
"""
BioASQ Task 14 — Agentic RAG with Local PubMed Baseline (SQLite)
=================================================================

TRUE agentic AI with ReAct-style reasoning loop:

  ┌────────────────────────────────────────────────────────────────┐
  │  THINK  → What do I need to answer this question?             │
  │  PLAN   → Decompose into sub-questions if multi-hop           │
  │  ACT    → Call tools: SEARCH_DB, SEARCH_SEMANTIC, EXPAND      │
  │  OBSERVE → Read results, evaluate relevance                   │
  │  REFLECT → Is evidence sufficient? Do I need more?            │
  │  ANSWER  → Generate answer from evidence chain                │
  │  VERIFY  → Self-check against evidence                        │
  │  CORRECT → Fix any errors found                               │
  └────────────────────────────────────────────────────────────────┘

The agent has 7 tools:
  1. SEARCH_DB(query)       — FTS5 keyword search on local SQLite (37M articles)
  2. SEARCH_SEMANTIC(query) — PubMedBERT embeddings + FAISS cosine similarity
  3. SEARCH_HYBRID(query)   — RRF fusion of both
  4. RERANK(passages)       — Cross-encoder reranker (ms-marco-MiniLM-L-12-v2)
  5. DECOMPOSE(question)    — Break complex questions into sub-questions
  6. EXPAND(pmid)           — Find related articles by content similarity
  7. SYNTHESIZE(answers)    — Combine sub-answers into final answer

All retrieval is LOCAL — your 33M article SQLite database.
No PubMed API calls. No internet needed. YOUR models do everything.

SETUP:
    pip install sentence-transformers faiss-cpu numpy rank-bm25 requests

START VLLM:
    CUDA_VISIBLE_DEVICES=0,1 \
    python -m vllm.entrypoints.openai.api_server \
        --model /panfs/g52-panfs/exp/FY25/models/gemma-3-27b-it \
        --served-model-name gemma-3-27b-it \
        --dtype bfloat16 --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 --max-model-len 8192 \
        --host 127.0.0.1 --port 8000

RUN:
    python bioasq_agentic_local.py \
        --test-input BioASQ-task14bPhaseA-testset1.json \
        --training training13b.json \
        --db pubmed_index.db \
        --model gemma-3-27b-it \
        --embed-device cpu \
        -o submission.json
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =====================================================================
# Data
# =====================================================================

@dataclass
class Passage:
    text: str
    pmid: str
    doc_url: str
    section: str = "abstract"
    offset_begin: int = 0
    offset_end: int = 0
    similarity_score: float = 0.0
    source_tool: str = ""  # which tool found this


@dataclass
class ReActStep:
    """One step in the agent's reasoning chain."""
    step_type: str   # THINK, ACT, OBSERVE, REFLECT
    content: str
    tool: str = ""
    result_count: int = 0


# =====================================================================
# Tool 1: SQLite FTS5 Search (keyword)
# =====================================================================

class SQLiteSearch:
    """Searches the local PubMed baseline via FTS5 full-text index."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        log.info("SQLite DB: %s (%d articles)", db_path, count)
        self.article_count = count

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        """FTS5 keyword search with BM25 ranking."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Build FTS5 query
        terms = re.findall(r'[A-Za-z0-9\-]+', query)
        terms = [t for t in terms if len(t) > 2]
        if not terms:
            conn.close()
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms[:10])

        sql = """
            SELECT a.pmid, a.title, a.abstract, a.journal, a.year,
                   a.mesh_terms, articles_fts.rank AS score
            FROM articles_fts
            JOIN articles a ON a.pmid = articles_fts.pmid
            WHERE articles_fts MATCH ?
              AND length(a.abstract) > 50
            ORDER BY articles_fts.rank
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, (fts_query, max_results)).fetchall()
            results = []
            for r in rows:
                results.append({
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "journal": r["journal"] or "",
                    "year": r["year"] or "",
                    "mesh_terms": r["mesh_terms"] or "",
                    "score": r["score"],
                    "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{r['pmid']}",
                })
            return results
        except sqlite3.OperationalError as e:
            log.warning("FTS5 search failed: %s", e)
            return []
        finally:
            conn.close()

    def fetch_by_pmid(self, pmid: str) -> dict | None:
        """Get a specific article by PMID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM articles WHERE pmid = ?", (pmid,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "pmid": row["pmid"],
            "title": row["title"],
            "abstract": row["abstract"],
            "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{row['pmid']}",
        }

    def find_related(self, pmid: str, max_results: int = 10) -> list[dict]:
        """Find articles with similar MeSH terms to a given article."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get the source article's MeSH terms
        src = conn.execute(
            "SELECT mesh_terms FROM articles WHERE pmid = ?", (pmid,)
        ).fetchone()
        if not src or not src["mesh_terms"]:
            conn.close()
            return []

        # Use MeSH terms as FTS query
        mesh = src["mesh_terms"]
        terms = [t.strip() for t in mesh.split(";") if len(t.strip()) > 3][:5]
        if not terms:
            conn.close()
            return []

        fts_q = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = conn.execute("""
                SELECT a.pmid, a.title, a.abstract, a.mesh_terms,
                       articles_fts.rank AS score
                FROM articles_fts
                JOIN articles a ON a.pmid = articles_fts.pmid
                WHERE articles_fts MATCH ?
                  AND a.pmid != ?
                  AND length(a.abstract) > 50
                ORDER BY articles_fts.rank
                LIMIT ?
            """, (fts_q, pmid, max_results)).fetchall()
            return [{"pmid": r["pmid"], "title": r["title"],
                     "abstract": r["abstract"],
                     "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{r['pmid']}"}
                    for r in rows]
        except:
            return []
        finally:
            conn.close()


# =====================================================================
# Tool 2: Semantic Search (FAISS + PubMedBERT)
# =====================================================================

class Embedder:
    def __init__(self, model="pritamdeka/S-PubMedBert-MS-MARCO", device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Embedding: %s (%s)", model, device)
        self.model = SentenceTransformer(model, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return np.array(self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ), dtype=np.float32)

    def encode1(self, t):
        return self.encode([t])[0]


class Reranker:
    """
    Cross-encoder reranker — processes (query, passage) pairs together.
    WAY more accurate than LLM scoring because:
      - Trained specifically for relevance scoring (MS-MARCO)
      - Sees query and passage as one input (cross-attention)
      - 30 passages in <1 second vs LLM taking 10-15 seconds
    """

    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu"):
        from sentence_transformers import CrossEncoder
        log.info("Reranker: %s (%s)", model, device)
        self.m = CrossEncoder(model, device=device)

    def rerank(self, query: str, passages: list[Passage], k: int = 15) -> list[Passage]:
        """Score each passage against the query and return top-k."""
        if not passages:
            return []
        # Truncate passages to 512 chars for speed
        pairs = [[query, p.text[:512]] for p in passages]
        scores = self.m.predict(pairs)
        for p, s in zip(passages, scores):
            p.similarity_score = float(s)
        ranked = sorted(passages, key=lambda p: p.similarity_score, reverse=True)
        return ranked[:k]


class LiveIndex:
    """On-the-fly FAISS + BM25 hybrid index built per question."""

    def __init__(self, emb: Embedder):
        import faiss
        self.faiss = faiss
        self.emb = emb
        self.index = faiss.IndexFlatIP(emb.dim)
        self.passages: list[Passage] = []
        self._seen: set[str] = set()
        self._bm25 = None
        self._dirty = True

    def add_article(self, art: dict, source_tool: str = ""):
        """Chunk and index an article: title + full abstract + overlapping 3-sent chunks."""
        pmid = art["pmid"]
        url = art.get("url", f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}")
        new = []

        t = art.get("title", "").strip()
        if t and t not in self._seen:
            self._seen.add(t)
            new.append(Passage(t, pmid, url, "title", 0, len(t), source_tool=source_tool))

        ab = art.get("abstract", "").strip()
        if ab:
            if ab not in self._seen and len(ab) > 50:
                self._seen.add(ab)
                new.append(Passage(ab, pmid, url, "abstract", 0, len(ab),
                                   source_tool=source_tool))
            sents = re.split(r'(?<=[.!?])\s+', ab)
            if len(sents) > 3:
                for i in range(len(sents) - 2):
                    ch = " ".join(s.strip() for s in sents[i:i+3]).strip()
                    if len(ch) < 50 or ch in self._seen:
                        continue
                    b = ab.find(sents[i])
                    self._seen.add(ch)
                    new.append(Passage(ch, pmid, url, "abstract",
                                       max(b, 0), max(b, 0)+len(ch),
                                       source_tool=source_tool))

        if not new:
            return
        embs = self.emb.encode([p.text for p in new])
        self.index.add(embs)
        self.passages.extend(new)
        self._dirty = True

    def search_semantic(self, query: str, k: int = 20):
        if self.index.ntotal == 0:
            return []
        qe = self.emb.encode1(query).reshape(1, -1)
        n = min(k * 3, self.index.ntotal)
        sc, ix = self.index.search(qe, n)
        return [(int(i), float(s)) for s, i in zip(sc[0], ix[0])
                if 0 <= i < len(self.passages)]

    def search_bm25(self, query: str, k: int = 20):
        if self._dirty:
            from rank_bm25 import BM25Okapi
            corpus = [re.findall(r'[a-z0-9\-]+', p.text.lower())
                      for p in self.passages]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._dirty = False
        if not self._bm25:
            return []
        tokens = re.findall(r'[a-z0-9\-]+', query.lower())
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k * 3]
        return [(i, s) for i, s in ranked if s > 0]

    def search_hybrid(self, query: str, k: int = 15):
        """Reciprocal Rank Fusion of semantic + BM25."""
        sem = self.search_semantic(query, k)
        bm = self.search_bm25(query, k)

        rrf = {}
        K = 60
        for r, (i, _) in enumerate(sem):
            rrf[i] = rrf.get(i, 0) + 0.6 / (K + r + 1)
        for r, (i, _) in enumerate(bm):
            rrf[i] = rrf.get(i, 0) + 0.4 / (K + r + 1)

        results, seen_sigs = [], set()
        for i, sc in sorted(rrf.items(), key=lambda x: -x[1]):
            p = self.passages[i]
            sig = p.text[:80].lower()
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            p.similarity_score = sc
            results.append(p)
            if len(results) >= k:
                break
        return results

    @property
    def size(self):
        return len(self.passages)

    @property
    def n_articles(self):
        return len({p.pmid for p in self.passages})


# =====================================================================
# LLM Client
# =====================================================================

class LLM:
    def __init__(self, url="http://localhost:8000", model="gemma-3-27b-it"):
        self.url = url.rstrip("/")
        self.model = model
        self.s = requests.Session()
        try:
            r = self.s.get(f"{self.url}/v1/models", timeout=10)
            r.raise_for_status()
            av = [m["id"] for m in r.json().get("data", [])]
            log.info("vLLM: %s", av)
            if av and self.model not in av:
                self.model = av[0]
        except Exception as e:
            log.error("vLLM: %s", e)
            sys.exit(1)

    def ask(self, prompt, max_tokens=1024, temp=0.3):
        for i in range(3):
            try:
                r = self.s.post(
                    f"{self.url}/v1/chat/completions",
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temp,
                          "top_p": 0.95},
                    timeout=180)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("LLM(%d): %s", i+1, e)
                time.sleep(3 * (i+1))
        return ""


# =====================================================================
# Few-Shot Bank
# =====================================================================

class FewShot:
    def __init__(self):
        self.ex = {"factoid": [], "list": [], "yesno": [], "summary": []}

    def load(self, path, n=40):
        with open(path) as f:
            data = json.load(f)
        for q in data.get("questions", []):
            qt = q.get("type", "").lower()
            if qt not in self.ex:
                continue
            sn = q.get("snippets", [])
            ea, ia = q.get("exact_answer"), q.get("ideal_answer")
            if not sn or not ia:
                continue
            if qt != "summary" and not ea:
                continue
            self.ex[qt].append({
                "body": q["body"],
                "snippets": [s.get("text", "") for s in sn[:5]],
                "exact_answer": ea,
                "ideal_answer": (ia if isinstance(ia, str)
                                 else ia[0] if isinstance(ia, list) and ia
                                 else ""),
            })
        for qt in self.ex:
            self.ex[qt] = sorted(self.ex[qt],
                                 key=lambda x: len(x["body"]))[:n]
            log.info("  FS %s: %d", qt, len(self.ex[qt]))

    def get(self, qt, n=2):
        e = self.ex.get(qt, [])
        s = [x for x in e if len(" ".join(x["snippets"])) < 1500]
        return (s or e)[:n]


# =====================================================================
# Prompt Templates
# =====================================================================

def _sb(texts, mx=5000):
    b = ""
    for i, s in enumerate(texts, 1):
        line = f"[{i}] {s.strip()}\n"
        if len(b) + len(line) > mx:
            break
        b += line
    return b.strip()


def _fmt(ea):
    if isinstance(ea, str):
        return ea
    if isinstance(ea, list):
        if ea and isinstance(ea[0], list):
            return "; ".join(", ".join(x) if isinstance(x, list)
                            else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)


def _cap(t, max_words=200):
    w = t.split()
    return " ".join(w[:max_words]) if len(w) > max_words else t


# ── Decomposition prompt ──
def prompt_decompose(question):
    return (
        "You are a biomedical expert. This question may require multiple "
        "pieces of information (multi-hop reasoning). Break it into 2-4 "
        "independent sub-questions that, when answered, will fully answer "
        "the original.\n\n"
        "If the question is simple (single fact), just return it as-is.\n\n"
        f"QUESTION: {question}\n\n"
        "Return sub-questions numbered, one per line. Nothing else.\n\n"
        "SUB-QUESTIONS:\n"
    )


# ── Query generation ──
def prompt_queries(question, prev=None):
    p = (f"Generate 3 short search queries (3-6 words, plain keywords) "
         f"for: {question}\n\n")
    if prev:
        p += "Previous didn't work, use DIFFERENT terms:\n"
        p += "\n".join(f"  - {q}" for q in prev[-3:]) + "\n\n"
    p += "Write ONLY 3 queries numbered. Nothing else.\n\n1."
    return p


# ── Evidence evaluation ──
def prompt_evaluate(question, texts):
    block = "\n".join(f"[{i+1}] {t[:200]}" for i, t in enumerate(texts[:10]))
    return (
        f"Can this question be fully answered from these passages?\n\n"
        f"QUESTION: {question}\n\nPASSAGES:\n{block}\n\n"
        f"Reply SUFFICIENT or INSUFFICIENT. If insufficient, say what's "
        f"missing in one sentence.\n\nVERDICT:"
    )


# ── Answer prompts ──
def prompt_answer(q, qt, ts, fs):
    if qt == "yesno":
        return _pr_yn(q, ts, fs)
    if qt == "factoid":
        return _pr_fac(q, ts, fs)
    if qt == "list":
        return _pr_lst(q, ts, fs)
    return _pr_sum(q, ts, fs)


def _pr_yn(q, ts, fs):
    p = (
        "You are an expert biomedical QA system.\n\n"
        "INSTRUCTIONS:\n"
        "1. Find evidence supporting YES.\n"
        "2. Find evidence supporting NO.\n"
        "3. Choose the side with STRONGER evidence.\n"
        "4. If evidence shows PROBLEMS (toxicity, failure, lack of evidence), answer NO.\n"
        "5. If mixed or insufficient, lean NO.\n"
        "6. 'Promising' or 'preclinical' does NOT mean YES.\n\n"
    )
    for ex in fs[:2]:
        ea = ex["exact_answer"]
        ea = ea[0] if isinstance(ea, list) and ea else ea
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'], 600)}\n"
        p += f"EXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\n"
    p += "EVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:"
    return p


def _pr_fac(q, ts, fs):
    p = (
        "You are an expert biomedical QA system.\n\n"
        "RULES:\n"
        "1. EXACT_ANSWER: 1-5 words MAX. Specific name/number/term.\n"
        "2. Use EXACT terminology from evidence.\n"
        "3. If no clear answer, write: unknown\n"
        "4. Prefer: drug names, gene names, disease names, numbers.\n"
        "5. Then write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n"
        "GOOD: 'transsphenoidal surgery', 'NF1', 'palivizumab'\n"
        "BAD: 'multiple factors', 'it involves mechanisms'\n\n"
    )
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'], 600)}\n"
        p += f"EXACT_ANSWER: {_fmt(ex['exact_answer'])}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nEXACT_ANSWER:"
    return p


def _pr_lst(q, ts, fs):
    p = (
        "You are an expert biomedical QA system.\n\n"
        "RULES:\n"
        "1. List EVERY relevant item. Be EXHAUSTIVE.\n"
        "2. Too many better than too few. Aim 5-15+ items.\n"
        "3. Each item: 1-5 words, specific term. Prefix '- '.\n"
        "4. After list write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n"
    )
    for ex in fs[:1]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'], 500)}\n"
        p += "EXACT_ANSWER:\n"
        ea = ex["exact_answer"]
        if isinstance(ea, list):
            for it in ea[:8]:
                if isinstance(it, list):
                    p += f"- {it[0]}\n"
                else:
                    p += f"- {it}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts, 6000)}\n"
    p += "List EVERY relevant item:\n\nEXACT_ANSWER:\n"
    return p


def _pr_sum(q, ts, fs):
    p = ("You are an expert biomedical QA system. "
         "Write a 3-6 sentence answer (max 200 words).\n\n")
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'], 800)}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:"
    return p


def prompt_verify(q, ts, a):
    return (
        f"Check this answer. Fix errors or unsupported claims.\n\n"
        f"Q: {q}\nEVIDENCE:\n{_sb(ts, 3000)}\n\n"
        f"CANDIDATE:\n{a}\n\nCORRECTED_ANSWER:"
    )


def prompt_synthesize(question, sub_answers):
    block = "\n".join(f"Sub-Q{i+1}: {sa['question']}\n"
                      f"Answer: {sa['answer']}\n"
                      for i, sa in enumerate(sub_answers))
    return (
        f"Combine these sub-answers into a single coherent answer.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"SUB-ANSWERS:\n{block}\n\n"
        f"COMBINED ANSWER:"
    )


# =====================================================================
# Parsers
# =====================================================================

def parse_factoid(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    e = parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is", "The exact answer is", "Answer:"]:
        if e.lower().startswith(pf.lower()):
            e = e[len(pf):].strip()
    ideal = parts[1].strip() if len(parts) == 2 else ""
    return [e] if e else ["unknown"], _cap(ideal or e)


def parse_yesno(r):
    m = re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)', r, re.I)
    if m:
        raw = m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        raw = ""
        for l in reversed(r.strip().split("\n")):
            if "yes" in l.lower()[:10] or "no" in l.lower()[:10]:
                raw = l.lower()
                break
        if not raw:
            raw = r.strip().split("\n")[-1].lower()

    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    ideal = parts[1].strip() if len(parts) == 2 else ""

    if "yes" in raw[:10] and "no" not in raw[:5]:
        exact = "yes"
    elif "no" in raw[:10]:
        exact = "no"
    else:
        lo = r.lower()
        neg = len(re.findall(
            r'insufficient|toxicity|not effective|no evidence|failed|'
            r'ineffective|not recommended|lack of', lo))
        pos = len(re.findall(
            r'effective|demonstrated|shown to|evidence supports|'
            r'approved|recommended', lo))
        exact = "no" if neg >= pos else "yes"
    return exact, _cap(ideal or f"The answer is {exact}.")


def parse_list(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    lr = parts[0].strip()
    ideal = parts[1].strip() if len(parts) == 2 else ""
    items = []
    for l in lr.split("\n"):
        l = re.sub(r'^[\-\*•]\s*', '', l.strip())
        l = re.sub(r'^\d+[\.\)]\s*', '', l).strip().strip('"\'').rstrip(".")
        if l and len(l) > 1 and len(l) <= 100:
            items.append(l)
    return items[:100] or ["unknown"], _cap(ideal)


def parse_summary(r):
    return _cap(re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', r,
                       flags=re.I).strip() or "No answer.")


# =====================================================================
# Consensus
# =====================================================================

def con_fac(c):
    flat = [x[0].lower().strip() for x in c if x]
    if not flat:
        return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip() == best:
            return x[:5]
    return [best]


def con_yn(c):
    return Counter(c).most_common(1)[0][0]


def con_lst(c):
    items = {}
    for cand in c:
        for s in cand:
            k = s.lower().strip() if isinstance(s, str) else s
            if k:
                items.setdefault(k, s)
    return list(items.values())[:100] or ["unknown"]


def con_ideal(c):
    if not c:
        return ""
    s = sorted(c, key=len)
    return s[len(s) // 2]


# =====================================================================
# Helpers
# =====================================================================

def simple_kw(q):
    STOP = set(
        "what is are the a an of in on to for with by from and or not how "
        "does do can which who whom where when why this that these those "
        "it its has have had been being please list describe common main "
        "types most often should".split()
    )
    return " ".join(
        t for t in re.findall(r"[A-Za-z0-9\-']+", q)
        if t.lower() not in STOP and len(t) > 2
    )[:80]


def parse_queries(resp):
    qs = []
    for l in resp.strip().split("\n"):
        l = re.sub(r'^\d+[\.\)]\s*', '', l.strip()).strip('"').strip()
        if not l or len(l) < 5 or len(l) > 80:
            continue
        if any(s in l.lower() for s in [
            "here are", "queries", "search", "i would", "let me",
            "following", "note:", "aim", "different"
        ]):
            continue
        qs.append(l)
    return qs[:3]


def parse_subquestions(resp):
    subs = []
    for l in resp.strip().split("\n"):
        l = re.sub(r'^\d+[\.\)]\s*', '', l.strip()).strip()
        if l and len(l) > 10 and len(l) < 200 and "?" in l:
            subs.append(l)
    return subs[:4]


# =====================================================================
# THE AGENT — ReAct-style reasoning with multi-hop support
# =====================================================================

class Agent:
    """
    ReAct-style agent with explicit Thought → Action → Observation loop.

    For multi-hop questions:
      1. DECOMPOSE into sub-questions
      2. Solve each sub-question independently
      3. SYNTHESIZE sub-answers into final answer

    For simple questions:
      1. Direct retrieval + answer pipeline

    All reasoning steps are logged as a trace for debugging/presentation.
    """

    def __init__(self, llm, db, emb, reranker, fs, passes=3, iters=3):
        self.llm = llm
        self.db = db
        self.emb = emb
        self.reranker = reranker
        self.fs = fs
        self.passes = passes
        self.iters = iters

    def solve(self, question: dict) -> dict:
        qid = question["id"]
        body = question["body"]
        qtype = question.get("type", "summary").lower()
        trace: list[ReActStep] = []

        log.info("━" * 60)
        log.info("  Question %s [%s]: %s", qid, qtype, body)
        log.info("━" * 60)

        # ── THINK: Is this multi-hop? ──
        trace.append(ReActStep("THINK", f"Analyzing question: {body}"))

        decomp_resp = self.llm.ask(prompt_decompose(body), max_tokens=300, temp=0.2)
        sub_questions = parse_subquestions(decomp_resp)

        is_multihop = len(sub_questions) >= 2
        if is_multihop:
            trace.append(ReActStep("PLAN", f"Multi-hop detected. Decomposed into "
                                   f"{len(sub_questions)} sub-questions"))
            for i, sq in enumerate(sub_questions):
                log.info("  Sub-Q%d: %s", i+1, sq)
        else:
            trace.append(ReActStep("PLAN", "Simple question — direct retrieval"))
            sub_questions = [body]

        # ── ACT: Retrieve evidence for each sub-question ──
        idx = LiveIndex(self.emb)
        seen_pmids: set[str] = set()
        all_queries: list[str] = []
        sub_answers: list[dict] = []

        # Ingest gold data if present
        for gs in question.get("snippets", []):
            t = gs.get("text", "").strip()
            if t:
                du = gs.get("document", "")
                pm = re.search(r'/pubmed/(\d+)', du)
                idx.passages.append(Passage(
                    text=t, pmid=pm.group(1) if pm else "gold",
                    doc_url=du, section="abstract",
                    source_tool="GOLD"))
        if idx.passages:
            embs = self.emb.encode([p.text for p in idx.passages])
            idx.index.add(embs)
            idx._dirty = True
            trace.append(ReActStep("OBSERVE",
                                   f"Ingested {len(idx.passages)} gold snippets"))

        for sq_idx, sub_q in enumerate(sub_questions):
            sq_label = f"Sub-Q{sq_idx+1}" if is_multihop else "Main-Q"
            log.info("\n  ── %s: %s ──", sq_label, sub_q[:60])

            # Keyword search on SQLite
            kw = simple_kw(sub_q)
            if kw:
                trace.append(ReActStep("ACT",
                    f"SEARCH_DB('{kw}')", tool="SEARCH_DB"))
                db_results = self.db.search(kw, max_results=20)
                new_count = 0
                for art in db_results:
                    if art["pmid"] not in seen_pmids:
                        seen_pmids.add(art["pmid"])
                        idx.add_article(art, source_tool="SEARCH_DB")
                        new_count += 1
                trace.append(ReActStep("OBSERVE",
                    f"Found {len(db_results)} articles, {new_count} new → "
                    f"{idx.size} passages indexed",
                    result_count=new_count))
                log.info("    SEARCH_DB('%s'): +%d articles", kw[:40], new_count)

            # Agentic retrieval loop
            for it in range(self.iters):
                # LLM generates queries
                qr = self.llm.ask(prompt_queries(sub_q, all_queries or None),
                                  max_tokens=200, temp=0.3)
                nq = parse_queries(qr)
                if not nq:
                    w = simple_kw(sub_q).split()
                    if len(w) >= 3:
                        nq = [" ".join(w[:3])]

                for q in nq:
                    if q in all_queries:
                        continue
                    all_queries.append(q)
                    trace.append(ReActStep("ACT",
                        f"SEARCH_DB('{q}')", tool="SEARCH_DB"))
                    results = self.db.search(q, max_results=15)
                    nc = 0
                    for art in results:
                        if art["pmid"] not in seen_pmids:
                            seen_pmids.add(art["pmid"])
                            idx.add_article(art, source_tool="SEARCH_DB")
                            nc += 1
                    trace.append(ReActStep("OBSERVE",
                        f"+{nc} articles from '{q[:40]}'", result_count=nc))
                    log.info("    SEARCH_DB('%s'): +%d", q[:50], nc)

                # Search our own index
                top = idx.search_hybrid(sub_q, k=10)
                if top:
                    trace.append(ReActStep("ACT",
                        f"SEARCH_HYBRID('{sub_q[:40]}...')",
                        tool="SEARCH_HYBRID"))
                    trace.append(ReActStep("OBSERVE",
                        f"Top passage (score={top[0].similarity_score:.4f}): "
                        f"{top[0].text[:80]}...",
                        result_count=len(top)))
                    log.info("    SEARCH_HYBRID: best=%.4f '%s...'",
                             top[0].similarity_score, top[0].text[:50])

                # Expand from best articles
                if top:
                    best_pmid = next(
                        (p.pmid for p in top[:3] if p.pmid != "gold"), None)
                    if best_pmid:
                        trace.append(ReActStep("ACT",
                            f"EXPAND(PMID:{best_pmid})", tool="EXPAND"))
                        related = self.db.find_related(best_pmid, 10)
                        nc = 0
                        for art in related:
                            if art["pmid"] not in seen_pmids:
                                seen_pmids.add(art["pmid"])
                                idx.add_article(art, source_tool="EXPAND")
                                nc += 1
                        trace.append(ReActStep("OBSERVE",
                            f"+{nc} related articles via MeSH similarity",
                            result_count=nc))
                        log.info("    EXPAND(%s): +%d related", best_pmid, nc)

                log.info("    Index: %d articles, %d passages",
                         idx.n_articles, idx.size)

                # Evaluate sufficiency
                if idx.size >= 15 and top:
                    trace.append(ReActStep("REFLECT",
                        "Evaluating evidence sufficiency..."))
                    v = self.llm.ask(
                        prompt_evaluate(sub_q, [p.text for p in top]),
                        max_tokens=50, temp=0.1)
                    if "SUFFICIENT" in v.upper():
                        trace.append(ReActStep("REFLECT",
                            f"SUFFICIENT — stopping retrieval for {sq_label}"))
                        log.info("    SUFFICIENT")
                        break
                    else:
                        missing = v.replace("INSUFFICIENT", "").strip()
                        trace.append(ReActStep("REFLECT",
                            f"INSUFFICIENT — {missing[:100]}"))
                        log.info("    INSUFFICIENT — %s", missing[:60])

            # For multi-hop: generate sub-answer
            if is_multihop:
                top = idx.search_hybrid(sub_q, k=10)
                ptexts = [p.text for p in top[:10]] or [sub_q]
                sub_ans = self.llm.ask(
                    f"Answer this briefly (2-3 sentences) from the evidence:\n\n"
                    f"Q: {sub_q}\nEVIDENCE:\n{_sb(ptexts, 3000)}\n\nANSWER:",
                    max_tokens=300, temp=0.2)
                sub_answers.append({"question": sub_q, "answer": sub_ans})
                trace.append(ReActStep("ACT",
                    f"Sub-answer for '{sub_q[:40]}...': {sub_ans[:80]}...",
                    tool="ANSWER"))
                log.info("    Sub-answer: %s...", sub_ans[:60])

        # ── Cross-encoder rerank final passages ──
        top = idx.search_hybrid(body, k=30)  # get more candidates for reranker
        if len(top) > 3:
            log.info("  RERANK: scoring %d passages with cross-encoder...", len(top))
            top = self.reranker.rerank(body, top, k=15)
            trace.append(ReActStep("ACT",
                f"RERANK — cross-encoder re-scored {len(top)} passages. "
                f"Top: (score={top[0].similarity_score:.4f}) {top[0].text[:60]}...",
                tool="RERANK"))
            log.info("  RERANK done. Top(%.4f): %s...",
                     top[0].similarity_score, top[0].text[:60])

        # ── ANSWER ──
        fsex = self.fs.get(qtype, 2)
        ptexts = [p.text for p in top[:15]] or [body]

        # For multi-hop: synthesize sub-answers first
        if is_multihop and sub_answers:
            synth = self.llm.ask(
                prompt_synthesize(body, sub_answers),
                max_tokens=500, temp=0.2)
            # Add synthesized answer as additional context
            ptexts.insert(0, f"[Synthesized] {synth}")
            trace.append(ReActStep("ACT",
                f"SYNTHESIZE — Combined {len(sub_answers)} sub-answers",
                tool="SYNTHESIZE"))

        ec, ic = [], []
        for pi in range(self.passes):
            temp = 0.15 + pi * 0.15
            log.info("  Answer %d/%d (t=%.2f)", pi+1, self.passes, temp)
            raw = self.llm.ask(prompt_answer(body, qtype, ptexts, fsex),
                               max_tokens=768, temp=temp)

            if qtype == "factoid":
                e, i = parse_factoid(raw); ec.append(e)
            elif qtype == "yesno":
                e, i = parse_yesno(raw); ec.append(e)
            elif qtype == "list":
                e, i = parse_list(raw); ec.append(e)
            else:
                i = parse_summary(raw)
            ic.append(i)

            if pi == 0:
                trace.append(ReActStep("ACT",
                    "VERIFY — Self-checking answer against evidence",
                    tool="VERIFY"))
                corr = self.llm.ask(prompt_verify(body, ptexts, raw),
                                     max_tokens=768, temp=0.2)
                if qtype == "factoid":
                    e2, i2 = parse_factoid(corr); ec.append(e2); ic.append(i2)
                elif qtype == "yesno":
                    e2, i2 = parse_yesno(corr); ec.append(e2); ic.append(i2)
                elif qtype == "list":
                    e2, i2 = parse_list(corr); ec.append(e2); ic.append(i2)
                else:
                    ic.append(parse_summary(corr))

        # ── Build result ──
        result = {"id": qid, "ideal_answer": con_ideal(ic)}
        if qtype == "factoid":
            result["exact_answer"] = con_fac(ec)
        elif qtype == "yesno":
            result["exact_answer"] = con_yn(ec)
        elif qtype == "list":
            result["exact_answer"] = con_lst(ec)

        # Documents + snippets for Phase A
        seen_u = set()
        result["documents"] = []
        for p in top[:10]:
            if p.doc_url and p.doc_url not in seen_u:
                seen_u.add(p.doc_url)
                result["documents"].append(p.doc_url)
        result["snippets"] = [{
            "document": p.doc_url, "text": p.text,
            "offsetInBeginSection": p.offset_begin,
            "offsetInEndSection": p.offset_end,
            "beginSection": "sections.0",
            "endSection": "sections.0",
        } for p in top[:10]]

        # Store trace for reporting
        result["_trace"] = [
            {"step": s.step_type, "content": s.content,
             "tool": s.tool, "results": s.result_count}
            for s in trace
        ]

        log.info("  ✓ exact=%s (%d trace steps)",
                 str(result.get("exact_answer", "N/A"))[:60], len(trace))
        return result


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="BioASQ 14 — Agentic RAG with Local PubMed SQLite")
    ap.add_argument("--test-input", "-t", required=True)
    ap.add_argument("--training", "-tr", default=None)
    ap.add_argument("--db", required=True, help="SQLite PubMed baseline DB")
    ap.add_argument("--output", "-o", default="submission.json")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--model", "-m", default="gemma-3-27b-it")
    ap.add_argument("--embedding-model",
                    default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--reranker-model",
                    default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                    help="Cross-encoder model for reranking")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--retrieval-iterations", type=int, default=3)
    a = ap.parse_args()

    # Load components
    fs = FewShot()
    if a.training:
        fs.load(a.training)

    with open(a.test_input) as f:
        qs = json.load(f).get("questions", [])
    log.info("Loaded %d questions", len(qs))

    llm = LLM(a.vllm_url, a.model)
    db = SQLiteSearch(a.db)
    emb = Embedder(a.embedding_model, a.embed_device)
    reranker = Reranker(a.reranker_model, device=a.embed_device or "cpu")
    agent = Agent(llm, db, emb, reranker, fs, a.passes, a.retrieval_iterations)

    # Process
    results = []
    for i, q in enumerate(qs):
        log.info("═══ Question %d / %d ═══", i+1, len(qs))
        try:
            results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s", q.get("id"), e, exc_info=True)
            results.append({"id": q["id"], "ideal_answer": "Unable to generate.",
                            "documents": [], "snippets": []})

    # ── Write Phase B submission (answers only) ──
    phase_b = {"questions": []}
    for r in results:
        entry = {"id": r["id"], "ideal_answer": r.get("ideal_answer", "")}
        if "exact_answer" in r:
            entry["exact_answer"] = r["exact_answer"]
        if entry.get("exact_answer") is None:
            entry.pop("exact_answer", None)
        phase_b["questions"].append(entry)

    with open(a.output, "w") as f:
        json.dump(phase_b, f, indent=2, ensure_ascii=False)
    log.info("Phase B → %s", a.output)

    # ── Write Phase A submission (docs + snippets + answers) ──
    phase_a = {"questions": []}
    for r in results:
        entry = {"id": r["id"],
                 "documents": r.get("documents", []),
                 "snippets": r.get("snippets", [])}
        if "exact_answer" in r:
            entry["exact_answer"] = r["exact_answer"]
        entry["ideal_answer"] = r.get("ideal_answer", "")
        if entry.get("exact_answer") is None:
            entry.pop("exact_answer", None)
        phase_a["questions"].append(entry)

    out_a = a.output.replace(".json", "_phaseA.json")
    with open(out_a, "w") as f:
        json.dump(phase_a, f, indent=2, ensure_ascii=False)
    log.info("Phase A → %s", out_a)

    # ── Write trace report ──
    trace_report = []
    for r in results:
        trace_report.append({
            "id": r["id"],
            "exact_answer": r.get("exact_answer"),
            "trace": r.get("_trace", []),
        })
    trace_path = a.output.replace(".json", "_trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace_report, f, indent=2, ensure_ascii=False)
    log.info("Trace → %s", trace_path)

    print(f"\nDone! {len(results)} questions")
    print(f"Types: {dict(Counter(q.get('type', '?') for q in qs))}")
    print(f"Phase B (answers):   {a.output}")
    print(f"Phase A (full):      {out_a}")
    print(f"Trace (for slides):  {trace_path}")


if __name__ == "__main__":
    main()
