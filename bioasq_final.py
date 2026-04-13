#!/usr/bin/env python3
"""
BioASQ Task 14 — Agentic RAG Solver (Production)
==================================================
Handles BOTH Phase A+ (retrieve documents & snippets) and
Phase B (generate exact + ideal answers).

SETUP (one time):
    pip install requests sentence-transformers faiss-cpu numpy rank-bm25

START VLLM (Terminal 1):
    CUDA_VISIBLE_DEVICES=0,1 \
    python -m vllm.entrypoints.openai.api_server \
        --model /panfs/g52-panfs/exp/FY25/models/gemma-3-27b-it \
        --served-model-name gemma-3-27b-it \
        --dtype bfloat16 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --host 127.0.0.1 --port 8000

RUN (Terminal 2):
    python bioasq_final.py \
        --test-input BioASQ-task14bPhaseA-testset1.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        --embed-device cpu \
        -o submission.json

OUTPUT:
    submission.json          — Phase B (exact + ideal answers)
    submission_phaseA.json   — Phase A+ (documents + snippets)
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


# =====================================================================
# PubMed Client — fetches and downloads articles
# =====================================================================

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


class PubMed:
    def __init__(self, api_key=None):
        self.session = requests.Session()
        self.api_key = api_key
        self._last = 0.0
        self._interval = 0.2 if api_key else 1.0
        self._backoff = 1.0

    def _wait(self):
        elapsed = time.time() - self._last
        w = max(self._interval, self._backoff) - elapsed
        if w > 0: time.sleep(w)
        self._last = time.time()

    def _get(self, url, params, retries=3):
        for i in range(retries):
            self._wait()
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    self._backoff = min(self._backoff * 2, 10)
                    time.sleep(self._backoff + i * 2)
                    continue
                r.raise_for_status()
                self._backoff = max(self._backoff * 0.8, self._interval)
                return r
            except Exception as e:
                log.warning("  PubMed request failed (%d/%d): %s", i+1, retries, e)
                time.sleep(2 * (i+1))
        return None

    def search(self, query, n=30):
        p = {"db": "pubmed", "term": query, "retmax": n, "retmode": "json"}
        if self.api_key: p["api_key"] = self.api_key
        r = self._get(ESEARCH, p)
        if not r: return []
        try: return r.json().get("esearchresult", {}).get("idlist", [])
        except: return []

    def related(self, pmid, n=15):
        p = {"dbfrom": "pubmed", "db": "pubmed", "id": pmid,
             "cmd": "neighbor_score", "retmode": "json"}
        if self.api_key: p["api_key"] = self.api_key
        r = self._get(ELINK, p)
        if not r: return []
        try:
            for ls in r.json().get("linksets", [{}])[0].get("linksetdbs", []):
                if ls.get("linkname") == "pubmed_pubmed":
                    return [str(l["id"]) for l in ls.get("links", [])][:n]
        except: pass
        return []

    def fetch(self, pmids):
        if not pmids: return []
        all_arts = []
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]
            p = {"db": "pubmed", "id": ",".join(batch),
                 "rettype": "xml", "retmode": "xml"}
            if self.api_key: p["api_key"] = self.api_key
            r = self._get(EFETCH, p)
            if r: all_arts.extend(self._parse(r.text))
        return all_arts

    def _parse(self, xml):
        arts = []
        try:
            root = ET.fromstring(xml)
            for el in root.findall(".//PubmedArticle"):
                pid = el.find(".//PMID")
                pid = pid.text.strip() if pid is not None and pid.text else ""
                tit = el.find(".//ArticleTitle")
                tit = "".join(tit.itertext()).strip() if tit is not None else ""
                abp = []
                for a in el.findall(".//AbstractText"):
                    lab = a.get("Label", "")
                    txt = "".join(a.itertext()).strip()
                    if lab and txt: abp.append(f"{lab}: {txt}")
                    elif txt: abp.append(txt)
                ab = " ".join(abp)
                if pid and (tit or ab):
                    arts.append({"pmid": pid, "title": tit, "abstract": ab,
                                 "url": f"http://www.ncbi.nlm.nih.gov/pubmed/{pid}"})
        except ET.ParseError as e:
            log.warning("XML error: %s", e)
        return arts


# =====================================================================
# Embedding + Hybrid Index (FAISS semantic + BM25 keyword)
# =====================================================================

class Embedder:
    def __init__(self, model="pritamdeka/S-PubMedBert-MS-MARCO", device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Embedding model: %s (%s)", model, device)
        self.model = SentenceTransformer(model, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64):
        return np.array(self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True), dtype=np.float32)

    def encode_one(self, text):
        return self.encode([text])[0]


class HybridIndex:
    """FAISS (semantic) + BM25 (keyword) with Reciprocal Rank Fusion."""

    def __init__(self, embedder):
        import faiss
        self.faiss = faiss
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.passages: list[Passage] = []
        self._seen: set[str] = set()
        self._bm25 = None
        self._bm25_dirty = True

    def _tok(self, t):
        return re.findall(r'[a-z0-9\-]+', t.lower())

    def _rebuild_bm25(self):
        from rank_bm25 import BM25Okapi
        if self.passages:
            self._bm25 = BM25Okapi([self._tok(p.text) for p in self.passages])
        self._bm25_dirty = False

    def add_article(self, art):
        pmid, url = art["pmid"], art["url"]
        new = []

        # Title
        t = art.get("title", "").strip()
        if t and t not in self._seen:
            self._seen.add(t)
            new.append(Passage(text=t, pmid=pmid, doc_url=url,
                               section="title", offset_end=len(t)))

        # Abstract: full + overlapping 3-sentence chunks
        ab = art.get("abstract", "").strip()
        if ab:
            if ab not in self._seen and len(ab) > 50:
                self._seen.add(ab)
                new.append(Passage(text=ab, pmid=pmid, doc_url=url,
                                   section="abstract", offset_end=len(ab)))
            sents = re.split(r'(?<=[.!?])\s+', ab)
            if len(sents) > 3:
                for i in range(len(sents) - 2):
                    chunk = " ".join(s.strip() for s in sents[i:i+3]).strip()
                    if len(chunk) < 50 or chunk in self._seen: continue
                    b = ab.find(sents[i])
                    self._seen.add(chunk)
                    new.append(Passage(text=chunk, pmid=pmid, doc_url=url,
                                       section="abstract",
                                       offset_begin=max(b, 0),
                                       offset_end=max(b, 0)+len(chunk)))

        if not new: return
        embs = self.embedder.encode([p.text for p in new])
        self.index.add(embs)
        self.passages.extend(new)
        self._bm25_dirty = True

    def add_passages(self, passages):
        if not passages: return
        embs = self.embedder.encode([p.text for p in passages])
        self.index.add(embs)
        self.passages.extend(passages)
        self._bm25_dirty = True

    def search(self, query, k=15):
        """Hybrid search with reciprocal rank fusion."""
        if not self.passages: return []

        # Semantic
        sem = []
        if self.index.ntotal > 0:
            qe = self.embedder.encode_one(query).reshape(1, -1)
            n = min(k * 3, self.index.ntotal)
            scores, idxs = self.index.search(qe, n)
            sem = [(int(idx), float(sc)) for sc, idx in zip(scores[0], idxs[0])
                   if 0 <= idx < len(self.passages)]

        # BM25
        bm = []
        if self._bm25_dirty: self._rebuild_bm25()
        if self._bm25:
            sc = self._bm25.get_scores(self._tok(query))
            bm = sorted(enumerate(sc), key=lambda x: -x[1])[:k*3]
            bm = [(i, s) for i, s in bm if s > 0]

        # RRF merge
        K = 60
        rrf = {}
        for rank, (idx, _) in enumerate(sem):
            rrf[idx] = rrf.get(idx, 0) + 0.6 / (K + rank + 1)
        for rank, (idx, _) in enumerate(bm):
            rrf[idx] = rrf.get(idx, 0) + 0.4 / (K + rank + 1)

        # Dedup and return
        results, seen_sigs = [], set()
        for idx, score in sorted(rrf.items(), key=lambda x: -x[1]):
            p = self.passages[idx]
            sig = p.text[:80].lower()
            if sig in seen_sigs: continue
            seen_sigs.add(sig)
            p.similarity_score = score
            results.append(p)
            if len(results) >= k: break
        return results

    @property
    def size(self): return len(self.passages)
    @property
    def num_articles(self): return len({p.pmid for p in self.passages})


# =====================================================================
# LLM Client (vLLM at localhost:8000)
# =====================================================================

class LLM:
    def __init__(self, url="http://localhost:8000", model="gemma-3-27b-it"):
        self.url = url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        try:
            r = self.session.get(f"{self.url}/v1/models", timeout=10)
            r.raise_for_status()
            avail = [m["id"] for m in r.json().get("data", [])]
            log.info("vLLM: %s — %s", self.url, avail)
            if avail and self.model not in avail:
                self.model = avail[0]
                log.info("Using: %s", self.model)
        except Exception as e:
            log.error("Cannot reach vLLM at %s: %s", self.url, e)
            sys.exit(1)

    def ask(self, prompt, max_tokens=1024, temp=0.3):
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95}
        for i in range(3):
            try:
                r = self.session.post(f"{self.url}/v1/chat/completions",
                                      json=payload, timeout=180)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("LLM error (%d/3): %s", i+1, e)
                time.sleep(3 * (i+1))
        return ""


# =====================================================================
# Few-Shot Bank
# =====================================================================

class FewShot:
    def __init__(self):
        self.ex = {"factoid": [], "list": [], "yesno": [], "summary": []}

    def load(self, path, n=40):
        with open(path) as f: data = json.load(f)
        for q in data.get("questions", []):
            qt = q.get("type", "").lower()
            if qt not in self.ex: continue
            sn = q.get("snippets", [])
            ea, ia = q.get("exact_answer"), q.get("ideal_answer")
            if not sn or not ia: continue
            if qt != "summary" and not ea: continue
            self.ex[qt].append({
                "body": q["body"],
                "snippets": [s.get("text", "") for s in sn[:5]],
                "exact_answer": ea,
                "ideal_answer": ia if isinstance(ia, str) else ia[0] if isinstance(ia, list) and ia else "",
            })
        for qt in self.ex:
            self.ex[qt] = sorted(self.ex[qt], key=lambda x: len(x["body"]))[:n]
            log.info("  Few-shot %s: %d", qt, len(self.ex[qt]))

    def get(self, qt, n=2):
        e = self.ex.get(qt, [])
        s = [x for x in e if len(" ".join(x["snippets"])) < 1500]
        return (s or e)[:n]


# =====================================================================
# Prompts — tuned from eval results
# =====================================================================

def _sblock(texts, maxc=4000):
    b = ""
    for i, s in enumerate(texts, 1):
        line = f"[{i}] {s.strip()}\n"
        if len(b) + len(line) > maxc: break
        b += line
    return b.strip()

def _fmt(ea):
    if isinstance(ea, str): return ea
    if isinstance(ea, list):
        if ea and isinstance(ea[0], list):
            return "; ".join(", ".join(x) if isinstance(x, list) else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)


def prompt_queries(question, prev=None):
    p = ("Generate 3 short PubMed search queries (3-6 words each) for this question. "
         "Plain keywords only — no AND/OR/NOT, no [Mesh].\n\n"
         f"QUESTION: {question}\n\n")
    if prev:
        p += "Previous queries didn't work. Use DIFFERENT terms:\n"
        p += "\n".join(f"  - {q}" for q in prev[-3:]) + "\n\n"
    p += "Write ONLY the 3 queries, numbered. Nothing else.\n\n1."
    return p


def prompt_evaluate(question, texts):
    block = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts[:10]))
    return (f"Can this question be answered with these passages?\n\n"
            f"QUESTION: {question}\n\nPASSAGES:\n{block}\n\n"
            f"Reply SUFFICIENT or INSUFFICIENT.\nVERDICT:")


def prompt_answer(question, qtype, texts, fewshot):
    if qtype == "yesno":
        return _prompt_yesno(question, texts, fewshot)
    elif qtype == "factoid":
        return _prompt_factoid(question, texts, fewshot)
    elif qtype == "list":
        return _prompt_list(question, texts, fewshot)
    return _prompt_summary(question, texts, fewshot)


def _prompt_yesno(q, texts, fs):
    p = (
        "You are an expert biomedical QA system answering a yes/no question.\n\n"
        "INSTRUCTIONS:\n"
        "1. Find evidence in the passages that supports YES.\n"
        "2. Find evidence that supports NO.\n"
        "3. Choose the side with STRONGER evidence.\n"
        "4. If the question asks about efficacy/safety and the evidence shows "
        "PROBLEMS (toxicity, failure, side effects, lack of evidence), answer NO.\n"
        "5. If evidence is mixed or insufficient, lean toward NO.\n"
        "6. 'Promising preclinical results' does NOT mean the answer is YES.\n\n"
    )
    for ex in fs[:2]:
        ea = ex["exact_answer"]
        if isinstance(ea, list): ea = ea[0] if ea else "yes"
        p += f"---\nQUESTION: {ex['body']}\nEVIDENCE:\n{_sblock(ex['snippets'], 600)}\n"
        p += f"EXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQUESTION: {q}\nEVIDENCE:\n{_sblock(texts)}\n\n"
    p += "EVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:"
    return p


def _prompt_factoid(q, texts, fs):
    p = (
        "You are an expert biomedical QA system.\n\n"
        "RULES:\n"
        "1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.\n"
        "2. Use EXACT terminology from the evidence.\n"
        "3. Do NOT explain or elaborate in the exact answer.\n"
        "4. If evidence has no clear answer, write: unknown\n"
        "5. Prefer specific entities: drug names, gene names, disease names, numbers.\n\n"
        "GOOD: 'transsphenoidal surgery', 'NF1', '45,X', 'palivizumab'\n"
        "BAD: 'multiple causative factors', 'it involves several mechanisms'\n\n"
    )
    for ex in fs[:2]:
        p += f"---\nQUESTION: {ex['body']}\nEVIDENCE:\n{_sblock(ex['snippets'], 600)}\n"
        p += f"EXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQUESTION: {q}\nEVIDENCE:\n{_sblock(texts)}\nEXACT_ANSWER:"
    return p


def _prompt_list(q, texts, fs):
    p = (
        "You are an expert biomedical QA system.\n\n"
        "RULES:\n"
        "1. List EVERY relevant item from the evidence. Be EXHAUSTIVE.\n"
        "2. Include too many rather than too few.\n"
        "3. Go through EACH passage and extract ALL relevant items.\n"
        "4. Each item: 1-5 words, specific name or term.\n"
        "5. Aim for 5-15+ items. Many questions have 10+ correct answers.\n"
        "6. Prefix each with '- ' on its own line.\n"
        "7. After the list, write IDEAL_ANSWER: with a 2-4 sentence summary.\n\n"
    )
    for ex in fs[:1]:
        p += f"---\nQUESTION: {ex['body']}\nEVIDENCE:\n{_sblock(ex['snippets'], 500)}\nEXACT_ANSWER:\n"
        ea = ex["exact_answer"]
        if isinstance(ea, list):
            for item in ea[:8]:
                if isinstance(item, list): p += f"- {', '.join(item)}\n"
                else: p += f"- {item}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQUESTION: {q}\nEVIDENCE:\n{_sblock(texts, 5000)}\n"
    p += "List EVERY relevant item:\n\nEXACT_ANSWER:\n"
    return p


def _prompt_summary(q, texts, fs):
    p = "You are an expert biomedical QA system. Write a 3-6 sentence answer.\n\n"
    for ex in fs[:2]:
        p += f"---\nQUESTION: {ex['body']}\nEVIDENCE:\n{_sblock(ex['snippets'], 800)}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQUESTION: {q}\nEVIDENCE:\n{_sblock(texts)}\nIDEAL_ANSWER:"
    return p


def prompt_verify(q, texts, ans):
    return (f"Check this answer against evidence. Fix any errors or unsupported claims.\n\n"
            f"QUESTION: {q}\nEVIDENCE:\n{_sblock(texts, 3000)}\n\n"
            f"CANDIDATE:\n{ans}\n\nCORRECTED_ANSWER:")


# =====================================================================
# Parsers
# =====================================================================

def parse_factoid(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    e = parts[0].strip().strip('"\'').rstrip(".")
    for pfx in ["The answer is", "The exact answer is", "Answer:"]:
        if e.lower().startswith(pfx.lower()): e = e[len(pfx):].strip()
    ideal = parts[1].strip() if len(parts) == 2 else ""
    return [e] if e else ["unknown"], ideal or e


def parse_yesno(r):
    ea_m = re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)', r, re.IGNORECASE)
    if ea_m:
        raw = ea_m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        lines = r.strip().split("\n")
        raw = ""
        for line in reversed(lines):
            l = line.strip().lower()
            if "yes" in l[:10] or "no" in l[:10]: raw = l; break
        if not raw: raw = lines[-1].strip().lower() if lines else ""

    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    ideal = parts[1].strip() if len(parts) == 2 else ""

    if "yes" in raw[:10] and "no" not in raw[:5]: exact = "yes"
    elif "no" in raw[:10]: exact = "no"
    else:
        lo = r.lower()
        neg = len(re.findall(r'insufficient|toxicity|not effective|no evidence|failed|ineffective|not recommended|lack of', lo))
        pos = len(re.findall(r'effective|demonstrated|shown to|evidence supports|approved|recommended', lo))
        exact = "no" if neg >= pos else "yes"
    return exact, ideal or f"The answer is {exact}."


def parse_list(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, maxsplit=1, flags=re.IGNORECASE)
    lr = parts[0].strip()
    ideal = parts[1].strip() if len(parts) == 2 else ""
    items = []
    for line in lr.split("\n"):
        line = re.sub(r'^[\-\*•]\s*', '', line.strip())
        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip().strip('"\'').rstrip(".")
        if line and len(line) > 1: items.append([line])
    return items or [["unknown"]], ideal


def parse_summary(r):
    return re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', r, flags=re.IGNORECASE).strip() or "No answer."


# =====================================================================
# Consensus
# =====================================================================

def con_factoid(c):
    flat = [x[0].lower().strip() for x in c if x]
    if not flat: return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip() == best: return x
    return [best]

def con_yesno(c): return Counter(c).most_common(1)[0][0]

def con_list(c):
    items = {}
    for cand in c:
        for s in cand:
            k = s[0].lower().strip() if s else ""
            if k: items.setdefault(k, s)
    return list(items.values()) or [["unknown"]]

def con_ideal(c):
    if not c: return ""
    s = sorted(c, key=len)
    return s[len(s)//2]


# =====================================================================
# Helpers
# =====================================================================

def simple_kw(q):
    STOP = set("what is are the a an of in on to for with by from and or not how does do can which who whom where when why this that these those it its has have had been being please list describe common main types most often typically usually associated caused present should".split())
    toks = re.findall(r"[A-Za-z0-9\-']+", q)
    return " ".join(t for t in toks if t.lower() not in STOP and len(t) > 2)[:80]


def parse_queries(response):
    queries = []
    for line in response.strip().split("\n"):
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip()).strip('"').strip()
        if not line or len(line) < 5 or len(line) > 80: continue
        if any(s in line.lower() for s in [
            "here are", "queries", "search for", "i would", "let me",
            "the following", "note:", "these", "aim", "different"
        ]): continue
        queries.append(line)
    return queries[:3]


# =====================================================================
# THE AGENT
# =====================================================================

class Agent:
    def __init__(self, llm, pubmed, embedder, fewshot,
                 passes=3, iters=3, articles=30):
        self.llm = llm
        self.pm = pubmed
        self.emb = embedder
        self.fs = fewshot
        self.passes = passes
        self.iters = iters
        self.n_articles = articles

    def solve(self, question):
        qid = question["id"]
        body = question["body"]
        qtype = question.get("type", "summary").lower()
        log.info("━━━ %s [%s]: %.70s", qid, qtype, body)

        idx = HybridIndex(self.emb)
        seen, qused = set(), []

        # ── Ingest gold data ──
        for gs in question.get("snippets", []):
            t = gs.get("text", "").strip()
            if not t: continue
            du = gs.get("document", "")
            m = re.search(r'/pubmed/(\d+)', du)
            idx.add_passages([Passage(
                text=t, pmid=m.group(1) if m else "gold", doc_url=du,
                section=gs.get("beginSection", "abstract"),
                offset_begin=gs.get("offsetInBeginSection", 0),
                offset_end=gs.get("offsetInEndSection", 0))])

        gp = []
        for du in question.get("documents", []):
            m = re.search(r'/pubmed/(\d+)', du)
            if m and m.group(1) not in seen:
                gp.append(m.group(1)); seen.add(m.group(1))
        if gp:
            for a in self.pm.fetch(gp): idx.add_article(a)
        log.info("  Gold: %d passages", idx.size)

        # ── Seed corpus ──
        kw = simple_kw(body)
        if kw:
            qused.append(kw)
            pids = self.pm.search(kw, self.n_articles)
            new = [p for p in pids if p not in seen]; seen.update(new)
            if new:
                for a in self.pm.fetch(new): idx.add_article(a)
            log.info("  Seed '%s': %d passages", kw[:40], idx.size)

        # ── Agentic loop ──
        for it in range(self.iters):
            log.info("  Iter %d/%d", it+1, self.iters)

            # LLM generates queries
            qr = self.llm.ask(prompt_queries(body, qused or None),
                              max_tokens=200, temp=0.3)
            nq = parse_queries(qr)
            if not nq:
                w = simple_kw(body).split()
                if len(w) >= 3: nq = [" ".join(w[:3])]

            for q in nq:
                if q in qused: continue
                qused.append(q)
                pids = self.pm.search(q, self.n_articles)
                new = [p for p in pids if p not in seen]; seen.update(new)
                if new:
                    for a in self.pm.fetch(new): idx.add_article(a)
                log.info("    '%s': +%d new", q[:50], len(new))

            # Search our index
            top = idx.search(body, k=10)
            if top:
                log.info("    Best (%.4f): %.60s...", top[0].similarity_score, top[0].text)

            # Expand from best articles
            if top and it < self.iters - 1:
                best_pmids = list(dict.fromkeys(
                    p.pmid for p in top[:3] if p.pmid != "gold"))
                for bp in best_pmids[:2]:
                    rel = self.pm.related(bp, 10)
                    new = [p for p in rel if p not in seen]; seen.update(new)
                    if new:
                        for a in self.pm.fetch(new): idx.add_article(a)
                        log.info("    Related(%s): +%d", bp, len(new))
                    break

            log.info("    Index: %d articles, %d passages", idx.num_articles, idx.size)

            # Evaluate
            if idx.size >= 15:
                v = self.llm.ask(prompt_evaluate(body, [p.text for p in top]),
                                 max_tokens=20, temp=0.1)
                if "SUFFICIENT" in v.upper():
                    log.info("    SUFFICIENT"); break

        # ── LLM rerank top passages ──
        top = idx.search(body, k=20)
        if len(top) > 5:
            block = "\n".join(f"[{i+1}] {p.text[:200]}" for i, p in enumerate(top[:20]))
            scores_resp = self.llm.ask(
                f"Rate each passage's relevance (0-2) to: {body}\n\n{block}\n\n"
                f"Return: [N] SCORE\nSCORES:\n", max_tokens=256, temp=0.1)
            for line in scores_resp.strip().split("\n"):
                m = re.match(r'\[(\d+)\]\s*(\d)', line)
                if m:
                    i, sc = int(m.group(1))-1, int(m.group(2))
                    if 0 <= i < len(top):
                        top[i].similarity_score = top[i].similarity_score * 0.4 + (sc/2)*0.6
            top = sorted(top, key=lambda p: p.similarity_score, reverse=True)

        log.info("  Final: %d articles, %d passages", idx.num_articles, idx.size)

        # ── Answer ──
        fsex = self.fs.get(qtype, 2)
        ptexts = [p.text for p in top[:15]] or [body]

        ec, ic = [], []
        for pi in range(self.passes):
            temp = 0.15 + pi * 0.15
            log.info("  Answer %d/%d (temp=%.2f)", pi+1, self.passes, temp)
            raw = self.llm.ask(prompt_answer(body, qtype, ptexts, fsex),
                               max_tokens=768, temp=temp)
            if qtype == "factoid": e, i = parse_factoid(raw); ec.append(e)
            elif qtype == "yesno": e, i = parse_yesno(raw); ec.append(e)
            elif qtype == "list": e, i = parse_list(raw); ec.append(e)
            else: i = parse_summary(raw)
            ic.append(i)

            # Verify first pass
            if pi == 0:
                corr = self.llm.ask(prompt_verify(body, ptexts, raw),
                                     max_tokens=768, temp=0.2)
                if qtype == "factoid": e2, i2 = parse_factoid(corr); ec.append(e2); ic.append(i2)
                elif qtype == "yesno": e2, i2 = parse_yesno(corr); ec.append(e2); ic.append(i2)
                elif qtype == "list": e2, i2 = parse_list(corr); ec.append(e2); ic.append(i2)
                else: ic.append(parse_summary(corr))

        # Consensus
        result = {"id": qid}
        if qtype == "factoid": result["exact_answer"] = con_factoid(ec)
        elif qtype == "yesno": result["exact_answer"] = con_yesno(ec)
        elif qtype == "list": result["exact_answer"] = con_list(ec)
        result["ideal_answer"] = con_ideal(ic)

        # Documents + snippets for Phase A
        seen_urls = set()
        result["_documents"] = []
        for p in top[:10]:
            if p.doc_url and p.doc_url not in seen_urls:
                seen_urls.add(p.doc_url); result["_documents"].append(p.doc_url)
        result["_snippets"] = [{"text": p.text, "document": p.doc_url,
            "beginSection": "sections.0" if p.section in ("abstract","title") else p.section,
            "endSection": "sections.0" if p.section in ("abstract","title") else p.section,
            "offsetInBeginSection": p.offset_begin,
            "offsetInEndSection": p.offset_end} for p in top[:10]]

        log.info("  ✓ exact=%s", str(result.get("exact_answer", "N/A"))[:60])
        return result


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="BioASQ 14 — Agentic RAG Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run on Phase A test set (retrieves articles + generates answers)
  python bioasq_final.py -t BioASQ-task14bPhaseA-testset1.json -tr training13b.json -o sub.json

  # Run on Phase B test set (gold snippets provided)
  python bioasq_final.py -t BioASQ-task14bPhaseB-testset1.json -tr training13b.json -o sub.json
""")
    ap.add_argument("--test-input", "-t", required=True, help="Test set JSON")
    ap.add_argument("--training", "-tr", default=None, help="Training JSON for few-shot")
    ap.add_argument("--output", "-o", default="submission.json", help="Output file")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--model", "-m", default="gemma-3-27b-it")
    ap.add_argument("--embedding-model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--embed-device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--passes", type=int, default=3, help="Answer generation passes")
    ap.add_argument("--retrieval-iterations", type=int, default=3)
    ap.add_argument("--articles-per-query", type=int, default=30)
    ap.add_argument("--api-key", default=None, help="NCBI API key (optional)")
    ap.add_argument("--question-ids", nargs="*", default=None)
    a = ap.parse_args()

    fs = FewShot()
    if a.training: fs.load(a.training)

    with open(a.test_input) as f: qs = json.load(f).get("questions", [])
    log.info("Loaded %d questions", len(qs))
    if a.question_ids: qs = [q for q in qs if q["id"] in set(a.question_ids)]

    llm = LLM(url=a.vllm_url, model=a.model)
    pm = PubMed(api_key=a.api_key)
    emb = Embedder(model=a.embedding_model, device=a.embed_device)
    agent = Agent(llm, pm, emb, fs, a.passes, a.retrieval_iterations, a.articles_per_query)

    results = []
    for i, q in enumerate(qs):
        log.info("═══ Question %d / %d ═══", i+1, len(qs))
        try:
            results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s", q.get("id"), e, exc_info=True)
            results.append({"id": q["id"], "ideal_answer": "Unable to generate."})

    # ── Write Phase B submission ──
    phase_b = {"questions": []}
    for r in results:
        entry = {"id": r["id"], "ideal_answer": r.get("ideal_answer", "")}
        if "exact_answer" in r: entry["exact_answer"] = r["exact_answer"]
        phase_b["questions"].append(entry)
    for entry in phase_b["questions"]:
        if entry.get("exact_answer") is None: entry.pop("exact_answer", None)

    with open(a.output, "w") as f:
        json.dump(phase_b, f, indent=2, ensure_ascii=False)
    log.info("Phase B → %s", a.output)

    # ── Write Phase A submission ──
    phase_a = {"questions": []}
    for r in results:
        phase_a["questions"].append({
            "id": r["id"],
            "documents": r.get("_documents", []),
            "snippets": r.get("_snippets", []),
        })
    out_a = a.output.replace(".json", "_phaseA.json")
    with open(out_a, "w") as f:
        json.dump(phase_a, f, indent=2, ensure_ascii=False)
    log.info("Phase A → %s", out_a)

    print(f"\nDone! {len(results)} questions")
    print(f"Types: {dict(Counter(q.get('type','?') for q in qs))}")
    print(f"Phase B (answers):   {a.output}")
    print(f"Phase A (articles):  {out_a}")


if __name__ == "__main__":
    main()
