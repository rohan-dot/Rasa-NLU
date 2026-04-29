#!/usr/bin/env python3
"""
BioASQ Task 14b — FINAL COMPETITION SOLVER (Batch 4)
=====================================================
The best of everything. Handles Phase A+ and Phase B.

PubMed retrieval → FAISS+BM25 hybrid indexing → LLM reranking
→ multi-pass answer generation → self-verification → consensus

RUN:
    python bioasq_competition.py \
        --test-input BioASQ-task14bPhaseA-testset4.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        --embed-device cpu \
        -o batch4.json

OUTPUT:
    batch4.json           → Phase B submission (answers)
    batch4_phaseA.json    → Phase A+ submission (docs + snippets + answers)
"""

import argparse, json, logging, re, sys, time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
import numpy as np, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

@dataclass
class Passage:
    text: str; pmid: str; doc_url: str; section: str = "abstract"
    offset_begin: int = 0; offset_end: int = 0; score: float = 0.0

# ═════════════════════════════════════════════════════════════════════
# PUBMED — retrieval + citation expansion
# ═════════════════════════════════════════════════════════════════════

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

class PubMed:
    def __init__(self, api_key=None):
        self.s = requests.Session()
        self.key = api_key
        self._last = 0.0
        self._min = 0.2 if api_key else 1.0
        self._bo = 1.0

    def _wait(self):
        w = max(self._min, self._bo) - (time.time() - self._last)
        if w > 0: time.sleep(w)
        self._last = time.time()

    def _get(self, url, p, retries=3):
        for i in range(retries):
            self._wait()
            try:
                r = self.s.get(url, params=p, timeout=30)
                if r.status_code == 429:
                    self._bo = min(self._bo * 2, 10)
                    log.warning("    429 — backing off %.1fs", self._bo)
                    time.sleep(self._bo + i * 2); continue
                r.raise_for_status()
                self._bo = max(self._bo * 0.8, self._min)
                return r
            except Exception as e:
                log.warning("    PubMed(%d/%d): %s", i+1, retries, e)
                time.sleep(2 * (i+1))
        return None

    def search(self, query, n=30):
        """Search PubMed for PMIDs."""
        p = {"db": "pubmed", "term": query, "retmax": n, "retmode": "json"}
        if self.key: p["api_key"] = self.key
        r = self._get(ESEARCH, p)
        try: return r.json()["esearchresult"]["idlist"] if r else []
        except: return []

    def fetch(self, pmids):
        """Fetch article data by PMIDs (batched)."""
        if not pmids: return []
        all_a = []
        for i in range(0, len(pmids), 50):
            p = {"db": "pubmed", "id": ",".join(pmids[i:i+50]),
                 "rettype": "xml", "retmode": "xml"}
            if self.key: p["api_key"] = self.key
            r = self._get(EFETCH, p)
            if r: all_a.extend(self._parse(r.text))
        return all_a

    def related(self, pmid, n=15):
        """Get related articles via NCBI elink (citation network)."""
        p = {"dbfrom": "pubmed", "db": "pubmed", "id": pmid,
             "cmd": "neighbor_score", "retmode": "json"}
        if self.key: p["api_key"] = self.key
        r = self._get(ELINK, p)
        if not r: return []
        try:
            for ls in r.json().get("linksets", [{}])[0].get("linksetdbs", []):
                if ls.get("linkname") == "pubmed_pubmed":
                    return [str(l["id"]) for l in ls.get("links", [])][:n]
        except: pass
        return []

    def citing(self, pmid, n=10):
        """Get articles that cite this one."""
        p = {"dbfrom": "pubmed", "db": "pubmed", "id": pmid,
             "cmd": "neighbor", "linkname": "pubmed_pubmed_citedin",
             "retmode": "json"}
        if self.key: p["api_key"] = self.key
        r = self._get(ELINK, p)
        if not r: return []
        try:
            for ls in r.json().get("linksets", [{}])[0].get("linksetdbs", []):
                if "citedin" in ls.get("linkname", ""):
                    return [str(l["id"]) for l in ls.get("links", [])][:n]
        except: pass
        return []

    def _parse(self, xml):
        arts = []
        try:
            for el in ET.fromstring(xml).findall(".//PubmedArticle"):
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
        except: pass
        return arts


# ═════════════════════════════════════════════════════════════════════
# EMBEDDER + HYBRID INDEX
# ═════════════════════════════════════════════════════════════════════

class Embedder:
    def __init__(self, model="pritamdeka/S-PubMedBert-MS-MARCO", device=None):
        import torch; from sentence_transformers import SentenceTransformer
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Embedder: %s (%s)", model, device)
        self.m = SentenceTransformer(model, device=device)
        self.dim = self.m.get_sentence_embedding_dimension()
    def encode(self, t):
        return np.array(self.m.encode(t, show_progress_bar=False,
                        normalize_embeddings=True), dtype=np.float32)
    def encode1(self, t): return self.encode([t])[0]


class Index:
    """Per-question FAISS+BM25 hybrid index with overlapping chunking."""
    def __init__(self, emb):
        import faiss
        self.faiss = faiss; self.emb = emb
        self.ix = faiss.IndexFlatIP(emb.dim)
        self.ps: list[Passage] = []; self._seen = set()
        self._bm = None; self._d = True

    def add_art(self, a):
        pmid, url = a["pmid"], a["url"]; new = []
        t = a.get("title", "").strip()
        if t and t not in self._seen:
            self._seen.add(t)
            new.append(Passage(t, pmid, url, "title", 0, len(t)))
        ab = a.get("abstract", "").strip()
        if ab:
            # Full abstract
            if ab not in self._seen and len(ab) > 50:
                self._seen.add(ab)
                new.append(Passage(ab, pmid, url, "abstract", 0, len(ab)))
            # Overlapping 3-sentence chunks, stride 1
            ss = re.split(r'(?<=[.!?])\s+', ab)
            if len(ss) > 3:
                for i in range(len(ss) - 2):
                    ch = " ".join(s.strip() for s in ss[i:i+3]).strip()
                    if len(ch) < 50 or ch in self._seen: continue
                    b = ab.find(ss[i])
                    self._seen.add(ch)
                    new.append(Passage(ch, pmid, url, "abstract",
                                       max(b, 0), max(b, 0) + len(ch)))
        if not new: return
        self.ix.add(self.emb.encode([p.text for p in new]))
        self.ps.extend(new); self._d = True

    def add_gold(self, passages):
        if not passages: return
        self.ix.add(self.emb.encode([p.text for p in passages]))
        self.ps.extend(passages); self._d = True

    def search(self, q, k=15):
        if not self.ps: return []
        # Semantic
        sem = []
        if self.ix.ntotal > 0:
            sc, ix = self.ix.search(self.emb.encode1(q).reshape(1, -1),
                                    min(k * 3, self.ix.ntotal))
            sem = [(int(i), float(s)) for s, i in zip(sc[0], ix[0])
                   if 0 <= i < len(self.ps)]
        # BM25
        if self._d:
            from rank_bm25 import BM25Okapi
            self._bm = BM25Okapi([re.findall(r'[a-z0-9\-]+', p.text.lower())
                                  for p in self.ps])
            self._d = False
        bm = []
        if self._bm:
            bsc = self._bm.get_scores(re.findall(r'[a-z0-9\-]+', q.lower()))
            bm = sorted(enumerate(bsc), key=lambda x: -x[1])[:k*3]
            bm = [(i, s) for i, s in bm if s > 0]
        # RRF
        rrf = {}
        for r, (i, _) in enumerate(sem): rrf[i] = rrf.get(i, 0) + 0.6/(60+r+1)
        for r, (i, _) in enumerate(bm):  rrf[i] = rrf.get(i, 0) + 0.4/(60+r+1)
        res, sn = [], set()
        for i, sc in sorted(rrf.items(), key=lambda x: -x[1]):
            p = self.ps[i]; sg = p.text[:80].lower()
            if sg in sn: continue
            sn.add(sg); p.score = sc; res.append(p)
            if len(res) >= k: break
        return res

    @property
    def size(self): return len(self.ps)
    @property
    def n_art(self): return len({p.pmid for p in self.ps})


# ═════════════════════════════════════════════════════════════════════
# LLM
# ═════════════════════════════════════════════════════════════════════

class LLM:
    def __init__(self, url="http://localhost:8000", model="gemma-3-27b-it"):
        self.url = url.rstrip("/"); self.model = model
        self.s = requests.Session()
        try:
            r = self.s.get(f"{self.url}/v1/models", timeout=10); r.raise_for_status()
            av = [m["id"] for m in r.json().get("data", [])]
            log.info("vLLM: %s", av)
            if av and self.model not in av: self.model = av[0]
        except Exception as e: log.error("vLLM: %s", e); sys.exit(1)

    def ask(self, p, mt=1024, t=0.3):
        for i in range(3):
            try:
                r = self.s.post(f"{self.url}/v1/chat/completions",
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": p}],
                          "max_tokens": mt, "temperature": t, "top_p": 0.95},
                    timeout=180)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("LLM(%d): %s", i+1, e); time.sleep(3*(i+1))
        return ""


# ═════════════════════════════════════════════════════════════════════
# FEW-SHOT
# ═════════════════════════════════════════════════════════════════════

class FS:
    def __init__(self): self.ex = {"factoid":[],"list":[],"yesno":[],"summary":[]}
    def load(self, path, n=40):
        with open(path) as f: data = json.load(f)
        for q in data.get("questions", []):
            qt = q.get("type","").lower()
            if qt not in self.ex: continue
            sn = q.get("snippets",[]); ea = q.get("exact_answer"); ia = q.get("ideal_answer")
            if not sn or not ia: continue
            if qt != "summary" and not ea: continue
            self.ex[qt].append({"body": q["body"],
                "snippets": [s.get("text","") for s in sn[:5]], "exact_answer": ea,
                "ideal_answer": ia if isinstance(ia,str) else ia[0] if isinstance(ia,list) and ia else ""})
        for qt in self.ex:
            self.ex[qt] = sorted(self.ex[qt], key=lambda x: len(x["body"]))[:n]
            log.info("  FS %s: %d", qt, len(self.ex[qt]))
    def get(self, qt, n=2):
        e = self.ex.get(qt,[]); s = [x for x in e if len(" ".join(x["snippets"]))<1500]
        return (s or e)[:n]


# ═════════════════════════════════════════════════════════════════════
# PROMPTS (battle-tested from eval iterations)
# ═════════════════════════════════════════════════════════════════════

def _sb(ts, mx=5000):
    b = ""
    for i, s in enumerate(ts, 1):
        l = f"[{i}] {s.strip()}\n"
        if len(b)+len(l) > mx: break
        b += l
    return b.strip()

def _fmt(ea):
    if isinstance(ea, str): return ea
    if isinstance(ea, list):
        if ea and isinstance(ea[0], list):
            return "; ".join(", ".join(x) if isinstance(x,list) else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)

def _cap(t):
    w = t.split(); return " ".join(w[:200]) if len(w) > 200 else t

def pr_queries(q, prev=None):
    p = f"Generate 3 short PubMed queries (3-6 words, plain keywords) for: {q}\n\n"
    if prev:
        p += "Previous didn't find enough. Use COMPLETELY DIFFERENT terms:\n"
        p += "\n".join(f"  - {x}" for x in prev[-3:]) + "\n\n"
    p += "Write ONLY 3 queries numbered. Nothing else.\n\n1."
    return p

def pr_eval(q, ts):
    return (f"Can '{q}' be answered from these?\n\n" +
            "\n".join(f"[{i+1}] {t[:150]}" for i,t in enumerate(ts[:8])) +
            "\n\nSUFFICIENT or INSUFFICIENT?")

def pr_answer(q, qt, ts, fs):
    if qt == "yesno": return _yn(q, ts, fs)
    if qt == "factoid": return _fac(q, ts, fs)
    if qt == "list": return _lst(q, ts, fs)
    return _sum(q, ts, fs)

def _yn(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\n"
         "INSTRUCTIONS:\n"
         "1. Find evidence supporting YES.\n"
         "2. Find evidence supporting NO.\n"
         "3. Choose the side with STRONGER direct evidence.\n"
         "4. If evidence shows PROBLEMS (toxicity, failure, side effects, "
         "contradictory results, lack of clinical evidence), answer NO.\n"
         "5. If evidence is mixed, unclear, or insufficient, answer NO.\n"
         "6. 'Promising preclinical results' or 'under investigation' does NOT mean YES.\n"
         "7. A drug being 'studied' or 'tested' does NOT mean it works.\n\n")
    for ex in fs[:2]:
        ea = ex["exact_answer"]
        if isinstance(ea, list): ea = ea[0] if ea else "yes"
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\n"
        p += f"EXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\n"
    p += "EVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:"
    return p

def _fac(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\n"
         "STRICT RULES:\n"
         "1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.\n"
         "2. Copy the EXACT terminology from the evidence passages.\n"
         "3. Do NOT paraphrase, explain, or elaborate in the exact answer.\n"
         "4. If the evidence does not contain a clear specific answer, write: unknown\n"
         "5. Prefer named entities: drug names, protein names, gene symbols, disease names, "
         "specific numbers/percentages.\n"
         "6. Then write IDEAL_ANSWER: a 2-4 sentence explanation (max 200 words).\n\n"
         "GOOD exact answers: 'transsphenoidal surgery', 'NF1', '45,X', 'palivizumab', "
         "'150 per million', 'mesenchymal'\n"
         "BAD exact answers: 'multiple causative factors', 'a type of bleeding', "
         "'transcriptional regulation involving several pathways'\n\n")
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\n"
        p += f"EXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nEXACT_ANSWER:"
    return p

def _lst(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\n"
         "STRICT RULES:\n"
         "1. List EVERY relevant item mentioned in the evidence. Be EXHAUSTIVE.\n"
         "2. It is MUCH better to include too many items than too few.\n"
         "3. Go through EACH evidence passage systematically and extract ALL relevant items.\n"
         "4. Each item should be a specific name or term (1-5 words).\n"
         "5. Aim for at least 5-15 items. Many questions have 10-20+ answers.\n"
         "6. Prefix each item with '- ' on its own line.\n"
         "7. Do NOT group or combine items. One item per line.\n"
         "8. After the list, write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n")
    for ex in fs[:1]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],500)}\nEXACT_ANSWER:\n"
        ea = ex["exact_answer"]
        if isinstance(ea, list):
            for it in ea[:10]:
                if isinstance(it, list): p += f"- {it[0]}\n"
                else: p += f"- {it}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts, 6000)}\n"
    p += "Now list EVERY relevant item from ALL passages:\n\nEXACT_ANSWER:\n"
    return p

def _sum(q, ts, fs):
    p = ("You are an expert biomedical QA system. Write a comprehensive "
         "3-6 sentence answer (max 200 words) using the evidence.\n\n")
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],800)}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:"
    return p

def pr_verify(q, ts, a):
    return (f"You are a biomedical reviewer. Check this answer against the evidence. "
            f"Fix any errors, unsupported claims, or missing key information.\n\n"
            f"Q: {q}\nEVIDENCE:\n{_sb(ts,3000)}\n\nCANDIDATE:\n{a}\n\nCORRECTED_ANSWER:")


# ═════════════════════════════════════════════════════════════════════
# PARSERS
# ═════════════════════════════════════════════════════════════════════

def p_fac(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    e = parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is","The exact answer is","Answer:","EXACT_ANSWER:"]:
        if e.lower().startswith(pf.lower()): e = e[len(pf):].strip()
    # Remove trailing explanation after newline
    if "\n" in e: e = e.split("\n")[0].strip()
    ideal = parts[1].strip() if len(parts) == 2 else ""
    return [e.strip()] if e.strip() else ["unknown"], _cap(ideal or e)

def p_yn(r):
    m = re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)', r, re.I)
    if m: raw = m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        raw = ""
        for l in reversed(r.strip().split("\n")):
            ll = l.strip().lower()
            if ll.startswith("yes") or ll.startswith("no"): raw = ll; break
        if not raw: raw = r.strip().split("\n")[-1].lower()
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    ideal = parts[1].strip() if len(parts) == 2 else ""
    # Determine yes/no
    if raw.startswith("yes"): exact = "yes"
    elif raw.startswith("no"): exact = "no"
    else:
        lo = r.lower()
        neg = len(re.findall(r'insufficient|toxicity|not effective|no evidence|failed|'
                             r'ineffective|not recommended|lack of|no direct|not been shown|'
                             r'no conclusive|adverse|contradicted|limited evidence', lo))
        pos = len(re.findall(r'effective|demonstrated efficacy|shown to be|evidence supports|'
                             r'fda.approved|recommended|clinically proven|established', lo))
        exact = "no" if neg >= pos else "yes"
    return exact, _cap(ideal or f"The answer is {exact}.")

def p_lst(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    lr = parts[0].strip(); ideal = parts[1].strip() if len(parts) == 2 else ""
    items = []
    for l in lr.split("\n"):
        l = re.sub(r'^[\-\*•]\s*', '', l.strip())
        l = re.sub(r'^\d+[\.\)]\s*', '', l).strip().strip('"\'').rstrip(".")
        if l and 1 < len(l) <= 100:
            items.append(l)
    return items[:100] or ["unknown"], _cap(ideal)

def p_sum(r):
    return _cap(re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', r, flags=re.I).strip() or "No answer.")


# ═════════════════════════════════════════════════════════════════════
# CONSENSUS
# ═════════════════════════════════════════════════════════════════════

def con_fac(c):
    flat = [x[0].lower().strip() for x in c if x and x[0].strip()]
    if not flat: return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip() == best: return x[:5]
    return [best]

def con_yn(c):
    if not c: return "no"
    return Counter(c).most_common(1)[0][0]

def con_lst(c):
    items = {}
    for cand in c:
        for s in cand:
            k = s.lower().strip() if isinstance(s, str) else s
            if k and k != "unknown": items.setdefault(k, s)
    return list(items.values())[:100] or ["unknown"]

def con_ideal(c):
    c = [x for x in c if x and len(x) > 10]
    if not c: return "No sufficient evidence to generate an answer."
    s = sorted(c, key=len)
    return s[len(s)//2]


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def kw(q):
    STOP = set("what is are the a an of in on to for with by from and or not how does do can "
               "which who whom where when why this that these those it its has have had been "
               "being please list describe common main types most often should typically".split())
    return " ".join(t for t in re.findall(r"[A-Za-z0-9\-']+", q)
                    if t.lower() not in STOP and len(t) > 2)[:80]

def parse_q(r):
    qs = []
    for l in r.strip().split("\n"):
        l = re.sub(r'^\d+[\.\)]\s*', '', l.strip()).strip('"').strip()
        if not l or len(l) < 5 or len(l) > 80: continue
        if any(s in l.lower() for s in ["here are","queries","search for","i would",
               "let me","following","note:","aim","different","these"]): continue
        qs.append(l)
    return qs[:3]


# ═════════════════════════════════════════════════════════════════════
# THE AGENT
# ═════════════════════════════════════════════════════════════════════

class Agent:
    def __init__(self, llm, pm, emb, fs, passes=3, iters=3, n_art=30):
        self.llm = llm; self.pm = pm; self.emb = emb; self.fs = fs
        self.passes = passes; self.iters = iters; self.n_art = n_art

    def solve(self, question):
        qid = question["id"]
        body = question["body"]
        qtype = question.get("type", "summary").lower()
        log.info("━━━ %s [%s]: %s", qid, qtype, body)

        idx = Index(self.emb)
        seen = set()
        qused = []

        # ── Ingest gold snippets (Phase B) ──
        gold_p = []
        for gs in question.get("snippets", []):
            t = gs.get("text", "").strip()
            if not t: continue
            du = gs.get("document", "")
            m = re.search(r'/pubmed/(\d+)', du)
            gold_p.append(Passage(t, m.group(1) if m else "gold", du, "abstract",
                                  gs.get("offsetInBeginSection", 0),
                                  gs.get("offsetInEndSection", 0)))
        if gold_p:
            idx.add_gold(gold_p)

        # Fetch gold doc abstracts
        gpmids = []
        for du in question.get("documents", []):
            m = re.search(r'/pubmed/(\d+)', du)
            if m and m.group(1) not in seen:
                gpmids.append(m.group(1)); seen.add(m.group(1))
        if gpmids:
            for a in self.pm.fetch(gpmids): idx.add_art(a)

        has_gold = len(gold_p) > 0
        log.info("  Gold: %d snippets, %d doc PMIDs", len(gold_p), len(gpmids))

        # ── Retrieval (always for A+, backup for B) ──
        if not has_gold or len(gold_p) < 3:
            # Seed search
            k = kw(body)
            if k:
                qused.append(k)
                pids = self.pm.search(k, self.n_art)
                new = [p for p in pids if p not in seen]; seen.update(new)
                if new:
                    for a in self.pm.fetch(new): idx.add_art(a)
                log.info("  Seed '%s': %d passages", k[:40], idx.size)

            # Agentic loop
            for it in range(self.iters):
                log.info("  Iter %d/%d", it+1, self.iters)

                qr = self.llm.ask(pr_queries(body, qused or None), mt=200, t=0.3)
                nq = parse_q(qr)
                if not nq:
                    w = kw(body).split()
                    if len(w) >= 3: nq = [" ".join(w[:3])]

                for q in nq:
                    if q in qused: continue
                    qused.append(q)
                    pids = self.pm.search(q, self.n_art)
                    new = [p for p in pids if p not in seen]; seen.update(new)
                    if new:
                        for a in self.pm.fetch(new): idx.add_art(a)
                    log.info("    '%s': +%d", q[:50], len(new))

                # Our retrieval
                top = idx.search(body, k=10)
                if top:
                    log.info("    Best(%.4f): %s...", top[0].score, top[0].text[:60])

                # Expand from best
                if top:
                    for p in top[:2]:
                        if p.pmid == "gold": continue
                        rel = self.pm.related(p.pmid, 10)
                        new = [r for r in rel if r not in seen]; seen.update(new)
                        if new:
                            for a in self.pm.fetch(new): idx.add_art(a)
                            log.info("    Related(%s): +%d", p.pmid, len(new))
                        # One citing expansion too
                        cit = self.pm.citing(p.pmid, 5)
                        new2 = [c for c in cit if c not in seen]; seen.update(new2)
                        if new2:
                            for a in self.pm.fetch(new2): idx.add_art(a)
                            log.info("    Citing(%s): +%d", p.pmid, len(new2))
                        break

                log.info("    Index: %d art, %d passages", idx.n_art, idx.size)

                if idx.size >= 20:
                    v = self.llm.ask(pr_eval(body, [p.text for p in (top or [])]),
                                     mt=20, t=0.1)
                    if "SUFFICIENT" in v.upper():
                        log.info("    SUFFICIENT"); break

        log.info("  Final: %d art, %d passages, %d queries",
                 idx.n_art, idx.size, len(qused))

        # ── Rerank ──
        top = idx.search(body, k=20)
        if len(top) > 5:
            block = "\n".join(f"[{i+1}] {p.text[:200]}" for i, p in enumerate(top[:20]))
            sr = self.llm.ask(
                f"Rate each passage's relevance (0-2) to answering: {body}\n\n"
                f"{block}\n\nReturn [N] SCORE per line:\n", mt=256, t=0.1)
            for line in sr.strip().split("\n"):
                m = re.match(r'\[(\d+)\]\s*(\d)', line)
                if m:
                    i, sc = int(m.group(1))-1, int(m.group(2))
                    if 0 <= i < len(top):
                        top[i].score = top[i].score * 0.4 + (sc/2) * 0.6
            top = sorted(top, key=lambda p: p.score, reverse=True)
            log.info("  Reranked: top(%.3f) %s...", top[0].score, top[0].text[:50])

        # ── Answer generation ──
        fsex = self.fs.get(qtype, 2)
        ptexts = [p.text for p in top[:15]] or [body]

        ec, ic = [], []
        for pi in range(self.passes):
            temp = 0.15 + pi * 0.15
            log.info("  Pass %d/%d (t=%.2f)", pi+1, self.passes, temp)
            raw = self.llm.ask(pr_answer(body, qtype, ptexts, fsex), mt=768, t=temp)

            if qtype == "factoid": e, i = p_fac(raw); ec.append(e)
            elif qtype == "yesno": e, i = p_yn(raw); ec.append(e)
            elif qtype == "list": e, i = p_lst(raw); ec.append(e)
            else: i = p_sum(raw)
            ic.append(i)

            # Verify first pass
            if pi == 0:
                corr = self.llm.ask(pr_verify(body, ptexts, raw), mt=768, t=0.2)
                if qtype == "factoid": e2, i2 = p_fac(corr); ec.append(e2); ic.append(i2)
                elif qtype == "yesno": e2, i2 = p_yn(corr); ec.append(e2); ic.append(i2)
                elif qtype == "list": e2, i2 = p_lst(corr); ec.append(e2); ic.append(i2)
                else: ic.append(p_sum(corr))

        # ── Result ──
        result = {"id": qid, "ideal_answer": con_ideal(ic)}
        if qtype == "factoid": result["exact_answer"] = con_fac(ec)
        elif qtype == "yesno": result["exact_answer"] = con_yn(ec)
        elif qtype == "list": result["exact_answer"] = con_lst(ec)

        # Docs + snippets for Phase A
        su = set(); result["documents"] = []
        for p in top[:10]:
            if p.doc_url and p.doc_url not in su:
                su.add(p.doc_url); result["documents"].append(p.doc_url)
        result["snippets"] = [{"document": p.doc_url, "text": p.text,
            "offsetInBeginSection": p.offset_begin,
            "offsetInEndSection": p.offset_end,
            "beginSection": "sections.0", "endSection": "sections.0"}
            for p in top[:10]]

        log.info("  ✓ exact=%s", str(result.get("exact_answer", "N/A"))[:60])
        return result


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="BioASQ 14 — Competition Solver")
    ap.add_argument("--test-input", "-t", required=True)
    ap.add_argument("--training", "-tr", default=None)
    ap.add_argument("--output", "-o", default="submission.json")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--model", "-m", default="gemma-3-27b-it")
    ap.add_argument("--embedding-model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--embed-device", default=None)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--retrieval-iterations", type=int, default=3)
    ap.add_argument("--articles-per-query", type=int, default=30)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()

    fs = FS()
    if a.training: fs.load(a.training)

    with open(a.test_input) as f: qs = json.load(f).get("questions", [])
    log.info("Loaded %d questions", len(qs))

    llm = LLM(a.vllm_url, a.model)
    pm = PubMed(a.api_key)
    emb = Embedder(a.embedding_model, a.embed_device)
    agent = Agent(llm, pm, emb, fs, a.passes, a.retrieval_iterations, a.articles_per_query)

    results = []
    for i, q in enumerate(qs):
        log.info("═══ Question %d / %d ═══", i+1, len(qs))
        try: results.append(agent.solve(q))
        except Exception as e:
            log.error("FAILED %s: %s", q.get("id"), e, exc_info=True)
            results.append({"id": q["id"], "ideal_answer": "Unable to generate.",
                            "documents": [], "snippets": []})

    # ── Phase B output (answers only) ──
    pb = {"questions": []}
    for r in results:
        e = {"id": r["id"], "ideal_answer": r.get("ideal_answer", "")}
        if "exact_answer" in r and r["exact_answer"] is not None:
            e["exact_answer"] = r["exact_answer"]
        pb["questions"].append(e)
    with open(a.output, "w") as f:
        json.dump(pb, f, indent=2, ensure_ascii=False)
    log.info("Phase B → %s", a.output)

    # ── Phase A+ output (docs + snippets + answers) ──
    pa = {"questions": []}
    for r in results:
        e = {"id": r["id"], "documents": r.get("documents", []),
             "snippets": r.get("snippets", []),
             "ideal_answer": r.get("ideal_answer", "")}
        if "exact_answer" in r and r["exact_answer"] is not None:
            e["exact_answer"] = r["exact_answer"]
        pa["questions"].append(e)
    oa = a.output.replace(".json", "_phaseA.json")
    with open(oa, "w") as f:
        json.dump(pa, f, indent=2, ensure_ascii=False)
    log.info("Phase A+ → %s", oa)

    print(f"\nDone! {len(results)} questions")
    print(f"Types: {dict(Counter(q.get('type','?') for q in qs))}")
    print(f"Phase B:  {a.output}")
    print(f"Phase A+: {oa}")

if __name__ == "__main__": main()
