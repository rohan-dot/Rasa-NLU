#!/usr/bin/env python3
"""
BioASQ Task 14 — Phase A+ Solver
==================================
Phase A+ = questions only (NO gold snippets provided)
Must return: documents, snippets, exact answers, AND ideal answers.

This script does EVERYTHING:
  1. Retrieve documents + snippets from PubMed (own indexing)
  2. Generate exact + ideal answers from retrieved evidence

FORMAT RULES (from BioASQ guidelines):
  - factoid exact_answer: list of up to 5 entity names, ordered by confidence
  - list exact_answer: flat list, max 100 entries, max 100 chars each
  - yesno exact_answer: "yes" or "no"
  - summary: no exact_answer
  - ideal_answer: max 200 words for all types
  - No synonyms for factoid/list (since BioASQ5)

SETUP:
    pip install requests sentence-transformers faiss-cpu numpy rank-bm25

RUN:
    python bioasq_phaseA.py \
        --test-input BioASQ-task14bPhaseA-testset1.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        --embed-device cpu \
        -o phaseA_submission.json
"""

import argparse, json, logging, re, sys, time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
import numpy as np, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Data ──
@dataclass
class Passage:
    text: str; pmid: str; doc_url: str; section: str = "abstract"
    offset_begin: int = 0; offset_end: int = 0; similarity_score: float = 0.0

# ── PubMed ──
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

class PubMed:
    def __init__(self, api_key=None):
        self.s = requests.Session(); self.api_key = api_key
        self._last = 0.0; self._int = 0.2 if api_key else 1.0; self._bo = 1.0

    def _wait(self):
        w = max(self._int, self._bo) - (time.time()-self._last)
        if w > 0: time.sleep(w)
        self._last = time.time()

    def _get(self, url, p, ret=3):
        for i in range(ret):
            self._wait()
            try:
                r = self.s.get(url, params=p, timeout=30)
                if r.status_code == 429:
                    self._bo = min(self._bo*2, 10); time.sleep(self._bo+i*2); continue
                r.raise_for_status(); self._bo = max(self._bo*0.8, self._int); return r
            except: time.sleep(2*(i+1))
        return None

    def search(self, q, n=30):
        p = {"db":"pubmed","term":q,"retmax":n,"retmode":"json"}
        if self.api_key: p["api_key"] = self.api_key
        r = self._get(ESEARCH, p)
        try: return r.json()["esearchresult"]["idlist"] if r else []
        except: return []

    def related(self, pmid, n=15):
        p = {"dbfrom":"pubmed","db":"pubmed","id":pmid,"cmd":"neighbor_score","retmode":"json"}
        if self.api_key: p["api_key"] = self.api_key
        r = self._get(ELINK, p)
        if not r: return []
        try:
            for ls in r.json().get("linksets",[{}])[0].get("linksetdbs",[]):
                if ls.get("linkname") == "pubmed_pubmed":
                    return [str(l["id"]) for l in ls.get("links",[])][:n]
        except: pass
        return []

    def fetch(self, pmids):
        if not pmids: return []
        all_a = []
        for i in range(0, len(pmids), 50):
            p = {"db":"pubmed","id":",".join(pmids[i:i+50]),"rettype":"xml","retmode":"xml"}
            if self.api_key: p["api_key"] = self.api_key
            r = self._get(EFETCH, p)
            if r: all_a.extend(self._px(r.text))
        return all_a

    def _px(self, xml):
        arts = []
        try:
            for el in ET.fromstring(xml).findall(".//PubmedArticle"):
                pid = el.find(".//PMID")
                pid = pid.text.strip() if pid is not None and pid.text else ""
                tit = el.find(".//ArticleTitle")
                tit = "".join(tit.itertext()).strip() if tit is not None else ""
                ab = " ".join(("".join(a.itertext()).strip()) for a in el.findall(".//AbstractText"))
                if pid and (tit or ab):
                    arts.append({"pmid":pid,"title":tit,"abstract":ab,
                                 "url":f"http://www.ncbi.nlm.nih.gov/pubmed/{pid}"})
        except: pass
        return arts

# ── Embedder + Hybrid Index ──
class Embedder:
    def __init__(self, model="pritamdeka/S-PubMedBert-MS-MARCO", device=None):
        import torch; from sentence_transformers import SentenceTransformer
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Embedding: %s (%s)", model, device)
        self.m = SentenceTransformer(model, device=device)
        self.dim = self.m.get_sentence_embedding_dimension()
    def encode(self, t): return np.array(self.m.encode(t, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
    def encode1(self, t): return self.encode([t])[0]

class Index:
    def __init__(self, emb):
        import faiss; self.faiss = faiss; self.emb = emb
        self.ix = faiss.IndexFlatIP(emb.dim); self.ps: list[Passage] = []; self._seen = set()
        self._bm = None; self._d = True

    def add_art(self, a):
        pmid, url = a["pmid"], a["url"]; new = []
        t = a.get("title","").strip()
        if t and t not in self._seen: self._seen.add(t); new.append(Passage(t, pmid, url, "title", 0, len(t)))
        ab = a.get("abstract","").strip()
        if ab:
            if ab not in self._seen and len(ab)>50:
                self._seen.add(ab); new.append(Passage(ab, pmid, url, "abstract", 0, len(ab)))
            ss = re.split(r'(?<=[.!?])\s+', ab)
            if len(ss) > 3:
                for i in range(len(ss)-2):
                    ch = " ".join(s.strip() for s in ss[i:i+3]).strip()
                    if len(ch)<50 or ch in self._seen: continue
                    b = ab.find(ss[i]); self._seen.add(ch)
                    new.append(Passage(ch, pmid, url, "abstract", max(b,0), max(b,0)+len(ch)))
        if not new: return
        self.ix.add(self.emb.encode([p.text for p in new])); self.ps.extend(new); self._d = True

    def search(self, q, k=15):
        if not self.ps: return []
        sem = []
        if self.ix.ntotal > 0:
            sc, ix = self.ix.search(self.emb.encode1(q).reshape(1,-1), min(k*3, self.ix.ntotal))
            sem = [(int(i),float(s)) for s,i in zip(sc[0],ix[0]) if 0<=i<len(self.ps)]
        if self._d:
            from rank_bm25 import BM25Okapi
            self._bm = BM25Okapi([re.findall(r'[a-z0-9\-]+', p.text.lower()) for p in self.ps]); self._d = False
        bm = []
        if self._bm:
            bsc = self._bm.get_scores(re.findall(r'[a-z0-9\-]+', q.lower()))
            bm = sorted(enumerate(bsc), key=lambda x: -x[1])[:k*3]; bm = [(i,s) for i,s in bm if s>0]
        rrf = {}
        for r,(i,_) in enumerate(sem): rrf[i] = rrf.get(i,0) + 0.6/(60+r+1)
        for r,(i,_) in enumerate(bm): rrf[i] = rrf.get(i,0) + 0.4/(60+r+1)
        res, sn = [], set()
        for i, sc in sorted(rrf.items(), key=lambda x: -x[1]):
            p = self.ps[i]; sg = p.text[:80].lower()
            if sg in sn: continue
            sn.add(sg); p.similarity_score = sc; res.append(p)
            if len(res)>=k: break
        return res

    @property
    def size(self): return len(self.ps)
    @property
    def n_art(self): return len({p.pmid for p in self.ps})

# ── LLM ──
class LLM:
    def __init__(self, url="http://localhost:8000", model="gemma-3-27b-it"):
        self.url = url.rstrip("/"); self.model = model; self.s = requests.Session()
        try:
            r = self.s.get(f"{self.url}/v1/models", timeout=10); r.raise_for_status()
            av = [m["id"] for m in r.json().get("data",[])]
            log.info("vLLM: %s", av)
            if av and self.model not in av: self.model = av[0]
        except Exception as e: log.error("vLLM: %s", e); sys.exit(1)

    def ask(self, p, mt=1024, t=0.3):
        for i in range(3):
            try:
                r = self.s.post(f"{self.url}/v1/chat/completions",
                    json={"model":self.model,"messages":[{"role":"user","content":p}],
                          "max_tokens":mt,"temperature":t,"top_p":0.95}, timeout=180)
                r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e: log.warning("LLM(%d): %s",i+1,e); time.sleep(3*(i+1))
        return ""

# ── Few-Shot ──
class FS:
    def __init__(self): self.ex = {"factoid":[],"list":[],"yesno":[],"summary":[]}
    def load(self, path, n=40):
        with open(path) as f: data = json.load(f)
        for q in data.get("questions",[]):
            qt = q.get("type","").lower()
            if qt not in self.ex: continue
            sn, ea, ia = q.get("snippets",[]), q.get("exact_answer"), q.get("ideal_answer")
            if not sn or not ia: continue
            if qt != "summary" and not ea: continue
            self.ex[qt].append({"body":q["body"],"snippets":[s.get("text","") for s in sn[:5]],
                "exact_answer":ea, "ideal_answer":ia if isinstance(ia,str) else ia[0] if isinstance(ia,list) and ia else ""})
        for qt in self.ex:
            self.ex[qt] = sorted(self.ex[qt], key=lambda x: len(x["body"]))[:n]
            log.info("  FS %s: %d", qt, len(self.ex[qt]))
    def get(self, qt, n=2):
        e = self.ex.get(qt,[]); s = [x for x in e if len(" ".join(x["snippets"]))<1500]; return (s or e)[:n]

# ── Prompts ──
def _sb(ts, mx=5000):
    b = ""
    for i,s in enumerate(ts,1):
        l = f"[{i}] {s.strip()}\n"
        if len(b)+len(l)>mx: break
        b += l
    return b.strip()

def _fmt(ea):
    if isinstance(ea, str): return ea
    if isinstance(ea, list):
        if ea and isinstance(ea[0], list): return "; ".join(", ".join(x) if isinstance(x,list) else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)

def pr_queries(q, prev=None):
    p = f"Generate 3 short PubMed queries (3-6 words, plain keywords) for: {q}\n\n"
    if prev: p += "Previous didn't work, use DIFFERENT terms:\n" + "\n".join(f"  - {x}" for x in prev[-3:]) + "\n\n"
    p += "Write ONLY 3 queries numbered. Nothing else.\n\n1."
    return p

def pr_eval(q, ts):
    return f"Can '{q}' be answered from these?\n\n" + "\n".join(f"[{i+1}] {t[:150]}" for i,t in enumerate(ts[:8])) + "\n\nSUFFICIENT or INSUFFICIENT?"

def pr_answer(q, qt, ts, fs):
    if qt == "yesno": return _pr_yn(q, ts, fs)
    if qt == "factoid": return _pr_fac(q, ts, fs)
    if qt == "list": return _pr_lst(q, ts, fs)
    return _pr_sum(q, ts, fs)

def _pr_yn(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\nINSTRUCTIONS:\n"
         "1. Find evidence supporting YES.\n2. Find evidence supporting NO.\n"
         "3. Choose the side with STRONGER evidence.\n"
         "4. If evidence shows PROBLEMS (toxicity, failure, lack of evidence), answer NO.\n"
         "5. If mixed or insufficient evidence, lean NO.\n"
         "6. 'Promising' or 'preclinical' does NOT mean YES.\n\n")
    for ex in fs[:2]:
        ea = ex["exact_answer"]; ea = ea[0] if isinstance(ea,list) and ea else ea
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\nEVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:"
    return p

def _pr_fac(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\nRULES:\n"
         "1. EXACT_ANSWER: 1-5 words MAX. Specific name/number/term.\n"
         "2. Use EXACT terminology from evidence.\n"
         "3. If no clear answer in evidence, write: unknown\n"
         "4. Prefer: drug names, gene names, disease names, numbers.\n"
         "5. Then write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n"
         "GOOD: 'transsphenoidal surgery', 'NF1', 'palivizumab'\n"
         "BAD: 'multiple causative factors', 'it involves several mechanisms'\n\n")
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nEXACT_ANSWER:"
    return p

def _pr_lst(q, ts, fs):
    p = ("You are an expert biomedical QA system.\n\nRULES:\n"
         "1. List EVERY relevant item. Be EXHAUSTIVE.\n"
         "2. Too many is better than too few.\n"
         "3. Each item: 1-5 words, specific term.\n"
         "4. Aim for 5-15+ items.\n"
         "5. Prefix each with '- '.\n"
         "6. After the list write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n")
    for ex in fs[:1]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],500)}\nEXACT_ANSWER:\n"
        ea = ex["exact_answer"]
        if isinstance(ea,list):
            for it in ea[:8]:
                if isinstance(it,list): p += f"- {it[0]}\n"
                else: p += f"- {it}\n"
        p += f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts,6000)}\nList EVERY relevant item:\n\nEXACT_ANSWER:\n"
    return p

def _pr_sum(q, ts, fs):
    p = "You are an expert biomedical QA system. Write a 3-6 sentence answer (max 200 words).\n\n"
    for ex in fs[:2]:
        p += f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],800)}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p += f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:"
    return p

def pr_verify(q, ts, a):
    return f"Check this answer. Fix errors or unsupported claims.\n\nQ: {q}\nEVIDENCE:\n{_sb(ts,3000)}\n\nCANDIDATE:\n{a}\n\nCORRECTED_ANSWER:"

# ── Parsers ──
def p_fac(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    e = parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is","The exact answer is","Answer:"]:
        if e.lower().startswith(pf.lower()): e = e[len(pf):].strip()
    ideal = parts[1].strip() if len(parts)==2 else ""
    return [e] if e else ["unknown"], _cap_ideal(ideal or e)

def p_yn(r):
    m = re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)', r, re.I)
    if m: raw = m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        raw = ""
        for l in reversed(r.strip().split("\n")):
            if "yes" in l.strip().lower()[:10] or "no" in l.strip().lower()[:10]: raw = l.strip().lower(); break
        if not raw: raw = r.strip().split("\n")[-1].lower()
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    ideal = parts[1].strip() if len(parts)==2 else ""
    if "yes" in raw[:10] and "no" not in raw[:5]: exact = "yes"
    elif "no" in raw[:10]: exact = "no"
    else:
        lo = r.lower()
        neg = len(re.findall(r'insufficient|toxicity|not effective|no evidence|failed|ineffective|not recommended|lack of', lo))
        pos = len(re.findall(r'effective|demonstrated|shown to|evidence supports|approved|recommended', lo))
        exact = "no" if neg >= pos else "yes"
    return exact, _cap_ideal(ideal or f"The answer is {exact}.")

def p_lst(r):
    parts = re.split(r'IDEAL_ANSWER\s*:', r, 1, re.I)
    lr = parts[0].strip(); ideal = parts[1].strip() if len(parts)==2 else ""
    items = []
    for l in lr.split("\n"):
        l = re.sub(r'^[\-\*•]\s*','', l.strip())
        l = re.sub(r'^\d+[\.\)]\s*','', l).strip().strip('"\'').rstrip(".")
        if l and len(l)>1 and len(l)<=100: items.append(l)
    return items[:100] or ["unknown"], _cap_ideal(ideal)

def p_sum(r):
    return _cap_ideal(re.sub(r'^(IDEAL_ANSWER\s*:?\s*)', '', r, flags=re.I).strip() or "No answer.")

def _cap_ideal(t):
    """Cap ideal answer at 200 words per BioASQ rules."""
    words = t.split()
    if len(words) > 200: return " ".join(words[:200])
    return t

# ── Consensus ──
def con_fac(c):
    flat = [x[0].lower().strip() for x in c if x]
    if not flat: return ["unknown"]
    best = Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip() == best: return x[:5]  # max 5 per rules
    return [best]

def con_yn(c): return Counter(c).most_common(1)[0][0]

def con_lst(c):
    items = {}
    for cand in c:
        for s in cand:
            k = s.lower().strip() if isinstance(s,str) else s[0].lower().strip()
            v = s if isinstance(s,str) else s[0]
            if k: items.setdefault(k, v)
    return list(items.values())[:100] or ["unknown"]

def con_ideal(c):
    if not c: return ""
    s = sorted(c, key=len); return s[len(s)//2]

# ── Helpers ──
def simple_kw(q):
    STOP = set("what is are the a an of in on to for with by from and or not how does do can which who whom where when why this that these those it its has have had been being please list describe common main types most often should".split())
    return " ".join(t for t in re.findall(r"[A-Za-z0-9\-']+", q) if t.lower() not in STOP and len(t)>2)[:80]

def parse_q(r):
    qs = []
    for l in r.strip().split("\n"):
        l = re.sub(r'^\d+[\.\)]\s*','',l.strip()).strip('"').strip()
        if not l or len(l)<5 or len(l)>80: continue
        if any(s in l.lower() for s in ["here are","queries","search","i would","let me","following","note:","aim","different"]): continue
        qs.append(l)
    return qs[:3]

# ── Agent ──
class Agent:
    def __init__(self, llm, pm, emb, fs, passes=3, iters=3, n_art=30):
        self.llm=llm; self.pm=pm; self.emb=emb; self.fs=fs
        self.passes=passes; self.iters=iters; self.n_art=n_art

    def solve(self, question):
        qid, body = question["id"], question["body"]
        qtype = question.get("type","summary").lower()
        log.info("━━━ %s [%s]: %.70s", qid, qtype, body)

        # ── RETRIEVAL ──
        idx = Index(self.emb); seen, qused = set(), []

        kw = simple_kw(body)
        if kw:
            qused.append(kw)
            pids = self.pm.search(kw, self.n_art)
            new = [p for p in pids if p not in seen]; seen.update(new)
            if new:
                for a in self.pm.fetch(new): idx.add_art(a)
            log.info("  Seed: %d passages", idx.size)

        for it in range(self.iters):
            qr = self.llm.ask(pr_queries(body, qused or None), mt=200, t=0.3)
            nq = parse_q(qr)
            if not nq:
                w = simple_kw(body).split()
                if len(w)>=3: nq = [" ".join(w[:3])]
            for q in nq:
                if q in qused: continue
                qused.append(q)
                pids = self.pm.search(q, self.n_art)
                new = [p for p in pids if p not in seen]; seen.update(new)
                if new:
                    for a in self.pm.fetch(new): idx.add_art(a)
                log.info("    '%s': +%d", q[:50], len(new))

            top = idx.search(body, k=10)
            # Expand from best
            if top:
                for p in top[:2]:
                    if p.pmid == "gold": continue
                    rel = self.pm.related(p.pmid, 10)
                    new = [r for r in rel if r not in seen]; seen.update(new)
                    if new:
                        for a in self.pm.fetch(new): idx.add_art(a)
                        log.info("    Related(%s): +%d", p.pmid, len(new))
                    break

            if idx.size >= 15:
                v = self.llm.ask(pr_eval(body, [p.text for p in top]), mt=20, t=0.1)
                if "SUFFICIENT" in v.upper(): log.info("    SUFFICIENT"); break

        log.info("  Index: %d articles, %d passages", idx.n_art, idx.size)

        # ── Rerank ──
        top = idx.search(body, k=20)

        # ── ANSWER GENERATION ──
        fsex = self.fs.get(qtype, 2)
        ptexts = [p.text for p in top[:15]] or [body]

        ec, ic = [], []
        for pi in range(self.passes):
            temp = 0.15 + pi*0.15
            log.info("  Pass %d/%d (t=%.2f)", pi+1, self.passes, temp)
            raw = self.llm.ask(pr_answer(body, qtype, ptexts, fsex), mt=768, t=temp)
            if qtype=="factoid": e,i = p_fac(raw); ec.append(e)
            elif qtype=="yesno": e,i = p_yn(raw); ec.append(e)
            elif qtype=="list": e,i = p_lst(raw); ec.append(e)
            else: i = p_sum(raw)
            ic.append(i)
            if pi==0:
                corr = self.llm.ask(pr_verify(body, ptexts, raw), mt=768, t=0.2)
                if qtype=="factoid": e2,i2=p_fac(corr); ec.append(e2); ic.append(i2)
                elif qtype=="yesno": e2,i2=p_yn(corr); ec.append(e2); ic.append(i2)
                elif qtype=="list": e2,i2=p_lst(corr); ec.append(e2); ic.append(i2)
                else: ic.append(p_sum(corr))

        # ── Build result ──
        result = {"id": qid}
        if qtype=="factoid": result["exact_answer"] = con_fac(ec)
        elif qtype=="yesno": result["exact_answer"] = con_yn(ec)
        elif qtype=="list": result["exact_answer"] = con_lst(ec)
        result["ideal_answer"] = con_ideal(ic)

        # Documents + snippets
        seen_u = set(); result["documents"] = []
        for p in top[:10]:
            if p.doc_url and p.doc_url not in seen_u:
                seen_u.add(p.doc_url); result["documents"].append(p.doc_url)
        result["snippets"] = [{"document":p.doc_url,"text":p.text,
            "offsetInBeginSection":p.offset_begin,"offsetInEndSection":p.offset_end,
            "beginSection":"sections.0","endSection":"sections.0"} for p in top[:10]]

        log.info("  ✓ exact=%s", str(result.get("exact_answer","N/A"))[:60])
        return result

# ── Main ──
def main():
    ap = argparse.ArgumentParser(description="BioASQ 14 Phase A+")
    ap.add_argument("--test-input","-t",required=True)
    ap.add_argument("--training","-tr",default=None)
    ap.add_argument("--output","-o",default="phaseA_submission.json")
    ap.add_argument("--vllm-url",default="http://localhost:8000")
    ap.add_argument("--model","-m",default="gemma-3-27b-it")
    ap.add_argument("--embedding-model",default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--embed-device",default=None)
    ap.add_argument("--passes",type=int,default=3)
    ap.add_argument("--retrieval-iterations",type=int,default=3)
    ap.add_argument("--articles-per-query",type=int,default=30)
    ap.add_argument("--api-key",default=None)
    a = ap.parse_args()

    fs = FS()
    if a.training: fs.load(a.training)
    with open(a.test_input) as f: qs = json.load(f).get("questions",[])
    log.info("Loaded %d questions", len(qs))

    llm = LLM(a.vllm_url, a.model)
    pm = PubMed(a.api_key)
    emb = Embedder(a.embedding_model, a.embed_device)
    agent = Agent(llm, pm, emb, fs, a.passes, a.retrieval_iterations, a.articles_per_query)

    results = []
    for i,q in enumerate(qs):
        log.info("═══ %d / %d ═══", i+1, len(qs))
        try: results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s", q.get("id"), e)
            results.append({"id":q["id"],"ideal_answer":"Unable to generate.",
                            "documents":[],"snippets":[]})

    for r in results:
        if r.get("exact_answer") is None: r.pop("exact_answer", None)

    with open(a.output, "w") as f:
        json.dump({"questions": results}, f, indent=2, ensure_ascii=False)
    print(f"\nDone! {len(results)} questions → {a.output}")
    print(f"Types: {dict(Counter(q.get('type','?') for q in qs))}")
    print("Submit to BioASQ Phase A+")

if __name__ == "__main__": main()
