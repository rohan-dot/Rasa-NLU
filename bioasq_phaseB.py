#!/usr/bin/env python3
"""
BioASQ Task 14 — Phase B Solver
=================================
Phase B = questions + gold documents + gold snippets provided.
Must return: exact answers + ideal answers.
Gold snippets are PRIMARY evidence (expert-curated).

FORMAT RULES:
  - factoid: list of up to 5 entity names, no synonyms
  - list: flat list, max 100 entries, max 100 chars each, no synonyms
  - yesno: "yes" or "no"
  - summary: no exact_answer
  - ideal_answer: max 200 words

RUN:
    python bioasq_phaseB.py \
        --test-input BioASQ-task14bPhaseB-testset1.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        -o phaseB_submission.json
"""

import argparse, json, logging, re, sys, time
import xml.etree.ElementTree as ET
from collections import Counter
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── PubMed (backup only) ──
class PubMed:
    def __init__(self, api_key=None):
        self.s = requests.Session(); self.api_key = api_key
        self._last = 0.0; self._int = 0.2 if api_key else 1.0; self._bo = 1.0
    def _wait(self):
        w = max(self._int, self._bo)-(time.time()-self._last)
        if w>0: time.sleep(w)
        self._last = time.time()
    def search_fetch(self, q, n=15):
        self._wait()
        p = {"db":"pubmed","term":q,"retmax":n,"retmode":"json"}
        if self.api_key: p["api_key"]=self.api_key
        try:
            r = self.s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=p, timeout=30)
            if r.status_code==429: time.sleep(5); return []
            pmids = r.json().get("esearchresult",{}).get("idlist",[])
        except: return []
        if not pmids: return []
        self._wait()
        p2 = {"db":"pubmed","id":",".join(pmids),"rettype":"xml","retmode":"xml"}
        if self.api_key: p2["api_key"]=self.api_key
        try:
            r2 = self.s.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=p2, timeout=60)
            texts = []
            for el in ET.fromstring(r2.text).findall(".//PubmedArticle"):
                ab = " ".join("".join(a.itertext()).strip() for a in el.findall(".//AbstractText"))
                if ab: texts.append(ab)
            return texts
        except: return []

# ── LLM ──
class LLM:
    def __init__(self, url="http://localhost:8000", model="gemma-3-27b-it"):
        self.url=url.rstrip("/"); self.model=model; self.s=requests.Session()
        try:
            r = self.s.get(f"{self.url}/v1/models", timeout=10); r.raise_for_status()
            av = [m["id"] for m in r.json().get("data",[])]
            log.info("vLLM: %s", av)
            if av and self.model not in av: self.model = av[0]
        except Exception as e: log.error("vLLM: %s",e); sys.exit(1)

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
    def __init__(self): self.ex={"factoid":[],"list":[],"yesno":[],"summary":[]}
    def load(self, path, n=40):
        with open(path) as f: data = json.load(f)
        for q in data.get("questions",[]):
            qt=q.get("type","").lower()
            if qt not in self.ex: continue
            sn,ea,ia = q.get("snippets",[]),q.get("exact_answer"),q.get("ideal_answer")
            if not sn or not ia: continue
            if qt!="summary" and not ea: continue
            self.ex[qt].append({"body":q["body"],"snippets":[s.get("text","") for s in sn[:5]],
                "exact_answer":ea,"ideal_answer":ia if isinstance(ia,str) else ia[0] if isinstance(ia,list) and ia else ""})
        for qt in self.ex:
            self.ex[qt]=sorted(self.ex[qt],key=lambda x:len(x["body"]))[:n]
            log.info("  FS %s: %d",qt,len(self.ex[qt]))
    def get(self,qt,n=2):
        e=self.ex.get(qt,[]); s=[x for x in e if len(" ".join(x["snippets"]))<1500]; return (s or e)[:n]

# ── Prompts ──
def _sb(ts, mx=5000):
    b=""
    for i,s in enumerate(ts,1):
        l=f"[{i}] {s.strip()}\n"
        if len(b)+len(l)>mx: break
        b+=l
    return b.strip()

def _fmt(ea):
    if isinstance(ea,str): return ea
    if isinstance(ea,list):
        if ea and isinstance(ea[0],list): return "; ".join(", ".join(x) if isinstance(x,list) else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)

def pr_answer(q, qt, ts, fs):
    if qt=="yesno": return _pr_yn(q,ts,fs)
    if qt=="factoid": return _pr_fac(q,ts,fs)
    if qt=="list": return _pr_lst(q,ts,fs)
    return _pr_sum(q,ts,fs)

def _pr_yn(q,ts,fs):
    p=("You are an expert biomedical QA system.\n\nINSTRUCTIONS:\n"
       "1. Find evidence supporting YES.\n2. Find evidence supporting NO.\n"
       "3. Choose the side with STRONGER evidence.\n"
       "4. If evidence shows PROBLEMS (toxicity, failure, lack of evidence), answer NO.\n"
       "5. If mixed or insufficient, lean NO.\n"
       "6. 'Promising' or 'preclinical' does NOT mean YES.\n\n")
    for ex in fs[:2]:
        ea=ex["exact_answer"]; ea=ea[0] if isinstance(ea,list) and ea else ea
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\nEVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:"
    return p

def _pr_fac(q,ts,fs):
    p=("You are an expert biomedical QA system.\n\nRULES:\n"
       "1. EXACT_ANSWER: 1-5 words MAX. Specific name/number/term.\n"
       "2. Use EXACT terminology from evidence.\n"
       "3. If no clear answer, write: unknown\n"
       "4. Prefer: drug names, gene names, disease names, numbers.\n"
       "5. Then write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n"
       "GOOD: 'transsphenoidal surgery', 'NF1', 'palivizumab'\nBAD: 'multiple factors', 'it involves mechanisms'\n\n")
    for ex in fs[:2]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nEXACT_ANSWER:"
    return p

def _pr_lst(q,ts,fs):
    p=("You are an expert biomedical QA system.\n\nRULES:\n"
       "1. List EVERY relevant item. Be EXHAUSTIVE.\n"
       "2. Too many better than too few. Aim 5-15+ items.\n"
       "3. Each item: 1-5 words, specific term.\n"
       "4. Prefix each with '- '.\n"
       "5. After list write IDEAL_ANSWER: 2-4 sentences (max 200 words).\n\n")
    for ex in fs[:1]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],500)}\nEXACT_ANSWER:\n"
        ea=ex["exact_answer"]
        if isinstance(ea,list):
            for it in ea[:8]:
                if isinstance(it,list): p+=f"- {it[0]}\n"
                else: p+=f"- {it}\n"
        p+=f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts,6000)}\nList EVERY relevant item:\n\nEXACT_ANSWER:\n"
    return p

def _pr_sum(q,ts,fs):
    p="You are an expert biomedical QA system. Write a 3-6 sentence answer (max 200 words).\n\n"
    for ex in fs[:2]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],800)}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:"
    return p

def pr_verify(q,ts,a):
    return f"Check this answer. Fix errors.\n\nQ: {q}\nEVIDENCE:\n{_sb(ts,3000)}\n\nCANDIDATE:\n{a}\n\nCORRECTED_ANSWER:"

# ── Parsers ──
def _cap(t):
    w=t.split(); return " ".join(w[:200]) if len(w)>200 else t

def p_fac(r):
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I)
    e=parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is","The exact answer is","Answer:"]:
        if e.lower().startswith(pf.lower()): e=e[len(pf):].strip()
    ideal=parts[1].strip() if len(parts)==2 else ""
    return [e] if e else ["unknown"], _cap(ideal or e)

def p_yn(r):
    m=re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)',r,re.I)
    if m: raw=m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        raw=""
        for l in reversed(r.strip().split("\n")):
            if "yes" in l.strip().lower()[:10] or "no" in l.strip().lower()[:10]: raw=l.strip().lower(); break
        if not raw: raw=r.strip().split("\n")[-1].lower()
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I)
    ideal=parts[1].strip() if len(parts)==2 else ""
    if "yes" in raw[:10] and "no" not in raw[:5]: exact="yes"
    elif "no" in raw[:10]: exact="no"
    else:
        lo=r.lower()
        neg=len(re.findall(r'insufficient|toxicity|not effective|no evidence|failed|ineffective|not recommended|lack of',lo))
        pos=len(re.findall(r'effective|demonstrated|shown to|evidence supports|approved|recommended',lo))
        exact="no" if neg>=pos else "yes"
    return exact, _cap(ideal or f"The answer is {exact}.")

def p_lst(r):
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I)
    lr=parts[0].strip(); ideal=parts[1].strip() if len(parts)==2 else ""
    items=[]
    for l in lr.split("\n"):
        l=re.sub(r'^[\-\*•]\s*','',l.strip())
        l=re.sub(r'^\d+[\.\)]\s*','',l).strip().strip('"\'').rstrip(".")
        if l and len(l)>1 and len(l)<=100: items.append(l)
    return items[:100] or ["unknown"], _cap(ideal)

def p_sum(r):
    return _cap(re.sub(r'^(IDEAL_ANSWER\s*:?\s*)','',r,flags=re.I).strip() or "No answer.")

# ── Consensus ──
def con_fac(c):
    flat=[x[0].lower().strip() for x in c if x]
    if not flat: return ["unknown"]
    best=Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip()==best: return x[:5]
    return [best]
def con_yn(c): return Counter(c).most_common(1)[0][0]
def con_lst(c):
    items={}
    for cand in c:
        for s in cand:
            k=s.lower().strip() if isinstance(s,str) else s
            if k: items.setdefault(k,s)
    return list(items.values())[:100] or ["unknown"]
def con_ideal(c):
    if not c: return ""
    s=sorted(c,key=len); return s[len(s)//2]

# ── Agent ──
class Agent:
    def __init__(self, llm, pm, fs, passes=3):
        self.llm=llm; self.pm=pm; self.fs=fs; self.passes=passes

    def solve(self, q):
        qid,body = q["id"],q["body"]
        qtype = q.get("type","summary").lower()
        log.info("━━━ %s [%s]: %.70s", qid, qtype, body)

        # ── Gold snippets = primary evidence ──
        ev = []
        for gs in q.get("snippets",[]):
            t = gs.get("text","").strip()
            if t: ev.append(t)
        log.info("  Gold snippets: %d", len(ev))

        # ── Backup from PubMed if gold is thin ──
        if len(ev) < 3:
            log.info("  Gold thin — fetching backup from PubMed")
            STOP=set("what is are the a an of in on to for with by from and or not how does do can which".split())
            kw=" ".join(t for t in re.findall(r"[A-Za-z0-9\-']+",body) if t.lower() not in STOP and len(t)>2)[:80]
            if kw:
                extras = self.pm.search_fetch(kw, 15)
                qtoks = set(t.lower() for t in re.findall(r"[A-Za-z0-9\-']+",body) if len(t)>2)
                for ab in extras:
                    for sent in re.split(r'(?<=[.!?])\s+', ab):
                        if len(sent)<40: continue
                        stoks = set(t.lower() for t in re.findall(r"[A-Za-z0-9\-']+",sent))
                        if len(qtoks&stoks)/max(len(qtoks),1) > 0.15: ev.append(sent)
                log.info("  After backup: %d snippets", len(ev))
        if not ev: ev = [body]

        # ── Multi-pass answer ──
        fsex = self.fs.get(qtype, 2)
        ec, ic = [], []
        for pi in range(self.passes):
            temp = 0.15 + pi*0.15
            log.info("  Pass %d/%d (t=%.2f)", pi+1, self.passes, temp)
            raw = self.llm.ask(pr_answer(body, qtype, ev, fsex), mt=768, t=temp)
            if qtype=="factoid": e,i=p_fac(raw); ec.append(e)
            elif qtype=="yesno": e,i=p_yn(raw); ec.append(e)
            elif qtype=="list": e,i=p_lst(raw); ec.append(e)
            else: i=p_sum(raw)
            ic.append(i)
            if pi==0:
                corr=self.llm.ask(pr_verify(body,ev,raw),mt=768,t=0.2)
                if qtype=="factoid": e2,i2=p_fac(corr); ec.append(e2); ic.append(i2)
                elif qtype=="yesno": e2,i2=p_yn(corr); ec.append(e2); ic.append(i2)
                elif qtype=="list": e2,i2=p_lst(corr); ec.append(e2); ic.append(i2)
                else: ic.append(p_sum(corr))

        result = {"id": qid, "ideal_answer": con_ideal(ic)}
        if qtype=="factoid": result["exact_answer"]=con_fac(ec)
        elif qtype=="yesno": result["exact_answer"]=con_yn(ec)
        elif qtype=="list": result["exact_answer"]=con_lst(ec)
        log.info("  ✓ exact=%s", str(result.get("exact_answer","N/A"))[:60])
        return result

# ── Main ──
def main():
    ap=argparse.ArgumentParser(description="BioASQ 14 Phase B")
    ap.add_argument("--test-input","-t",required=True)
    ap.add_argument("--training","-tr",default=None)
    ap.add_argument("--output","-o",default="phaseB_submission.json")
    ap.add_argument("--vllm-url",default="http://localhost:8000")
    ap.add_argument("--model","-m",default="gemma-3-27b-it")
    ap.add_argument("--passes",type=int,default=3)
    ap.add_argument("--api-key",default=None)
    a=ap.parse_args()

    fs=FS()
    if a.training: fs.load(a.training)
    with open(a.test_input) as f: qs=json.load(f).get("questions",[])
    log.info("Loaded %d questions",len(qs))

    llm=LLM(a.vllm_url,a.model); pm=PubMed(a.api_key)
    agent=Agent(llm,pm,fs,a.passes)

    results=[]
    for i,q in enumerate(qs):
        log.info("═══ %d / %d ═══",i+1,len(qs))
        try: results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s",q.get("id"),e)
            results.append({"id":q["id"],"ideal_answer":"Unable to generate."})

    for r in results:
        if r.get("exact_answer") is None: r.pop("exact_answer",None)

    with open(a.output,"w") as f:
        json.dump({"questions":results},f,indent=2,ensure_ascii=False)
    print(f"\nDone! {len(results)} questions → {a.output}")
    print("Submit to BioASQ Phase B")

if __name__=="__main__": main()
