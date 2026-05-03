#!/usr/bin/env python3
"""
BioASQ — Agentic RAG with YOUR SQLite PubMed DB
=================================================
Optimized for slow storage: 2-3 FTS5 queries per question instead of 10+.
Builds local FAISS+BM25 index from broad fetch, then searches instantly.

RUN:
    python bioasq_sqlite_agent.py \
        --test-input 13B1_golden.json \
        --training training13b.json \
        --db /scratch/ro31337/pubmed_index.db \
        --model gemma-4-31b-it \
        --embed-device cpu \
        -o results.json
"""
import argparse,json,logging,re,sys,time,sqlite3
from collections import Counter
from dataclasses import dataclass
import numpy as np,requests

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)

@dataclass
class Passage:
    text:str;pmid:str;doc_url:str;section:str="abstract"
    offset_begin:int=0;offset_end:int=0;score:float=0.0;source:str=""

@dataclass
class Step:
    kind:str;content:str;tool:str="";n:int=0

# ═══ SQLITE — BROAD CACHED SEARCH ═══
class DB:
    def __init__(self,path):
        self.path=path;self._cache={}
        conn=sqlite3.connect(path)
        self.count=conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        log.info("SQLite: %s (%d articles)",path,self.count)

    def fetch(self,query,n=50):
        key=query.lower().strip()
        if key in self._cache:
            log.info("    [cache] '%s'",query[:40]);return self._cache[key]
        conn=sqlite3.connect(self.path);conn.row_factory=sqlite3.Row
        terms=[t for t in re.findall(r'[A-Za-z0-9\-]+',query) if len(t)>2][:8]
        if not terms:conn.close();return []
        fts=" OR ".join(f'"{t}"' for t in terms)
        try:
            log.info("    [FTS5] '%s'...",query[:50])
            t0=time.time()
            rows=conn.execute("""
                SELECT a.pmid,a.title,a.abstract,a.mesh_terms,articles_fts.rank AS score
                FROM articles_fts JOIN articles a ON a.pmid=articles_fts.pmid
                WHERE articles_fts MATCH ? AND length(a.abstract)>50
                ORDER BY articles_fts.rank LIMIT ?
            """,(fts,n)).fetchall()
            log.info("    [FTS5] %d articles in %.1fs",len(rows),time.time()-t0)
            results=[{"pmid":r["pmid"],"title":r["title"] or "","abstract":r["abstract"] or "",
                      "mesh":r["mesh_terms"] or "","url":f"http://www.ncbi.nlm.nih.gov/pubmed/{r['pmid']}"} for r in rows]
            self._cache[key]=results;return results
        except sqlite3.OperationalError as e:
            log.warning("    FTS5 error: %s",e);return []
        finally:conn.close()

    def expand_mesh(self,pmid,n=10):
        conn=sqlite3.connect(self.path);conn.row_factory=sqlite3.Row
        src=conn.execute("SELECT mesh_terms FROM articles WHERE pmid=?",(pmid,)).fetchone()
        if not src or not src["mesh_terms"]:conn.close();return []
        terms=[t.strip() for t in src["mesh_terms"].split(";") if len(t.strip())>3][:4]
        if not terms:conn.close();return []
        fts=" OR ".join(f'"{t}"' for t in terms)
        try:
            rows=conn.execute("""
                SELECT a.pmid,a.title,a.abstract FROM articles_fts
                JOIN articles a ON a.pmid=articles_fts.pmid
                WHERE articles_fts MATCH ? AND a.pmid!=? AND length(a.abstract)>50
                ORDER BY articles_fts.rank LIMIT ?
            """,(fts,pmid,n)).fetchall()
            return [{"pmid":r["pmid"],"title":r["title"] or "","abstract":r["abstract"] or "",
                     "url":f"http://www.ncbi.nlm.nih.gov/pubmed/{r['pmid']}"} for r in rows]
        except:return []
        finally:conn.close()

# ═══ EMBEDDER + LOCAL INDEX ═══
class Embedder:
    def __init__(self,model="pritamdeka/S-PubMedBert-MS-MARCO",device=None):
        import torch;from sentence_transformers import SentenceTransformer
        if device is None:device="cuda" if torch.cuda.is_available() else "cpu"
        log.info("Embedder: %s (%s)",model,device)
        self.m=SentenceTransformer(model,device=device)
        self.dim=self.m.get_sentence_embedding_dimension()
    def encode(self,t):return np.array(self.m.encode(t,show_progress_bar=False,normalize_embeddings=True),dtype=np.float32)
    def encode1(self,t):return self.encode([t])[0]

class Reranker:
    def __init__(self,model="cross-encoder/ms-marco-MiniLM-L-12-v2",device="cpu"):
        from sentence_transformers import CrossEncoder
        log.info("Reranker: %s (%s)",model,device)
        self.m=CrossEncoder(model,device=device)
    def rerank(self,q,ps,k=15):
        if not ps:return []
        scores=self.m.predict([[q,p.text[:512]] for p in ps])
        for p,s in zip(ps,scores):p.score=float(s)
        return sorted(ps,key=lambda p:p.score,reverse=True)[:k]

class LocalIndex:
    def __init__(self,emb):
        import faiss;self.faiss=faiss;self.emb=emb
        self.ix=faiss.IndexFlatIP(emb.dim);self.ps=[];self._seen=set()
        self._bm=None;self._d=True

    def add(self,art):
        pmid=art["pmid"];url=art["url"];new=[]
        t=art.get("title","").strip()
        if t and t not in self._seen:
            self._seen.add(t);new.append(Passage(t,pmid,url,"title",0,len(t)))
        ab=art.get("abstract","").strip()
        if ab:
            if ab not in self._seen and len(ab)>50:
                self._seen.add(ab);new.append(Passage(ab,pmid,url,"abstract",0,len(ab)))
            ss=re.split(r'(?<=[.!?])\s+',ab)
            if len(ss)>3:
                for i in range(len(ss)-2):
                    ch=" ".join(s.strip() for s in ss[i:i+3]).strip()
                    if len(ch)<50 or ch in self._seen:continue
                    b=ab.find(ss[i]);self._seen.add(ch)
                    new.append(Passage(ch,pmid,url,"abstract",max(b,0),max(b,0)+len(ch)))
        if not new:return
        self.ix.add(self.emb.encode([p.text for p in new]))
        self.ps.extend(new);self._d=True

    def search(self,q,k=15):
        if not self.ps:return []
        sem=[]
        if self.ix.ntotal>0:
            sc,ix=self.ix.search(self.emb.encode1(q).reshape(1,-1),min(k*3,self.ix.ntotal))
            sem=[(int(i),float(s)) for s,i in zip(sc[0],ix[0]) if 0<=i<len(self.ps)]
        if self._d:
            from rank_bm25 import BM25Okapi
            self._bm=BM25Okapi([re.findall(r'[a-z0-9\-]+',p.text.lower()) for p in self.ps])
            self._d=False
        bm=[]
        if self._bm:
            bsc=self._bm.get_scores(re.findall(r'[a-z0-9\-]+',q.lower()))
            bm=sorted(enumerate(bsc),key=lambda x:-x[1])[:k*3]
            bm=[(i,s) for i,s in bm if s>0]
        rrf={}
        for r,(i,_) in enumerate(sem):rrf[i]=rrf.get(i,0)+0.6/(60+r+1)
        for r,(i,_) in enumerate(bm):rrf[i]=rrf.get(i,0)+0.4/(60+r+1)
        res=[];sn=set()
        for i,sc in sorted(rrf.items(),key=lambda x:-x[1]):
            p=self.ps[i];sg=p.text[:80].lower()
            if sg in sn:continue
            sn.add(sg);p.score=sc;res.append(p)
            if len(res)>=k:break
        return res

    @property
    def size(self):return len(self.ps)
    @property
    def n_art(self):return len({p.pmid for p in self.ps})

# ═══ LLM ═══
class LLM:
    def __init__(self,url="http://localhost:8000",model="gemma-4-31b-it"):
        self.url=url.rstrip("/");self.model=model;self.s=requests.Session()
        try:
            r=self.s.get(f"{self.url}/v1/models",timeout=10);r.raise_for_status()
            av=[m["id"] for m in r.json().get("data",[])]
            log.info("vLLM: %s",av)
            if av and self.model not in av:self.model=av[0]
        except Exception as e:log.error("vLLM: %s",e);sys.exit(1)
    def ask(self,p,mt=1024,t=0.3):
        for i in range(3):
            try:
                r=self.s.post(f"{self.url}/v1/chat/completions",json={"model":self.model,
                    "messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":t,"top_p":0.95},timeout=180)
                r.raise_for_status();return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:log.warning("LLM(%d):%s",i+1,e);time.sleep(3*(i+1))
        return ""

# ═══ FEW-SHOT ═══
class FS:
    def __init__(self):self.ex={"factoid":[],"list":[],"yesno":[],"summary":[]}
    def load(self,path,n=40):
        with open(path) as f:data=json.load(f)
        for q in data.get("questions",[]):
            qt=q.get("type","").lower()
            if qt not in self.ex:continue
            sn=q.get("snippets",[]);ea=q.get("exact_answer");ia=q.get("ideal_answer")
            if not sn or not ia:continue
            if qt!="summary" and not ea:continue
            self.ex[qt].append({"body":q["body"],"snippets":[s.get("text","") for s in sn[:5]],
                "exact_answer":ea,"ideal_answer":ia if isinstance(ia,str) else ia[0] if isinstance(ia,list) and ia else ""})
        for qt in self.ex:self.ex[qt]=sorted(self.ex[qt],key=lambda x:len(x["body"]))[:n];log.info("  FS %s: %d",qt,len(self.ex[qt]))
    def get(self,qt,n=2):
        e=self.ex.get(qt,[]);s=[x for x in e if len(" ".join(x["snippets"]))<1500];return(s or e)[:n]

# ═══ PROMPTS ═══
def _sb(ts,mx=5000):
    b=""
    for i,s in enumerate(ts,1):
        l=f"[{i}] {s.strip()}\n"
        if len(b)+len(l)>mx:break
        b+=l
    return b.strip()
def _fmt(ea):
    if isinstance(ea,str):return ea
    if isinstance(ea,list):
        if ea and isinstance(ea[0],list):return "; ".join(", ".join(x) if isinstance(x,list) else str(x) for x in ea)
        return ", ".join(str(x) for x in ea)
    return str(ea)
def _cap(t):w=t.split();return " ".join(w[:200]) if len(w)>200 else t

def pr_decompose(q):return f"Break this biomedical question into 2-4 sub-questions. If simple, return as-is.\n\nQ: {q}\n\nSUB-QUESTIONS:\n"
def pr_queries(q,prev=None):
    p=f"Generate 3 short search queries (3-6 words) for: {q}\n\n"
    if prev:p+="Previous:\n"+"\n".join(f"  - {x}" for x in prev[-3:])+"\nUse DIFFERENT terms.\n\n"
    p+="Write ONLY 3 queries.\n\n1.";return p
def pr_eval(q,ts):return f"Can '{q}' be answered?\n\n"+"\n".join(f"[{i+1}] {t[:150]}" for i,t in enumerate(ts[:8]))+"\n\nSUFFICIENT or INSUFFICIENT?"
def pr_answer(q,qt,ts,fs):
    if qt=="yesno":return _yn(q,ts,fs)
    if qt=="factoid":return _fac(q,ts,fs)
    if qt=="list":return _lst(q,ts,fs)
    return _sum(q,ts,fs)

def _yn(q,ts,fs):
    p=("Expert biomedical QA.\n\n1. Find evidence for YES.\n2. Find evidence for NO.\n"
       "3. Stronger side wins.\n4. Problems/mixed/insufficient → NO.\n5. 'Promising' ≠ YES.\n\n")
    for ex in fs[:2]:
        ea=ex["exact_answer"];ea=ea[0] if isinstance(ea,list) and ea else ea
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\nEVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:";return p

def _fac(q,ts,fs):
    p=("Expert biomedical QA.\n\nRULES:\n"
       "1. EXACT_ANSWER: 1-5 words MAX.\n"
       "2. COPY the exact term from evidence — do NOT paraphrase.\n"
       "3. '3-8%' not 'approximately 5%'. 'POLR2M' not 'RNA polymerase subunit'.\n"
       "4. If unclear: unknown\n5. Then IDEAL_ANSWER: 2-4 sentences.\n\n")
    for ex in fs[:2]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nCopy exact term:\nEXACT_ANSWER:";return p

def _lst(q,ts,fs):
    p=("Expert biomedical QA.\n\nRULES:\n"
       "1. Go through EACH passage ONE BY ONE.\n"
       "2. Extract EVERY relevant item. Missing items = lost points.\n"
       "3. Aim 10-20+ items. Each: 1-5 words, prefix '- '.\n"
       "4. List each drug/gene/disease SEPARATELY.\n"
       "5. After list: IDEAL_ANSWER: 2-4 sentences.\n\n")
    for ex in fs[:1]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],500)}\nEXACT_ANSWER:\n"
        ea=ex["exact_answer"]
        if isinstance(ea,list):
            for it in ea[:12]:
                if isinstance(it,list):p+=f"- {it[0]}\n"
                else:p+=f"- {it}\n"
        p+=f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts,6000)}\nGo through each passage:\n\nEXACT_ANSWER:\n";return p

def _sum(q,ts,fs):
    p="Expert biomedical QA. Write 3-6 sentences (max 200 words).\n\n"
    for ex in fs[:2]:p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],800)}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:";return p

def pr_verify(q,ts,a):return f"Check answer. Fix errors.\n\nQ: {q}\nEVIDENCE:\n{_sb(ts,3000)}\n\nCANDIDATE:\n{a}\n\nCORRECTED_ANSWER:"
def pr_synth(q,subs):return f"Combine:\n\nORIGINAL: {q}\n\n"+"\n".join(f"Sub-Q{i+1}: {s['question']}\nAnswer: {s['answer']}\n" for i,s in enumerate(subs))+"\nCOMBINED:"

# ═══ PARSERS ═══
def p_fac(r):
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I);e=parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is","The exact answer is","Answer:","EXACT_ANSWER:","Copy exact term:"]:
        if e.lower().startswith(pf.lower()):e=e[len(pf):].strip()
    if "\n" in e:e=e.split("\n")[0].strip()
    return [e.strip()] if e.strip() else ["unknown"],_cap(parts[1].strip() if len(parts)==2 else e)

def p_yn(r):
    m=re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)',r,re.I)
    raw=m.group(1).strip().lower().strip('"\'').rstrip(".") if m else ""
    if not raw:
        for l in reversed(r.strip().split("\n")):
            ll=l.strip().lower()
            if ll.startswith("yes") or ll.startswith("no"):raw=ll;break
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I);ideal=parts[1].strip() if len(parts)==2 else ""
    if raw.startswith("yes"):exact="yes"
    elif raw.startswith("no"):exact="no"
    else:
        lo=r.lower()
        neg=len(re.findall(r'insufficient|toxicity|not effective|no evidence|failed|ineffective|not recommended|lack of|limited evidence',lo))
        pos=len(re.findall(r'effective|demonstrated efficacy|shown to be|evidence supports|fda.approved|recommended|clinically proven',lo))
        exact="no" if neg>=pos else "yes"
    return exact,_cap(ideal or f"The answer is {exact}.")

def p_lst(r):
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I);lr=parts[0].strip();ideal=parts[1].strip() if len(parts)==2 else ""
    items=[]
    for l in lr.split("\n"):
        l=re.sub(r'^[\-\*•]\s*','',l.strip());l=re.sub(r'^\d+[\.\)]\s*','',l).strip().strip('"\'').rstrip(".")
        if not l or len(l)<=1 or len(l)>100:continue
        if any(s in l.upper() for s in ["CORRECTED","EXACT_ANSWER","IDEAL_ANSWER","EVIDENCE","QUESTION","NOTE:","MISSED"]):continue
        items.append(l)
    return items[:100] or ["unknown"],_cap(ideal)

def p_sum(r):return _cap(re.sub(r'^(IDEAL_ANSWER\s*:?\s*)','',r,flags=re.I).strip() or "No answer.")

# ═══ CONSENSUS ═══
def con_fac(c):
    flat=[x[0].lower().strip() for x in c if x and x[0].strip()]
    if not flat:return ["unknown"]
    best=Counter(flat).most_common(1)[0][0]
    for x in c:
        if x and x[0].lower().strip()==best:return x[:5]
    return [best]
def con_yn(c):return Counter(c).most_common(1)[0][0] if c else "no"
def con_lst(c):
    items={}
    for cand in c:
        for s in cand:
            k=s.lower().strip() if isinstance(s,str) else s
            if k and k!="unknown":items.setdefault(k,s)
    return list(items.values())[:100] or ["unknown"]
def con_ideal(c):
    c=[x for x in c if x and len(x)>10]
    if not c:return "No sufficient evidence."
    s=sorted(c,key=len);return s[len(s)//2]

# ═══ HELPERS ═══
def kw(q):
    STOP=set("what is are the a an of in on to for with by from and or not how does do can which who whom where when why this that these those it its has have had been being please list describe common main types most often should typically".split())
    return " ".join(t for t in re.findall(r"[A-Za-z0-9\-']+",q) if t.lower() not in STOP and len(t)>2)[:80]
def parse_qs(r):
    qs=[]
    for l in r.strip().split("\n"):
        l=re.sub(r'^\d+[\.\)]\s*','',l.strip()).strip('"').strip()
        if not l or len(l)<5 or len(l)>80:continue
        if any(s in l.lower() for s in ["here are","queries","search","i would","let me","following","note:","aim","different","these"]):continue
        qs.append(l)
    return qs[:3]
def parse_subqs(r):
    subs=[]
    for l in r.strip().split("\n"):
        l=re.sub(r'^\d+[\.\)]\s*','',l.strip()).strip()
        if l and len(l)>10 and len(l)<200 and "?" in l:subs.append(l)
    return subs[:4]

# ═══ AGENT ═══
class Agent:
    def __init__(self,llm,db,emb,reranker,fs,passes=3):
        self.llm=llm;self.db=db;self.emb=emb;self.reranker=reranker;self.fs=fs;self.passes=passes

    def solve(self,question):
        qid=question["id"];body=question["body"];qtype=question.get("type","summary").lower()
        trace=[]
        log.info("━"*60);log.info("  %s [%s]: %s",qid,qtype,body)

        # THINK + DECOMPOSE
        trace.append(Step("THINK",f"Analyzing: {body}"))
        decomp=self.llm.ask(pr_decompose(body),mt=300,t=0.2)
        sub_qs=parse_subqs(decomp);is_mh=len(sub_qs)>=2
        if is_mh:
            trace.append(Step("PLAN",f"Multi-hop: {len(sub_qs)} sub-Qs"))
            for i,sq in enumerate(sub_qs):log.info("  Sub-Q%d: %s",i+1,sq)
        else:trace.append(Step("PLAN","Simple"));sub_qs=[body]

        # BUILD LOCAL INDEX
        idx=LocalIndex(self.emb);sub_ans=[]

        # Gold snippets
        for gs in question.get("snippets",[]):
            t=gs.get("text","").strip()
            if t:
                du=gs.get("document","");m=re.search(r'/pubmed/(\d+)',du)
                p=Passage(t,m.group(1) if m else "gold",du,source="GOLD")
                idx.ps.append(p)
        if idx.ps:
            idx.ix.add(self.emb.encode([p.text for p in idx.ps]));idx._d=True
            trace.append(Step("OBSERVE",f"Gold: {len(idx.ps)} snippets"))

        for sq_i,sub_q in enumerate(sub_qs):
            lab=f"Sub-Q{sq_i+1}" if is_mh else "Main"
            log.info("\n  ── %s: %s ──",lab,sub_q[:60])

            # BROAD FETCH — one big FTS5 query
            k=kw(sub_q)
            if k:
                trace.append(Step("ACT",f"BROAD_FETCH('{k[:40]}')",tool="BROAD_FETCH"))
                arts=self.db.fetch(k,n=50)
                for a in arts:idx.add(a)
                trace.append(Step("OBSERVE",f"{len(arts)} articles → {idx.size} passages",n=len(arts)))
                log.info("    BROAD_FETCH: %d articles → %d passages",len(arts),idx.size)

            # LOCAL SEARCH (instant)
            top=idx.search(sub_q,k=15)
            if top:
                trace.append(Step("ACT",f"SEARCH_LOCAL('{sub_q[:40]}')",tool="SEARCH_LOCAL"))
                log.info("    LOCAL: top(%.4f) %s...",top[0].score,top[0].text[:50])

            # EXPAND via MeSH from best hit
            if top and top[0].pmid!="gold":
                trace.append(Step("ACT",f"EXPAND_MESH({top[0].pmid})",tool="EXPAND"))
                related=self.db.expand_mesh(top[0].pmid,10)
                for a in related:idx.add(a)
                trace.append(Step("OBSERVE",f"+{len(related)} related",n=len(related)))
                log.info("    EXPAND: +%d related",len(related))

            # EVALUATE — need more?
            top=idx.search(sub_q,k=10)
            if top:
                v=self.llm.ask(pr_eval(sub_q,[p.text[:150] for p in top[:8]]),mt=30,t=0.1)
                if "SUFFICIENT" in v.upper():
                    trace.append(Step("REFLECT","SUFFICIENT"));log.info("    SUFFICIENT")
                else:
                    trace.append(Step("REFLECT","INSUFFICIENT — extra fetch"))
                    qr=self.llm.ask(pr_queries(sub_q,[k] if k else None),mt=200,t=0.3)
                    for q in parse_qs(qr)[:2]:
                        arts=self.db.fetch(q,n=30)
                        for a in arts:idx.add(a)
                        log.info("    EXTRA '%s': +%d articles",q[:40],len(arts))

            # Sub-answer for multi-hop
            if is_mh:
                top=idx.search(sub_q,k=10)
                sa=self.llm.ask(f"Answer briefly:\n\nQ: {sub_q}\nEVIDENCE:\n{_sb([p.text for p in top],3000)}\n\nANSWER:",mt=300,t=0.2)
                sub_ans.append({"question":sub_q,"answer":sa})
                trace.append(Step("ACT",f"Sub-answer: {sa[:60]}...",tool="ANSWER"))

        log.info("  Index: %d art, %d passages",idx.n_art,idx.size)

        # RERANK
        top=idx.search(body,k=30)
        if len(top)>5:
            trace.append(Step("ACT","RERANK top 30",tool="RERANK"))
            top=self.reranker.rerank(body,top,k=15)
            log.info("  RERANK: top(%.4f) %s...",top[0].score,top[0].text[:50])

        # ANSWER
        fsex=self.fs.get(qtype,2);ptexts=[p.text for p in top[:15]] or [body]
        if is_mh and sub_ans:
            synth=self.llm.ask(pr_synth(body,sub_ans),mt=500,t=0.2)
            ptexts.insert(0,f"[Synthesized] {synth}")
            trace.append(Step("ACT","SYNTHESIZE",tool="SYNTHESIZE"))

        ec,ic=[],[]
        for pi in range(self.passes):
            temp=0.15+pi*0.15;log.info("  Answer %d/%d (t=%.2f)",pi+1,self.passes,temp)
            raw=self.llm.ask(pr_answer(body,qtype,ptexts,fsex),mt=768,t=temp)
            if qtype=="factoid":e,i=p_fac(raw);ec.append(e)
            elif qtype=="yesno":e,i=p_yn(raw);ec.append(e)
            elif qtype=="list":e,i=p_lst(raw);ec.append(e)
            else:i=p_sum(raw)
            ic.append(i)
            if pi==0:
                trace.append(Step("ACT","VERIFY",tool="VERIFY"))
                corr=self.llm.ask(pr_verify(body,ptexts,raw),mt=768,t=0.2)
                if qtype=="factoid":e2,i2=p_fac(corr);ec.append(e2);ic.append(i2)
                elif qtype=="yesno":e2,i2=p_yn(corr);ec.append(e2);ic.append(i2)
                elif qtype=="list":e2,i2=p_lst(corr);ec.append(e2);ic.append(i2)
                else:ic.append(p_sum(corr))

        # LIST second pass
        if qtype=="list" and ec:
            items_str=", ".join(ec[0][:20])
            sp=self.llm.ask(f"You listed: {items_str}\n\nQ: {body}\nEVIDENCE:\n{_sb(ptexts,5000)}\n\nAny MISSED items? '- ' prefix. If none: NONE\n\nMISSED:\n",mt=512,t=0.2)
            if "NONE" not in sp.upper():
                extra,_=p_lst(sp)
                if extra and extra!=["unknown"]:
                    ec.append(extra);trace.append(Step("ACT",f"LIST_REVIEW: +{len(extra)}",tool="LIST_REVIEW"))

        # RESULT
        result={"id":qid,"ideal_answer":con_ideal(ic),"_type":qtype}
        if qtype=="factoid":result["exact_answer"]=con_fac(ec)
        elif qtype=="yesno":result["exact_answer"]=con_yn(ec)
        elif qtype=="list":result["exact_answer"]=con_lst(ec)
        su=set();result["documents"]=[]
        for p in top[:10]:
            if p.doc_url and p.doc_url not in su:su.add(p.doc_url);result["documents"].append(p.doc_url)
        result["snippets"]=[{"document":p.doc_url,"text":p.text,"offsetInBeginSection":p.offset_begin,
            "offsetInEndSection":p.offset_end,"beginSection":"sections.0","endSection":"sections.0"} for p in top[:10]]
        result["_trace"]=[{"step":s.kind,"content":s.content,"tool":s.tool,"results":s.n} for s in trace]
        log.info("  ✓ exact=%s (%d steps)",str(result.get("exact_answer","N/A"))[:60],len(trace))
        return result

# ═══ MAIN ═══
def main():
    ap=argparse.ArgumentParser(description="BioASQ — Agentic RAG with SQLite")
    ap.add_argument("--test-input","-t",required=True)
    ap.add_argument("--training","-tr",default=None)
    ap.add_argument("--db",required=True)
    ap.add_argument("--output","-o",default="results.json")
    ap.add_argument("--vllm-url",default="http://localhost:8000")
    ap.add_argument("--model","-m",default="gemma-4-31b-it")
    ap.add_argument("--embedding-model",default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--reranker-model",default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--embed-device",default="cpu")
    ap.add_argument("--passes",type=int,default=3)
    ap.add_argument("--question-ids",nargs="*",default=None)
    a=ap.parse_args()

    fs=FS()
    if a.training:fs.load(a.training)
    with open(a.test_input) as f:qs=json.load(f).get("questions",[])
    log.info("Loaded %d questions",len(qs))
    if a.question_ids:qs=[q for q in qs if q["id"] in set(a.question_ids)]

    llm=LLM(a.vllm_url,a.model)
    db=DB(a.db)
    emb=Embedder(a.embedding_model,a.embed_device)
    reranker=Reranker(a.reranker_model,a.embed_device)
    agent=Agent(llm,db,emb,reranker,fs,a.passes)

    results=[]
    for i,q in enumerate(qs):
        log.info("═══ %d / %d ═══",i+1,len(qs))
        try:results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s",q.get("id"),e,exc_info=True)
            results.append({"id":q["id"],"ideal_answer":"Unable to generate.","documents":[],"snippets":[]})

    def fmt(ea,qt):
        if qt=="yesno":return ea
        if qt in("factoid","list"):
            if isinstance(ea,list):return[[x] if isinstance(x,str) else x for x in ea[:100 if qt=="list" else 5]]
            return[[str(ea)]]
        return ea

    pb={"questions":[]}
    for r in results:
        e={"id":r["id"],"ideal_answer":r.get("ideal_answer","")}
        if "exact_answer" in r and r["exact_answer"] is not None:e["exact_answer"]=fmt(r["exact_answer"],r.get("_type",""))
        pb["questions"].append(e)
    with open(a.output,"w") as f:json.dump(pb,f,indent=2,ensure_ascii=False)
    log.info("Phase B → %s",a.output)

    pa={"questions":[]}
    for r in results:
        e={"id":r["id"],"documents":r.get("documents",[]),"snippets":r.get("snippets",[]),"ideal_answer":r.get("ideal_answer","")}
        if "exact_answer" in r and r["exact_answer"] is not None:e["exact_answer"]=fmt(r["exact_answer"],r.get("_type",""))
        pa["questions"].append(e)
    oa=a.output.replace(".json","_phaseA.json")
    with open(oa,"w") as f:json.dump(pa,f,indent=2,ensure_ascii=False)

    tr=[{"id":r["id"],"exact_answer":r.get("exact_answer"),"trace":r.get("_trace",[])} for r in results]
    tp=a.output.replace(".json","_trace.json")
    with open(tp,"w") as f:json.dump(tr,f,indent=2,ensure_ascii=False)

    print(f"\nDone! {len(results)} questions")
    print(f"Phase B:  {a.output}\nPhase A+: {oa}\nTrace:    {tp}")

if __name__=="__main__":main()
