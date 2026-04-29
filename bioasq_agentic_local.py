#!/usr/bin/env python3
"""
BioASQ Task 14 — Agentic RAG with Pre-Built PubMed Indexes
============================================================
Uses pre-built FAISS + BM25 indexes over ENTIRE PubMed 2026 baseline.
Search 37M articles in <1 second. No on-the-fly indexing.

RUN:
    python bioasq_agentic_local.py \
        --test-input BioASQ-task14bPhaseA-testset4.json \
        --training training13b.json \
        --faiss-index /scratch/da32459/data/rag-cpu/pubmed-2026.IVF8192-Flat.NeuML-pubmedbert-base-embeddings.faiss \
        --bm25-index /scratch/da32459/data/rag-cpu/pubmed_2026.bm25 \
        --corpus-db /scratch/da32459/data/rag-cpu/corpus/pubmed_2026.db \
        --model gemma-4-31b-it \
        --embed-device cpu \
        -o batch4_local.json
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
    offset_begin:int=0;offset_end:int=0;similarity_score:float=0.0;source_tool:str=""

@dataclass
class ReActStep:
    step_type:str;content:str;tool:str="";result_count:int=0

# ═══ PRE-BUILT INDEX SEARCHERS ═══

class FAISSSearcher:
    def __init__(self,index_path,embed_model="NeuML/pubmedbert-base-embeddings",device="cpu"):
        import faiss;from sentence_transformers import SentenceTransformer
        log.info("Loading FAISS: %s",index_path)
        self.index=faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
        log.info("FAISS: %d vectors, dim=%d",self.index.ntotal,self.index.d)
        log.info("Loading embedder: %s (%s)",embed_model,device)
        self.emb=SentenceTransformer(embed_model,device=device)
    def search(self,query,k=30):
        qv=np.array(self.emb.encode([query],normalize_embeddings=True),dtype=np.float32)
        sc,ix=self.index.search(qv,k)
        return [(int(i),float(s)) for s,i in zip(sc[0],ix[0]) if i>=0]

class BM25Searcher:
    def __init__(self,index_path):
        self.active = False
        if index_path == "NONE" or not index_path:
            log.info("BM25: disabled")
            return
        log.info("Loading BM25: %s",index_path)
        try:
            import bm25s;self.bm25=bm25s.BM25.load(index_path,load_corpus=False);self.backend="bm25s";self.active=True
            log.info("BM25 loaded (bm25s)")
        except Exception as e:
            log.warning("bm25s failed (%s), trying pickle",e)
            try:
                import pickle
                with open(index_path,"rb") as f:self.bm25=pickle.load(f)
                self.backend="pickle";self.active=True
            except Exception as e2:
                log.warning("BM25 load failed entirely: %s — running FAISS only",e2)
    def search(self,query,k=30):
        if not self.active: return []
        if self.backend=="bm25s":
            import bm25s;tokens=bm25s.tokenize([query])
            r,s=self.bm25.retrieve(tokens,k=k)
            return [(int(i),float(sc)) for i,sc in zip(r[0],s[0]) if sc>0]
        else:
            tokens=re.findall(r'[a-z0-9\-]+',query.lower())
            sc=self.bm25.get_scores(tokens)
            return sorted(enumerate(sc),key=lambda x:-x[1])[:k]

class CorpusDB:
    def __init__(self,db_path):
        self.db=db_path;log.info("Corpus: %s",db_path)
        conn=sqlite3.connect(db_path)
        tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        log.info("  Tables: %s",tables)
        self.table=tables[0] if tables else "documents"
        cols=[r[1] for r in conn.execute(f"PRAGMA table_info({self.table})").fetchall()]
        log.info("  Columns: %s",cols)
        cols_l=[c.lower() for c in cols]
        self.pmid_col=next((c for c,cl in zip(cols,cols_l) if cl in ("pmid","id","doc_id")),cols[0] if cols else "id")
        self.title_col=next((c for c,cl in zip(cols,cols_l) if "title" in cl),None)
        self.abstract_col=next((c for c,cl in zip(cols,cols_l) if any(x in cl for x in ("abstract","text","content"))),None)
        cnt=conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        log.info("  %d articles, pmid=%s, title=%s, abstract=%s",cnt,self.pmid_col,self.title_col,self.abstract_col)
        conn.close()

    def fetch_by_indices(self,indices):
        if not indices:return []
        conn=sqlite3.connect(self.db);conn.row_factory=sqlite3.Row;results=[]
        for idx in indices:
            try:
                row=conn.execute(f"SELECT * FROM {self.table} WHERE rowid=?",(idx+1,)).fetchone()
                if row:
                    d=dict(row)
                    pmid=str(d.get(self.pmid_col,str(idx)))
                    title=str(d.get(self.title_col,"") or "") if self.title_col else ""
                    abstract=str(d.get(self.abstract_col,"") or "") if self.abstract_col else ""
                    if not abstract:
                        for v in d.values():
                            if isinstance(v,str) and len(v)>100:abstract=v;break
                    results.append({"pmid":pmid,"title":title,"abstract":abstract,
                                    "url":f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}","_idx":idx})
            except:pass
        conn.close();return results

class HybridSearcher:
    def __init__(self,faiss_s,bm25_s,corpus):
        self.faiss=faiss_s;self.bm25=bm25_s;self.corpus=corpus
        self.bm25_active = bm25_s.active if hasattr(bm25_s,'active') else True
    def search(self,query,k=30):
        sem=self.faiss.search(query,k*2)
        bm=self.bm25.search(query,k*2) if self.bm25_active else []
        rrf={};K=60
        sw = 1.0 if not bm else 0.6  # full weight to FAISS if no BM25
        bw = 0.0 if not bm else 0.4
        for r,(i,_) in enumerate(sem):rrf[i]=rrf.get(i,0)+sw/(K+r+1)
        for r,(i,_) in enumerate(bm):rrf[i]=rrf.get(i,0)+bw/(K+r+1)
        ranked=sorted(rrf.items(),key=lambda x:-x[1])[:k]
        idx_map=dict(ranked);arts=self.corpus.fetch_by_indices([i for i,_ in ranked])
        i2a={a["_idx"]:a for a in arts}
        passages=[];seen=set()
        for idx,sc in ranked:
            a=i2a.get(idx);
            if not a:continue
            t=a["abstract"] or a["title"]
            if not t:continue
            sig=t[:80].lower()
            if sig in seen:continue
            seen.add(sig)
            passages.append(Passage(t,a["pmid"],a["url"],"abstract" if a["abstract"] else "title",
                                    0,len(t),sc,"SEARCH_HYBRID"))
        return passages

class Reranker:
    def __init__(self,model="cross-encoder/ms-marco-MiniLM-L-12-v2",device="cpu"):
        from sentence_transformers import CrossEncoder
        log.info("Reranker: %s (%s)",model,device)
        self.m=CrossEncoder(model,device=device)
    def rerank(self,query,passages,k=15):
        if not passages:return []
        pairs=[[query,p.text[:512]] for p in passages]
        scores=self.m.predict(pairs)
        for p,s in zip(passages,scores):p.similarity_score=float(s)
        return sorted(passages,key=lambda p:p.similarity_score,reverse=True)[:k]

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
    def ask(self,p,max_tokens=1024,temp=0.3):
        for i in range(3):
            try:
                r=self.s.post(f"{self.url}/v1/chat/completions",json={"model":self.model,
                    "messages":[{"role":"user","content":p}],"max_tokens":max_tokens,
                    "temperature":temp,"top_p":0.95},timeout=180)
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

def pr_decompose(q):
    return f"Break this biomedical question into 2-4 sub-questions. If simple, return as-is.\n\nQUESTION: {q}\n\nSUB-QUESTIONS:\n"
def pr_queries(q,prev=None):
    p=f"Generate 3 short search queries (3-6 words) for: {q}\n\n"
    if prev:p+="Previous didn't work:\n"+"\n".join(f"  - {x}" for x in prev[-3:])+"\n\n"
    p+="Write ONLY 3 queries.\n\n1.";return p
def pr_eval(q,ts):
    return f"Can '{q}' be answered from these?\n\n"+"\n".join(f"[{i+1}] {t[:150]}" for i,t in enumerate(ts[:8]))+"\n\nSUFFICIENT or INSUFFICIENT?"
def pr_answer(q,qt,ts,fs):
    if qt=="yesno":return _yn(q,ts,fs)
    if qt=="factoid":return _fac(q,ts,fs)
    if qt=="list":return _lst(q,ts,fs)
    return _sum(q,ts,fs)
def _yn(q,ts,fs):
    p="You are an expert biomedical QA system.\n\n1. Find evidence for YES.\n2. Find evidence for NO.\n3. Choose stronger side.\n4. Problems/mixed → NO.\n5. 'Promising' ≠ YES.\n\n"
    for ex in fs[:2]:
        ea=ex["exact_answer"];ea=ea[0] if isinstance(ea,list) and ea else ea
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {ea}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\n\nEVIDENCE FOR YES:\nEVIDENCE FOR NO:\nEXACT_ANSWER:";return p
def _fac(q,ts,fs):
    p="You are an expert biomedical QA system.\n\nRULES: EXACT_ANSWER 1-5 words MAX. Use EXACT terms from evidence. If unclear: unknown. Then IDEAL_ANSWER 2-4 sentences.\n\n"
    for ex in fs[:2]:p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],600)}\nEXACT_ANSWER: {_fmt(ex['exact_answer'])}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nEXACT_ANSWER:";return p
def _lst(q,ts,fs):
    p="You are an expert biomedical QA system.\n\nRULES: List EVERY item. 5-15+ items. '- ' prefix. Then IDEAL_ANSWER.\n\n"
    for ex in fs[:1]:
        p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],500)}\nEXACT_ANSWER:\n"
        ea=ex["exact_answer"]
        if isinstance(ea,list):
            for it in ea[:10]:
                if isinstance(it,list):p+=f"- {it[0]}\n"
                else:p+=f"- {it}\n"
        p+=f"IDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts,6000)}\nList EVERY item:\n\nEXACT_ANSWER:\n";return p
def _sum(q,ts,fs):
    p="Expert biomedical QA. Write 3-6 sentences (max 200 words).\n\n"
    for ex in fs[:2]:p+=f"---\nQ: {ex['body']}\nEVIDENCE:\n{_sb(ex['snippets'],800)}\nIDEAL_ANSWER: {ex['ideal_answer']}\n\n"
    p+=f"---\nQ: {q}\nEVIDENCE:\n{_sb(ts)}\nIDEAL_ANSWER:";return p
def pr_verify(q,ts,a):return f"Check answer. Fix errors.\n\nQ: {q}\nEVIDENCE:\n{_sb(ts,3000)}\n\nCANDIDATE:\n{a}\n\nCORRECTED_ANSWER:"
def pr_synth(q,subs):
    return f"Combine sub-answers:\n\nORIGINAL: {q}\n\n"+"\n".join(f"Sub-Q{i+1}: {s['question']}\nAnswer: {s['answer']}\n" for i,s in enumerate(subs))+"\nCOMBINED:"

# ═══ PARSERS ═══
def p_fac(r):
    parts=re.split(r'IDEAL_ANSWER\s*:',r,1,re.I);e=parts[0].strip().strip('"\'').rstrip(".")
    for pf in ["The answer is","The exact answer is","Answer:","EXACT_ANSWER:"]:
        if e.lower().startswith(pf.lower()):e=e[len(pf):].strip()
    if "\n" in e:e=e.split("\n")[0].strip()
    ideal=parts[1].strip() if len(parts)==2 else "";return [e.strip()] if e.strip() else ["unknown"],_cap(ideal or e)
def p_yn(r):
    m=re.search(r'EXACT_ANSWER\s*:\s*(.*?)(?:\n|IDEAL|$)',r,re.I)
    if m:raw=m.group(1).strip().lower().strip('"\'').rstrip(".")
    else:
        raw=""
        for l in reversed(r.strip().split("\n")):
            ll=l.strip().lower()
            if ll.startswith("yes") or ll.startswith("no"):raw=ll;break
        if not raw:raw=r.strip().split("\n")[-1].lower()
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
        if any(skip in l.upper() for skip in ["CORRECTED_ANSWER","EXACT_ANSWER","IDEAL_ANSWER","EVIDENCE","QUESTION","NOTE:"]):continue
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
def simple_kw(q):
    STOP=set("what is are the a an of in on to for with by from and or not how does do can which who whom where when why this that these those it its has have had been being please list describe common main types most often should".split())
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

# ═══ THE AGENT ═══
class Agent:
    def __init__(self,llm,searcher,reranker,fs,passes=3,iters=3):
        self.llm=llm;self.searcher=searcher;self.reranker=reranker;self.fs=fs;self.passes=passes;self.iters=iters

    def solve(self,question):
        qid=question["id"];body=question["body"];qtype=question.get("type","summary").lower()
        trace=[]
        log.info("━"*60);log.info("  %s [%s]: %s",qid,qtype,body);log.info("━"*60)

        trace.append(ReActStep("THINK",f"Analyzing: {body}"))
        decomp=self.llm.ask(pr_decompose(body),max_tokens=300,temp=0.2)
        sub_qs=parse_subqs(decomp);is_mh=len(sub_qs)>=2
        if is_mh:
            trace.append(ReActStep("PLAN",f"Multi-hop: {len(sub_qs)} sub-questions"))
            for i,sq in enumerate(sub_qs):log.info("  Sub-Q%d: %s",i+1,sq)
        else:trace.append(ReActStep("PLAN","Simple question"));sub_qs=[body]

        # Gold snippets
        gold=[]
        for gs in question.get("snippets",[]):
            t=gs.get("text","").strip()
            if t:
                du=gs.get("document","");m=re.search(r'/pubmed/(\d+)',du)
                gold.append(Passage(t,m.group(1) if m else "gold",du,source_tool="GOLD"))
        if gold:trace.append(ReActStep("OBSERVE",f"Gold: {len(gold)} snippets"))

        all_p=list(gold);sub_ans=[];all_q=[]

        for sq_i,sub_q in enumerate(sub_qs):
            lab=f"Sub-Q{sq_i+1}" if is_mh else "Main"
            log.info("\n  ── %s: %s ──",lab,sub_q[:60])

            # Direct search
            trace.append(ReActStep("ACT",f"SEARCH_HYBRID('{sub_q[:50]}...')",tool="SEARCH_HYBRID"))
            r=self.searcher.search(sub_q,k=20);all_p.extend(r)
            trace.append(ReActStep("OBSERVE",f"{len(r)} passages",result_count=len(r)))
            log.info("    SEARCH_HYBRID: %d passages",len(r))
            if r:log.info("      Best(%.4f): %s...",r[0].similarity_score,r[0].text[:60])

            # LLM queries
            for it in range(self.iters):
                qr=self.llm.ask(pr_queries(sub_q,all_q or None),max_tokens=200,temp=0.3)
                nq=parse_qs(qr)
                if not nq:
                    w=simple_kw(sub_q).split()
                    if len(w)>=3:nq=[" ".join(w[:3])]
                for q in nq:
                    if q in all_q:continue
                    all_q.append(q)
                    trace.append(ReActStep("ACT",f"SEARCH_HYBRID('{q[:50]}')",tool="SEARCH_HYBRID"))
                    r2=self.searcher.search(q,k=15);all_p.extend(r2)
                    trace.append(ReActStep("OBSERVE",f"+{len(r2)} passages",result_count=len(r2)))
                    log.info("    '%s': +%d",q[:50],len(r2))
                if len(all_p)>=20:
                    top_t=[p.text[:150] for p in sorted(all_p,key=lambda p:p.similarity_score,reverse=True)[:8]]
                    v=self.llm.ask(pr_eval(sub_q,top_t),max_tokens=30,temp=0.1)
                    if "SUFFICIENT" in v.upper():trace.append(ReActStep("REFLECT","SUFFICIENT"));log.info("    SUFFICIENT");break
                    else:trace.append(ReActStep("REFLECT","INSUFFICIENT"))

            if is_mh:
                top=sorted(all_p,key=lambda p:p.similarity_score,reverse=True)[:10]
                sa=self.llm.ask(f"Answer briefly:\n\nQ: {sub_q}\nEVIDENCE:\n{_sb([p.text for p in top],3000)}\n\nANSWER:",max_tokens=300,temp=0.2)
                sub_ans.append({"question":sub_q,"answer":sa})
                trace.append(ReActStep("ACT",f"Sub-answer: {sa[:60]}...",tool="ANSWER"))

        # Dedup
        seen=set();unique=[]
        for p in all_p:
            sig=p.text[:80].lower()
            if sig not in seen:seen.add(sig);unique.append(p)
        log.info("  Unique passages: %d",len(unique))

        # Rerank
        if len(unique)>5:
            trace.append(ReActStep("ACT",f"RERANK {min(len(unique),30)} passages",tool="RERANK"))
            top=self.reranker.rerank(body,unique[:30],k=15)
            log.info("  RERANK: top(%.4f) %s...",top[0].similarity_score,top[0].text[:60])
        else:top=unique

        # Answer
        fsex=self.fs.get(qtype,2);ptexts=[p.text for p in top[:15]] or [body]
        if is_mh and sub_ans:
            synth=self.llm.ask(pr_synth(body,sub_ans),max_tokens=500,temp=0.2)
            ptexts.insert(0,f"[Synthesized] {synth}")
            trace.append(ReActStep("ACT","SYNTHESIZE",tool="SYNTHESIZE"))

        ec,ic=[],[]
        for pi in range(self.passes):
            temp=0.15+pi*0.15;log.info("  Answer %d/%d (t=%.2f)",pi+1,self.passes,temp)
            raw=self.llm.ask(pr_answer(body,qtype,ptexts,fsex),max_tokens=768,temp=temp)
            if qtype=="factoid":e,i=p_fac(raw);ec.append(e)
            elif qtype=="yesno":e,i=p_yn(raw);ec.append(e)
            elif qtype=="list":e,i=p_lst(raw);ec.append(e)
            else:i=p_sum(raw)
            ic.append(i)
            if pi==0:
                trace.append(ReActStep("ACT","VERIFY",tool="VERIFY"))
                corr=self.llm.ask(pr_verify(body,ptexts,raw),max_tokens=768,temp=0.2)
                if qtype=="factoid":e2,i2=p_fac(corr);ec.append(e2);ic.append(i2)
                elif qtype=="yesno":e2,i2=p_yn(corr);ec.append(e2);ic.append(i2)
                elif qtype=="list":e2,i2=p_lst(corr);ec.append(e2);ic.append(i2)
                else:ic.append(p_sum(corr))

        result={"id":qid,"ideal_answer":con_ideal(ic),"_type":qtype}
        if qtype=="factoid":result["exact_answer"]=con_fac(ec)
        elif qtype=="yesno":result["exact_answer"]=con_yn(ec)
        elif qtype=="list":result["exact_answer"]=con_lst(ec)
        su=set();result["documents"]=[]
        for p in top[:10]:
            if p.doc_url and p.doc_url not in su:su.add(p.doc_url);result["documents"].append(p.doc_url)
        result["snippets"]=[{"document":p.doc_url,"text":p.text,"offsetInBeginSection":p.offset_begin,
            "offsetInEndSection":p.offset_end,"beginSection":"sections.0","endSection":"sections.0"} for p in top[:10]]
        result["_trace"]=[{"step":s.step_type,"content":s.content,"tool":s.tool,"results":s.result_count} for s in trace]
        log.info("  ✓ exact=%s (%d steps)",str(result.get("exact_answer","N/A"))[:60],len(trace))
        return result

# ═══ MAIN ═══
def main():
    ap=argparse.ArgumentParser(description="BioASQ — Agentic RAG with Pre-Built Indexes")
    ap.add_argument("--test-input","-t",required=True)
    ap.add_argument("--training","-tr",default=None)
    ap.add_argument("--faiss-index",required=True)
    ap.add_argument("--bm25-index",required=True)
    ap.add_argument("--corpus-db",required=True)
    ap.add_argument("--output","-o",default="submission.json")
    ap.add_argument("--vllm-url",default="http://localhost:8000")
    ap.add_argument("--model","-m",default="gemma-4-31b-it")
    ap.add_argument("--embed-model",default="NeuML/pubmedbert-base-embeddings")
    ap.add_argument("--reranker-model",default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--embed-device",default="cpu")
    ap.add_argument("--passes",type=int,default=3)
    ap.add_argument("--retrieval-iterations",type=int,default=3)
    ap.add_argument("--question-ids",nargs="*",default=None)
    a=ap.parse_args()

    fs=FS()
    if a.training:fs.load(a.training)
    with open(a.test_input) as f:qs=json.load(f).get("questions",[])
    log.info("Loaded %d questions",len(qs))
    if a.question_ids:qs=[q for q in qs if q["id"] in set(a.question_ids)]

    faiss_s=FAISSSearcher(a.faiss_index,a.embed_model,a.embed_device)
    bm25_s=BM25Searcher(a.bm25_index)
    corpus=CorpusDB(a.corpus_db)
    searcher=HybridSearcher(faiss_s,bm25_s,corpus)
    reranker=Reranker(a.reranker_model,device=a.embed_device)
    llm=LLM(a.vllm_url,a.model)
    agent=Agent(llm,searcher,reranker,fs,a.passes,a.retrieval_iterations)

    results=[]
    for i,q in enumerate(qs):
        log.info("═══ Question %d / %d ═══",i+1,len(qs))
        try:results.append(agent.solve(q))
        except Exception as e:
            log.error("Failed %s: %s",q.get("id"),e,exc_info=True)
            results.append({"id":q["id"],"ideal_answer":"Unable to generate.","documents":[],"snippets":[]})

    def fmt(ea,qt):
        if qt=="yesno":return ea
        if qt in ("factoid","list"):
            if isinstance(ea,list):return [[x] if isinstance(x,str) else x for x in ea[:100 if qt=="list" else 5]]
            return [[str(ea)]]
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
    log.info("Phase A+ → %s",oa)

    tr=[{"id":r["id"],"exact_answer":r.get("exact_answer"),"trace":r.get("_trace",[])} for r in results]
    tp=a.output.replace(".json","_trace.json")
    with open(tp,"w") as f:json.dump(tr,f,indent=2,ensure_ascii=False)

    print(f"\nDone! {len(results)} questions")
    print(f"Phase B:  {a.output}\nPhase A+: {oa}\nTrace:    {tp}")

if __name__=="__main__":main()
