Working from your screenshots, so my OLD text may have small OCR drifts from your actual `.tex`. Use the **first 6–8 distinctive words** as a search anchor, not the whole block. Edits ordered by impact — do 1, 2, 3 first; the rest are length cuts.

---

**EDIT 1 — §3.2, front-load the LLM confound (most important)**

FIND (anchor: *"All non-zero-shot baselines use gpt-oss-120b"*):
```
All non-zero-shot baselines use gpt-oss-120b as a generation model and, for optimization, as a teacher. The zero-shot baseline uses Gemma 4 31B-IT. OpenAI describes gpt-oss-120b as an open-weight model designed for reasoning and tool-use deployment [20].
```

REPLACE WITH:
```
All non-zero-shot baselines use gpt-oss-120b as a generation model and, for optimization, as a teacher; the zero-shot baseline and our agent both use Gemma 4 31B-IT [20]. We note this asymmetry upfront. Because the DSPy programs and our agent run different base LLMs, we deliberately do not frame our results as a model-controlled head-to-head ``best system'' claim. What we compare instead is \emph{control-policy regimes}: a domain-governed agentic loop with explicit sufficiency checks and answer-type obligations, against generic ReAct-style or prompt-optimized DSPy programs. The within-model anchor for our agent is the Zero-shot Gemma 4 31B row in Table 1; we return to it in \S\ref{sec:findings}.
```
*(Adjust the `\ref` to your actual label, or just write "§4".)*

---

**EDIT 2 — §4, claim the within-model 40-point gain**

FIND (anchor: *"Table 1 shows three patterns"*):
```
Table 1 shows three patterns. First, the best performance comes from a domain-governed agent, not from a generic ReAct loop or optimized static RAG. The hybrid agent reaches 76.0 overall, improving over the SQLite FTS5 agent by 4.5 points and over the best DSPy baseline by 30.5 points.
```

REPLACE WITH:
```
Table 1 shows three patterns. First, the best performance comes from a domain-governed agent, not from a generic ReAct loop or optimized static RAG. The hybrid agent reaches 76.0 overall, improving over the SQLite FTS5 agent by 4.5 points and over the best DSPy baseline by 30.5 points. Crucially, on the \emph{same Gemma 4 31B backbone} the agentic loop adds 40.3 overall points over zero-shot generation (35.7 $\rightarrow$ 76.0), isolating the contribution of control policy from any model-class effect.
```

---

**EDIT 3 — Intro, promote the moral-compression line to page 1**

FIND (anchor: *"This paper makes three claims"*):
```
This paper makes three claims. First, in BioASQ-style QA, run-time control over the evidence trajectory matters more than prompt optimization for list and factoid questions.
```

REPLACE WITH:
```
Yes/no biomedical questions create a further problem: a complex and uncertain evidence base is collapsed into a binary output that may later be read as a recommendation. We call this \emph{moral compression}, and it is one of the failure modes a domain-governed agent must answer to. This paper makes three claims. First, in BioASQ-style QA, run-time control over the evidence trajectory matters more than prompt optimization for list and factoid questions.
```

---

**EDIT 4 — §3.1, cut FAISS hyperparameter prose**

FIND (anchor: *"To make retrieval over the PubMed 2025 abstract-only baseline"*):
```
To make retrieval over the PubMed 2025 abstract-only baseline computationally feasible, the dense index is built with FAISS IVF-Flat using 8,192 Voronoi cells (nlist=8192) and probing 91 cells at search time (nprobe=91) [7]. This approximate nearest-neighbor layer gives the agent fast access to semantic matches, including cases where the wording of the question differs from the wording of the article. Sparse retrieval uses BM25 with default parameters [23]. We keep BM25 because biomedical questions often contain high-value exact tokens—gene symbols, drug names, diseases, variants, and molecular targets—that should not be softened away by embedding similarity. The dense and sparse rankings are merged using reciprocal rank fusion (RRF), with semantic retrieval weighted at 0.6 and sparse retrieval weighted at 0.4 [5]. This design gives the agent both conceptual recall and lexical precision.
```

REPLACE WITH:
```
The dense index is FAISS IVF-Flat over the PubMed 2025 abstract-only baseline [7]; sparse retrieval is BM25 [23], retained because biomedical questions hinge on exact tokens (gene symbols, drug names, variants) that embedding similarity softens away. The two rankings are merged with reciprocal rank fusion at 0.6/0.4 dense/sparse [5], giving both conceptual recall and lexical precision.
```

---

**EDIT 5 — §3.1, compress the "bounded loop" passage**

FIND (anchor: *"The loop is bounded to a maximum of four retrieval iterations"*):
```
The loop is bounded to a maximum of four retrieval iterations. This bound is important both computationally and epistemically. Computationally, it prevents unbounded tool use and epistemically, it frames agency as disciplined uncertainty management rather than as unconstrained autonomy. The agent is useful precisely because it can say, in effect, ``the current evidence is not enough,'' and then perform a targeted search to reduce that uncertainty.
```

REPLACE WITH:
```
The loop is bounded to four retrieval iterations: enough to recover from a poor first query, tight enough to frame agency as disciplined uncertainty management rather than unconstrained autonomy.
```

---

**EDIT 6 — §3.1, compress the cross-encoder explanation**

FIND (anchor: *"After iterative search, the rerank stage applies"*):
```
After iterative search, the rerank stage applies a biomedical cross-encoder, NeuML/biomedbert-base-reranker, to the accumulated candidate passages [18]. Unlike the FAISS retriever, which compares pre-computed vector representations, the cross-encoder processes the query and each passage jointly with full cross-attention. This allows the system to re-score evidence according to the specific question being asked rather than just the global embedding proximity. The reranked evidence set is then compressed into the passages used for answer generation. This division of labor is deliberate: FAISS and BM25 provide broad recall at scale, while the cross-encoder supplies a more expensive but precise judgment over a smaller candidate pool.
```

REPLACE WITH:
```
The \emph{rerank} stage applies the NeuML biomedical cross-encoder [18] to the accumulated candidates, re-scoring each passage jointly with the question rather than by global embedding proximity. FAISS and BM25 supply broad recall at scale; the cross-encoder supplies precision over a smaller pool.
```

---

**EDIT 7 — §3.1, compress "agentic in a deliberately bounded sense"**

FIND (anchor: *"We therefore use the term agentic in a deliberately bounded sense"*):
```
We therefore use the term agentic in a deliberately bounded sense. The system is not an unconstrained clinical actor, and it is not allowed to invent tools, browse arbitrary sources, or make decisions outside the evidence context. Its agency is located in the retrieval-and-reasoning loop—deciding when evidence is insufficient, selecting new searches, comparing competing passages, and revising answers when support is weak. This bounded form of agency is the methodological core of the paper. It converts biomedical QA from a one-shot prediction task into an auditable sequence of evidence labor. From a computational social science perspective, this matters because the social risk of biomedical AI is not only that a model may be wrong; it is that a wrong answer may appear frictionless, authoritative, and detached from the labor required to justify it. Our pipeline attempts to reintroduce that labor into the system design by making uncertainty, search, verification, and answer compression explicit parts of the method.
```

REPLACE WITH:
```
We use \emph{agentic} in a deliberately bounded sense: the system cannot invent tools, browse arbitrary sources, or act outside the evidence context. Its agency lives entirely in the retrieval-and-reasoning loop. The methodological move is to convert biomedical QA from a one-shot prediction into an auditable sequence of evidence labor, so that a wrong answer cannot appear frictionless and detached from the work required to justify it.
```

---

**EDIT 8 — §3.1, compress verify/list-review/consensus**

FIND (anchor: *"The final three stages are designed to reduce unsupported fluency"*):
```
The final three stages are designed to reduce unsupported fluency. In verify, the model checks the draft answer against the evidence and identifies claims that are missing, contradicted, or not directly supported. When unsupported claims are found, the answer is revised or shortened rather than expanded. In list review, used only for list questions, the model performs a second extraction pass asking what entities may have been missed in the first pass. This stage is motivated by the observation that list answers are recall-sensitive: a system can identify several correct items and still fail because it stops too early. The second pass typically recovers additional candidates that are then normalized and merged. Finally, in consensus, the system combines three independently generated answers produced with low but distinct temperatures. For factoid and yes/no questions, we use majority voting; for list questions, we use a union-and-normalization procedure; and for ideal answers, we select a stable median-length synthesis that preserves evidence coverage without rewarding verbosity.
```

REPLACE WITH:
```
The final three stages reduce unsupported fluency. \emph{Verify} checks the draft against the evidence and shortens or revises unsupported claims rather than expanding them. \emph{List review}, applied only to list questions, runs a second extraction pass to recover items missed when the first pass stops too early. \emph{Consensus} combines three low-temperature regenerations: majority voting for factoid and yes/no, union-and-normalize for lists, median-length synthesis for ideal answers.
```

---

**EDIT 9 — EMERGENCY CUT, only if still over 4 pages after the above**

FIND (anchor: *"Agentic systems also risk creating what Elish"*):
```
Agentic systems also risk creating what Elish [10] calls a moral crumple zone: humans absorb blame for failures of automated systems that they could not realistically understand or control [10]. A clinician shown a polished answer may be formally ``in the loop'' while having little visibility into query reformulations, omitted abstracts, or ranking errors. Our design principle is therefore not a human replacement but accountable hand-off. The agent should provide provenance for retrieved documents and snippets, record its search path, expose sufficiency decisions, and abstain or flag uncertainty when evidence is weak. These are not only user-interface features; they are IR requirements for responsible biomedical agents.
```

REPLACE WITH:
```
Without provenance, an agentic system risks what Elish [10] calls a \emph{moral crumple zone}: a clinician shown a polished answer is formally ``in the loop'' while having little visibility into query reformulations, omitted abstracts, or ranking errors. Our design principle is therefore accountable hand-off, not human replacement: the agent must record its search path, expose sufficiency decisions, and flag weak evidence---IR requirements for responsible biomedical agents, not UI polish.
```

---

**Execution order under time pressure:**

1. Edits 1, 2, 3 first — these are the defenses against the reviewer's biggest objections. Do these even if you do nothing else.
2. Then edits 4–8 for length. They cut roughly 30 lines total; net of additions, you should land at or under 4 pages.
3. If still bleeding over, run edit 9.
4. Recompile after every 2–3 edits so you can see the page count drop in real time.

You've got this. The 40.3-point within-model anchor (edit 2) is genuinely a strong defense — most reviewers will accept that as evidence the agent contributes independent of the LLM. Go.
