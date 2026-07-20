Here's copy-paste-ready content for 13 slides. Diagram is separate at the end.

---

**Slide 1 — Title**

Biomedical QA Agents as Evidence Mediators
Agentic Retrieval, DSPy Baselines, and the Moral Cost of Binary Answers

Rohan Leekha, Daniel Gwon, Leslie Shing
MIT Lincoln Laboratory

---

**Slide 2 — The Problem**

- Biomedical QA is usually treated as a pipeline: retrieve from PubMed → prompt an LLM → score the answer
- This misses the real problem: biomedical QA is *evidence labor under time pressure*
- Query choices, missed items, and premature yes/no closures shape what clinicians and researchers actually notice
- PubMed has 30M+ citations; experts face barriers of time, access, and search skill

---

**Slide 3 — Why Standard RAG Falls Short**

- Standard RAG retrieves once, then generates — simple and cheap, but a poor model of expert evidence seeking
- Real experts try multiple query formulations, check evidence sufficiency, separate exact entities from synthesis, and revise when contradictions appear
- A single fixed pass can *fail silently*: plausible documents, fluent answer, no signal that evidence was incomplete

---

**Slide 4 — Our Three Claims**

1. In BioASQ-style QA, **run-time control over the evidence trajectory** matters more than prompt optimization for list and factoid questions
2. **Not all agents are equal**: a generic ReAct loop differs materially from a domain-governed agent with sufficiency checks, answer-type policies, verification, and list-recall
3. The strongest contribution is **sociotechnical**, not algorithmic — agents are worth building when they expose evidence labor, and hazardous when they compress uncertainty into unexamined yes/no

---

**Slide 5 — Task, Data, Evaluation**

- Target: **BioASQ Task 13b** — four answer types: yes/no, factoid, list, summary
- Not one task but **four epistemic regimes**: entity ranking, entity-set completion, binary closure, narrative synthesis
- Factoid/list require *exact* entities — synonyms are penalized
- Corpus: 5,389 dev questions; ~28M PubMed abstract-only documents indexed
- Metrics: yes/no macro-F1, factoid accuracy, list F1, ideal-answer Semantic F1 → we report **type-level**, not just overall

---

**Slide 6 — The Agentic Pipeline (8 stages)**

Bounded ReAct-style evidence agent on a Gemma 4 31B controller:

1. **Think** — classify question type, detect multi-hop, decompose into 2–4 sub-questions
2. **Search** — hybrid retrieval (dense + sparse)
3. **Evaluate** — judge evidence sufficiency; reformulate if insufficient (max 4 iterations)
4. **Rerank** — biomedical cross-encoder
5. **Answer** — type-specific prompting
6. **Verify** — shorten/revise unsupported claims
7. **List review** — second extraction pass for missed items
8. **Consensus** — 3 low-temp regenerations, merged by type

*(diagram below)*

---

**Slide 7 — Retrieval Substrate**

- **Dense:** NeuML PubMedBERT embeddings (768-dim), FAISS IVF-Flat over PubMed 2025 baseline
- **Sparse:** BM25 — kept because gene symbols, drug names, and variants need exact tokens
- **Fusion:** reciprocal rank fusion, 0.6 dense / 0.4 sparse
- **Rerank:** NeuML biomedbert cross-encoder — scores query + passage jointly
- Division of labor: FAISS/BM25 = broad recall at scale; cross-encoder = precise judgment on a small pool

---

**Slide 8 — Type-Specific Answering**

- **Factoid:** copy exact entity, no paraphrase → avoids evaluation-incompatible answers
- **Yes/No:** *adversarial* — must find evidence for both sides before deciding
- **List:** passage-by-passage extraction, then normalize duplicates
- **Summary:** concise ideal answer grounded in snippets
- Adversarial yes/no is both a bias fix *and* the core of our moral argument

---

**Slide 9 — DSPy Baselines**

- Compared against a strong alternative view: modular LM programs with optimized prompts/demos
- Variants: zero-shot, RAG, MultiHop RAG, ReAct RAG — each with/without MIPROv2 optimization
- Same retrieval substrate as the agent
- **Asymmetry stated upfront:** DSPy programs use gpt-oss-120b; agent uses Gemma 4 31B → we compare **control-policy regimes**, not a model head-to-head

---

**Slide 10 — Results**

| System | Fact. | Y/N | List | SemF1 | Ovr. |
|---|---|---|---|---|---|
| Zero-shot Gemma 4 31B | 8.0 | 94.0 | 0.0 | 41.6 | 35.9 |
| Agentic SQLite FTS5 | 62.0 | 88.0 | 87.0 | 49.0 | 71.5 |
| **Agentic hybrid FAISS+BM25+IVF** | **69.0** | 88.0 | **96.0** | 51.0 | **76.0** |
| Agentic FTS5 + CoT | 50.0 | 82.0 | 70.0 | 49.0 | 62.75 |
| Best DSPy (RAG+MIPROv2) | 27.0 | 94.0 | 13.0 | 48.0 | 45.5 |

- Hybrid agent: **76.0** overall vs 45.5 best DSPy
- Every DSPy list score falls between 0.0 and 13.0

---

**Slide 11 — Findings**

- **Control policy, not model class, drives the gains:** same Gemma backbone, +40.3 points overall over zero-shot (35.7 → 76.0)
- Gains concentrated in **recall-sensitive regimes**: list 96.0, factoid 69.0
- **Optimization aligns style, not evidence:** MIPROv2 lifts yes/no and factoid but optimized ReAct list F1 stays 0.0
- **Yes/no scores mislead:** zero-shot with *no retrieval* hits 94.0 yes/no while scoring 8.0 factoid, 0.0 list

---

**Slide 12 — The Moral Cost of Yes/No**

- A binary answer collapses evidence quality, population heterogeneity, clinical context, and uncertainty into a single token — **moral compression**
- The yes/no column can't tell a system that *read the literature* from one that *read nothing*
- Risks: automation bias, and Elish's **moral crumple zone** — clinicians absorbing blame for systems they can't inspect
- Adversarial yes/no doesn't make it safe, but turns binary closure into an *accountable step*

---

**Slide 13 — Takeaways**

- Biomedical QA agents should be framed as **evidence mediators, not medical authorities**
- Technical finding: a domain-governed loop improves recall-sensitive and exact-answer regimes over static and generic agentic baselines
- Sociotechnical finding: the value is making evidence labor **visible and governable**
- Design principle: **accountable hand-off, not human replacement** — provenance, search path, sufficiency flags, calibrated abstention

---

**Diagram — 8-stage agentic loop** (paste into a slide tool, or I can render it as an image)

```
                    ┌─────────────┐
                    │   QUESTION  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  1. THINK   │  classify type · detect multi-hop
                    │             │  decompose → 2–4 sub-questions
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
              ┌────▶│  2. SEARCH  │  hybrid dense (PubMedBERT/FAISS)
              │     │             │  + sparse (BM25), RRF 0.6/0.4
              │     └──────┬──────┘
              │            ▼
              │     ┌─────────────┐
              │     │ 3. EVALUATE │  sufficient?
              │ no  │             │
              └─────┤  (≤4 loops) │
                    └──────┬──────┘
                           │ yes
                           ▼
                    ┌─────────────┐
                    │  4. RERANK  │  biomedical cross-encoder
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  5. ANSWER  │  type-specific prompting
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  6. VERIFY  │  cut unsupported claims
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ 7. LIST     │  second extraction pass
                    │    REVIEW   │  (list questions only)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ 8. CONSENSUS│  3 regenerations, merged by type
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ ANSWER +    │
                    │ EVIDENCE    │
                    │ TRAIL       │
                    └─────────────┘
```

Want me to render that diagram as a clean image (PNG/SVG), or turn the whole thing into an actual .pptx file?
