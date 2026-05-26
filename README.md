Here's everything in one block, ordered as it appears in the paper — §3.1 addition first, then the full §4 Findings rewrite.

---

## Add to the end of §3.1, right before §3.2 begins:

```latex
We evaluate four variants of this pipeline that differ only in
the retrieval substrate or the answer-generation prompt, holding
the eight-stage control loop fixed. The \textbf{hybrid
FAISS+BM25+IVF} configuration is the system described above. The
\textbf{SQLite FTS5} variant replaces FAISS dense retrieval and
the cross-encoder reranker with SQLite's built-in BM25-based
full-text search index over the same PubMed corpus, isolating the
contribution of dense retrieval and reranking against an
otherwise identical agentic loop. The \textbf{SQLite FTS5 with
CoT prompt} variant uses the same FTS5 backend but replaces the
type-specific answer prompts with a single chain-of-thought
prompt, isolating the contribution of answer-type-specific
generation against an otherwise identical agent. Together the
three agentic rows in Table~1 separate the contribution of the
control loop (visible against zero-shot Gemma), the retrieval
substrate (FAISS+BM25 vs.\ FTS5), and type-specific generation
(type-specific vs.\ CoT). The \textbf{Zero-shot Gemma 4 31B} row
in Table~1 runs the same Gemma 4 31B controller without any
agentic loop and serves as the within-model anchor.
```

---

## Replace §4 Findings in full with:

```latex
\section{Findings}

Table~1 shows four patterns. First, the strongest performance
comes from a domain-governed agent, not from a generic ReAct loop
or optimized static RAG. The hybrid FAISS+BM25+IVF agent reaches
76.0 overall, ahead of the SQLite FTS5 agent (71.5) and the best
DSPy baseline, DSPy RAG + MIPROv2 (45.5), by 30.5 points.
Crucially, on the same Gemma 4 31B backbone the agentic loop adds
40.3 overall points over zero-shot generation
($35.7 \rightarrow 76.0$), isolating the contribution of control
policy from any model-class effect.

Second, the improvement is concentrated in recall-sensitive
regimes. On list questions, the hybrid agent reaches 96.0, the
SQLite FTS5 agent reaches 87.0, and every DSPy variant lands
between 0.0 and 13.0, with three of the six DSPy systems scoring
0.0 outright. On factoid, the hybrid agent reaches 69.0 against a
best-DSPy of 31.0 (ReAct RAG + MIPROv2). The SQLite FTS5
ablation, which removes dense retrieval and cross-encoder
reranking but keeps the agentic loop, still reaches 62.0 on
factoid and 87.0 on list, showing that the loop itself carries
most of the lift over DSPy; the dense+reranker stack adds a
further 7 and 9 points respectively. The CoT-prompt ablation
(SQLite FTS5, CoT) drops to 50.0 on factoid and 70.0 on list,
indicating that type-specific generation contributes roughly 12
and 17 points on top of the loop alone. List and factoid are
the regimes that punish a single missed entity, and they are
exactly where iterative query reformulation,
evidence-sufficiency checks, and a second-pass list review
should help.

Third, prompt and program optimization improves benchmark
behavior on the regimes it can reach, but does not address
evidence completeness. MIPROv2 lifts DSPy RAG factoid from 12.0
to 27.0 and yes/no from 82.0 to 94.0; DSPy ReAct RAG + MIPROv2
reaches 94.0 on yes/no and 55.0 on SemanticF1. Both numbers are
competitive on those columns, yet list F1 for optimized ReAct
remains at 0.0 and optimized RAG only reaches 13.0. Optimization
can align answer style and binary outputs to benchmark
expectations without changing whether the system has actually
retrieved the full evidence set behind a question.

Fourth, and most consequential for the discussion that follows,
yes/no and summary scores can be misleading on their own. The
zero-shot Gemma 4 31B baseline---a system that performs no
retrieval whatsoever---reaches 94.0 on yes/no and 41.63 on
SemanticF1 while scoring 8.0 on factoid and 0.0 on list. The two
regimes where a system can perform well without doing any
evidence work are the same two regimes where the output gives
the reader nothing concrete to verify. We return to this
asymmetry in \S5.2 as the empirical basis for the moral
compression argument and in \S5.3 as the basis for the
accountability argument.
```

---

Two ablation numbers worth flagging because I derived them from your table — verify against your records before submitting:

- **FTS5 vs. hybrid lift:** I claimed dense+reranker adds 7 points on factoid (62 → 69) and 9 on list (87 → 96). Direct subtraction from your table.
- **CoT vs. type-specific lift:** I claimed type-specific generation adds 12 on factoid (50 → 62) and 17 on list (70 → 87). Also direct subtraction.

Both are arithmetic on the rows you submitted, so they should be safe, but worth a quick eye-pass before the deadline.

The §3.1 paragraph is roughly 8 lines and §4 is about 4 lines longer than the current version. If you're tight on space after this, the simplest cut is the final sentence of §4 paragraph 4 (the *"We return to this asymmetry…"* forward-reference) — losing it costs nothing because §5 is right there on the next page anyway.
