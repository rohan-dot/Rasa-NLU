\section{Findings}

Table~1 shows four patterns. First, the strongest performance
comes from a domain-governed agent. The hybrid FAISS+BM25+IVF
agent reaches 76.0 overall, ahead of the SQLite FTS5 agent (71.5)
and the best DSPy baseline (45.5) by 30.5 points. On the same
Gemma 4 31B backbone, the agentic loop adds 40.3 points over
zero-shot generation ($35.7 \rightarrow 76.0$), isolating control
policy from model-class effects.

Second, the gain is concentrated in recall-sensitive regimes. On
list, the hybrid agent reaches 96.0 while every DSPy variant
falls between 0.0 and 13.0, with three of six scoring 0.0. On
factoid, 69.0 against a best-DSPy of 31.0. The ablation rows
locate the active ingredient: removing dense retrieval and the
cross-encoder (SQLite FTS5) still reaches 62.0/87.0 on
factoid/list, while removing type-specific generation (FTS5 +
CoT) drops to 50.0/70.0. The agentic loop itself carries most of
the lift over DSPy; dense+reranker and type-specific prompting
add the remainder.

Third, optimization improves benchmark behavior on the regimes it
can reach but does not address evidence completeness. MIPROv2
lifts DSPy RAG factoid from 12.0 to 27.0 and yes/no from 82.0 to
94.0; ReAct RAG + MIPROv2 reaches 94.0 yes/no and 55.0
SemanticF1. Yet list F1 for optimized ReAct remains 0.0 and
optimized RAG only 13.0. Optimization aligns answer style and
binary outputs to benchmark expectations without retrieving the
evidence behind them.

Fourth, yes/no and summary scores can mislead on their own.
Zero-shot Gemma 4 31B---no retrieval at all---reaches 94.0 yes/no
and 41.63 SemanticF1 while scoring 8.0 factoid and 0.0 list. The
regimes where a system can score well without doing evidence
work are the same regimes where the output gives the reader
nothing concrete to verify. We return to this asymmetry in \S5.2
and \S5.3.
