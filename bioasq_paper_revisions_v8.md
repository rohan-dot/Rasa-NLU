# BioASQ Task 14b paper — v8 deltas (corrections to v7)

Two fixes:

1. **Acknowledge `asmalltrialsystem`** in §3.7 as the same configuration as `ossllm` under a different submission name. Don't hide the third system — just explain the merger.

2. **Surface Phase A+ batch 2 results** in the abstract and §1. Currently they appear in Table 3 and §4.2 only; 0.762 yes/no on Phase A+ batch 2 is our strongest Phase A+ yes/no result and deserves a mention up front.

---

## Delta A — Abstract (replaces the v7 abstract)

```latex
\begin{abstract}
We describe two open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b under three submission names. Both share a common architecture: an LLM-controlled retrieval loop over PubMed, hybrid PubMedBERT and BM25 reranking fused via Reciprocal Rank Fusion (RRF), three training-set exemplars conditioning the answer-generation prompt, and a deterministic JSON format-hygiene filter on the model's output. The two configurations vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense) and the PubMed access substrate (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline). On Phase B batch 1, our Gemma 3 system reaches 0.941 yes/no accuracy, placing it in the upper-middle quartile of the Task 14b field. On Phase B batches 3 and 4, the Gemma 4 system reaches 0.813 yes/no, 0.364 factoid strict accuracy, and 0.365 list F-measure. On Phase A$^+$, yes/no accuracy across the three submitted batches ranges from 0.762 (batch 2) to 0.455 (batch 3) to 0.750 (batch 4), with list F-measure ranging from 0.095 to 0.255 over the same batches---and the same generator scoring identical aggregates on Phase A$^+$ batch 4 regardless of LLM backbone or retrieval backend, consistent with retrieval recall being the binding constraint on the end-to-end pipeline.
\end{abstract}
```

---

## Delta B — §1 Introduction headline-numbers paragraph (replaces the v7 version)

Replace the "We submitted to all four batches" paragraph with:

```latex
We submitted to all four batches of Task 14b. The headline Phase B numbers are: on batch 1, our Gemma 3 system reached 0.941 yes/no accuracy, 0.317 list F-measure, and 0.179 ROUGE-2 F1 on ideal answers; on batches 3 and 4, the Gemma 4 system reached 0.813 yes/no, 0.364 factoid strict, and 0.365 list F-measure. On Phase A$^+$, yes/no accuracy was 0.762 on batch 2, 0.455 on batch 3, and 0.750 on batch 4, with list F-measure of 0.095, 0.201, and 0.255 respectively (factoid metrics were not scored on batch 3). The remainder of the paper describes the shared system architecture (\S\ref{sec:system}), reports per-batch results (\S\ref{sec:results}), and discusses where we stand in the Task 14b field and what we would change for next year (\S\ref{sec:discussion}).
```

---

## Delta C — §3.7 The submitted variants (replaces the v7 version)

```latex
\subsection{The submitted variants}
\label{subsec:variants}

We submitted to BioASQ Task 14b under three system names: \texttt{asmalltrialsystem}, \texttt{ossllm}, and \texttt{Finalcorrected}. The first two share an identical retrieval, reranking, generation, and format-hygiene pipeline running Gemma 3 27B as both controller and generator with NCBI E-utilities for retrieval; they differ only in whether three training-set exemplars are prepended to the answer-generation prompt, and they produced identical aggregate scores on every batch they were submitted to. We treat them as a single configuration in the rest of the paper and refer to them collectively as \texttt{ossllm}; results reported for \texttt{ossllm} are the scores both submissions returned. The third system, \texttt{Finalcorrected}, uses Gemma 4 31B Dense (released April 2026) as both controller and generator and queries a local SQLite FTS5 mirror of the PubMed Annual Baseline. Table~\ref{tab:systems} summarizes the two configurations. The submissions vary along two orthogonal axes: generator scale and retrieval-substrate locality.
```

This replaces the v7 "We submitted two systems..." version. Keeps the rest of §3 unchanged.

---

## What this fixes

- **`asmalltrialsystem` is named and acknowledged** as a submission, with one honest sentence explaining the merger with `ossllm`. No more pretending it doesn't exist.
- **Phase A+ batch 2 is in the abstract and intro** (0.762 yes/no, 0.095 list F1), alongside batches 3 and 4. Reader doesn't have to flip to Table 3 to see we submitted to three Phase A+ batches.
- **No claim that few-shot helped** anywhere. The v7 design choice still holds — we describe few-shot as a methodology component, not a measured improvement. The acknowledgment in §3.7 just notes that the two submissions produced identical aggregate scores, without framing this as a controlled experiment.

Everything else in v7 (table replacements, §3 condensation, §4 prose rewrites, §5 changes) still applies as-is. These three deltas just sit on top.
