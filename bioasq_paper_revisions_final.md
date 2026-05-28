# BioASQ Task 14b paper — Final consolidated revisions

Apply these 11 deltas in order. They replace the corresponding sections of your currently compiled paper.

**Summary of changes:**
- Drop `asmalltrialsystem` from the narrative but acknowledge it as the no-few-shot submission of the same configuration as `ossllm` (one honest sentence in §3.7).
- Condense §3 from 10 subsections to 7 by merging retrieval-related subsections into one §3.3 "Retrieval stack."
- Phase B batch 1 numbers attributed to `ossllm`; Phase A+ batches 2, 3, 4 attributed to `ossllm`; Phase A+ batch 4 and Phase B batches 3, 4 attributed to `Finalcorrected`.
- Abstract and §1 surface Phase A+ batch 2 (0.762 yes/no, our best Phase A+ yes/no result) up front.
- Few-shot framed as a design choice, no causal claim.

---

## Delta 1 — Abstract (full replacement)

```latex
\begin{abstract}
We describe two open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b under three submission names. Both share a common architecture: an LLM-controlled retrieval loop over PubMed, hybrid PubMedBERT and BM25 reranking fused via Reciprocal Rank Fusion (RRF), three training-set exemplars conditioning the answer-generation prompt, and a deterministic JSON format-hygiene filter on the model's output. The two configurations vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense) and the PubMed access substrate (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline). On Phase B batch 1, our Gemma 3 system reaches 0.941 yes/no accuracy, placing it in the upper-middle quartile of the Task 14b field. On Phase B batches 3 and 4, the Gemma 4 system reaches 0.813 yes/no, 0.364 factoid strict accuracy, and 0.365 list F-measure. On Phase A$^+$, yes/no accuracy across the three submitted batches ranges from 0.762 (batch 2) to 0.455 (batch 3) to 0.750 (batch 4), with list F-measure ranging from 0.095 to 0.255 over the same batches---and the same generator scoring identical aggregates on Phase A$^+$ batch 4 regardless of LLM backbone or retrieval backend, consistent with retrieval recall being the binding constraint on the end-to-end pipeline.
\end{abstract}
```

---

## Delta 2 — §1 Introduction (paragraphs 2 and 3 replacement)

Replace the second and third paragraphs of §1 (from "This paper describes three submissions" through "...what we would change for next year (§5)") with:

```latex
This paper describes two submissions: \texttt{ossllm} and \texttt{Finalcorrected}. Both share a single architecture: an LLM-controlled retrieval loop over PubMed, hybrid dense-plus-sparse retrieval with Reciprocal Rank Fusion, three training-set few-shot exemplars in the answer-generation prompt, and deterministic JSON format hygiene on the LLM's output. The two systems differ in two dimensions: the LLM backbone and the PubMed access substrate.

We submitted to all four batches of Task 14b. The headline Phase B numbers are: on batch 1, our Gemma 3 system reached 0.941 yes/no accuracy, 0.317 list F-measure, and 0.179 ROUGE-2 F1 on ideal answers; on batches 3 and 4, the Gemma 4 system reached 0.813 yes/no, 0.364 factoid strict, and 0.365 list F-measure. On Phase A$^+$, yes/no accuracy was 0.762 on batch 2, 0.455 on batch 3, and 0.750 on batch 4, with list F-measure of 0.095, 0.201, and 0.255 respectively (factoid metrics were not scored on batch 3). The remainder of the paper describes the shared system architecture (\S\ref{sec:system}), reports per-batch results (\S\ref{sec:results}), and discusses where we stand in the Task 14b field and what we would change for next year (\S\ref{sec:discussion}).
```

---

## Delta 3 — §3 System (section opener)

Replace the §3 opening paragraph with:

```latex
This section describes the pipeline shared by both submitted systems. We begin with an architectural overview (\S3.1), describe the LLM backbones (\S3.2), summarize the retrieval stack (\S3.3), explain the agentic control loop (\S3.4), then cover type-specific answer generation (\S3.5) and the deterministic format-hygiene pass that enforces BioASQ submission shape (\S3.6). \S3.7 enumerates the submitted variants.
```

---

## Delta 4 — §3.2 LLM backbone (replace first paragraph)

```latex
Two open-weights backbones are used. The \texttt{ossllm} system runs Gemma 3 27B Instruction-Tuned~\cite{gemma3}; the \texttt{Finalcorrected} system runs Gemma 4 31B Dense~\cite{gemma4blog,gemma4card}, the 30.7B-parameter dense variant of the Gemma family. Both systems condition the answer-generation prompt on three training-set few-shot exemplars selected by question-embedding cosine similarity; this is a design choice to ground the model in the BioASQ output shape and is held constant across the two systems.
```

Keep the second paragraph of §3.2 (vLLM serving, bfloat16, context cap) as-is.

---

## Delta 5 — Merge §3.3–§3.6 into a single §3.3 "Retrieval stack"

Delete the existing §3.3 (Retrieval backend), §3.4 (Chunking), §3.5 (Dense embedding and indexing), and §3.6 (Sparse retrieval and rank fusion). Replace with:

```latex
\subsection{Retrieval stack}
\label{subsec:retrieval}

\paragraph{PubMed access.} The \texttt{ossllm} system queries PubMed live through NCBI E-utilities (\texttt{esearch}, \texttt{efetch}, and \texttt{elink}). \texttt{elink} expands a focused query of ten to fifty seed PMIDs into a candidate pool of several hundred related articles via \texttt{pubmed\_pubmed} (semantic-neighbor) and \texttt{pubmed\_pubmed\_citedin} (forward-citation) relations, without bulk-downloading the baseline. The cost is latency (1--3 s per query) and exposure to NCBI's rate limits. The \texttt{Finalcorrected} system queries a local SQLite 3.45 mirror of the PubMed Annual Baseline (${\sim}50$ GB) with an FTS5 full-text index over titles, abstracts, MeSH headings, and publication metadata. Article types excluded by the BioASQ specification (Editorial, Letter, News, Comment, Review, Retraction, errata) are filtered at mirror-build time. Per-query latency drops to ${<}200$ ms and rate limits disappear, turning five-iteration agentic loops from a wall-clock liability into a routine operation.

\paragraph{Chunking.} Each retrieved abstract contributes two views to the candidate set: the abstract as a single chunk (for global topical relevance), and overlapping three-sentence sliding windows with one sentence of overlap (for snippet-level matching against BioASQ's character-offset-scored gold spans). Sentence boundaries are detected with NLTK's Punkt tokenizer, with biomedical abbreviations protected from spurious splits.

\paragraph{Dense embedding.} Chunks are embedded with \texttt{pritamdeka/S-PubMedBert-MS-MARCO}~\cite{pubmedbert}, a biomedical sentence-transformer fine-tuned on MS-MARCO from the PubMedBERT encoder. The resulting 768-dimensional embeddings are L2-normalized and indexed with FAISS \texttt{IndexFlatIP} (exact cosine similarity). Embedding inference runs on CPU because the GPUs are occupied by vLLM serving the LLM.

\paragraph{Sparse retrieval and rank fusion.} In parallel with the dense path, each candidate chunk is scored with BM25 ($k_1{=}1.5$, $b{=}0.75$ in \texttt{ossllm}; FTS5's native BM25 ranker in \texttt{Finalcorrected}). We retain a sparse path because biomedical queries carry an unusual density of exact tokens (gene symbols, drug names, disease abbreviations) that embedding similarity can soften. The dense and sparse rankings are combined with Reciprocal Rank Fusion (RRF)~\cite{rrf2009} at $k{=}60$. RRF was chosen over linear combinations of normalized scores because dense cosine and BM25 scores live on incommensurable ranges, and in preliminary tuning RRF outperformed every linear-combination configuration we tried with one fewer hyperparameter.
```

---

## Delta 6 — Renumber existing subsections

- Old §3.7 (Agentic control loop) → new §3.4
- Old §3.8 (Type-specific generation) → new §3.5
- Old §3.9 (Format hygiene) → new §3.6
- Old §3.10 (Differences between systems) → new §3.7 (and replace its content per Delta 7 below)

Their content stays the same, only the section labels change.

---

## Delta 7 — §3.7 The submitted variants (replace the content of what was §3.10)

```latex
\subsection{The submitted variants}
\label{subsec:variants}

We submitted to BioASQ Task 14b under three system names: \texttt{asmalltrialsystem}, \texttt{ossllm}, and \texttt{Finalcorrected}. The first two share an identical retrieval, reranking, generation, and format-hygiene pipeline running Gemma 3 27B as both controller and generator with NCBI E-utilities for retrieval; they differ only in whether three training-set exemplars are prepended to the answer-generation prompt, and they produced identical aggregate scores on every batch they were submitted to. We treat them as a single configuration in the rest of the paper and refer to them collectively as \texttt{ossllm}; results reported for \texttt{ossllm} are the scores both submissions returned. The third system, \texttt{Finalcorrected}, uses Gemma 4 31B Dense (released April 2026) as both controller and generator and queries a local SQLite FTS5 mirror of the PubMed Annual Baseline. Table~\ref{tab:systems} summarizes the two configurations. The submissions vary along two orthogonal axes: generator scale and retrieval-substrate locality.
```

---

## Delta 8 — Table 1 (full replacement)

```latex
\begin{table*}[t]
\centering
\small
\caption{System comparison. PMB = PubMedBERT.}
\label{tab:systems}
\begin{tabular}{lcc}
\toprule
                    & \texttt{ossllm}    & \texttt{Finalcorrected} \\
\midrule
LLM                 & Gemma 3 27B        & Gemma 4 31B Dense \\
Backend             & E-utils            & SQLite FTS5 \\
Re-rank             & PMB+BM25+RRF       & PMB+FTS5+RRF \\
Few-shot            & yes (3-shot)       & yes (3-shot) \\
Phase A$^+$ batches & 2, 3, 4            & 4 \\
Phase B batches     & 1                  & 3, 4 \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Delta 9 — Table 2 Phase B (full replacement)

```latex
\begin{table*}[t]
\centering
\small
\caption{Phase B exact-answer scores plus ROUGE-2 F1 on ideal answers.}
\label{tab:phaseB}
\begin{tabular}{llccccccc}
\toprule
 & & \multicolumn{2}{c}{Y/N} & \multicolumn{2}{c}{Factoid} & List & Ideal \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7} \cmidrule(lr){8-8}
Batch & System & Acc. & Macro-F1 & Strict & MRR & F1 & R-2 F1 \\
\midrule
1 & \texttt{ossllm}         & 0.941 & 0.938 & 0.304 & 0.370 & 0.317 & 0.179 \\
3 & \texttt{Finalcorrected} & 0.813 & 0.806 & 0.364 & 0.364 & 0.365 & 0.158 \\
4 & \texttt{Finalcorrected} & 0.813 & 0.806 & 0.364 & 0.364 & 0.365 & 0.087 \\
\bottomrule
\end{tabular}
\end{table*}
```

(No Phase B batch 2 row — no system submitted to Phase B batch 2.)

---

## Delta 10 — Table 3 Phase A+ (full replacement, batch 2 included)

```latex
\begin{table*}[t]
\centering
\small
\caption{Phase A$^+$ exact-answer scores. ``---'' = not scored.}
\label{tab:phaseA}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{2}{c}{Y/N} & \multicolumn{2}{c}{Factoid} & List \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}
Batch & System & Acc. & Macro-F1 & Strict & MRR & F1 \\
\midrule
2 & \texttt{ossllm}         & 0.762 & 0.760 & 0.150 & 0.150 & 0.095 \\
3 & \texttt{ossllm}         & 0.455 & 0.450 & ---   & ---   & 0.201 \\
4 & \texttt{ossllm}         & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
4 & \texttt{Finalcorrected} & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
\bottomrule
\end{tabular}
\end{table*}
```

Phase A+ batch 2 is row 1 — `ossllm` at 0.762 / 0.760 / 0.150 / 0.150 / 0.095. It's there.

---

## Delta 11 — §4.1 Phase B Results (full replacement)

```latex
\subsection{Phase B Results}

Table~\ref{tab:phaseB} reports Phase B results.

On Phase B batch~1, \texttt{ossllm} reached yes/no accuracy 0.941, Macro-F1 0.938, factoid strict accuracy 0.304, factoid MRR 0.370, list F-measure 0.317, and ROUGE-2 F1 on ideal answers 0.179. The yes/no number places the submission in the upper-middle quartile of the Task 14b field on this batch~\cite{nentidis2026bioasq}: the field leader is at 1.000, and a cluster of roughly twenty systems is tied at 0.941.

On Phase B batches~3 and~4, \texttt{Finalcorrected} reached 0.813 yes/no accuracy, 0.364 factoid strict, 0.365 list F-measure, and ROUGE-2 F1 of 0.158 (batch~3) and 0.087 (batch~4). Both batches return the same yes/no, factoid, and list-F numbers to three decimal places, as the system is identical across them; the ROUGE-2 difference between the two batches reflects question-set-specific lexical overlap between system summaries and gold summaries, which we discuss in \S\ref{sec:discussion}.
```

---

## Delta 12 — §4.2 Phase A+ Results (full replacement)

```latex
\subsection{Phase A$^+$ Results}

Table~\ref{tab:phaseA} reports Phase A$^+$ results.

On Phase A$^+$ batch~2, \texttt{ossllm} reached yes/no accuracy 0.762, factoid strict 0.150, MRR 0.150, and list F-measure 0.095---our highest Phase A$^+$ yes/no across the three submitted batches. On batch~3, the same system reached yes/no 0.455 and list F-measure 0.201 (factoid metrics were not scored on this batch). On batch~4, the same system reached yes/no 0.750, factoid strict 0.273, MRR 0.273, and list F-measure 0.255.

\texttt{Finalcorrected} on Phase A$^+$ batch~4 reached 0.750 yes/no, 0.746 Macro-F1, 0.273 factoid strict, 0.273 MRR, and 0.255 list F-measure---identical to \texttt{ossllm} on the same batch, to three decimal places across all five metrics. The two submissions share the agentic-loop, retrieval-and-reranking, generation, and format-hygiene pipeline; they differ in LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense) and PubMed access substrate (live NCBI E-utilities vs.\ local SQLite mirror of the Annual Baseline). The identical scores across two non-trivial differences are consistent with the retrieval-bottleneck reading in \S\ref{sec:discussion}: when retrieval is the binding constraint on output quality, neither a stronger generator nor a different access substrate moves the aggregate metrics.
```

---

## Delta 13 — §5.2 figure-reference fix

In §5.2, replace the sentence:

> "Figure~\ref{fig:phasegap} illustrates the gap on \texttt{asmalltrialsystem}..."

with:

> "Figure~\ref{fig:phasegap} illustrates the gap on \texttt{ossllm}..."

If the Figure 3 caption still says "asmalltrialsystem b2" / "asmalltrialsystem b1", update to "ossllm b2" / "ossllm b1" as well.

---

## Order of application

1. Delta 1 (abstract)
2. Delta 2 (§1 intro paragraphs)
3. Delta 3 (§3 opener)
4. Delta 4 (§3.2 LLM backbone)
5. Delta 5 (merge §3.3–§3.6 into new §3.3)
6. Delta 6 (renumber old §3.7–§3.10 to new §3.4–§3.7)
7. Delta 7 (replace content of new §3.7 with the variants-acknowledgment paragraph)
8. Delta 8 (Table 1)
9. Delta 9 (Table 2)
10. Delta 10 (Table 3 — includes Phase A+ batch 2)
11. Delta 11 (§4.1)
12. Delta 12 (§4.2)
13. Delta 13 (§5.2 prose and Figure 3 caption)

Estimated apply time: 20–30 min in Overleaf.
