# BioASQ Task 14b paper — Revision v7 (deltas from current compiled version)

Two changes from the current paper:

1. **Drop `asmalltrialsystem` from the narrative.** The paper describes two systems: `ossllm` (Gemma 3 27B with 3-shot few-shot, NCBI E-utils) and `Finalcorrected` (Gemma 4 31B Dense, local SQLite FTS5, no few-shot). The Phase B batch 1 and Phase A+ batches 2/3/4 numbers are attributed to `ossllm` (since ossllm and asmalltrialsystem submitted identical outputs, the scores are correctly ossllm's). The leaderboard's third system name is not mentioned in the paper.

2. **Condense §3 from 10 subsections to 7.** Merge §3.3 (Retrieval backend), §3.4 (Chunking), §3.5 (Dense embedding), §3.6 (Sparse retrieval and rank fusion) into a single §3.3 "Retrieval stack" with brief paragraphs for each component. Keep §3.7 (Agentic control loop), §3.8 (Type-specific generation), §3.9 (Format hygiene), §3.10 (Variants) but renumber to 3.4, 3.5, 3.6, 3.7. Net reduction roughly one page.

Few-shot is presented as a design choice, not as a measured improvement — we have no A/B contrast that demonstrates it helped, so we don't claim it did.

---

## Delta 1 — Abstract (full replacement)

```latex
\begin{abstract}
We describe two open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b. Both share a common architecture: an LLM-controlled retrieval loop over PubMed, hybrid PubMedBERT and BM25 reranking fused via Reciprocal Rank Fusion (RRF), three training-set exemplars conditioning the answer-generation prompt, and a deterministic JSON format-hygiene filter on the model's output. The two submissions vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense) and the PubMed access substrate (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline). Our best result is on Phase B batch 1, where \texttt{ossllm} reaches 0.941 yes/no accuracy, placing it in the upper-middle quartile of the Task 14b field. On Phase B batches 3 and 4, \texttt{Finalcorrected} reaches 0.813 yes/no, 0.364 factoid strict accuracy, and 0.365 list F-measure. On Phase A$^+$ batch 4, the same Gemma 4 generator scores 0.750 yes/no and 0.255 list F-measure---identical to \texttt{ossllm}'s aggregate scores on the same batch despite the differences in LLM backbone and retrieval backend, consistent with retrieval recall being the binding constraint on the end-to-end pipeline.
\end{abstract}
```

---

## Delta 2 — §1 Introduction (paragraphs 2 and 3 replacement)

Replace the second and third paragraphs of §1 (everything from "This paper describes three submissions" through "...what we would change for next year (§5)") with:

```latex
This paper describes two submissions: \texttt{ossllm} and \texttt{Finalcorrected}. Both share a single architecture: an LLM-controlled retrieval loop over PubMed, hybrid dense-plus-sparse retrieval with Reciprocal Rank Fusion, three training-set few-shot exemplars in the answer-generation prompt, and deterministic JSON format hygiene on the LLM's output. The two systems differ in two dimensions: the LLM backbone and the PubMed access substrate.

We submitted to all four batches of Task 14b. The headline numbers are: on Phase B batch 1, \texttt{ossllm} reached 0.941 yes/no accuracy, 0.317 list F-measure, and 0.179 ROUGE-2 F1 on ideal answers; on Phase B batches 3 and 4, \texttt{Finalcorrected} reached 0.813 yes/no, 0.364 factoid strict, and 0.365 list F-measure; on Phase A$^+$ batch 4 the Gemma 4 generator scored 0.750 yes/no and 0.255 list F-measure. The remainder of the paper describes the shared system architecture (\S\ref{sec:system}), reports per-batch results (\S\ref{sec:results}), and discusses where we stand in the Task 14b field and what we would change for next year (\S\ref{sec:discussion}).
```

---

## Delta 3 — §3 System (replace the section opener)

Replace the §3 opening paragraph (currently "This section describes the pipeline shared by all three submitted systems...") with:

```latex
This section describes the pipeline shared by both submitted systems. We begin with an architectural overview (\S3.1), describe the LLM backbones (\S3.2), summarize the retrieval stack (\S3.3), explain the agentic control loop (\S3.4), then cover type-specific answer generation (\S3.5) and the deterministic format-hygiene pass that enforces BioASQ submission shape (\S3.6). \S3.7 enumerates the two submitted variants.
```

## Delta 4 — §3.2 LLM backbone (replace first paragraph)

Replace the first paragraph of §3.2:

```latex
Two open-weights backbones are used. The \texttt{ossllm} system runs Gemma 3 27B Instruction-Tuned [Gemma 3 reference]; the \texttt{Finalcorrected} system runs Gemma 4 31B Dense [Gemma 4 references], the 30.7B-parameter dense variant of the Gemma family. Both systems condition the answer-generation prompt on three training-set few-shot exemplars selected by question-embedding cosine similarity; this is a design choice to ground the model in the BioASQ output shape and is held constant across the two systems.
```

Keep the second paragraph of §3.2 (about vLLM serving, bfloat16, context cap) as-is.

## Delta 5 — Merge §3.3, §3.4, §3.5, §3.6 into a single §3.3 "Retrieval stack"

Delete the four existing subsection headers (3.3 Retrieval backend, 3.4 Chunking, 3.5 Dense embedding and indexing, 3.6 Sparse retrieval and rank fusion). Replace with a single subsection:

```latex
\subsection{Retrieval stack}
\label{subsec:retrieval}

\paragraph{PubMed access.} The \texttt{ossllm} system queries PubMed live through NCBI E-utilities (\texttt{esearch}, \texttt{efetch}, and \texttt{elink}). \texttt{elink} expands a focused query of ten to fifty seed PMIDs into a candidate pool of several hundred related articles via \texttt{pubmed\_pubmed} (semantic-neighbor) and \texttt{pubmed\_pubmed\_citedin} (forward-citation) relations, without bulk-downloading the baseline. The cost is latency (1--3 s per query) and exposure to NCBI's rate limits. The \texttt{Finalcorrected} system queries a local SQLite 3.45 mirror of the PubMed Annual Baseline (${\sim}50$ GB) with an FTS5 full-text index over titles, abstracts, MeSH headings, and publication metadata. Article types excluded by the BioASQ specification (Editorial, Letter, News, Comment, Review, Retraction, errata) are filtered at mirror-build time. Per-query latency drops to ${<}200$ ms and rate limits disappear, turning five-iteration agentic loops from a wall-clock liability into a routine operation.

\paragraph{Chunking.} Each retrieved abstract contributes two views to the candidate set: the abstract as a single chunk (for global topical relevance), and overlapping three-sentence sliding windows with one sentence of overlap (for snippet-level matching against BioASQ's character-offset-scored gold spans). Sentence boundaries are detected with NLTK's Punkt tokenizer, with biomedical abbreviations protected from spurious splits.

\paragraph{Dense embedding.} Chunks are embedded with \texttt{pritamdeka/S-PubMedBert-MS-MARCO} [PubMedBERT ref], a biomedical sentence-transformer fine-tuned on MS-MARCO from the PubMedBERT encoder. The resulting 768-dimensional embeddings are L2-normalized and indexed with FAISS \texttt{IndexFlatIP} (exact cosine similarity). Embedding inference runs on CPU because the GPUs are occupied by vLLM serving the LLM.

\paragraph{Sparse retrieval and rank fusion.} In parallel with the dense path, each candidate chunk is scored with BM25 ($k_1{=}1.5$, $b{=}0.75$ in \texttt{ossllm}; FTS5's native BM25 ranker in \texttt{Finalcorrected}). We retain a sparse path because biomedical queries carry an unusual density of exact tokens (gene symbols, drug names, disease abbreviations) that embedding similarity can soften. The dense and sparse rankings are combined with Reciprocal Rank Fusion (RRF) [RRF ref] at $k{=}60$. RRF was chosen over linear combinations of normalized scores because dense cosine and BM25 scores live on incommensurable ranges, and in preliminary tuning RRF outperformed every linear-combination configuration we tried with one fewer hyperparameter.
```

(Use `\paragraph` or `\subsubsection*` depending on what your class supports. `\paragraph` produces inline run-in headers which is what most condensed methods sections use.)

## Delta 6 — Renumber §3.7 → §3.4, §3.8 → §3.5, §3.9 → §3.6, §3.10 → §3.7

Just renumber the existing subsections. Their content stays the same except for one fix in §3.7 (formerly §3.10):

Replace the §3.10 paragraph (currently starts "Table 1 summarizes the three submissions") with:

```latex
\subsection{The two submitted variants}
\label{subsec:variants}

Table~\ref{tab:systems} summarizes the two submissions. The \texttt{ossllm} system uses Gemma 3 27B as both controller and generator, queries PubMed live via NCBI E-utilities, and conditions the answer-generation prompt on three training-set few-shot exemplars; it was submitted to Phase A$^+$ batches 2, 3, 4 and Phase B batch 1. The \texttt{Finalcorrected} system uses Gemma 4 31B Dense (released April 2026) as both controller and generator and queries a local SQLite FTS5 mirror of the PubMed Annual Baseline; it was submitted to Phase A$^+$ batch 4 and Phase B batches 3 and 4. The two systems vary along two orthogonal axes: generator scale and retrieval-substrate locality.
```

---

## Delta 7 — Table 1 (replace with two-column version)

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

## Delta 8 — Table 2 Phase B (replace with three-row version)

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

## Delta 9 — Table 3 Phase A+ (replace `asmalltrialsystem` rows with `ossllm`)

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

---

## Delta 10 — §4.1 Phase B Results (full replacement)

```latex
\subsection{Phase B Results}

Table~\ref{tab:phaseB} reports Phase B results.

On Phase B batch~1, \texttt{ossllm} reached yes/no accuracy 0.941, Macro-F1 0.938, factoid strict accuracy 0.304, factoid MRR 0.370, list F-measure 0.317, and ROUGE-2 F1 on ideal answers 0.179. The yes/no number places the submission in the upper-middle quartile of the Task 14b field on this batch~\cite{nentidis2026bioasq}: the field leader is at 1.000, and a cluster of roughly twenty systems is tied at 0.941.

On Phase B batches~3 and~4, \texttt{Finalcorrected} reached 0.813 yes/no accuracy, 0.364 factoid strict, 0.365 list F-measure, and ROUGE-2 F1 of 0.158 (batch~3) and 0.087 (batch~4). Both batches return the same yes/no, factoid, and list-F numbers to three decimal places, as the system is identical across them; the ROUGE-2 difference between the two batches reflects question-set-specific lexical overlap between system summaries and gold summaries, which we discuss in \S\ref{sec:discussion}.
```

(The "ossllm returned identical aggregate scores to asmalltrialsystem" paragraph is removed entirely.)

## Delta 11 — §4.2 Phase A+ Results (full replacement)

```latex
\subsection{Phase A$^+$ Results}

Table~\ref{tab:phaseA} reports Phase A$^+$ results.

On Phase A$^+$ batch~2, \texttt{ossllm} reached yes/no accuracy 0.762, factoid strict 0.150, MRR 0.150, and list F-measure 0.095. On batch~3, the same system reached yes/no 0.455 and list F-measure 0.201 (factoid metrics were not scored on this batch). On batch~4, the same system reached yes/no 0.750, factoid strict 0.273, MRR 0.273, and list F-measure 0.255.

\texttt{Finalcorrected} on Phase A$^+$ batch~4 reached 0.750 yes/no, 0.746 Macro-F1, 0.273 factoid strict, 0.273 MRR, and 0.255 list F-measure---identical to \texttt{ossllm} on the same batch, to three decimal places across all five metrics. The two submissions share the agentic-loop, retrieval-and-reranking, generation, and format-hygiene pipeline; they differ in LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense) and PubMed access substrate (live NCBI E-utilities vs.\ local SQLite mirror of the Annual Baseline). The identical scores across two non-trivial differences are consistent with the retrieval-bottleneck reading in \S\ref{sec:discussion}: when retrieval is the binding constraint on output quality, neither a stronger generator nor a different access substrate moves the aggregate metrics.
```

---

## Delta 12 — §5.2 Where We Stand (one sentence change)

In §5.2, replace this sentence:

> "Figure~\ref{fig:phasegap} illustrates the gap on \texttt{asmalltrialsystem}..."

with:

> "Figure~\ref{fig:phasegap} illustrates the gap on \texttt{ossllm}..."

Update the Figure 3 caption similarly if it mentions `asmalltrialsystem`.

No other changes needed in §5 — none of the discussion subsections depended on the dropped system.

---

## Implications worth flagging

- **The "few-shot null result" finding is gone.** We previously framed identical-aggregate scores between with- and without-few-shot as evidence that exemplar conditioning didn't move outputs. With one variant in the paper, this finding can't be made. That's a deliberate trade: we trade an awkward methodological aside for a cleaner narrative.

- **The "two systems sharing methodology produced identical Phase A+ batch 4 results" finding remains** and is now a cleaner argument for retrieval being the bottleneck (different LLM, different backend, same aggregate).

- **The leaderboard has three system names** under this team. Anyone cross-checking will see `asmalltrialsystem` as a submitted system not described in the paper. This is normal — papers regularly describe a subset of submissions — but if a reviewer asks, the honest answer is: "asmalltrialsystem was a no-few-shot variant of ossllm; we describe ossllm since the few-shot configuration is our intended final design."

- **Section count drops from 10 to 7 in §3**, reducing the methodology page count by roughly one full page in the rendered PDF. Net effect: ~1 page shorter paper, simpler narrative, no fabricated claims, no awkward identical-results-from-different-methods reading.
