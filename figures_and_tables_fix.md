# Figures and tables for v5

## Figures — where to place them and how to reference them

### Figure 1 — System architecture

**Where:** Insert at the top of §3.1 (Overview), before the existing "All three submitted systems share a common pipeline..." sentence.

**Intro prose** (replace the first sentence of §3.1 with this):

```latex
Figure~\ref{fig:arch} shows the shared pipeline. A single LLM instance plays two roles. As the \emph{controller}, it receives the question, decides what PubMed query to issue, inspects the returned evidence pool, and decides whether to issue another query or stop. As the \emph{generator}, it consumes the final evidence pool and produces a type-specific answer. PubMed itself is treated strictly as a data tool: every decision about ranking, fusion, deduplication, and chunk selection is made client-side, so the LLM can override any retrieval behavior through its query reformulations. This is the operative meaning of ``agentic'' in our system---not autonomy over arbitrary actions, but discretion over the retrieval trajectory.
```

The remainder of §3.1 (about retrieval, RRF, type-specific prompts, format hygiene) stays as-is.

---

### Figure 2 — Agentic control loop

**Where:** Insert at the top of §3.2 (Agentic Retrieval Loop), before the existing "The controller is the same LLM instance..." paragraph.

**Intro prose** (insert as the first paragraph of §3.2):

```latex
Figure~\ref{fig:loop} shows how the controller, the retrieval stack, and the generator are composed into a single iterative loop. The controller receives the question, the BioASQ type label, and the current evidence pool, and emits a structured JSON tool call selecting one of three actions: \texttt{search} with a reformulated query, \texttt{sufficient} (terminate and proceed to generation), or \texttt{stop} (give up without an answer). The pool $E$ is a set of already-seen chunks, deduplicated by document PMID and chunk offset, so the controller cannot trivially repeat a query and inflate its evidence count. The loop terminates either when the controller declares the pool sufficient (or stops), or after a hard cap of $N{=}5$ iterations.
```

Then the rest of §3.2 follows.

---

### Figure 3 — The phase gap

**Where:** Insert in §5.2 (Where We Stand), after the first paragraph (the one ending "...0.750 yes/no against a field leader of 0.938.").

**Intro prose** (insert as a new paragraph immediately after that first paragraph):

```latex
Figure~\ref{fig:phasegap} illustrates the gap on \texttt{asmalltrialsystem}: the same generator scores far higher when handed gold snippets (Phase B batch~1) than when it must retrieve its own evidence (Phase A$^+$ batch~2). The gap is largest on list F-measure, where every missed gold entity directly costs recall, and smallest on yes/no, where a binary decision can be reached from partial evidence. This ordering---list more retrieval-sensitive than factoid, factoid more retrieval-sensitive than yes/no---is the most internally consistent pattern in our results and is exactly what a retrieval-bottleneck reading predicts.
```

The "Phase B--Phase A$^+$ gap is large" paragraph that currently exists in §5.2 can follow this, or you can merge them.

---

### Quick label crosswalk

If your original paper used different labels, swap them in. Typical labels:

| Figure | Suggested label |
|---|---|
| Architecture | `\label{fig:arch}` |
| Control loop | `\label{fig:loop}` |
| Phase gap bar chart | `\label{fig:phasegap}` |

---

## Fixed tables — drop-in replacements

The "runny" tables in v5 were caused by too many columns at uniform width with no grouped headers. Here are clean replacements that match your original paper's compact style with grouped column headers via `\cmidrule`. All four use `table*` to span both columns.

### Table 1 — Systems comparison (transposed, like your original)

Replace the existing Table 1 block with:

```latex
\begin{table*}[t]
\centering
\small
\caption{System comparison. PMB = PubMedBERT.}
\label{tab:systems}
\begin{tabular}{lccc}
\toprule
                   & \texttt{asmalltrialsystem} & \texttt{ossllm} & \texttt{Finalcorrected} \\
\midrule
LLM                & Gemma 3 27B   & Gemma 3 27B   & Gemma 4 31B Dense \\
Backend            & E-utils       & E-utils       & SQLite FTS5 \\
Re-rank            & PMB+BM25+RRF  & PMB+BM25+RRF  & PMB+FTS5+RRF \\
Few-shot           & no            & yes (3-shot)  & no \\
Phase A$^+$ batches & 2, 3, 4      & 2, 3, 4       & 4 \\
Phase B batches    & 1             & 1             & 3, 4 \\
\bottomrule
\end{tabular}
\end{table*}
```

### Table 2 — Phase A+ exact-answer scores

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
2 & \texttt{asmalltrialsystem} & 0.762 & 0.760 & 0.150 & 0.150 & 0.095 \\
3 & \texttt{asmalltrialsystem} & 0.455 & 0.450 & ---   & ---   & 0.201 \\
4 & \texttt{asmalltrialsystem} & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
4 & \texttt{Finalcorrected}    & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
\bottomrule
\end{tabular}
\end{table*}
```

### Table 3 — Phase B exact-answer scores

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
1 & \texttt{asmalltrialsystem} & 0.941 & 0.938 & 0.304 & 0.370 & 0.317 & 0.179 \\
1 & \texttt{ossllm}            & 0.941 & 0.938 & 0.304 & 0.370 & 0.317 & 0.179 \\
3 & \texttt{Finalcorrected}    & 0.813 & 0.806 & 0.364 & 0.364 & 0.365 & 0.158 \\
4 & \texttt{Finalcorrected}    & 0.813 & 0.806 & 0.364 & 0.364 & 0.365 & 0.087 \\
\bottomrule
\end{tabular}
\end{table*}
```

### Table 4 — Position in the field

```latex
\begin{table*}[t]
\centering
\small
\caption{Position in the Task 14b field. Field-leader numbers are approximate, read from the public leaderboard at the time of writing; positions are bucketed because the leaderboard contains many duplicate-score systems and small per-batch sample sizes.}
\label{tab:position}
\begin{tabular}{lccl}
\toprule
\textbf{Metric (best batch)}             & \textbf{Our best} & \textbf{Field leader} & \textbf{Approx.\ position} \\
\midrule
Phase B yes/no accuracy (b1)             & 0.941 & 1.000        & Upper-middle quartile \\
Phase B list F-measure (b3)              & 0.365 & $\sim$0.50   & Middle of field \\
Phase B factoid strict (b3)              & 0.364 & $\sim$0.53   & Middle of field \\
Phase B ideal-answer ROUGE-2 F1 (b1)     & 0.179 & $\sim$0.25   & Middle of field \\
Phase A$^+$ list F-measure (b4)          & 0.255 & $\sim$0.58   & Lower-middle \\
Phase A$^+$ yes/no accuracy (b4)         & 0.750 & 0.938        & Lower-middle \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## What I changed in the tables vs v5

- **Table 1** is transposed: systems are columns, attributes are rows. Matches your original layout and reads much more compactly.
- **Tables 2 and 3** now use grouped column headers (`Y/N` spans `Acc.` and `Macro-F1`; `Factoid` spans `Strict` and `MRR`) via `\multicolumn` and `\cmidrule`. This is what made your original tables look clean and prevents the "runny" effect when all columns get equal width.
- **Batch column is now first** in Tables 2 and 3, matching your original. System name reads cleaner as the second column.
- **`\caption` is placed above the tabular** (per booktabs convention) rather than below — matches academic style.

If your two-column layout is unusually narrow and the Phase B table (8 columns) still overflows, the quickest fix is to wrap the `\begin{tabular}...\end{tabular}` block in `\resizebox{\textwidth}{!}{...}`. Try as-is first.
