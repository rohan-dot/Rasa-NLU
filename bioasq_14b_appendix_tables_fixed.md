# BioASQ Appendix A.1–A.5 — non-floating tables (drag-and-drop)

The appendix tables were `\begin{table}[h]` **floats**, so LaTeX drifted them down the
page and left the big gaps (A.2/A.3 headers with nothing under them). This replaces them
with non-floating `center` blocks that sit exactly where you type them. Captions become
bold inline labels.

**Replace your appendix region from §A.1 through the end of §A.5** (everything up to but
not including `\subsection{Prompts}`) with the block below.

**Check the table numbers first:** these are numbered 5–8, assuming your last body table
was Table 4. If your last numbered table before the appendix is different, adjust 5/6/7/8.

```latex
\section{Reproducibility Details}
\label{app:repro}

A public code repository accompanying this submission was not available within the
camera-ready window. This appendix documents the prompts, hyperparameters, and
configuration in full so that the system can be re-implemented without access to the
original source.

\subsection{Decoding settings}
\label{app:decoding}

All LLM calls are served by a local vLLM instance. Decoding parameters vary by call type
(Table~5). Each call retries up to three times with exponential backoff (3, 6, 9 seconds)
under a 180-second timeout.

\begin{center}
\small
\begin{tabular}{lrrr}
\toprule
Call type & \texttt{max\_tokens} & \texttt{temp.} & \texttt{top\_p} \\
\midrule
Query generation (controller)  & 200  & 0.3 & 0.95 \\
Sufficiency check (controller) & 20   & 0.1 & 0.95 \\
Relevance rating (LLM rerank)  & 256  & 0.1 & 0.95 \\
Answer generation (generator)  & 1024 & 0.3 & 0.95 \\
\bottomrule
\end{tabular}
\end{center}
\noindent\textbf{Table 5.} Per-call-type decoding settings.
\vspace{8pt}

\subsection{Model checkpoints}
\label{app:checkpoints}

\begin{center}
\small
\begin{tabular}{ll}
\toprule
Component & Checkpoint \\
\midrule
ossllm / asmalltrialsystem & \texttt{google/gemma-3-27b-it} \\
Finalcorrected             & \texttt{google/gemma-4-31b-it} (Dense) \\
Dense embedding            & \texttt{pritamdeka/S-PubMedBert-MS-MARCO} \\
Sparse (ossllm)            & \texttt{BM25Okapi} ($k_1{=}1.5$, $b{=}0.75$) \\
Sparse (Finalcorrected)    & SQLite FTS5 native BM25 \\
\bottomrule
\end{tabular}
\end{center}
\noindent\textbf{Table 6.} Model and retrieval-component checkpoints.
\vspace{8pt}

\subsection{Retrieval hyperparameters}
\label{app:retrieval}

\begin{center}
\small
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Dense index         & FAISS \texttt{IndexFlatIP}, 768-dim, L2-norm. \\
Per-question index   & rebuilt fresh per question \\
BM25 rebuild         & on every new-chunk insertion \\
RRF constant $k$     & 60 \\
RRF dense weight     & 0.6 \\
RRF sparse weight    & 0.4 \\
Loop cap $N$         & 5 iterations \\
Sufficiency check    & enabled when index $\geq 20$ passages \\
Neighbour expansion  & $\leq 10$ \texttt{pubmed\_pubmed} / top hit \\
Citation expansion   & $\leq 5$ \texttt{pubmed\_pubmed\_citedin} / top hit \\
Expansion scope      & top 2 passages per iteration \\
\bottomrule
\end{tabular}
\end{center}
\noindent\textbf{Table 7.} Retrieval and agentic-loop hyperparameters.
\vspace{8pt}

\subsection{Chunking}
\label{app:chunking}

Every retrieved abstract contributes three chunk types to the per-question index:
(i)~the title as a single chunk; (ii)~the full abstract as a single chunk (length
$\geq 50$ characters); and (iii)~overlapping three-sentence sliding windows with
one-sentence stride, where sentences are split on the regular expression
\texttt{(?<=[.!?])\textbackslash s+}. Windows shorter than 50 characters or duplicates
of previously seen chunks are dropped. Chunks are deduplicated within each per-question
index; there is no cross-question deduplication.

\subsection{Top-\texorpdfstring{$k$}{k} schedule}
\label{app:topk}

\begin{center}
\small
\begin{tabular}{lr}
\toprule
Stage & $k$ \\
\midrule
Per-search FAISS retrieval (pre-fusion)  & $3k$, capped at index size \\
Per-search BM25 ranking                  & $3k$ \\
Per-search output (post-RRF)             & 15 \\
Agentic-loop sufficiency inspection      & top 10 \\
Pre-rerank candidate pool                & 20 \\
Post-LLM-rerank candidate pool           & re-sorted top 20 \\
Generator context                        & top 5--10 \\
\bottomrule
\end{tabular}
\end{center}
\noindent\textbf{Table 8.} Top-$k$ at each retrieval stage.
\vspace{8pt}
```

---

## What changed vs. what stayed

- **Each `\begin{table}[h] ... \end{table}` float → `\begin{center} ... \end{center}`** non-floating block. This is the whole fix — the tables now sit where you type them.
- **`\caption{...}` + `\label{...}` → `\noindent\textbf{Table N.} ...`** bold inline label.
  You lose auto-numbering, so the numbers are hardcoded 5–8.
- **The `tabular` content is identical** — same columns, same `\toprule`/`\midrule`/`\bottomrule`, same data. Nothing inside the tables moved.
- **A.4 Chunking** had no table, just prose — unchanged.

## After pasting

- If anything in your body text says `\ref{tab:decoding}`, `\ref{tab:checkpoints}`,
  `\ref{tab:retrieval}`, or `\ref{tab:topk}`, those references will now break (the labels
  are gone). Search for them; you almost certainly don't reference the appendix tables, but
  check. If you do, just replace the `\ref{...}` with the literal number (5/6/7/8).
- Recompile once — no second pass needed since there are no floats or auto-numbers left to
  resolve here.
- The big gaps and the empty A.2/A.3 headers will be gone; every table now sits directly
  under its subsection heading.
```
