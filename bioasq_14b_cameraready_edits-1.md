# BioASQ Task 14b — Camera-Ready Edit Guide (single source of truth)

Everything you need, in order. Each block says **WHERE** to make the change and gives **drag-and-drop LaTeX**. Work top to bottom.

---

## PART 0 — Author block (de-anonymise)

**Good news:** your `\documentclass` has no `anonymous` option, so nothing to change there.

**WHERE:** lines ~50–66 of `main.tex`, the `\author` / `\address` block.

**Problems in the current block:**
1. The `%url=https://yamadharma.github.io/` and `%url=https://kmitd.github.io/ilaria/` lines are leftovers from the CEUR sample template. They're commented out so they won't render, but delete them to avoid confusion.
2. Verify the ORCIDs are really yours (the sample template ships with placeholder ORCIDs that look real). If they're not yours, delete the `orcid=` lines. If they are, **uncomment** them (remove the leading `%`) so they render.
3. Leslie Shing may still be missing — add her.

**Replace the whole author/address block with this:**

```latex
%%
%% The "author" command and its associated commands are used to define
%% the authors and their affiliations.
\author[1]{Rohan Leekha}[%
  orcid=0000-0002-0877-7063,
  email=rohan.leekha@ll.mit.edu,
]
\author[1]{Daniel Gwon}[%
  orcid=0000-0001-7116-9338,
  email=daniel.gwon@ll.mit.edu,
]
\author[1]{Leslie Shing}[%
  email=leslie.shing@ll.mit.edu,
]

\address[1]{MIT Lincoln Laboratory, 244 Wood Street, Lexington, MA 02421, USA}
```

Notes:
- All three share affiliation `[1]`, so there is exactly one `\address[1]`.
- If an ORCID above is not actually yours, delete that whole `orcid=...,` line. Don't ship someone else's ORCID.
- CEUR-ART marks the first author as corresponding by default; that matches your EasyChair record (Rohan = corresponding).

---

## PART 1 — Trivial edits (do these first, 5 minutes total)

### 1A. Replace "outperform" (Reviewer 2.7)

**WHERE:** §3.3, "Sparse retrieval and rank fusion" paragraph.

**FIND:**
```
RRF outperformed every linear-combination configuration we tried
```
**REPLACE WITH:**
```
RRF scored higher than every linear-combination configuration we tested in preliminary tuning; we did not test for statistical significance
```

Then Ctrl-F the whole document for `outperform`, `beats`, `exceeds`, `better than` and soften any others the same way.

### 1B. Reword the yes-bias claim (Reviewer 2.3)

**WHERE:** §3.5, "Yes/no questions" paragraph, first sentence.

**FIND:**
```
Instruction-tuned LLMs exhibit a pronounced yes-bias: in our preliminary runs
on the BioASQ training set, the base Gemma 3 27B prompt answered ``yes'' to roughly 70\%
of yes/no questions regardless of evidence polarity.
```
**REPLACE WITH:**
```
In our preliminary runs on the BioASQ training set, the base Gemma 3 27B prompt
answered ``yes'' to roughly 70\% of yes/no questions regardless of the polarity of
the supporting evidence. We treat this as a calibration target for our prompt design
rather than as a property of all instruction-tuned LLMs.
```

### 1C. Add citations for the exact-token claim (Reviewer 2.3)

**WHERE:** §3.3, "Sparse retrieval and rank fusion" paragraph.

**FIND:**
```
biomedical queries carry an unusual density of exact tokens (gene symbols, drug
names, disease abbreviations) that embedding similarity can soften
```
**REPLACE WITH:**
```
biomedical queries carry an unusual density of exact tokens (gene symbols, drug
names, disease abbreviations) that embedding similarity can soften~\cite{lee2020biobert,gu2021pubmedbert}
```
(Both keys are already in your `ref.bib` from the BioASQ paper — confirm they exist; if not, copy them across.)

---

## PART 2 — Abstract & intro framing (Reviewer 1.2)

### 2A. Abstract

**WHERE:** the `\begin{abstract}` block.

**FIND:**
```
placing it in the upper-middle quartile of the Task 14b field
```
**REPLACE WITH:**
```
placing it in the upper-middle quartile of the Task 14b field on this batch; results on subsequent batches and on Phase~A+ are mid-field or lower-middle
```

### 2B. Introduction — add one sentence after the per-batch numbers

**WHERE:** §1, immediately after the sentence ending "...factoid metrics were not scored on batch 3)."

**INSERT this sentence:**
```
Of the six metric-by-batch combinations we report, one is upper-middle (Phase~B yes/no batch~1), three are middle of field, and two are lower-middle. The contribution of this paper is therefore not a leaderboard result but the diagnostic finding that retrieval recall is the binding constraint on the end-to-end pipeline, supported by the within-system isolation evidence in \S\ref{sec:phaseaplus}.
```
(Adjust `\ref{sec:phaseaplus}` to whatever label your Phase A+ results subsection has, or write "\S4.2".)

---

## PART 3 — Contribution paragraph (Reviewer 1.1)

**WHERE:** end of §3.1 "Architecture", as a new final paragraph.

**INSERT:**
```
The contribution of this paper is not a new retrieval component or a new generator
architecture. It is the integration of three specific choices: a controller prompt
that withholds the controller's own query history to discourage paraphrase-cycling,
a deterministic format-hygiene filter that recovers silently-malformed outputs against
BioASQ's strict scorer, and a per-question hybrid index design that allows a small
($N{=}5$) iteration cap to be sufficient on questions that decompose cleanly. The
empirical finding we report --- that the resulting system reaches upper-middle yes/no
performance on Phase~B without domain fine-tuning, while remaining bounded on Phase~A+
by retrieval recall --- is the result of those choices interacting on the BioASQ~14b
benchmark.
```

---

## PART 4 — Expanded Related Work (Reviewers 1.1, 2.2, 2.6)

**WHERE:** replace the entire current §2 "Related Work" body (keep the `\section{Related Work}` header).

```
BioASQ Task 13b in 2025 saw a range of retrieval-augmented architectures applied to
biomedical question answering. BIT.UA (Universidade de Aveiro)~\cite{bituabioasq}
combined dense retrieval with cross-encoder reranking and supervised fine-tuning of
the generator, reaching Phase~B yes/no Macro-F1 of approximately 0.96 on the strongest
batch and list F-measure approximately 0.55. UNITOR~\cite{unitorbioasq} used a
dense-plus-rerank pipeline with a different reranker and reached comparable yes/no
performance. Ateia and Kruschwitz~\cite{ateiaselffeedback} introduced an iterative
self-feedback loop where the LLM critiques and regenerates its own answer, with
reported yes/no accuracy in the 0.85--0.92 range and list F-measure up to approximately
0.40. AQAMS~\cite{aqamsbioasq} composed multiple LLM agents that exchanged messages
while answering. The 2025 field leaders on Phase~B factoid strict and ideal-answer
ROUGE-2 F1 were at approximately 0.50 and 0.25 respectively. The ReAct pattern~\cite{react}
is the general template of LLM-driven action loops that we specialise to PubMed search.

Biomedical question answering has also been studied on adjacent benchmarks: PubMedQA~\cite{pubmedqa}
(research-question yes/no), MedQA~\cite{medqa} (medical licensing exams), and BioRED~\cite{biored}
(biomedical relation extraction). RAG over PubMed has been a recurring approach in the
broader biomedical NLP literature~\cite{lee2020biobert,gu2021pubmedbert}, and agentic
LLM patterns (ReAct~\cite{react}, Toolformer~\cite{toolformer}) have transferred from
general-domain QA to biomedical IR with mixed results --- most prior agentic biomedical
systems use the loop for query reformulation but not for explicit evidence-pool
sufficiency reasoning.

Our submission concentrates agency in a single LLM instance that plays both controller
and generator roles within a forward retrieval loop. We do not iterate on the generated
answer (unlike Ateia and Kruschwitz), we do not distribute agency across multiple LLM
agents (unlike AQAMS), we do not perform supervised fine-tuning of the generator (unlike
BIT.UA), and we use unsupervised RRF rather than a learned cross-encoder for first-stage
fusion (unlike UNITOR). The combination is intended to test how far a stripped-down,
single-model agentic system without fine-tuning can reach against the BioASQ scorer, and
to identify which component of the pipeline is the binding constraint.

BioASQ's exact-match scorer rewards two things simultaneously: correct evidence retrieval
and conformance to a strict output shape (nested-list-of-lists for factoid and list, no
synonyms, bare lowercase yes/no). Most prior systems treat these as one combined
optimisation target. We treat them as separable: the agentic loop targets retrieval
recall, and the format-hygiene filter targets output-shape conformance. The contribution
is to evaluate the two separately and to report the resulting position in the field as a
diagnostic, not as a leaderboard claim.
```

**New `\cite` keys you must add to `ref.bib`:** `pubmedqa`, `medqa`, `biored`, `toolformer`. If you don't want to chase these four down, delete the second paragraph (the "adjacent benchmarks" one) — the other three paragraphs use keys you already have.

---

## PART 5 — Ablation framing (Reviewer 1.3)

**WHERE:** end of §3.7 "The submitted variants", as a new paragraph.

```
Although the three submitted systems were configured for the BioASQ submission rather
than as a controlled ablation study, they vary along three orthogonal axes.
asmalltrialsystem and ossllm differ only in whether three training-set exemplars are
prepended to the answer-generation prompt; on every batch they were both submitted to,
they produced identical aggregate scores, indicating that few-shot conditioning of this
form did not change the system's behaviour on the BioASQ test distribution.
asmalltrialsystem/ossllm and Finalcorrected differ along two axes simultaneously --- LLM
backbone (Gemma~3 27B vs.\ Gemma~4 31B Dense) and retrieval substrate (live NCBI
E-utilities vs.\ a local SQLite FTS5 mirror of the PubMed Annual Baseline). On Phase~A+
batch~4, the only batch where both systems were submitted, they returned identical scores
to three decimal places across five metrics, indicating that neither change moved aggregate
performance in the retrieval-bottlenecked end-to-end pipeline. We did not control for
individual components within each axis, and we cannot decompose the headline Phase~B yes/no
result (0.941) into contributions from the agentic loop, the format-hygiene filter, or the
sufficiency-check mechanism; we acknowledge this in \S\ref{sec:limited}.
```
(Adjust `\ref{sec:limited}` to your §5.3 label, or write "\S5.3".)

---

## PART 6 — Limitations block (Reviewers 1.3, 2.1, 2.4)

**WHERE:** §5.3 "Where We Are Limited", append these three short paragraphs.

```
We did not run controlled ablations on the agentic loop's iteration cap, the deterministic
format-hygiene filter, or the sufficiency-check early-stop mechanism, and we cannot decompose
our headline Phase~B numbers into the contribution of each component. This is a methodological
limitation of the submission; we report the three submitted systems' as-submitted differences
as the only ablation evidence available to us.

We did not benchmark our submitted systems against BioASQ 13b (2025) gold data, which would
have provided cross-year calibration of our Phase~B and Phase~A+ scores against a known
leaderboard distribution. This is a limitation of the present submission rather than an
intentional design choice.

A public code repository accompanying this submission was not available within the camera-ready
window; the configuration and prompts in Appendix~\ref{app:repro} are intended to allow
re-implementation but are not a substitute for the original source. Releasing the code is a
priority for follow-up work.
```

---

## PART 7 — §5.1 citations (Reviewer 2.6)

**WHERE:** §5.1 "Where We Are Strong", three design-choice paragraphs.

- In the **agentic-loop** paragraph, append: `This places our finding within the agentic-RAG line of work~\cite{react,toolformer}.`
- In the **format-hygiene** paragraph, append: `Deterministic post-processing is a lightweight alternative to constrained decoding~\cite{willard2023outlines} for enforcing structured output.`
- In the **RRF** paragraph, append: `The positional damping in RRF~\cite{rrf} is theoretically suited to the heavy-tailed score distributions that dense and sparse biomedical rankers produce.`

**New key:** `willard2023outlines` (Outlines / constrained decoding). If you don't want to add it, drop that one sentence; `react`, `toolformer`, `rrf` you already have.

---

## PART 8 — THE APPENDIX (Reviewers 1.4, 2.1, 2.5)

**WHERE:** paste this entire block **after** your bibliography (`\bibliography{ref}` or `\end{thebibliography}`) and **before** `\end{document}`.

It uses `listings` for the prompts, which you already load (`\usepackage{listings}` + `\lstset{breaklines=true}` are in your preamble), so it will compile as-is.

```latex
\appendix

\section{Reproducibility Details}
\label{app:repro}

A public code repository accompanying this submission was not available within the
camera-ready window. This appendix documents the prompts, hyperparameters, and
configuration in full so that the system can be re-implemented without access to the
original source.

\subsection{Decoding settings}
\label{app:decoding}

All LLM calls are served by a local vLLM instance. Decoding parameters vary by call type
(Table~\ref{tab:decoding}). Each call retries up to three times with exponential backoff
(3, 6, 9 seconds) under a 180-second timeout.

\begin{table}[h]
\centering
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
\caption{Per-call-type decoding settings.}
\label{tab:decoding}
\end{table}

\subsection{Model checkpoints}
\label{app:checkpoints}

\begin{table}[h]
\centering
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
\caption{Model and retrieval-component checkpoints.}
\label{tab:checkpoints}
\end{table}

\subsection{Retrieval hyperparameters}
\label{app:retrieval}

\begin{table}[h]
\centering
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
\caption{Retrieval and agentic-loop hyperparameters.}
\label{tab:retrieval}
\end{table}

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

\begin{table}[h]
\centering
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
\caption{Top-$k$ at each retrieval stage.}
\label{tab:topk}
\end{table}

\subsection{Prompts}
\label{app:prompts}

The following prompts are reproduced verbatim. Placeholders in braces are replaced at
run time with the question, the retrieved evidence, and the few-shot exemplars.

\paragraph{Query generation.}
\begin{lstlisting}
Generate 3 short PubMed queries (3-6 words, plain keywords) for: {question}

[If previous queries exist:]
Previous didn't find enough. Use COMPLETELY DIFFERENT terms:
  - {prev_query_1}
  - {prev_query_2}
  - {prev_query_3}

Write ONLY 3 queries numbered. Nothing else.

1.
\end{lstlisting}

\paragraph{Sufficiency check.}
\begin{lstlisting}
Can '{question}' be answered from these?

[1] {passage_1_truncated_to_150_chars}
...
[8] {passage_8_truncated_to_150_chars}

SUFFICIENT or INSUFFICIENT?
\end{lstlisting}
The controller terminates the loop early if the response contains the substring
\texttt{SUFFICIENT} (case-insensitive).

\paragraph{Yes/no generation.}
\begin{lstlisting}
You are an expert biomedical QA system.

INSTRUCTIONS:
1. Find evidence supporting YES.
2. Find evidence supporting NO.
3. Choose the side with STRONGER direct evidence.
4. If evidence shows PROBLEMS (toxicity, failure, side effects,
   contradictory results, lack of clinical evidence), answer NO.
5. If evidence is mixed, unclear, or insufficient, answer NO.
6. 'Promising preclinical results' or 'under investigation' does NOT mean YES.
7. A drug being 'studied' or 'tested' does NOT mean it works.

[two few-shot examples: Q / EVIDENCE / EXACT_ANSWER / IDEAL_ANSWER]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}

EVIDENCE FOR YES:
EVIDENCE FOR NO:
EXACT_ANSWER:
\end{lstlisting}

\paragraph{Factoid generation.}
\begin{lstlisting}
You are an expert biomedical QA system.

STRICT RULES:
1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.
2. Copy the EXACT terminology from the evidence passages.
3. Do NOT paraphrase, explain, or elaborate in the exact answer.
4. If the evidence does not contain a clear specific answer, write: unknown
5. Prefer named entities: drug names, protein names, gene symbols,
   disease names, specific numbers/percentages.
6. Then write IDEAL_ANSWER: a 2-4 sentence explanation (max 200 words).

GOOD: 'transsphenoidal surgery', 'NF1', '45,X', 'palivizumab', 'mesenchymal'
BAD:  'multiple causative factors', 'a type of bleeding'

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
EXACT_ANSWER:
\end{lstlisting}

\paragraph{List generation.}
\begin{lstlisting}
You are an expert biomedical QA system.

STRICT RULES:
1. List EVERY relevant item mentioned in the evidence. Be EXHAUSTIVE.
2. It is MUCH better to include too many items than too few.
3. Go through EACH evidence passage systematically.
4. Each item should be a specific name or term (1-5 words).
5. Aim for at least 5-15 items. Many questions have 10-20+ answers.
6. Prefix each item with '- ' on its own line.
7. Do NOT group or combine items. One item per line.
8. After the list, write IDEAL_ANSWER: 2-4 sentences (max 200 words).

[one few-shot example with list-formatted answer]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}

Now list EVERY relevant item from ALL passages:

EXACT_ANSWER:
\end{lstlisting}

\paragraph{Summary generation.}
\begin{lstlisting}
You are an expert biomedical QA system. Write a comprehensive
3-6 sentence answer (max 200 words) using the evidence.

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
IDEAL_ANSWER:
\end{lstlisting}

\paragraph{LLM relevance reranking.}
\begin{lstlisting}
Rate each passage's relevance (0-2) to answering: {question}

[1] {passage_1_truncated_to_200_chars}
...
[20] {passage_20_truncated_to_200_chars}

Return [N] SCORE per line:
\end{lstlisting}
The relevance score (0, 1, or 2) is fused with the RRF score using weights 0.6 (RRF)
and 0.4 (LLM relevance $/2$).

\subsection{Format-hygiene filter}
\label{app:hygiene}

The deterministic filter applies the following operations, in order, to every model
output before JSON serialisation:
\begin{enumerate}
\item Strip markdown tokens (\texttt{**}, \texttt{\_\_}, backtick, \texttt{>}, \texttt{\#}).
\item Strip scratchpad headers (e.g.\ \texttt{**Reasoning:**}, \texttt{Thought:}) and
      bracketed citation tokens (\texttt{[1]}, \texttt{[1, 2]}).
\item For factoid and list outputs, strip leading prefixes
      (\texttt{Answer:}, \texttt{EXACT\_ANSWER:}, \texttt{Final answer:}).
\item For factoid outputs, truncate to the first noun phrase if the model returns a full
      sentence, then verify length is 1--5 tokens.
\item For list outputs, parse \texttt{-}-prefixed lines and deduplicate
      case-insensitively after normalisation (strip trailing parenthetical synonyms,
      collapse whitespace, unify common abbreviation variants).
\item Enforce BioASQ output shape: factoid and list exact answers as a list of
      single-element lists (BioASQ$\geq$5 format), capped at 5 outer entries for factoid
      and 100 for list; yes/no as a bare lowercase string; summary outputs omit the
      \texttt{exact\_answer} field.
\end{enumerate}

\subsection{Hardware and serving}
\label{app:hardware}

Two NVIDIA A100 80GB GPUs serve the LLM via vLLM with
\texttt{tensor\_parallel\_size=2}, \texttt{gpu\_memory\_utilization=0.85}, and
\texttt{bfloat16} precision; the context window is capped at 8{,}192 tokens during
generation. PubMedBERT embeddings are computed on CPU (batch size 32, FP32) due to GPU
memory contention with the served 27B/31B model. The Finalcorrected system queries a
$\sim$50\,GB SQLite~3.45 mirror of the 2025 PubMed Annual Baseline (release 18~December
2024) with an FTS5 full-text index over titles, abstracts, MeSH headings, and publication
metadata; \texttt{Editorial}, \texttt{Letter}, \texttt{News}, \texttt{Comment},
\texttt{Review}, and \texttt{Retraction} article types are excluded at mirror-build time.

\subsection{Randomness}
\label{app:random}

Controller calls run at \texttt{temperature} 0.1--0.3; the sufficiency check at 0.1 is
effectively deterministic on the prompts we observed. Generator calls run at
\texttt{temperature} 0.3; we did not fix a serving-level random seed at submission time,
which we estimate contributes $\sim$1--2 points of aggregate-metric variance from informal
repeated runs. Few-shot exemplar selection is deterministic given a fixed development set
(cosine similarity to the test question, ties broken by training-set order).
```

---

## Order of operations in Overleaf

1. Fix the author block (Part 0). Recompile — confirm three authors + MIT LL affiliation render.
2. Do the trivial edits (Part 1). Recompile.
3. Abstract + intro framing (Part 2).
4. Paste contribution paragraph (Part 3), Related Work (Part 4), ablation paragraph (Part 5), limitations (Part 6), §5.1 citations (Part 7).
5. Paste the appendix (Part 8) before `\end{document}`. Recompile **twice** so the `\label{app:repro}` reference in Part 6 resolves.
6. Add the missing `ref.bib` keys flagged in Parts 4 and 7 (`pubmedqa`, `medqa`, `biored`, `toolformer`, `willard2023outlines`) — or delete the sentences that use them if you'd rather not chase citations.
7. Final recompile. Check page count and that no `??` cross-refs remain.

---

## New `ref.bib` keys to add (or delete the using sentence)

| Key | Used in | If you skip it |
|---|---|---|
| `pubmedqa` | Part 4 ¶2 | delete Part 4 paragraph 2 |
| `medqa` | Part 4 ¶2 | delete Part 4 paragraph 2 |
| `biored` | Part 4 ¶2 | delete Part 4 paragraph 2 |
| `toolformer` | Part 4 ¶2, Part 7 | likely already in your bib (it was in the Agent4IR paper) |
| `willard2023outlines` | Part 7 | delete the format-hygiene sentence |
| `lee2020biobert` | Part 1C, Part 4 | already in your bib |
| `gu2021pubmedbert` | Part 1C, Part 4 | already in your bib (it's your embedding cite, ref [10]) |
