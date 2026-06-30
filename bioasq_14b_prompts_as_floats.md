# BioASQ Appendix §A.6 — All Prompts as Floats (drag-and-drop)

This replaces your entire **§A.6 Prompts** section. Each prompt is now wrapped in a
`figure` float with a caption, so it behaves exactly like your tables: it never splits
across a page, and it snaps to a clean position.

---

## Step 1 — Preamble style (paste once, after `\lstset{breaklines=true}`)

If you already pasted the earlier style block, **replace it** with this one. If `xcolor`
is already loaded elsewhere, delete the first line to avoid a duplicate-package warning.

```latex
\usepackage{xcolor}
\definecolor{promptbg}{RGB}{246,247,249}
\definecolor{promptframe}{RGB}{200,204,210}
\lstdefinestyle{prompt}{
  backgroundcolor=\color{promptbg},
  frame=single,
  rulecolor=\color{promptframe},
  framesep=5pt,
  basicstyle=\ttfamily\scriptsize,
  breaklines=true,
  breakindent=0pt,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  xleftmargin=2pt,
  xrightmargin=2pt,
  aboveskip=2pt,
  belowskip=2pt
}
```

---

## Step 2 — Replace the whole §A.6 with this

Delete everything from `\subsection{Prompts}` through the end of the last prompt, and
paste this in its place.

```latex
\subsection{Prompts}
\label{app:prompts}

The following prompts are reproduced verbatim (Figures~\ref{prompt:query}--\ref{prompt:rerank}).
Placeholders in braces are replaced at run time with the question, the retrieved evidence,
and the few-shot exemplars.

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
Generate 3 short PubMed queries (3-6 words, plain keywords) for: {question}

[If previous queries exist:]
Previous didn't find enough. Use COMPLETELY DIFFERENT terms:
  - {prev_query_1}
  - {prev_query_2}
  - {prev_query_3}

Write ONLY 3 queries numbered. Nothing else.

1.
\end{lstlisting}
\caption{Query generation prompt.}
\label{prompt:query}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
Can '{question}' be answered from these?

[1] {passage_1_truncated_to_150_chars}
...
[8] {passage_8_truncated_to_150_chars}

SUFFICIENT or INSUFFICIENT?
\end{lstlisting}
\caption{Sufficiency check prompt. The controller terminates the loop early if the
response contains the substring \texttt{SUFFICIENT} (case-insensitive).}
\label{prompt:suff}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
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
\caption{Yes/no generation prompt.}
\label{prompt:yesno}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
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
\caption{Factoid generation prompt.}
\label{prompt:factoid}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
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
\caption{List generation prompt.}
\label{prompt:list}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
You are an expert biomedical QA system. Write a comprehensive
3-6 sentence answer (max 200 words) using the evidence.

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
IDEAL_ANSWER:
\end{lstlisting}
\caption{Summary generation prompt.}
\label{prompt:summary}
\end{figure}

\begin{figure}[t]
\begin{lstlisting}[style=prompt]
Rate each passage's relevance (0-2) to answering: {question}

[1] {passage_1_truncated_to_200_chars}
...
[20] {passage_20_truncated_to_200_chars}

Return [N] SCORE per line:
\end{lstlisting}
\caption{LLM relevance-reranking prompt. The relevance score (0, 1, or 2) is fused with
the RRF score using weights 0.6 (RRF) and 0.4 (LLM relevance $/2$).}
\label{prompt:rerank}
\end{figure}
```

---

## Notes

- **Each prompt is now a `figure` float** — it cannot split across a page, which is what was causing the half-empty boxes before.
- **The two explanatory sentences** (the SUFFICIENT-substring note and the RRF-fusion note) are now folded into their figure captions, so they stay attached to the right prompt and don't float away.
- **`\scriptsize`** keeps the longer prompts (yes/no, factoid, list) inside one box.
- If LaTeX bunches several prompt figures at the end of the appendix (float pile-up), add `\clearpage` after every two or three prompts to flush them, or change a couple of `[t]` to `[h]` to nudge them in place.
- Recompile **twice** so the `\ref{prompt:...}` numbers in the intro sentence resolve.
```

