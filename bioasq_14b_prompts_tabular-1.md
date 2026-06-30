# BioASQ §A.6 Prompts — overflow-proof boxes, no preamble changes

The previous version overflowed because `\newline` inside a plain `tabular` cell does not
wrap long lines. This version puts each prompt inside `\begin{verbatim}`, inside a
`minipage` fixed at `0.92\linewidth`, inside one bordered `tabular` cell. The minipage
caps the width so nothing runs off the page, and verbatim needs no character escaping. No
preamble changes.

Replace your whole §A.6 with the block below.

```latex
\subsection{Prompts}
\label{app:prompts}

The following prompts are reproduced verbatim. Placeholders in braces are replaced at run
time with the question, the retrieved evidence, and the few-shot exemplars.

\paragraph{Query generation.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
Generate 3 short PubMed queries (3-6 words, plain
keywords) for: {question}

[If previous queries exist:]
Previous didn't find enough. Use COMPLETELY
DIFFERENT terms:
  - {prev_query_1}
  - {prev_query_2}
  - {prev_query_3}

Write ONLY 3 queries numbered. Nothing else.

1.
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\paragraph{Sufficiency check.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
Can '{question}' be answered from these?

[1] {passage_1_truncated_to_150_chars}
...
[8] {passage_8_truncated_to_150_chars}

SUFFICIENT or INSUFFICIENT?
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\noindent The controller terminates the loop early if the response contains the substring
\texttt{SUFFICIENT} (case-insensitive).

\paragraph{Yes/no generation.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
You are an expert biomedical QA system.

INSTRUCTIONS:
1. Find evidence supporting YES.
2. Find evidence supporting NO.
3. Choose the side with STRONGER direct evidence.
4. If evidence shows PROBLEMS (toxicity, failure,
   side effects, contradictory results, lack of
   clinical evidence), answer NO.
5. If evidence is mixed, unclear, or insufficient,
   answer NO.
6. 'Promising preclinical results' or 'under
   investigation' does NOT mean YES.
7. A drug being 'studied' or 'tested' does NOT
   mean it works.

[two few-shot examples: Q / EVIDENCE /
 EXACT_ANSWER / IDEAL_ANSWER]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}

EVIDENCE FOR YES:
EVIDENCE FOR NO:
EXACT_ANSWER:
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\paragraph{Factoid generation.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
You are an expert biomedical QA system.

STRICT RULES:
1. EXACT_ANSWER must be 1-5 words. A specific
   name, number, or term.
2. Copy the EXACT terminology from the evidence.
3. Do NOT paraphrase or elaborate in the answer.
4. If the evidence does not contain a clear
   specific answer, write: unknown
5. Prefer named entities: drug names, protein
   names, gene symbols, disease names, numbers.
6. Then write IDEAL_ANSWER: a 2-4 sentence
   explanation (max 200 words).

GOOD: 'transsphenoidal surgery', 'NF1', '45,X',
      'palivizumab', 'mesenchymal'
BAD:  'multiple causative factors',
      'a type of bleeding'

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
EXACT_ANSWER:
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\paragraph{List generation.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
You are an expert biomedical QA system.

STRICT RULES:
1. List EVERY relevant item in the evidence.
   Be EXHAUSTIVE.
2. Better to include too many than too few.
3. Go through EACH passage systematically.
4. Each item is a specific name or term (1-5
   words).
5. Aim for 5-15 items. Many questions have
   10-20+ answers.
6. Prefix each item with '- ' on its own line.
7. Do NOT group or combine items. One per line.
8. After the list, write IDEAL_ANSWER: 2-4
   sentences (max 200 words).

[one few-shot example with list-formatted answer]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}

Now list EVERY relevant item from ALL passages:

EXACT_ANSWER:
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\paragraph{Summary generation.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
You are an expert biomedical QA system. Write a
comprehensive 3-6 sentence answer (max 200 words)
using the evidence.

[two few-shot examples]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
IDEAL_ANSWER:
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\paragraph{LLM relevance reranking.}
\begin{center}\footnotesize
\begin{tabular}{|l|}
\hline
\begin{minipage}{0.92\linewidth}\vspace{2pt}
\begin{verbatim}
Rate each passage's relevance (0-2) to
answering: {question}

[1] {passage_1_truncated_to_200_chars}
...
[20] {passage_20_truncated_to_200_chars}

Return [N] SCORE per line:
\end{verbatim}
\end{minipage} \\
\hline
\end{tabular}
\end{center}

\noindent The relevance score (0, 1, or 2) is fused with the RRF score using weights 0.6
(RRF) and 0.4 (LLM relevance $/2$).
```

---

## Why this one does not overflow

- Text lives inside `\begin{verbatim}` — raw, no escaping (your `{`, `}`, `_`, `[`, `]`
  all print as typed).
- The verbatim sits inside a `minipage` fixed at `0.92\linewidth`, so the box can never be
  wider than the column.
- Long lines were pre-wrapped by hand (the indented continuation lines) so no single line
  runs past the box edge, since verbatim itself does not auto-wrap.
- The whole thing is one bordered `tabular` cell, so it reads like your tables.

If a box is taller than the space left on a page, add `\clearpage` immediately before that
`\paragraph{...}` to push it onto a fresh page.
```
