# BioASQ §A.6 Prompts — plain tabular boxes, no preamble changes

Replace your whole §A.6 with this. Every prompt is a one-cell `tabular`. Nothing new in
the preamble. The `\\` at the end of each line is what keeps the line breaks inside the
cell.

```latex
\subsection{Prompts}
\label{app:prompts}

The following prompts are reproduced verbatim. Placeholders in braces are replaced at run
time with the question, the retrieved evidence, and the few-shot exemplars.

\paragraph{Query generation.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
Generate 3 short PubMed queries (3-6 words, plain keywords) for: \{question\} \newline
\newline
{[}If previous queries exist:{]} \newline
Previous didn't find enough. Use COMPLETELY DIFFERENT terms: \newline
\hspace*{1em}- \{prev\_query\_1\} \newline
\hspace*{1em}- \{prev\_query\_2\} \newline
\hspace*{1em}- \{prev\_query\_3\} \newline
\newline
Write ONLY 3 queries numbered. Nothing else. \newline
\newline
1. \\
\hline
\end{tabular}

\paragraph{Sufficiency check.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
Can '\{question\}' be answered from these? \newline
\newline
{[}1{]} \{passage\_1\_truncated\_to\_150\_chars\} \newline
... \newline
{[}8{]} \{passage\_8\_truncated\_to\_150\_chars\} \newline
\newline
SUFFICIENT or INSUFFICIENT? \\
\hline
\end{tabular}

\noindent The controller terminates the loop early if the response contains the substring
\texttt{SUFFICIENT} (case-insensitive).

\paragraph{Yes/no generation.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
You are an expert biomedical QA system. \newline
\newline
INSTRUCTIONS: \newline
1. Find evidence supporting YES. \newline
2. Find evidence supporting NO. \newline
3. Choose the side with STRONGER direct evidence. \newline
4. If evidence shows PROBLEMS (toxicity, failure, side effects, contradictory results, lack of clinical evidence), answer NO. \newline
5. If evidence is mixed, unclear, or insufficient, answer NO. \newline
6. 'Promising preclinical results' or 'under investigation' does NOT mean YES. \newline
7. A drug being 'studied' or 'tested' does NOT mean it works. \newline
\newline
{[}two few-shot examples: Q / EVIDENCE / EXACT\_ANSWER / IDEAL\_ANSWER{]} \newline
\newline
--- \newline
Q: \{question\} \newline
EVIDENCE: \newline
\{retrieved\_snippets\} \newline
\newline
EVIDENCE FOR YES: \newline
EVIDENCE FOR NO: \newline
EXACT\_ANSWER: \\
\hline
\end{tabular}

\paragraph{Factoid generation.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
You are an expert biomedical QA system. \newline
\newline
STRICT RULES: \newline
1. EXACT\_ANSWER must be 1-5 words. A specific name, number, or term. \newline
2. Copy the EXACT terminology from the evidence passages. \newline
3. Do NOT paraphrase, explain, or elaborate in the exact answer. \newline
4. If the evidence does not contain a clear specific answer, write: unknown \newline
5. Prefer named entities: drug names, protein names, gene symbols, disease names, specific numbers/percentages. \newline
6. Then write IDEAL\_ANSWER: a 2-4 sentence explanation (max 200 words). \newline
\newline
GOOD: 'transsphenoidal surgery', 'NF1', '45,X', 'palivizumab', 'mesenchymal' \newline
BAD:  'multiple causative factors', 'a type of bleeding' \newline
\newline
{[}two few-shot examples{]} \newline
\newline
--- \newline
Q: \{question\} \newline
EVIDENCE: \newline
\{retrieved\_snippets\} \newline
EXACT\_ANSWER: \\
\hline
\end{tabular}

\paragraph{List generation.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
You are an expert biomedical QA system. \newline
\newline
STRICT RULES: \newline
1. List EVERY relevant item mentioned in the evidence. Be EXHAUSTIVE. \newline
2. It is MUCH better to include too many items than too few. \newline
3. Go through EACH evidence passage systematically. \newline
4. Each item should be a specific name or term (1-5 words). \newline
5. Aim for at least 5-15 items. Many questions have 10-20+ answers. \newline
6. Prefix each item with '- ' on its own line. \newline
7. Do NOT group or combine items. One item per line. \newline
8. After the list, write IDEAL\_ANSWER: 2-4 sentences (max 200 words). \newline
\newline
{[}one few-shot example with list-formatted answer{]} \newline
\newline
--- \newline
Q: \{question\} \newline
EVIDENCE: \newline
\{retrieved\_snippets\} \newline
\newline
Now list EVERY relevant item from ALL passages: \newline
\newline
EXACT\_ANSWER: \\
\hline
\end{tabular}

\paragraph{Summary generation.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
You are an expert biomedical QA system. Write a comprehensive 3-6 sentence answer (max 200 words) using the evidence. \newline
\newline
{[}two few-shot examples{]} \newline
\newline
--- \newline
Q: \{question\} \newline
EVIDENCE: \newline
\{retrieved\_snippets\} \newline
IDEAL\_ANSWER: \\
\hline
\end{tabular}

\paragraph{LLM relevance reranking.}

\noindent\begin{tabular}{|p{0.95\linewidth}|}
\hline
\ttfamily\footnotesize
Rate each passage's relevance (0-2) to answering: \{question\} \newline
\newline
{[}1{]} \{passage\_1\_truncated\_to\_200\_chars\} \newline
... \newline
{[}20{]} \{passage\_20\_truncated\_to\_200\_chars\} \newline
\newline
Return {[}N{]} SCORE per line: \\
\hline
\end{tabular}

\noindent The relevance score (0, 1, or 2) is fused with the RRF score using weights 0.6
(RRF) and 0.4 (LLM relevance $/2$).
```

---

## Two things that matter

1. **Special characters are escaped** because tabular reads them as LaTeX, not raw text:
   `{` `}` became `\{` `\}`, `_` became `\_`, and `[` `]` are wrapped as `{[}` `{]}`.
   This is already done for you above — just paste.

2. **Line breaks inside the cell** are `\newline`. The very last line of each cell ends in
   `\\` (the real row end). Don't mix them up.

If any box runs off the bottom of a page, that's a tabular not breaking — just add
`\clearpage` before that `\paragraph{...}` to push it to the next page.
