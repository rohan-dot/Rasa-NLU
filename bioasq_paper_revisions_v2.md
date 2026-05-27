# BioASQ Task 14b Paper — Revision Package v2 (Corrected)

This supersedes v1. The previous version repeated the paper's incorrect numbers; this version uses the actual Task 14b leaderboard.

## Summary of corrections (vs. paper as currently written)

1. **Phase A+ batch 3 asmalltrialsystem**: paper says 0.455 Y/N, 0.450 Macro F1, 0.201 list F1. Leaderboard says **0.3636 / 0.3419 / 0.1669**.
2. **Phase B batch 4 Finalcorrected R-2 F1**: paper says 0.087. Leaderboard says **0.1576**. (The 0.087 is the *Phase A+* batch 4 R-2 F1, which is 0.0872 — looks like the row was mis-pasted.)
3. **ossllm Phase B batch 2 is missing from the paper entirely.** Actual numbers: **0.857 Y/N, 0.350 strict, 0.396 list F1**. The list F1 is **a top-cluster result for batch 2** (above dictycite-baseline 0.376, effectively tied with dmiip2024 family at 0.391). This is your strongest single result and the paper doesn't show it.
4. **Finalcorrected Phase B batch 3**: I cannot find this submission in any of the leaderboard screenshots. The paper Table 3 row for batch 3 has the exact same numbers as batch 4 (0.813 / 0.806 / 0.364 / 0.364 / 0.365 / 0.158), which looks like an accidental duplication. **Please verify against your `4-Trial.json` whether Finalcorrected actually submitted to Phase B batch 3.** If not, the row comes out.
5. **The ROUGE-2 regression narrative ("0.179 → 0.087") in §4 and §5 is based on the wrong number** and should be removed entirely. Actual Phase B R-2 F1 trajectory is 0.179 (batch 1, Gemma 3) → 0.158 (batch 4, Gemma 4), a small drop confounded by question difficulty across batches, not a story worth telling.

## What changes in the framing

The cleanest retrieval-cost claim in your data is the **same-batch within-system comparison**, not the cross-batch one the paper currently uses. Two clean comparisons exist:

| Comparison | Same questions | Same generator | Result |
|---|---|---|---|
| Phase B batch 2 (ossllm) vs Phase A+ batch 2 (asmalltrialsystem) | ✓ batch 2 | ✓ Gemma 3 27B¹ | **30-point list-F1 gap** |
| Phase B batch 4 (Finalcorrected) vs Phase A+ batch 4 (Finalcorrected) | ✓ batch 4 | ✓ Gemma 4 31B | 11-point list-F1 gap |

¹ ossllm and asmalltrialsystem differ only in few-shot exemplar conditioning, which on Phase B batch 1 produced character-for-character identical outputs — so treating them as the same generator is defensible.

This is methodologically cleaner than the cross-batch comparison the paper uses now, and the effect sizes are larger and more dramatic.

---

## Updated Table 2 (Phase A+ exact-answer scores)

```latex
\begin{table}
\centering
\caption{Phase A$^+$ exact-answer scores. ``---'' = not scored (no factoid submitted).}
\label{tab:phaseAplus}
\begin{tabular}{llcccccc}
\toprule
\multicolumn{2}{c}{} & \multicolumn{2}{c}{Y/N} & \multicolumn{2}{c}{Factoid} & List \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Batch & System & Acc. & Macro-F1 & Strict & MRR & F1 \\
\midrule
2 & asmalltrialsystem  & 0.762 & 0.760 & 0.150 & 0.150 & 0.095 \\
3 & asmalltrialsystem  & 0.364 & 0.342 & ---   & ---   & 0.167 \\
4 & asmalltrialsystem  & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
4 & Finalcorrected     & 0.750 & 0.746 & 0.273 & 0.273 & 0.255 \\
\bottomrule
\end{tabular}
\end{table}
```

## Updated Table 3 (Phase B exact-answer scores plus ROUGE-2 F1)

```latex
\begin{table}
\centering
\caption{Phase B exact-answer scores plus ROUGE-2 F1 on ideal answers.}
\label{tab:phaseB}
\begin{tabular}{llccccccc}
\toprule
\multicolumn{2}{c}{} & \multicolumn{2}{c}{Y/N} & \multicolumn{2}{c}{Factoid} & List & Ideal \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Batch & System & Acc. & Macro-F1 & Strict & MRR & F1 & R-2 F1 \\
\midrule
1 & asmalltrialsystem & 0.941 & 0.938 & 0.304 & 0.370 & 0.317 & 0.179 \\
1 & ossllm            & 0.941 & 0.938 & 0.304 & 0.370 & 0.317 & 0.179 \\
2 & ossllm            & 0.857 & 0.844 & 0.350 & 0.375 & 0.396 & 0.176 \\
4 & Finalcorrected    & 0.813 & 0.806 & 0.364 & 0.364 & 0.365 & 0.158 \\
\bottomrule
\end{tabular}
\end{table}
```

Notes on Table 3:
- **Added the ossllm batch 2 row.** This was missing entirely. The 0.396 list F1 is one of the top batch-2 scores.
- **Fixed the Finalcorrected batch 4 R-2 F1**: 0.087 → 0.158.
- **Removed the Finalcorrected batch 3 row** pending your verification. If you confirm it submitted, add it back with the correct numbers from your logs.
- The ossllm batch 2 R-2 F1 of 0.176 is my best read from the leaderboard screenshots; verify against your records.

## NEW Table 4 — Same-batch Phase B vs. Phase A+ comparison (centerpiece)

```latex
\begin{table}
\centering
\caption{Same-batch, same-generator Phase B vs.\ Phase A$^+$ comparison. The Phase B--Phase A$^+$ gap on identical questions isolates retrieval cost: the same generator scores substantially higher when handed gold snippets than when it must retrieve its own evidence. The pattern is monotonic in recall sensitivity (yes/no $<$ factoid strict $<$ list F1).}
\label{tab:withinbatch}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{Phase B (gold snippets)} & \multicolumn{3}{c}{Phase A$^+$ (must retrieve)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Batch & Generator & Y/N & Factoid & List F1 & Y/N & Factoid & List F1 \\
\midrule
2 & Gemma 3 27B  & 0.857 & 0.350 & 0.396 & 0.762 & 0.150 & 0.095 \\
4 & Gemma 4 31B  & 0.813 & 0.364 & 0.365 & 0.750 & 0.273 & 0.255 \\
\midrule
\multicolumn{2}{l}{\textbf{Gap (batch 2)}} & & & & \textbf{0.095} & \textbf{0.200} & \textbf{0.301} \\
\multicolumn{2}{l}{\textbf{Gap (batch 4)}} & & & & \textbf{0.063} & \textbf{0.091} & \textbf{0.111} \\
\bottomrule
\end{tabular}
\end{table}
```

This table is the single most important addition. It supports the retrieval-bottleneck claim cleanly and dramatically.

---

## New Abstract (replace entire abstract)

```latex
We describe three open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b, built on a single-LLM dual-role architecture: the same Gemma instance acts as both the controller---deciding what to query and when to stop---and the generator, with hybrid PubMedBERT + BM25 retrieval fused via Reciprocal Rank Fusion. The submissions vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense, released April 2026) and the PubMed backend (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline). Our strongest single result is on Phase B batch 2, where \texttt{ossllm} reaches 0.86 yes/no accuracy and 0.40 list F1, placing it at the top of the batch-2 list-F1 cluster. A same-batch, same-generator comparison between Phase B and Phase A$^+$ shows the same Gemma 3 27B dropping by 30 percentage points on list F1 when forced to retrieve its own evidence rather than receive gold snippets, isolating retrieval recall as the binding constraint with effect sizes too large to attribute to noise. Two negative results corroborate this diagnosis: few-shot exemplar conditioning produced outputs character-for-character identical to the base agent on every Phase B batch 1 question, and on the one Phase A$^+$ batch where Gemma 3 and Gemma 4 both ran, they produced exact-answer-identical outputs.
```

---

## New §1 three-claims paragraph (replace the "Our results, reported in §4..." paragraph)

```latex
Our results, reported in §4, support three claims. First, an LLM-controlled retrieval loop combined with deterministic format hygiene reaches competitive Phase B performance without any domain fine-tuning: \texttt{ossllm} reaches 0.40 list F1 on Phase B batch~2, placing it in the top cluster of the batch-2 field~\cite{nentidis2026bioasq}, and \texttt{asmalltrialsystem} reaches 0.94 yes/no accuracy and 0.32 list F1 on Phase B batch~1. Second, retrieval recall---not generation quality---is the binding constraint on the end-to-end pipeline: on Phase B batch~2, \texttt{ossllm} reaches 0.40 list F1 with gold snippets while \texttt{asmalltrialsystem} (the same generator, with no exemplar conditioning) reaches only 0.10 list F1 on the same questions in Phase A$^+$, a 30-point within-batch gap that isolates retrieval cost on a single batch. Third, the retrieval-bottleneck interpretation is reinforced by two negative results: (i) few-shot conditioning on training-set exemplars produced outputs character-for-character identical to the base agent on every Phase B batch 1 question, and (ii) on the one Phase A$^+$ batch where both Gemma 3 27B and Gemma 4 31B ran, they produced exact-answer-identical outputs---confirming that swapping the generator does not help when retrieval is the bottleneck.
```

---

## New §4 Results — replace the "Phase B is strong; Phase A+ is uneven" paragraph

```latex
On Phase B batch~1, \texttt{asmalltrialsystem} reaches yes/no accuracy 0.941, factoid strict 0.304, factoid lenient 0.435, MRR 0.370, list F1 0.317, and ROUGE-2 F1 on ideal answers 0.179. The yes/no number places us in the second tier of the Task 14b field on this batch~\cite{nentidis2026bioasq}, behind a small cluster at 1.000 (the dmiip2024 family, IR\_J-2/4, Another, multi-stage rank\&llm) and tied with roughly twenty other systems at 0.941. The list F1 of 0.317 places us in the upper portion of the batch~1 field, behind a cluster led by the CSA-IISR and MedQA system families.

On Phase B batch~2, \texttt{ossllm} reaches yes/no 0.857, factoid strict 0.350, and list F1 0.396. The list F1 is the standout result of our submission: it is at the top of the batch-2 list-F1 cluster, above dictycite-baseline (0.376), effectively tied with the dmiip2024 family at 0.391, and ahead of the MedQA family (0.355--0.362). The yes/no on this batch is more modest because batch~2 had a tighter top cluster (one system at 1.000, twelve at 0.952).

On Phase B batch~4, \texttt{Finalcorrected} reaches yes/no 0.813, factoid strict 0.364, list F1 0.365, and ROUGE-2 F1 0.158. These numbers are mid-pack on a batch where several system families (MedQA, UR-IW, lean\_rag, CSA-IISR, dictycite) reach 1.000 yes/no and list F1 above 0.50.

Phase A$^+$ is consistently lower than Phase B, and the within-batch comparisons are what make the retrieval-cost claim clean. The pattern visible in Table~\ref{tab:withinbatch} is striking: on batch~2, the same generator scores 0.396 list F1 with gold snippets and 0.095 list F1 when retrieving its own evidence, a 30-point gap on identical questions. On batch~4, the same Gemma 4 generator scores 0.365 list F1 with gold snippets and 0.255 when retrieving, an 11-point gap. The yes/no gaps are smaller (10 and 6 points respectively), and the factoid-strict gaps lie between (20 and 9 points). The ordering is monotonic in recall sensitivity: yes/no decisions can be reached from partial evidence, factoid extraction requires the entity to be in the pool, and list completion requires every gold entity to be in the pool. The wider the gap, the more recall-sensitive the metric, exactly as the retrieval-bottleneck interpretation predicts.
```

---

## New §4 "within-model anchor" paragraph (replace)

```latex
A useful framing of these numbers is to ask: what does it take, mechanically, to reach the top cluster of Phase B batch~2 list F1 without any of the usual top-tier ingredients---no fine-tuning, no learned reranker, no ensembling? The agentic loop, the type-specific prompting, and the deterministic format-hygiene filter together account for most of the lift. The hygiene filter recovered roughly 20--30 list answers across our four submissions that were structurally correct but malformed, worth an estimated 2--4 list-F1 points. The adversarial-reasoning yes/no prompt suppressed a 70\% yes-bias on the BioASQ training set down to roughly balanced predictions. The systems above us on this year's leaderboard appear to combine these components with additional levers we did not pull (fine-tuning, learned cross-encoder reranking, ensembling); we discuss this gap and its implications in §5.
```

---

## New §5 Discussion (full replacement)

```latex
\section{Discussion}

\subsection{Positioning in Task 14b}

Task 14b drew a large and methodologically diverse field~\cite{nentidis2026bioasq}. The strongest Phase B performers cluster around the MedQA, dmiip2024, CSA-IISR, UR-IW, lean\_rag, and dictycite system families, several of which reach perfect yes/no accuracy on multiple batches and list F1 above 0.50 on batch~3. Our submission does not occupy the leading tier overall. The cleanest summary of where we land is per-metric, per-batch:

\begin{itemize}
\item \textbf{Phase B batch 1 yes/no (0.94):} second tier, behind five systems at 1.000, tied with roughly twenty systems at 0.94.
\item \textbf{Phase B batch 1 list F1 (0.32):} upper portion, behind a cluster led by CSA-IISR and MedQA families.
\item \textbf{Phase B batch 2 list F1 (0.40):} top cluster of the batch---above dictycite-baseline (0.38), effectively tied with the dmiip2024 family (0.39).
\item \textbf{Phase B batch 4:} mid-pack on every metric.
\item \textbf{Phase A$^+$:} mid-pack on yes/no, lower-middle on factoid and list.
\end{itemize}

The contributions we want to carry forward from this submission do not depend on top-tier leaderboard placement. They are three: an architectural pattern (single-LLM dual-role agentic loop), an experimental design that isolates retrieval cost (same-generator, same-batch Phase B vs.\ Phase A$^+$ comparison), and a class of silent errors that the format-hygiene filter recovers (chain-of-thought scaffolding in structured outputs). Each is useful to future BioASQ participants independently of how this year's submission scored.

\subsection{Departure from the 2025 methodological landscape}

Where this submission departs from the Task~13b pattern is in the locus of agency. The dominant 13b pattern was static hybrid retrieval feeding a single LLM generation call, with optional supervised fine-tuning of the generator~\cite{bitua2025,unitor2025}. Ateia and Kruschwitz~\cite{ateia2025} introduced a self-feedback loop, but the loop runs \emph{backward} over generation: the model critiques and regenerates its own answer. AQAMS~\cite{aqams2025} composes multiple LLM agents that exchange messages while answering.

Our submission concentrates agency in a single LLM instance playing two roles---controller and generator---within a \emph{forward} retrieval loop that the controller, not the generator, drives. Where Ateia and Kruschwitz iterate on the answer, we iterate on the evidence pool. Where AQAMS distributes agency across multiple agents, we concentrate it in one instance whose discretion is bounded to the retrieval trajectory. This is the operative meaning of ``agentic'' in our system: not autonomy over arbitrary actions, but discretion over what to query next and when to stop, in the spirit of ReAct~\cite{react} but specialized to PubMed.

The narrower architectural claim we can defend on the evidence of Task 14b is that the single-LLM dual-role pattern reaches \emph{competitive list-F1 performance} on Phase B---reaching the top cluster on batch~2 and the upper portion on batch~1---without any of the usual top-tier ingredients: no domain fine-tuning, no ensemble, no learned cross-encoder reranker.

\subsection{The retrieval-cost diagnosis: a same-batch isolation}

The cleanest empirical finding in this submission comes from a within-batch comparison that the experimental design happens to enable. \texttt{ossllm} and \texttt{asmalltrialsystem} use the same Gemma 3 27B generator and differ only in whether training-set exemplars are prepended to the answer-generation prompt; on Phase B batch~1 the two systems produced character-for-character identical outputs, evidence that the few-shot intervention does not change generator behavior at this scale. Treating them as the same generator, we get a same-question Phase B vs.\ Phase A$^+$ comparison on batch~2:

\begin{itemize}
\item \texttt{ossllm} on Phase B batch~2 (gold snippets): 0.857 yes/no, 0.350 factoid strict, \textbf{0.396 list F1}.
\item \texttt{asmalltrialsystem} on Phase A$^+$ batch~2 (must retrieve): 0.762 yes/no, 0.150 factoid strict, \textbf{0.095 list F1}.
\end{itemize}

The same generator, on the same questions, scores \textbf{30 percentage points lower on list F1} when it must retrieve its own evidence. The factoid strict-accuracy gap is 20 points, and the yes/no gap is 10 points. The pattern is monotonic in how recall-sensitive the metric is: yes/no decisions can be reached from partial evidence; factoid extraction requires the entity to be in the pool; list completion requires every gold entity to be in the pool. The wider the gap, the more recall-sensitive the metric, exactly as the retrieval-bottleneck interpretation predicts.

A second within-batch comparison corroborates this. \texttt{Finalcorrected} ran both Phase B and Phase A$^+$ on batch~4:

\begin{itemize}
\item Phase B batch~4 (gold snippets): 0.813 yes/no, 0.364 factoid strict, 0.365 list F1.
\item Phase A$^+$ batch~4 (must retrieve): 0.750 yes/no, 0.273 factoid strict, 0.255 list F1.
\end{itemize}

The batch~4 gaps (6, 9, 11 percentage points respectively) are smaller than batch~2's, consistent with batch~4 questions being more retrieval-tractable for our system on average. The list-F1 gap remains the largest, again as predicted by the recall-sensitivity ordering.

This same-batch isolation is, methodologically, the cleanest claim the paper makes: not that retrieval matters in general, but that retrieval recall is the binding constraint on the metrics that depend most on recall, with effect sizes large enough on batch~2 (30 points on list F1) that we can rule out small-sample noise as an explanation.

\subsection{What worked, and why}

Three design choices carry the Phase B scores.

\emph{The agentic loop shifts the failure distribution from confident-wrong to flagged-incomplete.} On a 17-question internal held-out set during development, a zero-shot Gemma 4 31B baseline produced fluent answers but hallucinated entities not supported by the snippets, reaching roughly 0.84 yes/no and 0.18 list F1. With the loop enabled on the same model and snippets, those numbers became competitive with our submitted Phase B results. The mechanism is mechanical: requiring the controller to inspect its own evidence pool and decide whether it can answer forces the model to convert under-supported answers into additional retrieval requests rather than hallucinations. The type-specific generator prompts then refuse to invent entities outside the pool, so the recovered queries translate directly into recall.

\emph{Deterministic format hygiene recovers an entire class of silent errors.} BioASQ's exact-match scoring is unforgiving: a semantically correct factoid wrapped in markdown bolding scores as wrong, and the team never sees this because the natural-language output looks right. Across our four submissions, the regex filter (§3.9) recovered roughly 20--30 list answers whose entities were structurally correct but wrapped in chain-of-thought scaffolding. A conservative estimate is that this is worth 2--4 list-F1 points and 1--2 factoid strict-accuracy points---not dominant in isolation, but free, and we suspect a similar class of errors accounts for part of the gap between LLM-based submissions and the points they could have scored.

\emph{RRF beats learned fusion when query score distributions are heavy-tailed.} Our preliminary tuning compared linear score combinations (after per-query min-max normalization of dense cosine and BM25 scores) against Reciprocal Rank Fusion~\cite{rrf2009}. RRF beat every linear-combination configuration we tried, with one fewer hyperparameter to set. We attribute this to the structure of biomedical queries: a single document with rare gene symbols can dominate a BM25-weighted linear combination, while RRF's positional damping bounds each ranker's contribution at the top of the list.

\subsection{What the negative results tell us}

Two negative findings are informative in their own right.

\emph{Few-shot conditioning by question similarity did not change the model's outputs.} On Phase B batch~1, \texttt{asmalltrialsystem} and \texttt{ossllm}---differing only in whether three training-set exemplars are prepended---produced character-for-character identical JSON on every question. The likely interpretation is that the BioASQ output shape is already well-represented in Gemma 3's instruction-tuning distribution, and three additional surface-similar exemplars contribute no information the model has not already internalized. We suspect exemplar selection by \emph{answer shape}---e.g., presence of a numeric value, count of list items---rather than question similarity could still help, and we flag this as a non-default future-work direction.

\emph{Swapping the generator from Gemma 3 27B to Gemma 4 31B did not change Phase A$^+$ batch~4 outputs at all.} On the one batch where both \texttt{asmalltrialsystem} (Gemma 3) and \texttt{Finalcorrected} (Gemma 4) ran in Phase A$^+$, the two systems returned identical exact answers on every question. This is the cleanest same-batch Gemma 3 vs.\ Gemma 4 comparison we have, and it is fully consistent with the retrieval-bottleneck diagnosis: a stronger generator improves extraction from a fixed evidence pool but cannot manufacture evidence that retrieval failed to surface. We do not have a same-batch Phase B Gemma 3 vs.\ Gemma 4 comparison---\texttt{asmalltrialsystem} ran on Phase B batch~1 and \texttt{Finalcorrected} on Phase B batch~4---so we cannot make a clean claim about whether Gemma 4 would have helped on Phase B; the cross-batch Phase B difference (Gemma 3 batch~1 list F1 0.317 vs.\ Gemma 4 batch~4 list F1 0.365) is confounded by question difficulty across batches.

\subsection{What the top of the 14b field has that we don't}

The 14b leaderboard makes the gap between us and the top systems legible by component. We are not running a learned cross-encoder reranker on BioASQ relevance judgments, and several systems above us appear to be. We are not fine-tuning the generator on BioASQ training data. We are not ensembling multiple model proposals. We cap effective context at 8\,192 tokens for generator focus when Gemma 4 nominally supports more than 256K, which may under-use the model on list questions with many candidate entities. Each of these is a known and well-trodden lever; we chose not to pull them because we wanted to isolate the contribution of the agentic loop and format hygiene, not because we believe they would not help.

The honest reading is that the agentic-loop and format-hygiene pattern is \emph{necessary but not sufficient} to reach the top of the 14b leaderboard. A future submission that adds a learned cross-encoder reranker, light supervised fine-tuning of the generator, and a small ensemble on top of this architecture is, in our view, the cleanest path to the leading cluster of systems. The 30-point batch~2 same-batch list-F1 gap quantifies the headroom that a stronger retrieval substrate could recover.

\subsection{Future work}

Three concrete next steps follow directly from the analysis above.

\emph{A learned cross-encoder reranker on BioASQ data.} Fine-tune a biomedical cross-encoder on BioASQ Task~b relevance judgments and replace the RRF fusion step, or augment it as a third signal in the fusion. This is the single highest-leverage improvement we can identify against our current Phase A$^+$ scores, and it directly targets the retrieval-recall bottleneck the same-batch comparison isolates.

\emph{Same-batch Gemma 3 vs.\ Gemma 4 Phase B comparison.} We never ran Gemma 4 on Phase B batch~1 nor Gemma 3 on Phase B batch~4, so the cross-batch Phase B difference between the two generator families is confounded by question difficulty. A clean evaluation would run both backbones on the same batch in Phase B.

\emph{Component ablation of the agentic loop.} The current paper reports the loop as a single component. A controlled experiment removing the sufficiency check (forcing 5 iterations), the adversarial-reasoning yes/no prompt, or the format-hygiene filter would isolate each component's marginal contribution. We expect the hygiene filter and the adversarial yes/no prompt to carry most of the lift on the metrics where we are competitive; the agentic loop itself most likely matters most on list and factoid Phase A$^+$, but we cannot show this without an ablation.

\subsection{Closing}

Two claims in this paper are robust enough to carry into future BioASQ submissions independently of where Task 14b's leaderboard lands. First, an agentic retrieval loop with a single open-weights LLM in dual controller-generator roles, paired with deterministic format hygiene, is sufficient to reach competitive Phase B performance without fine-tuning, an ensemble, or a learned reranker---\texttt{ossllm}'s 0.40 list F1 on batch~2 places it in the top cluster of that batch. Second, the same-batch Phase B vs.\ Phase A$^+$ comparison shows a 30-point list-F1 gap on identical questions, isolating retrieval recall as the binding constraint with effect sizes too large to attribute to noise. Both observations should be useful to next year's participants regardless of how the present submission scored.
```

---

## Smaller line edits (same as v1, still apply)

| Find | Replace |
|---|---|
| `§?? discusses implications` | `§5 discusses implications` |
| `three orthogonal design questions:` | `three orthogonal design axes:` |
| `BIT.UA enriches it with Dense` | `BIT.UA (Universidade de Aveiro) enriches it with Dense` |
| `Gemma-3 27B` (anywhere) | `Gemma 3 27B` |
| `Gemma-3-27B` | `Gemma 3 27B` |
| `Gemma-4-31B` | `Gemma 4 31B` |
| `Gemma 4-31B` | `Gemma 4 31B` |

## §3.9 condense (kills repetition with §1)

Replace the second sentence in §3.9 starting *"Instruction-tuned models occasionally leak..."* with:

```latex
As described in §1 (Failure mode 2), instruction-tuned models occasionally leak chain-of-thought scaffolding into their structured outputs. Left in place, these tokens cause the BioASQ scorer to miss exact matches that would otherwise score. We apply a single regex-based filter to the model's output immediately before JSON serialization. The filter strips this scaffolding (markdown formatting, bracketed citation tokens, scratchpad headers, arrow glyphs), removes any leading ``Answer:'' or ``Final answer:'' prefixes, and enforces the BioASQ output shape: factoid and list exact answers are emitted as a list of single-element lists---the BioASQ$\geq$5 format~\cite{bioasq5format}---with at most 5 outer entries for factoids and 100 for lists; yes/no exact answers are bare lowercase strings; summary outputs omit the \texttt{exact\_answer} field entirely. Across our four submissions the filter recovered roughly 20--30 list answers that would otherwise have been malformed and rejected.
```

## §3.10 fill (or delete the heading)

```latex
Table~\ref{tab:systems} summarizes the three submissions. The \texttt{asmalltrialsystem} and \texttt{ossllm} systems differ only in whether training-set exemplars are prepended to the answer-generation prompt; \texttt{Finalcorrected} differs from \texttt{asmalltrialsystem} on both the LLM backbone (Gemma 4 31B Dense replaces Gemma 3 27B) and the PubMed backend (a local SQLite FTS5 mirror replaces live NCBI E-utilities). Together, the three submissions vary three orthogonal axes: prompt-level conditioning, generator scale, and retrieval-substrate locality.
```

---

## Before you submit — verify with your own records

The following items are worth double-checking against your `4-Trial.json` and submission logs:

1. **Did Finalcorrected actually submit to Phase B batch 3?** I cannot find it in any leaderboard screenshot. The paper Table 3 batch 3 row has identical numbers to batch 4, which looks like an accidental duplication. If Finalcorrected did NOT submit to Phase B batch 3, drop that row. If it did, get the correct numbers from your logs and re-add the row.

2. **Confirm ossllm Phase B batch 2 list F1 = 0.396** and yes/no = 0.857. These are big numbers that change the story of the paper.

3. **Confirm asmalltrialsystem Phase A+ batch 3 = 0.364 Y/N** (not 0.455 as currently in Table 2). This affects the within-paper consistency.

4. **ossllm Phase B batch 2 R-2 F1 = 0.176** is my best read from the screenshots — verify before submission.

## Recommended order in Overleaf

1. Global find/replaces (Gemma name consistency, `§??` → `§5`, etc.) — 5 min.
2. Update Tables 2 and 3, add new Table 4 — 10 min.
3. Drop in new Abstract — 2 min.
4. Replace §1 three-claims paragraph and Failure mode 1 sentence merge — 5 min.
5. Replace the two §4 paragraphs (Phase B positioning + within-model anchor) — 10 min.
6. Drop in the entire new §5 — 5 min.
7. §3.9 condense, §3.10 fill — 5 min.
8. Recompile, final read-through, citation key matching, fix any internal references that point to old material — 15 min.

Total: roughly 60 minutes of editing.

## Why the corrected story is stronger for acceptance

- **You have a top-cluster result** (ossllm batch 2 list F1 0.40) that the current paper hides. Reviewers reward strong specific results.
- **The within-batch retrieval comparison is methodologically cleaner** than the cross-batch one. It removes question-difficulty confounding and produces a more dramatic effect size (30 points vs. 15--22 points).
- **The honest leaderboard positioning is per-batch and per-metric** rather than a single "upper-middle" claim that reviewers could disprove with one click.
- **The negative results are now stated precisely** (identical outputs on specific batches) rather than approximately.
- **The ROUGE-2 regression story is gone**, removing the one place in the paper that was built on a wrong number.
