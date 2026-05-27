# BioASQ Task 14b Paper — Revision Package v3 (Final)

All numbers in this file come from the user's authoritative submission portal. This supersedes v1 and v2.

**What changed from the paper as currently written:** The paper's Tables 2 and 3 numbers are correct as-is. What needs revising is mostly stylistic: abstract tightening, advisor-flagged sentence rewrites, the broken `§??` cross-reference, name consistency, §3.9 redundancy, and honest field positioning (the paper currently claims "upper-middle of the field" overall, but per your own ranking table you're upper-middle only on Phase B batch 1 yes/no; middle-of-field on list, factoid, and ideal R-2; and lower-middle on Phase A+).

---

## 1. Abstract (full replacement)

```latex
We describe three open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b. All three share an LLM-controlled retrieval loop with hybrid PubMedBERT + BM25 reranking fused via Reciprocal Rank Fusion. They differ along three axes: the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense, released April 2026), the PubMed backend (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline), and whether training-set exemplars condition the answer-generation prompt. On Phase B (gold snippets) our best system reaches 0.94 yes/no accuracy on batch 1, placing it in the upper-middle quartile of the Task 14b field, and 0.37 list F1 on batch 3 (middle of field). On Phase A$^+$ batch 4, the same Gemma 4 generator drops to 0.75 yes/no and 0.26 list F1---an 11-point list-F1 gap on identical questions that isolates retrieval recall as the binding constraint on the end-to-end pipeline. Two negative results corroborate this diagnosis: training-set exemplar conditioning produced outputs character-for-character identical to the base agent on Phase B batch 1, and on the one Phase A$^+$ batch where both Gemma 3 27B and Gemma 4 31B ran, they returned identical exact answers across all questions.
```

---

## 2. §1 — three-claims paragraph (replace "Our results, reported in §4, support three claims.")

```latex
Our results, reported in §4, support three claims. First, an LLM-controlled retrieval loop combined with deterministic format hygiene reaches competitive Phase B performance without any domain fine-tuning: our best system attains 0.94 yes/no accuracy on Phase B batch~1, placing it in the upper-middle quartile of the Task 14b field~\cite{nentidis2026bioasq}, while list F1, factoid strict, and ideal-answer ROUGE-2 land in the middle of the field. Second, retrieval recall---not generation quality---is the binding constraint on the end-to-end pipeline: on the one batch where the same Gemma 4 generator ran both phases, the Phase B--Phase A$^+$ gap is 11 points on list F1 (0.365 vs.\ 0.255), 9 points on factoid strict, and 6 points on yes/no on identical questions, and the ordering is monotonic in how recall-sensitive the metric is. Third, the retrieval-bottleneck interpretation is reinforced by two negative results: (i) few-shot conditioning on training-set exemplars produced outputs character-for-character identical to the base agent on every Phase B batch~1 question, and (ii) on Phase A$^+$ batch~4 the Gemma 3 27B and Gemma 4 31B variants of our system returned exact-answer-identical outputs---confirming that a stronger generator does not help when retrieval is the bottleneck.
```

---

## 3. §1 — Failure mode 1 sentence merge (advisor's comment)

Replace the two sentences beginning *"A static retrieval pipeline lacks the feedback loop..."* through *"...is precisely the set of moves a biomedical expert performs when searching manually."* with:

```latex
This bottleneck is structural rather than cognitive: a static pipeline cannot replicate the iterative search behavior biomedical experts rely on---decomposing compound questions into sub-queries, detecting when the evidence pool is insufficient, or backing off from over-narrow phrasings. Generator fine-tuning cannot recover these moves either, because the bottleneck is not in reasoning over the snippets the model receives; it is in which snippets reach the model in the first place.
```

---

## 4. §1 small fixes (find/replace)

| Find | Replace |
|---|---|
| `§?? discusses implications` | `§5 discusses implications` |
| `three orthogonal design questions:` | `three orthogonal design axes:` |
| `BIT.UA enriches it with Dense` | `BIT.UA (Universidade de Aveiro) enriches it with Dense` |

## 5. Global Gemma name consistency

| Find | Replace |
|---|---|
| `Gemma-3 27B` | `Gemma 3 27B` |
| `Gemma-3-27B` | `Gemma 3 27B` |
| `Gemma-4-31B` | `Gemma 4 31B` |
| `Gemma 4-31B` | `Gemma 4 31B` |

---

## 6. §4 Results — replace the "Phase B is strong; Phase A+ is uneven" paragraph

```latex
On Phase B batch~1, \texttt{asmalltrialsystem} reaches yes/no accuracy 0.941, factoid strict 0.304, factoid lenient 0.435, MRR 0.370, list F1 0.317, and ROUGE-2 F1 on ideal answers 0.179. The yes/no number places us in the upper-middle quartile of the Task 14b field on this batch~\cite{nentidis2026bioasq}, behind a small cluster of systems at perfect 1.000 accuracy (the dmiip2024 family, IR\_J variants, and others) and tied with roughly twenty other submissions at 0.941. The list F1, factoid strict, and ideal-answer ROUGE-2 F1 numbers sit in the middle of the field. On Phase B batches~3 and~4, \texttt{Finalcorrected} reaches 0.813 yes/no, 0.364 factoid strict, 0.365 list F1, and ROUGE-2 F1 of 0.158 and 0.087 respectively---mid-pack on yes/no and list F1 across a field where the strongest performers (MedQA, UR-IW, lean\_rag, CSA-IISR, dictycite families) reach 1.000 yes/no and list F1 above 0.50 on batch~3. The within-model ROUGE-2 drop from batch~3 to batch~4 is large; we discuss it in §5.

Phase A$^+$ is consistently lower than Phase B. The cleanest comparison is on batch~4, the only batch where \texttt{Finalcorrected} ran both phases: the same Gemma 4 generator scores 0.813 vs.\ 0.750 on yes/no, 0.364 vs.\ 0.273 on factoid strict, and 0.365 vs.\ 0.255 on list F1 between Phase B and Phase A$^+$. Because the generator is held fixed, this within-batch gap isolates retrieval cost in something close to isolation. The gaps grow with recall sensitivity: 6 points on yes/no (binary decisions reachable from partial evidence), 9 points on factoid strict (entity must be in the pool), 11 points on list F1 (every gold entity must be in the pool). The Task 14b field-level rankings confirm this diagnosis by component: our Phase A$^+$ list F1 (0.255, lower-middle) and yes/no (0.750, lower-middle) are explicitly flagged in the leaderboard as bounded by retrieval recall and retrieval-induced noise.
```

## 7. §4 Results — replace the "Few-shot conditioning is a null result" paragraph

```latex
\texttt{asmalltrialsystem} and \texttt{ossllm} differ only in whether three training-set exemplars (selected by question-embedding cosine similarity) are prepended to the answer-generation prompt. On Phase B batch~1, the two systems produced character-for-character identical JSON on every question and therefore identical scores across every metric (0.941 yes/no, 0.317 list F1, 0.179 ROUGE-2 F1). We interpret this as the model being already well-calibrated by its instruction tuning to the BioASQ output shape, so that the exemplars contribute information the model has already internalized. The broader implication---worth reporting because the BioASQ literature contains several single-batch few-shot claims---is that exemplar conditioning by surface (question-embedding) similarity is not a robust intervention at this scale. An exemplar-selection method based on answer shape rather than question similarity might still help, and is left for future work.
```

## 8. §4 Results — replace the "Same evidence pool ⇒ same answer" paragraph

```latex
\texttt{Finalcorrected} and \texttt{asmalltrialsystem} differ on two non-trivial axes simultaneously: a Gemma 3 27B controller becomes a Gemma 4 31B controller, and a live NCBI E-utilities backend becomes a local SQLite FTS5 mirror of the Annual Baseline. On Phase A$^+$ batch~4, the two systems return identical exact answers on every question, despite this double change. The reading is that on this batch the converged evidence pool was small enough and the questions narrow enough that both pipelines collapsed to the same top-$k$ chunks after RRF fusion, and that on tightly bounded biomedical questions Gemma 3 and Gemma 4 extract the same exact entity from the same evidence. This is the cleanest same-batch Gemma 3 vs.\ Gemma 4 comparison we have, and it is fully consistent with the retrieval-bottleneck diagnosis: a stronger generator improves extraction from a fixed evidence pool but cannot manufacture evidence that retrieval failed to surface.
```

## 9. §4 Results — replace the "Gemma 4 helps factoid and list, hurts ROUGE-2" paragraph

```latex
The cross-batch comparison between Gemma 3 (\texttt{asmalltrialsystem} on Phase B batch~1) and Gemma 4 (\texttt{Finalcorrected} on Phase B batches~3 and~4) is confounded by question difficulty, since the batches contain different questions, but two patterns are visible. On factoid strict and list F1, Gemma 4 scores higher (0.364 vs.\ 0.304 strict; 0.365 vs.\ 0.317 list F1), consistent with Gemma 4's stronger entity-extraction style: it more reliably outputs the noun phrase the BioASQ scorer expects rather than wrapping it in a clause. On ideal-answer ROUGE-2 F1, the picture is more complicated: Gemma 3 on batch~1 scored 0.179, Gemma 4 on batch~3 scored 0.158 (a small drop), and Gemma 4 on batch~4 scored 0.087 (a large drop). The within-Gemma-4 drop from batch~3 to batch~4 suggests that batch difficulty, not the model upgrade, is the dominant factor, and we caution against attributing the regression to Gemma 4 alone. The likely contributing cause is Gemma 4's longer default generation style: ROUGE-2 F1 includes precision against the gold summary, and longer model summaries pay an n-gram precision penalty. We did not impose a stricter word cap on Gemma 4's summary generation; in retrospect the prompt should have been re-tuned when the model was swapped.
```

---

## 10. §5 Discussion — full replacement

```latex
\section{Discussion}

\subsection{Positioning in Task 14b}

Task 14b drew a large and methodologically diverse field~\cite{nentidis2026bioasq}. The strongest Phase B performers cluster around the MedQA, dmiip2024, CSA-IISR, UR-IW, lean\_rag, and dictycite system families, several of which reach perfect yes/no accuracy on multiple batches and list F1 above 0.50 on batch~3. Our submission sits in the field as follows, by metric and batch:

\begin{itemize}
\item \textbf{Phase B yes/no, batch~1:} 0.941, upper-middle quartile.
\item \textbf{Phase B list F1, batch~3:} 0.365 (field leader $\sim$0.50), middle of field. The dominant gap to the leaders is coverage of long-tail entities.
\item \textbf{Phase B factoid strict, batch~3:} 0.364 (field leader $\sim$0.53), middle of field. The dominant gap is short-form extraction precision.
\item \textbf{Phase B ideal-answer ROUGE-2 F1, batch~1:} 0.179 (field leader $\sim$0.25), middle of field. The dominant gap is verbose generation against terse gold summaries.
\item \textbf{Phase A$^+$ list F1, batch~4:} 0.255 (field leader $\sim$0.58), lower-middle. The dominant gap is retrieval recall.
\item \textbf{Phase A$^+$ yes/no, batch~4:} 0.750 (field leader 0.938), lower-middle. The dominant gap is retrieval-induced noise.
\end{itemize}

The contributions we want to carry forward from this submission do not depend on top-tier leaderboard placement. They are three: an architectural pattern (single-LLM dual-role agentic loop), an experimental design that isolates retrieval cost (same-generator, same-batch Phase B vs.\ Phase A$^+$ comparison on batch~4), and a class of silent errors that the format-hygiene filter recovers (chain-of-thought scaffolding in structured outputs). Each is useful to future BioASQ participants independently of how this year's submission scored.

\subsection{Departure from the 2025 methodological landscape}

Where this submission departs from the Task~13b pattern is in the locus of agency. The dominant 13b pattern was static hybrid retrieval feeding a single LLM generation call, with optional supervised fine-tuning of the generator~\cite{bitua2025,unitor2025}. Ateia and Kruschwitz~\cite{ateia2025} introduced a self-feedback loop, but the loop runs \emph{backward} over generation: the model critiques and regenerates its own answer. AQAMS~\cite{aqams2025} composes multiple LLM agents that exchange messages while answering.

Our submission concentrates agency in a single LLM instance playing two roles---controller and generator---within a \emph{forward} retrieval loop that the controller, not the generator, drives. Where Ateia and Kruschwitz iterate on the answer, we iterate on the evidence pool. Where AQAMS distributes agency across multiple agents, we concentrate it in one instance whose discretion is bounded to the retrieval trajectory. This is the operative meaning of ``agentic'' in our system: not autonomy over arbitrary actions, but discretion over what to query next and when to stop, in the spirit of ReAct~\cite{react} but specialized to PubMed.

\subsection{The retrieval-cost diagnosis}

The cleanest empirical claim in this paper is the within-batch Phase B vs.\ Phase A$^+$ comparison on batch~4, the only batch where \texttt{Finalcorrected} ran both phases. Because the generator is held constant (same Gemma 4 31B, same prompts, same format-hygiene filter), the Phase B--Phase A$^+$ gap on identical questions isolates the cost of having to retrieve evidence rather than receiving gold snippets:

\begin{itemize}
\item Yes/no: 0.813 (Phase B) vs.\ 0.750 (Phase A$^+$) --- 6-point gap.
\item Factoid strict: 0.364 vs.\ 0.273 --- 9-point gap.
\item List F1: 0.365 vs.\ 0.255 --- 11-point gap.
\end{itemize}

The ordering is monotonic in how recall-sensitive the metric is: yes/no decisions can be reached from partial evidence; factoid extraction requires the answer entity to be in the pool; list completion requires every gold entity to be in the pool. The wider the gap, the more recall-sensitive the metric, exactly as the retrieval-bottleneck interpretation predicts. The leaderboard's per-metric bottleneck attribution on Phase A$^+$ batch~4---retrieval recall for list, retrieval-induced noise for yes/no---confirms this externally.

A cross-batch view of the same generator (Gemma 3 27B \texttt{asmalltrialsystem} on Phase B batch~1 vs.\ Phase A$^+$ batches~2, 3, 4) shows substantial Phase B--Phase A$^+$ differences (0.941 yes/no vs.\ 0.455--0.762; 0.317 list F1 vs.\ 0.095--0.255), but those differences are confounded by question difficulty across batches. The clean methodological claim is the within-batch one.

\subsection{What worked, and why}

Three design choices carry the Phase B scores.

\emph{The agentic loop shifts the failure distribution from confident-wrong to flagged-incomplete.} On a 17-question internal held-out set during development, a zero-shot Gemma 4 31B baseline produced fluent answers but hallucinated entities not supported by the snippets, reaching roughly 0.84 yes/no and 0.18 list F1. With the loop enabled on the same model and snippets, those numbers became competitive with our submitted Phase B results. Requiring the controller to inspect its own evidence pool and decide whether it can answer forces the model to convert under-supported answers into additional retrieval requests rather than hallucinations. The type-specific generator prompts then refuse to invent entities outside the pool, so the recovered queries translate directly into recall.

\emph{Deterministic format hygiene recovers an entire class of silent errors.} BioASQ's exact-match scoring is unforgiving: a semantically correct factoid wrapped in markdown bolding scores as wrong. Across our four submissions, the regex filter (§3.9) recovered roughly 20--30 list answers whose entities were structurally correct but wrapped in chain-of-thought scaffolding---worth an estimated 2--4 list-F1 points and 1--2 factoid strict-accuracy points across the four submissions.

\emph{RRF beats learned fusion when query score distributions are heavy-tailed.} Our preliminary tuning compared linear score combinations (after per-query min-max normalization of dense cosine and BM25 scores) against Reciprocal Rank Fusion~\cite{rrf2009}. RRF beat every linear-combination configuration we tried, with one fewer hyperparameter to set. A single document with rare gene symbols can dominate a BM25-weighted linear combination, while RRF's positional damping bounds each ranker's contribution at the top of the list.

\subsection{What the negative results tell us}

Two negative findings are informative in their own right.

\emph{Few-shot conditioning by question similarity did not change the model's outputs.} \texttt{asmalltrialsystem} and \texttt{ossllm} produced character-for-character identical JSON on every Phase B batch~1 question, tying at exactly 0.941 / 0.938 / 0.304 / 0.370 / 0.317 / 0.179 across every metric. The likely interpretation is that the BioASQ output shape is already well-represented in Gemma 3's instruction-tuning distribution, and three additional surface-similar exemplars contribute no information the model has not already internalized. We suspect exemplar selection by \emph{answer shape}---e.g., presence of a numeric value, count of list items---could still help, and we flag this as a non-default future-work direction.

\emph{Swapping Gemma 3 27B for Gemma 4 31B did not change Phase A$^+$ batch~4 outputs.} On the one batch where both \texttt{asmalltrialsystem} (Gemma 3) and \texttt{Finalcorrected} (Gemma 4) ran in Phase A$^+$, the two systems returned identical exact answers on every question (0.750 / 0.746 / 0.273 / 0.273 / 0.255), despite differing on both the generator backbone and the PubMed backend. This is fully consistent with the retrieval-bottleneck diagnosis: a stronger generator improves extraction from a fixed evidence pool but cannot manufacture evidence that retrieval failed to surface.

We do not have a same-batch Phase B Gemma 3 vs.\ Gemma 4 comparison---\texttt{asmalltrialsystem} ran on Phase B batch~1 and \texttt{Finalcorrected} on Phase B batches~3 and~4---so we cannot make a clean claim about whether Gemma 4 would have helped on Phase B. The cross-batch Phase B differences (Gemma 3 batch~1 list F1 0.317 vs.\ Gemma 4 batch~3 list F1 0.365; ROUGE-2 F1 0.179 vs.\ 0.158 and 0.087) are confounded by question difficulty, as the within-Gemma-4 batch~3-to-batch~4 ROUGE-2 drop (0.158 to 0.087) makes clear: the model is constant across those two numbers, yet ROUGE-2 nearly halves.

\subsection{What the top of the 14b field has that we don't}

The 14b leaderboard makes the gap between us and the top systems legible by component. We are not running a learned cross-encoder reranker on BioASQ relevance judgments, and several systems above us appear to be. We are not fine-tuning the generator on BioASQ training data. We are not ensembling multiple model proposals. We cap effective context at 8\,192 tokens for generator focus when Gemma 4 nominally supports more than 256K, which may under-use the model on list questions with many candidate entities. Each of these is a known and well-trodden lever; we chose not to pull them because we wanted to isolate the contribution of the agentic loop and format hygiene, not because we believe they would not help.

The honest reading is that the agentic-loop and format-hygiene pattern is \emph{necessary but not sufficient} to reach the top of the 14b leaderboard. A future submission that adds a learned cross-encoder reranker, light supervised fine-tuning of the generator, and a small ensemble on top of this architecture is, in our view, the cleanest path to the leading cluster of systems. The 11-point batch~4 within-batch list-F1 gap quantifies the headroom that a stronger retrieval substrate could recover.

\subsection{Future work}

Three concrete next steps follow directly from the analysis above.

\emph{A learned cross-encoder reranker on BioASQ data.} Fine-tune a biomedical cross-encoder on BioASQ Task~b relevance judgments and replace the RRF fusion step, or augment it as a third signal. This is the single highest-leverage improvement we can identify against our current Phase A$^+$ scores, and it directly targets the retrieval-recall bottleneck the within-batch comparison isolates.

\emph{Type-aware context budgeting and prompt re-tuning per model.} Re-calibrate per-type prompts when swapping the model backbone, with explicit word caps re-set for each model's verbosity prior. This would likely shrink the cross-batch ROUGE-2 differences attributable to Gemma 4's longer default generation style.

\emph{Component ablation of the agentic loop.} The current paper reports the loop as a single component. A controlled experiment removing the sufficiency check (forcing 5 iterations), the adversarial-reasoning yes/no prompt, or the format-hygiene filter would isolate each component's marginal contribution.

\subsection{Closing}

Two claims in this paper are robust enough to carry into future BioASQ submissions independently of where Task 14b's leaderboard lands. First, an agentic retrieval loop with a single open-weights LLM in dual controller-generator roles, paired with deterministic format hygiene, is sufficient to reach upper-middle quartile Phase B yes/no performance without fine-tuning, an ensemble, or a learned reranker. Second, holding the generator constant across Phases B and A$^+$ on batch~4 yields an 11-point within-batch list-F1 gap on identical questions, isolating retrieval recall as the binding constraint with effect sizes that scale monotonically with metric recall-sensitivity. Both observations should be useful to next year's participants regardless of how the present submission scored.
```

(Replace `\cite{...}` keys with your actual BibTeX keys: `nentidis2026bioasq` = ref [2], `bitua2025` = [4], `unitor2025` = [5], `ateia2025` = [6], `aqams2025` = [11], `react` = [13], `rrf2009` = [7].)

---

## 11. §3.9 condense (kills repetition with §1)

Replace the second sentence in §3.9 starting *"Instruction-tuned models occasionally leak..."* with:

```latex
As described in §1 (Failure mode 2), instruction-tuned models occasionally leak chain-of-thought scaffolding into their structured outputs. Left in place, these tokens cause the BioASQ scorer to miss exact matches that would otherwise score. We apply a single regex-based filter to the model's output immediately before JSON serialization. The filter strips this scaffolding (markdown formatting, bracketed citation tokens, scratchpad headers, arrow glyphs), removes any leading ``Answer:'' or ``Final answer:'' prefixes, and enforces the BioASQ output shape: factoid and list exact answers are emitted as a list of single-element lists---the BioASQ$\geq$5 format~\cite{bioasq5format}---with at most 5 outer entries for factoids and 100 for lists; yes/no exact answers are bare lowercase strings; summary outputs omit the \texttt{exact\_answer} field entirely. Across our four submissions the filter recovered roughly 20--30 list answers that would otherwise have been malformed and rejected.
```

## 12. §3.10 fill (or delete the heading)

```latex
Table~\ref{tab:systems} summarizes the three submissions. The \texttt{asmalltrialsystem} and \texttt{ossllm} systems differ only in whether training-set exemplars are prepended to the answer-generation prompt; \texttt{Finalcorrected} differs from \texttt{asmalltrialsystem} on both the LLM backbone (Gemma 4 31B Dense replaces Gemma 3 27B) and the PubMed backend (a local SQLite FTS5 mirror replaces live NCBI E-utilities). Together, the three submissions vary three orthogonal axes: prompt-level conditioning, generator scale, and retrieval-substrate locality.
```

---

## 13. Tables — no changes needed

Your existing Tables 1, 2, and 3 are correct as printed. No changes required.

---

## 14. Recommended order in Overleaf

1. Global find/replaces (Gemma name consistency, `§??` → `§5`, "questions" → "axes") — 3 min.
2. Drop in the new Abstract — 2 min.
3. Replace the §1 three-claims paragraph and the Failure mode 1 sentence merge — 5 min.
4. Replace the four §4 Results paragraphs (Phase B positioning + few-shot null + same-evidence + Gemma comparison) — 15 min.
5. Drop in the entire new §5 — 5 min.
6. §3.9 condense, §3.10 fill — 5 min.
7. Recompile and final read-through; fix any internal `\ref`/`\cite` keys that point to old material — 15 min.

Total: roughly 50 minutes.

---

## Summary of what's actually changing

- **Numbers in the paper itself**: nothing changes. Tables 1, 2, 3 stay as-is.
- **Abstract**: tightened, honest about per-metric positioning, drops the "≈0.2 gap" overclaim, drops "Few-shot conditioning on training-set exemplars gave no measurable lift over the base agent" (replaced with the stronger "character-for-character identical" framing).
- **§1**: three-claims rewrite, failure-mode-1 sentence merge, broken cross-reference fix, "questions" → "axes", BIT.UA gloss.
- **§4**: four paragraph rewrites that align the prose with your own ranking-table positioning (upper-middle quartile on yes/no batch 1, middle on others, lower-middle on Phase A+), use the cleanest within-batch retrieval comparison (Finalcorrected batch 4) as the centerpiece, and walk back the ROUGE-2 regression narrative from "Gemma 4 caused this" to "batch difficulty is the dominant factor, Gemma 4 may contribute."
- **§5**: restructured to lead with honest per-metric positioning, then the within-batch retrieval diagnosis, then the negative results, then what the top of the field has that we don't.
- **§3.9**: condensed to remove repetition with §1.
- **§3.10**: filled in.
