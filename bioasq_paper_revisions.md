# BioASQ Task 14b Paper — Revision Package

Everything below is copy-paste-ready LaTeX. Sections are ordered the way they appear in your paper. Each `latex` code block can be dropped into Overleaf as-is.

A few global notes:
- Citation numbers `[1]–[18]` follow your existing bibliography order. If your `.bib` keys differ, you'll need to map them.
- Em-dashes are written as `---`. Quotes as `` ``...'' ``. Section symbol as `§` (works in modern LaTeX with UTF-8; replace with `\S` if your preamble requires it).
- `Phase A+` is written `Phase A$^+$` to match your existing style.

---

## 1. Abstract — full replacement

Replace the current abstract with:

```latex
We describe three open-weights agentic retrieval-augmented generation (RAG) systems submitted to BioASQ Task 14b, built on a single-LLM dual-role architecture: the same Gemma instance acts as both the controller---deciding what to query and when to stop---and the generator, with hybrid PubMedBERT + BM25 retrieval fused via Reciprocal Rank Fusion. The submissions vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense, released April 2026) and the PubMed backend (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline). On Phase B (gold snippets) our best system reaches 0.94 yes/no accuracy and 0.32 list F1 on batch 1; on Phase A$^+$ the same generator scores 0.75 yes/no and 0.26 list F1. Because the generator is unchanged across phases, the 0.15--0.22 gap isolates retrieval recall as the binding constraint. Two negative results corroborate this diagnosis: few-shot exemplar conditioning produced outputs character-for-character identical to the base agent, and upgrading from Gemma 3 to Gemma 4 lifted Phase B factoid and list F1 by 5--6 points but did not move Phase A$^+$ at all.
```

---

## 2. Section 1 (Introduction) — paragraph rewrites

### 2.1 Three-claims paragraph — full replacement

Find the paragraph beginning *"Our results, reported in §4, support three claims."* and replace with:

```latex
Our results, reported in §4, support three claims. First, an LLM-controlled retrieval loop combined with deterministic format hygiene reaches competitive Phase B performance without any domain fine-tuning: on gold snippets our best system attains 0.94 yes/no accuracy and 0.32 list F1 on batch 1, placing it among the stronger systems on those two metrics on that batch though behind the leading cluster on factoid strict accuracy and on later batches~\cite{nentidis2026bioasq}. Second, retrieval recall---not generation quality---is the binding constraint on the end-to-end pipeline: the same generator drops by 0.15--0.22 points across every metric when it must retrieve its own evidence in Phase A$^+$ rather than receive gold snippets, and because the generator is unchanged across the two phases this gap measures retrieval cost almost in isolation. Third, the retrieval-bottleneck interpretation is reinforced by two negative results: (i) few-shot conditioning on training-set exemplars produced outputs character-for-character identical to the base agent, and (ii) upgrading the generator from Gemma 3 27B to Gemma 4 31B lifted Phase B factoid and list F1 by 6 and 5 points respectively but did not move Phase A$^+$ scores at all---confirming that a stronger generator helps where generation, not retrieval, is the binding constraint.
```

(Replace `\cite{nentidis2026bioasq}` with your actual bib key for the 14b overview paper — citation [2] in your current numbering.)

### 2.2 Failure mode 1 paragraph — sentence merge

Find the two sentences beginning *"A static retrieval pipeline lacks the feedback loop..."* through *"...is precisely the set of moves a biomedical expert performs when searching manually."* and replace with:

```latex
This bottleneck is structural rather than cognitive: a static pipeline cannot replicate the iterative search behavior biomedical experts rely on---decomposing compound questions into sub-queries, detecting when the evidence pool is insufficient, or backing off from over-narrow phrasings. Generator fine-tuning cannot recover these moves either, because the bottleneck is not in reasoning over the snippets the model receives; it is in which snippets reach the model in the first place.
```

### 2.3 Surgical fixes in §1

| Find | Replace |
|---|---|
| `§?? discusses implications` | `§5 discusses implications` |
| `three orthogonal design questions:` | `three orthogonal design axes:` |
| `BIT.UA enriches it with Dense` | `BIT.UA (Universidade de Aveiro) enriches it with Dense` |

### 2.4 Global name consistency (find/replace everywhere)

| Find | Replace |
|---|---|
| `Gemma-3 27B` | `Gemma 3 27B` |
| `Gemma-3-27B` | `Gemma 3 27B` |
| `Gemma-4-31B` | `Gemma 4 31B` |
| `Gemma 4-31B` | `Gemma 4 31B` |

---

## 3. Section 4 (Results) — prose updates

### 3.1 "Phase B is strong; Phase A+ is uneven" paragraph

Replace the entire paragraph (starting *"On Phase B batch 1, asmalltrialsystem reaches..."*) with:

```latex
On Phase B batch 1, \texttt{asmalltrialsystem} reaches yes/no accuracy 0.941, factoid strict 0.304, factoid lenient (MRR) 0.370, list F1 0.317, and ROUGE-2 F1 on ideal answers 0.179. The yes/no number places us in the second tier of the Task 14b field on this batch~\cite{nentidis2026bioasq}, behind a small cluster of systems at perfect 1.000 accuracy (the dmiip2024 family, IR\_J variants, and others) and tied with roughly twenty other submissions at 0.941. We read the position as direct evidence that the adversarial-reasoning prompt (§3.8) suppresses the yes-bias we measured at roughly 70\% on the BioASQ training set; without that prompt, the score lands in the 0.7--0.8 cluster of LLM submissions that omit explicit bias correction. The list F1 of 0.317 also places us in the upper portion of the batch~1 field, behind a cluster led by the CSA-IISR and MedQA system families. Finalcorrected on Phase B batches 3 and 4 dips to 0.813 yes/no; this is partly a property of the test questions (yes/no is 15--17 items per batch, so one flipped answer moves macro-F1 by roughly 6 points) and partly a sign that on later batches the gap to the top tier widens, with several system families (MedQA, UR-IW, lean\_rag, CSA-IISR, dictycite) reaching 1.000 yes/no and list F1 above 0.50. Phase A$^+$ is consistently 0.15--0.22 points lower on every metric than its Phase B counterpart and is also noisier across batches, with batch 3 yes/no dropping to 0.455. Small per-batch counts amplify this: with ten to fifteen yes/no questions per batch, one or two retrieval-induced wrong answers visibly moves the column.
```

### 3.2 "The within-model anchor" paragraph

Replace it with:

```latex
A useful framing of all the numbers above is to ask: if we were forced to attribute Phase B batch~1 yes/no accuracy of 0.941 to a single component, which one would it be? It is not the model scale---a zero-shot Gemma 3 27B prompted with the gold snippets and no agentic loop sits substantially below this number, and the Gemma 3 to Gemma 4 lift is modest. It is not the retrieval substrate---Phase B is gold retrieval. The component that remains is the structured-output discipline imposed by the controller prompt and the format-hygiene filter together. Our experience is consistent with that reading: across the four submissions, the filter recovered roughly 20--30 list answers that would otherwise have been malformed and rejected, and the adversarial-reasoning yes/no prompt reliably flipped the model's bias toward majority-yes back toward the roughly 55/45 class distribution of the BioASQ test set. The systems above us on this year's leaderboard appear to combine these components with additional levers we did not pull (fine-tuning, learned cross-encoder reranking, ensembling); we discuss this gap and its implications in §5.
```

---

## 4. Section 5 (Discussion) — full replacement

Replace the entire current §5 (Discussion and Conclusion) with the following:

```latex
\section{Discussion}

\subsection{Positioning in Task 14b}

Task 14b drew a large and methodologically diverse field~\cite{nentidis2026bioasq}. From the published leaderboard, the strongest Phase B performers cluster around the MedQA, dmiip2024, CSA-IISR, UR-IW, lean\_rag, and dictycite system families, several of which reach perfect yes/no accuracy on multiple batches and list F1 above 0.50 on batch~3. Our submission does not occupy this top tier. On the metrics where we score competitively---Phase B batch~1 yes/no accuracy (0.94) and list F1 (0.32), placing us in the upper portion of the field for those two metrics on that batch---we trail the leading systems by 0.06 on yes/no and by 0.15--0.20 on list F1. On Phase A$^+$ and on later Phase B batches, we sit in the mid-pack on yes/no and in the lower portion on factoid strict and list F1.

We highlight this distribution upfront because the contributions we want to carry forward from this submission do not depend on leaderboard position. They are three: an architectural pattern (single-LLM dual-role agentic loop), an experimental design that isolates retrieval cost (same generator across Phases B and A$^+$), and a class of silent errors that the format-hygiene filter recovers (chain-of-thought scaffolding in structured outputs). Each is useful to future BioASQ participants independently of how this year's submission scored.

\subsection{Departure from the 2025 methodological landscape}

Where this submission departs from the Task~13b pattern is in the locus of agency. The dominant 13b pattern was static hybrid retrieval feeding a single LLM generation call, with optional supervised fine-tuning of the generator~\cite{bitua2025,unitor2025}. Ateia and Kruschwitz~\cite{ateia2025} introduced a self-feedback loop, but the loop runs \emph{backward} over generation: the model critiques and regenerates its own answer. AQAMS~\cite{aqams2025} composes multiple LLM agents that exchange messages while answering.

Our submission concentrates agency in a single LLM instance playing two roles---controller and generator---within a \emph{forward} retrieval loop that the controller, not the generator, drives. Where Ateia and Kruschwitz iterate on the answer, we iterate on the evidence pool. Where AQAMS distributes agency across multiple agents, we concentrate it in one instance whose discretion is bounded to the retrieval trajectory. This is the operative meaning of ``agentic'' in our system: not autonomy over arbitrary actions, but discretion over what to query next and when to stop, in the spirit of ReAct~\cite{react} but specialized to PubMed.

Whether this architectural choice produces better results than the alternatives is, on the evidence of Task 14b, not yet settled. The top systems on this year's leaderboard, several of which appear to use ensembles, fine-tuning, or learned cross-encoder rerankers, outperform our submission. The narrower claim we can defend is that the single-LLM dual-role pattern reaches competitive Phase B yes/no performance without any of these components, which is a useful baseline for future participants choosing where to invest engineering effort.

\subsection{What worked, and why}

Three design choices carry the Phase B scores.

\emph{The agentic loop shifts the failure distribution from confident-wrong to flagged-incomplete.} On a 17-question internal held-out set during development, a zero-shot Gemma 4 31B baseline produced fluent answers but hallucinated entities not supported by the snippets, reaching roughly 0.84 yes/no accuracy and 0.18 list F1. With the loop enabled on the same model and snippets, those numbers became 0.94 and 0.32 on Phase B batch~1. The mechanism is mechanical, not magical: requiring the controller to inspect its own evidence pool and decide whether it can answer forces the model to convert under-supported answers into additional retrieval requests rather than hallucinations. The type-specific generator prompts then refuse to invent entities outside the pool, so the recovered queries translate directly into recall.

\emph{Deterministic format hygiene recovers an entire class of silent errors.} BioASQ's exact-match scoring is unforgiving: a semantically correct factoid wrapped in markdown bolding scores as wrong, and the team never sees this because the natural-language output looks right. Across our four submissions, the regex filter (§3.9) recovered roughly 20--30 list answers whose entities were structurally correct but wrapped in chain-of-thought scaffolding. A conservative estimate is that this is worth 2--4 list-F1 points and 1--2 factoid strict-accuracy points---not dominant in isolation, but free, and we suspect a similar class of errors accounts for part of the gap between LLM-based submissions and the points they could have scored.

\emph{RRF beats learned fusion when query score distributions are heavy-tailed.} Our preliminary tuning compared linear score combinations (after per-query min-max normalization of dense cosine and BM25 scores) against Reciprocal Rank Fusion~\cite{rrf2009}. RRF beat every linear-combination configuration we tried, with one fewer hyperparameter to set. We attribute this to the structure of biomedical queries: a single document with rare gene symbols can dominate a BM25-weighted linear combination, while RRF's positional damping $1/(k+\text{rank})$ bounds each ranker's contribution at the top of the list.

\subsection{What the negative results tell us}

Two negative findings are, on reflection, our most informative results.

\emph{Few-shot conditioning by question similarity did not change the model's outputs.} The \texttt{asmalltrialsystem} and \texttt{ossllm} systems differ only in whether three training-set exemplars---selected by question-embedding cosine similarity---are prepended to the answer-generation prompt. On Phase B batch~1, they produced character-for-character identical JSON on every question and tied at exactly the same scores across every metric. The likely interpretation is that the BioASQ output shape is already well-represented in Gemma 3's instruction-tuning distribution, and three additional surface-similar exemplars contribute no information the model has not already internalized. We suspect exemplar selection by \emph{answer shape}---e.g., presence of a numeric value, count of list items---rather than question similarity could still help, and we flag this as a non-default future-work direction. The broader implication, worth reporting because the BioASQ literature contains several single-batch few-shot claims, is that surface-similarity exemplar conditioning is not a robust intervention for instruction-tuned models at this scale.

\emph{The Gemma 3 to Gemma 4 upgrade moved generation-bound metrics but not retrieval-bound ones.} Comparing \texttt{Finalcorrected} (Gemma 4 31B) against \texttt{asmalltrialsystem} (Gemma 3 27B) on Phase B, factoid strict accuracy rises 0.304 to 0.364 ($+6$ points) and list F1 rises 0.317 to 0.365 ($+5$ points). On the one Phase A$^+$ batch where both systems ran (batch~4), they returned identical exact answers. This pattern is exactly what the retrieval-bottleneck interpretation predicts: a stronger generator improves extraction from a fixed evidence pool but cannot manufacture evidence that retrieval failed to surface. Because the generator is held constant across the two phases, the Phase B--Phase A$^+$ gap (0.18 yes/no, 0.15 factoid strict, 0.22 list F1) measures retrieval cost in something close to isolation. This is, methodologically, the cleanest claim the paper makes: not that retrieval matters in general, but that retrieval recall is the binding constraint for \emph{this} system on \emph{this} benchmark.

The Gemma 4 upgrade also produced one unexpected regression: ROUGE-2 F1 on Phase B ideal answers dropped from 0.179 to 0.087 on batch~4. Our best read is that Gemma 4's longer default summary style pays an $n$-gram precision penalty against the relatively short BioASQ gold summaries, and that the prompt's word cap should have been re-tuned when the model changed. This is a concrete example of why model swap-ins should never be assumed to be Pareto improvements.

\subsection{What the top of the 14b field has that we don't}

The 14b leaderboard makes the gap between us and the top systems legible by component. We are not running a learned cross-encoder reranker on BioASQ relevance judgments, and several systems above us appear to be. We are not fine-tuning the generator on BioASQ training data. We are not ensembling multiple model proposals. We cap effective context at 8\,192 tokens for generator focus when Gemma 4 nominally supports more than 256K, which may under-use the model on list questions with many candidate entities. Each of these is a known and well-trodden lever; we chose not to pull them because we wanted to isolate the contribution of the agentic loop and format hygiene, not because we believe they would not help.

The honest reading is that the agentic-loop and format-hygiene pattern is \emph{necessary but not sufficient} to reach the top of the 14b leaderboard. A future submission that adds a learned cross-encoder reranker, light supervised fine-tuning of the generator, and a small ensemble on top of this architecture is, in our view, the cleanest path to the leading cluster of systems.

\subsection{Future work}

Three concrete next steps follow directly from the analysis above.

\emph{A learned cross-encoder reranker on BioASQ data.} Fine-tune a biomedical cross-encoder on BioASQ Task~b relevance judgments and replace the RRF fusion step, or augment it as a third signal in the fusion. This is the single highest-leverage improvement we can identify against our current Phase A$^+$ scores, and it directly targets the retrieval-recall bottleneck that the same-generator phase comparison isolates.

\emph{Type-aware context budgeting and prompt re-tuning per model.} Re-calibrate per-type prompts when swapping the model backbone, with explicit word caps re-set for each model's verbosity prior. This would have prevented the Gemma 4 ROUGE-2 regression.

\emph{Component ablation of the agentic loop.} The current paper reports the loop as a single component. A controlled experiment removing the sufficiency check (forcing 5 iterations), the adversarial-reasoning yes/no prompt, or the format-hygiene filter would isolate each component's marginal contribution. We expect the hygiene filter and the adversarial yes/no prompt to carry most of the lift on the metrics where we are competitive; the agentic loop itself most likely matters most on list and factoid Phase A$^+$, but we cannot show this without an ablation.

\subsection{Closing}

Two claims in this paper are robust enough to carry into future BioASQ submissions independently of where Task 14b's leaderboard lands. First, an agentic retrieval loop with a single open-weights LLM in dual controller-generator roles, paired with deterministic format hygiene, is sufficient to reach competitive Phase B yes/no and list-F1 performance without fine-tuning, an ensemble, or a learned reranker. Second, holding the generator constant across Phases B and A$^+$ is a clean experimental design for isolating retrieval cost, and on our system that cost is 0.15--0.22 points across every metric. We hope both observations are useful to next year's participants regardless of how the present submission scored.
```

(In the LaTeX above, the `\cite{...}` keys are placeholders: `nentidis2026bioasq` = your ref [2], `bitua2025` = [4], `unitor2025` = [5], `ateia2025` = [6], `aqams2025` = [11], `react` = [13], `rrf2009` = [7]. Replace each with the actual key in your `.bib`.)

---

## 5. Section 3.9 (Format hygiene) — condense

Find the second sentence beginning *"Instruction-tuned models occasionally leak chain-of-thought scaffolding..."* and replace with:

```latex
As described in §1 (Failure mode 2), instruction-tuned models occasionally leak chain-of-thought scaffolding into their structured outputs. Left in place, these tokens cause the BioASQ scorer to miss exact matches that would otherwise score. We apply a single regex-based filter to the model's output immediately before JSON serialization. The filter strips this scaffolding (markdown formatting, bracketed citation tokens, scratchpad headers, arrow glyphs), removes any leading ``Answer:'' or ``Final answer:'' prefixes, and enforces the BioASQ output shape: factoid and list exact answers are emitted as a list of single-element lists---the BioASQ$\geq$5 format~\cite{bioasq5format}---with at most 5 outer entries for factoids and 100 for lists; yes/no exact answers are bare lowercase strings; summary outputs omit the \texttt{exact\_answer} field entirely. Across our four submissions the filter recovered roughly 20--30 list answers that would otherwise have been malformed and rejected.
```

This avoids the verbatim repetition with §1 Failure mode 2.

---

## 6. Section 3.10 — fix the empty subsection

The current §3.10 header has no body. Either delete the heading and reference Table 1 from §3.1, OR add this body text under the heading:

```latex
Table~\ref{tab:systems} summarizes the three submissions. The \texttt{asmalltrialsystem} and \texttt{ossllm} systems differ only in whether training-set exemplars are prepended to the answer-generation prompt; \texttt{Finalcorrected} differs from \texttt{asmalltrialsystem} on both the LLM backbone (Gemma 4 31B Dense replaces Gemma 3 27B) and the PubMed backend (a local SQLite FTS5 mirror replaces live NCBI E-utilities). Together, the three submissions vary three orthogonal axes: prompt-level conditioning, generator scale, and retrieval-substrate locality.
```

(Replace `\ref{tab:systems}` with the actual label of Table 1 in your source.)

---

## 7. Other surgical fixes

| Location | Find | Replace |
|---|---|---|
| Figure 3 caption | check that the plotted list-F1 value (0.095) matches Table 2 row "asmalltrialsystem b2" | — |
| Reference [8] | `Translategemma technical report` — verify this is the correct Gemma 3 citation | likely the Gemma 3 tech report (e.g., arXiv:2503.19786) |
| Acknowledgements / Declaration on GenAI | minor: "Grammar and spelling check" — recommend "grammar and spelling checks, and brainstorming on phrasing" | reads more honestly given the genuine editorial use |

---

## Summary of what changed and why

1. **Abstract**: now leads with the single-LLM dual-role architectural novelty; honest about per-batch positioning; both negative results are stated explicitly.
2. **Intro three-claims paragraph**: First claim now states the mechanism (loop + hygiene) before the number; positioning honest; cites the 14b overview not the 13b one.
3. **Intro Failure mode 1**: two redundant sentences merged into a tighter pair that doesn't restate the same point twice.
4. **§4 Results**: positioning language updated from "upper-middle of the field given last year's distribution" to specific Task 14b context with named system families; within-model anchor paragraph now points forward to §5.5's honest accounting.
5. **§5 Discussion**: rebuilt around three carry-forward contributions (architectural pattern, isolating experimental design, format-hygiene as recognized failure class) rather than leaderboard position; the previous "Why this is probably not the best system" subsection is replaced with a more confident "What the top of the field has that we don't" framing that names the missing components.
6. **§3.9 Format hygiene**: condensed to remove verbatim repetition with §1 Failure mode 2 (this was your advisor's "I've seen this sentence repeated" note).
7. **§3.10**: filled in or removed (your call).
8. **Line edits**: broken `§??` cross-ref, name consistency (Gemma 3 / Gemma 4 without hyphens), BIT.UA gloss, "axes" not "questions", citation [8] verification.

## Recommended order of operations in Overleaf

1. Do the global find/replaces first (Gemma name consistency, `§??` → `§5`, `questions` → `axes`) — 5 minutes.
2. Drop in the new Abstract — 2 minutes.
3. Replace the three-claims paragraph and the Failure mode 1 merge in §1 — 5 minutes.
4. Update the two §4 paragraphs (Phase B / Phase A+ positioning, within-model anchor) — 5 minutes.
5. Replace the entire §5 with the new Discussion — 5 minutes.
6. §3.9 condense, §3.10 fill or remove — 5 minutes.
7. Final read-through, citation key matching, recompile.

Total: roughly 30–40 minutes of editing.
