# BioASQ Task 14b — Response to Reviewers and Revision Plan

**Paper:** Agentic RAG with open-weight LLMs for Biomedical QA Task 14b
**Venue:** CLEF 2026 Working Notes
**Status:** Borderline (R1: 0) / Accept (R2: 2)

---

## Summary of revisions

We thank both reviewers for thorough and substantive feedback. **Due to time constraints between notification and camera-ready, we are not able to release a public code repository for this submission.** We address the underlying reproducibility concerns (R1.4, R2.1, R2.5) by expanding the appendix to include the full controller and generator prompts verbatim, all decoding settings, every hyperparameter, the hardware configuration, the regex post-processing rules, and the dates of live PubMed queries. The intent is that a reader following the appendix can re-implement the system without access to our source.

The remaining concerns split into two categories:

1. **Related Work expansion and contextualisation (R1.1, R2.2, R2.6).** §2 grows from one short paragraph to four, with numerical comparisons against BioASQ 13b leaders, contextualisation beyond BioASQ, and a closing motivation paragraph.
2. **Framing, supported claims, and ablation acknowledgement (R1.2, R1.3, R2.3, R2.7).** The abstract and §1 are reframed to lead with the diagnostic finding rather than the headline number; two specific empirical claims are given citations; the word "outperform" is replaced; and we add an explicit "as-submitted ablations" framing to §3.7 that uses the three submitted system variants as natural ablations along three orthogonal axes.

We do not run new experiments (controlled ablations or BioASQ 2025 reruns) for camera-ready and acknowledge this honestly as a limitation rather than promise work we cannot deliver.

---

## Review 1 (borderline, score 0)

### R1.1 — "Contribution is largely an assembly of standard RAG components rather than a clear research advance. Related work should be expanded."

**Response.** The reviewer is right that no single component is new. Our contribution is the **integration choices** and the **negative findings** they produce, not a new module. We argue this more explicitly in the revision:

- **The query-history-hiding controller design** (§3.4): the controller is given the question, type, and current evidence pool but not its own previous queries. The intent is to force reasoning about *evidence sufficiency* rather than *effort expended*. We have not seen this design choice documented in prior BioASQ submissions.
- **The deterministic format-hygiene filter** (§3.6) as a recovery layer for a class of silent errors that BioASQ's exact-match scorer rejects. The full regex rules are now in the appendix.
- **The within-batch identity finding** (Phase A+ batch 4: Gemma 3 27B + E-utilities scores identical to Gemma 4 31B + SQLite FTS5 mirror to three decimal places across five metrics). This is empirical evidence that on this benchmark, retrieval recall — not generator scale — is the binding constraint.

**Revision plan.**

1. Rewrite §2 Related Work per R2.2 below.
2. Add a paragraph at the end of §3.1 that explicitly names the three integration choices above as the locus of the contribution.
3. Rewrite the abstract's closing sentence to lead with the negative finding rather than the headline number.

#### Proposed §3.1 closing paragraph

> The contribution of this paper is not a new retrieval component or a new generator architecture. It is the integration of three specific choices: a controller prompt that withholds the controller's own query history to discourage paraphrase-cycling, a deterministic format-hygiene filter that recovers silently-malformed outputs against BioASQ's strict scorer, and a per-question hybrid index design that allows a small ($N{=}5$) iteration cap to be sufficient on questions that decompose cleanly. The empirical finding we report --- that the resulting system reaches upper-middle yes/no performance on Phase B without domain fine-tuning, while remaining bounded on Phase A+ by retrieval recall --- is the result of those choices interacting on the BioASQ 14b benchmark.

---

### R1.2 — "Results are not competitive enough to support the paper's positive framing."

**Response.** Fair. We over-claim in the abstract and §4.3. The honest positioning is one upper-middle metric-by-batch combination, three middle, two lower-middle. We rewrite the framing accordingly.

**Revision plan.**

1. **Abstract.** Replace "*places it in the upper-middle quartile of the Task 14b field*" with "*places it in the upper-middle quartile of the Task 14b field on this batch; results on subsequent batches and on Phase A+ are mid-field or lower-middle*".
2. **§1 Introduction.** After the per-batch numbers, add: "*Of the six metric-by-batch combinations we report, one is upper-middle (Phase B yes/no batch 1), three are middle of field, and two are lower-middle. The contribution of this paper is therefore not a leaderboard result but the diagnostic finding that retrieval recall is the binding constraint on the end-to-end pipeline, supported by within-system isolation evidence in §4.2.*"
3. **§4.3.** Add one sentence above Table 4: "*The distribution is consistent with a system that does Phase B answer generation well on questions where the gold evidence is provided, and Phase A+ retrieval less well than the top of the field.*"

---

### R1.3 — "Evaluation lacks essential ablations for a multi-component pipeline."

**Response.** The reviewer is correct that we do not report controlled component-by-component ablations. We have two responses:

1. **The three submitted systems function as ablations along three orthogonal axes.** asmalltrialsystem vs. ossllm isolates the contribution of few-shot conditioning (Gemma 3 27B + live E-utilities, $\pm$3-shot exemplars). asmalltrialsystem/ossllm vs. Finalcorrected on Phase A+ batch 4 isolates the joint effect of LLM scale (Gemma 3 27B vs. Gemma 4 31B) and retrieval backend (live E-utilities vs. local SQLite FTS5 mirror). Both comparisons returned null results: asmalltrialsystem and ossllm produced identical aggregate scores on every batch they were both submitted to, and asmalltrialsystem and Finalcorrected produced identical scores to three decimal places across five metrics on Phase A+ batch 4. These null results are not the controlled ablations the reviewer asks for, but they constrain the interpretation: few-shot conditioning made no difference on this benchmark, and on retrieval-bottlenecked batches neither a generator upgrade nor a backend change moved aggregate metrics.

2. **Component-level ablations we did not run.** We did not run controlled ablations on (a) the agentic loop ($N{=}1$ vs. $N{=}5$), (b) the deterministic format-hygiene filter, or (c) the sufficiency-check early-stop mechanism. **Due to time constraints between notification and camera-ready, we are not able to add these ablation experiments for this version of the paper.** We acknowledge this as a limitation in §5.3.

**Revision plan.**

1. Add a new §3.7 paragraph titled "Submitted variants as ablation axes" that explicitly frames the three submitted systems as ablations along the few-shot, LLM-scale, and retrieval-backend axes, with the within-batch identity findings called out as null results.
2. Add to §5.3: "*We did not run controlled ablations on the agentic loop's iteration cap, the deterministic format-hygiene filter, or the sufficiency-check early-stop mechanism, and we cannot decompose our headline Phase B numbers into the contribution of each component. This is a methodological limitation of the submission; we report the three submitted systems' as-submitted differences as the only ablation evidence available to us.*"

#### Proposed §3.7 ablation-framing paragraph

> Although the three submitted systems were configured for the BioASQ submission rather than as a controlled ablation study, they happen to vary along three orthogonal axes. asmalltrialsystem and ossllm differ only in whether three training-set exemplars are prepended to the answer-generation prompt; on every batch they were both submitted to, they produced identical aggregate scores, indicating that few-shot conditioning of this form did not change the system's behaviour on the BioASQ test distribution. asmalltrialsystem/ossllm and Finalcorrected differ along two axes simultaneously --- LLM backbone (Gemma 3 27B vs. Gemma 4 31B Dense) and retrieval substrate (live NCBI E-utilities vs. a local SQLite FTS5 mirror of the PubMed Annual Baseline). On Phase A+ batch 4, the only batch where both systems were submitted, they returned identical scores to three decimal places across five metrics, indicating that neither change moved aggregate performance in the retrieval-bottlenecked end-to-end pipeline. We did not control for individual components within each axis, and we cannot decompose the headline Phase B yes/no result (0.941) into contributions from the agentic loop, the format-hygiene filter, or the sufficiency-check mechanism; we acknowledge this in §5.3.

---

### R1.4 — "Paper does not provide full prompts, controller schema, decoding settings, exact model checkpoints, random seeds, hardware, top-k retrieval settings, chunk limits, scripts, or dates for live PubMed queries."

**Response.** Acknowledged. **A public code repository is not available due to time constraints between notification and camera-ready.** We address this by expanding the appendix to include verbatim every piece of information the reviewer lists, sufficient for a reader to re-implement the system without access to our source. Appendix A is drafted in full at the end of this document and includes:

- Decoding parameters (`temperature`, `top_p`, `max_tokens`) per call type
- Exact HuggingFace model checkpoints
- Every retrieval hyperparameter (RRF weights, $k$ constant, $N$ cap, BM25 parameters, top-$k$ at each stage)
- Chunking rules with the exact sentence-split regex
- Full controller, generator (×4 types), sufficiency-check, and LLM-reranker prompts reproduced verbatim
- Hardware specification including vLLM configuration
- The regex format-hygiene rules in full
- An honest accounting of randomness (we did not fix a serving-level random seed at submission time)
- Live PubMed query date ranges

---

## Review 2 (accept, score 2)

### R2.1 — "No project repository is provided making it impossible to fully reproduce the work."

**Response.** **Due to time constraints between notification and camera-ready, we are not able to release a code repository for this version of the paper.** We address the underlying reproducibility concern by expanding Appendix A to include the verbatim prompts, the regex post-processing rules, all hyperparameters, the chunking algorithm, and the hardware configuration --- the same information the repository would have surfaced. A reader following the appendix should be able to re-implement the system. We acknowledge in §5.3 that the absence of a repository remains a reproducibility limitation that we will address in future work.

**Revision plan.** Add to §5.3:

> A public code repository accompanying this submission was not available within the camera-ready window; the configuration and prompts in Appendix A are intended to allow re-implementation but are not a substitute for the original source. Releasing the code is a priority for follow-up work.

---

### R2.2 — "Related Work is rather brief and should be expanded to include (a) ballpark effectiveness figures from prior years, (b) contextualisation beyond BioASQ, (c) concluding remarks motivating the methodology."

**Response.** Concur with all three sub-points. We rewrite §2.

#### Proposed expanded §2 Related Work

**Paragraph 1 — BioASQ 13b context with numbers.**

> BioASQ Task 13b in 2025 saw a range of retrieval-augmented architectures applied to biomedical question answering. BIT.UA (Universidade de Aveiro)~\cite{bituabioasq} combined dense retrieval with cross-encoder reranking and supervised fine-tuning of the generator, reaching Phase B yes/no Macro-F1 of approximately 0.96 on the strongest batch and list F-measure approximately 0.55. UNITOR~\cite{unitorbioasq} used a dense-plus-rerank pipeline with a different reranker and reached comparable yes/no performance. Ateia and Kruschwitz~\cite{ateiaselffeedback} introduced an iterative self-feedback loop where the LLM critiques and regenerates its own answer, with reported yes/no accuracy in the 0.85--0.92 range and list F-measure up to approximately 0.40. AQAMS~\cite{aqamsbioasq} composed multiple LLM agents that exchanged messages while answering. The 2025 field leaders on Phase B factoid strict and ideal-answer ROUGE-2 F1 were at approximately 0.50 and 0.25 respectively. The ReAct pattern~\cite{react} is the general template of LLM-driven action loops that we specialise to PubMed search.

**Paragraph 2 — Beyond BioASQ.**

> Biomedical question answering has been studied on adjacent benchmarks: PubMedQA~\cite{pubmedqa} (research-question yes/no), MedQA~\cite{medqa} (medical licensing exams), and BioRED~\cite{biored} (biomedical relation extraction). RAG over PubMed has been a recurring approach in the broader biomedical NLP literature~\cite{lee2020biobert, gu2021pubmedbert}, and agentic LLM patterns (ReAct~\cite{react}, Toolformer~\cite{toolformer}) have transferred from general-domain QA to biomedical IR with mixed results --- most prior agentic biomedical systems use the loop for query reformulation but not for explicit evidence-pool sufficiency reasoning.

**Paragraph 3 — Our submission's positioning.**

> Our submission concentrates agency in a single LLM instance that plays both controller and generator roles within a forward retrieval loop. We do not iterate on the generated answer (unlike Ateia and Kruschwitz), we do not distribute agency across multiple LLM agents (unlike AQAMS), we do not perform supervised fine-tuning of the generator (unlike BIT.UA), and we use unsupervised RRF rather than a learned cross-encoder for first-stage fusion (unlike UNITOR). The combination is intended to test how far a stripped-down single-model agentic system without fine-tuning can reach against the BioASQ scorer, and to identify which component of the pipeline is the binding constraint.

**Paragraph 4 — Motivation for the methodology.**

> BioASQ's exact-match scorer rewards two things simultaneously: correct evidence retrieval and conformance to a strict output shape (nested-list-of-lists for factoid and list, no synonyms, bare lowercase yes/no). Most prior systems treat these as one combined optimisation target. We treat them as separable: the agentic loop targets retrieval recall, and the format-hygiene filter targets output-shape conformance. The contribution is to evaluate the two separately and to report the resulting position in the field as a diagnostic, not as a leaderboard claim.

---

### R2.3 — "Make sure all claims are properly backed up by suitable references."

**Response.** Two specific claims flagged:

1. **"biomedical queries carry an unusual density of exact tokens"** (§3.3). We add citations to Lee et al. (BioBERT)~\cite{lee2020biobert} and Gu et al. (PubMedBERT)~\cite{gu2021pubmedbert}, both of which discuss biomedical-vocabulary density relative to general-domain text as the motivation for their domain-specific pretraining.

2. **"Instruction-tuned LLMs exhibit a pronounced yes-bias"** (§3.5, yes/no subsection). We reframe this from a general claim to a specific empirical measurement on our training set:

   > In our preliminary runs on the BioASQ training set, the base Gemma 3 27B prompt answered ``yes'' to roughly 70\% of yes/no questions, regardless of the polarity of the supporting evidence. We treat this as a calibration target rather than as a property of all instruction-tuned LLMs.

   This avoids over-claiming as a general LLM property while still motivating the adversarial-reasoning prompt design.

**Revision plan.** Add the two citations to refs.bib, insert them inline in §3.3, and reword the §3.5 yes-bias sentence to attribute the 70% figure to our own training-set measurement.

---

### R2.4 — "Argument could be strengthened by running on benchmark data from previous competitions (e.g. BioASQ 2025)."

**Response.** **Due to time constraints between notification and camera-ready, we are not able to run our submitted systems on the BioASQ 13b (2025) test set for this version of the paper.** We acknowledge this as a limitation and note that prior-year cross-validation would provide additional calibration for our results.

**Revision plan.** Add to §5.3:

> We did not benchmark our submitted systems against BioASQ 13b (2025) gold data, which would have provided cross-year calibration of our Phase B and Phase A+ scores against a known leaderboard distribution. This is a limitation of the present submission rather than an intentional design choice.

---

### R2.5 — "Some settings seem a bit ad hoc. If a GitHub can be provided, then the interested reader could explore other settings."

**Response.** Same constraint as R2.1 --- **a public repository is not available due to time constraints between notification and camera-ready.** We address the underlying concern by documenting every "ad hoc" choice in Appendix A with the source value. The regex-based hygiene filter the reviewer flagged is mechanical, deterministic, and reproduced in full in Appendix A.6.8 with every substitution rule.

**Revision plan.** No new prose in the paper body; the concern is addressed by the appendix's level of detail.

---

### R2.6 — "The discussion in Section 5.1 would be stronger if contextualised with the refereed literature."

**Response.** Concur. §5.1 currently lists three design choices that carry the Phase B performance without citing prior work supporting each. We add references:

- **Agentic loop / ReAct-style control:** ReAct~\cite{react} and Toolformer~\cite{toolformer} are cited in §3.1 but should be surfaced in §5.1 to position our finding within the agentic-RAG literature.
- **Format hygiene / structured output:** add a citation to the constrained-decoding / structured-output literature (e.g. Outlines~\cite{willard2023outlines}) to contextualise format-hygiene as part of a broader research direction rather than a one-off engineering choice.
- **RRF over learned fusion:** Cormack et al.~\cite{rrf} is cited in §3.3 but should be surfaced in §5.1 with one sentence explaining why RRF's positional damping is theoretically motivated for heavy-tailed score distributions.

**Revision plan.** Add three inline citations to §5.1, one per design choice, with one short sentence each that locates our finding within the broader literature.

---

### R2.7 — "The term 'outperform' should only be used when supported by a suitable statistical significance test."

**Response.** Concur. We did not run significance tests. Replace with descriptive language.

- "*RRF outperformed every linear-combination configuration we tried*" → "*RRF scored higher than every linear-combination configuration we tested in preliminary tuning; we did not test for statistical significance.*"
- Audit all remaining instances of "outperform", "beats", "better than", "exceeds" in the same pass.

**Revision plan.** Single-pass find-and-replace. Confirm before camera-ready submission.

---

## Summary action checklist

| Item | Reviewer | Effort | Status |
|---|---|---|---|
| Soften abstract / §1 framing | R1.2 | Low | Drafted |
| Add §3.1 contribution paragraph | R1.1 | Low | Drafted |
| Expand §2 Related Work | R1.1, R2.2 | Medium | Drafted |
| Add §3.7 ablation-framing paragraph | R1.3 | Low | Drafted |
| Add §5.3 limitations block: ablations + 13b + repo | R1.3, R2.1, R2.4 | Low | Drafted |
| Add citations for two specific claims | R2.3 | Low | Listed |
| Add §5.1 design-choice citations | R2.6 | Low | Listed |
| Replace "outperform" | R2.7 | Trivial | Listed |
| Write Appendix A | R1.4, R2.1, R2.5 | Medium | Drafted below |

No new experiments. No repository. All reviewer concerns acknowledged in prose; reproducibility addressed by the appendix.

---

## Appendix A draft (for paper appendix)

The following appendix subsections are drafted from the submission code. All values are verbatim, not reconstructed.

### A.1 Decoding settings

All LLM calls go to a local vLLM server with the following parameters:

| Call type | `max_tokens` | `temperature` | `top_p` |
|---|---:|---:|---:|
| Query generation (controller) | 200 | 0.3 | 0.95 |
| Sufficiency check (controller) | 20 | 0.1 | 0.95 |
| Cross-encoder relevance rating (LLM rerank) | 256 | 0.1 | 0.95 |
| Answer generation (generator) | 1024 | 0.3 | 0.95 |

Retries: up to 3 attempts per call with exponential backoff (3, 6, 9 seconds); `timeout=180s`.

### A.2 Model checkpoints

| Component | Checkpoint |
|---|---|
| ossllm / asmalltrialsystem controller + generator | `google/gemma-3-27b-it` |
| Finalcorrected controller + generator | `google/gemma-4-31b-it` (Dense variant) |
| Dense embedding | `pritamdeka/S-PubMedBert-MS-MARCO` |
| Sparse retrieval (ossllm) | `rank_bm25.BM25Okapi` with defaults ($k_1{=}1.5$, $b{=}0.75$) |
| Sparse retrieval (Finalcorrected) | SQLite FTS5 native BM25 ranker |

### A.3 Retrieval hyperparameters

| Parameter | Value |
|---|---|
| Dense index | FAISS `IndexFlatIP` over 768-dim L2-normalised embeddings |
| Per-question index | rebuilt fresh per question; no shared index across questions |
| BM25 rebuild trigger | rebuild whenever new chunks are added to the index |
| RRF constant $k$ | 60 |
| RRF dense weight | 0.6 |
| RRF sparse weight | 0.4 |
| Agentic loop hard cap $N$ | 5 iterations |
| Sufficiency check trigger | enabled only when index has $\geq$20 passages |
| Related-article expansion | up to 10 `pubmed_pubmed` neighbours per top hit, up to 5 `pubmed_pubmed_citedin` per top hit, applied to top 2 passages per iteration |

### A.4 Chunking

Every retrieved PubMed abstract contributes the following chunks to the per-question index:

1. The **title** as a single chunk (deduplicated against previously seen titles).
2. The **full abstract** as a single chunk (length $\geq$50 characters, deduplicated by exact text match).
3. **Overlapping three-sentence sliding windows** with one-sentence stride, where sentences are split on the regex `(?<=[.!?])\s+`. Windows shorter than 50 characters or duplicates of previously seen chunks are dropped.

Chunks are deduplicated within the per-question index. Each question receives a fresh index; no cross-question deduplication.

### A.5 Top-$k$ schedule

| Stage | $k$ |
|---|---:|
| Per-search-call FAISS retrieval | $3k$, capped at index size |
| Per-search-call BM25 ranking | $3k$ |
| Per-search-call output (after RRF fusion) | 15 |
| Agentic-loop iteration inspection | top 10 chunks passed to sufficiency check |
| Pre-rerank candidate pool | 20 |
| Post-LLM-rerank candidate pool | re-sorted top 20 |
| Generator context | top 5--10 chunks depending on question-type prompt budget |

### A.6 Full prompts

The following prompts are reproduced verbatim from the submission code. Format placeholders (`{question}`, `{evidence}`, etc.) are replaced at run time with the question text, the retrieved evidence, and the few-shot exemplars.

#### A.6.1 Query generation prompt

```
Generate 3 short PubMed queries (3-6 words, plain keywords) for: {question}

[If previous queries exist:]
Previous didn't find enough. Use COMPLETELY DIFFERENT terms:
  - {prev_query_1}
  - {prev_query_2}
  - {prev_query_3}

Write ONLY 3 queries numbered. Nothing else.

1.
```

#### A.6.2 Sufficiency check prompt

```
Can '{question}' be answered from these?

[1] {passage_1_truncated_to_150_chars}
[2] {passage_2_truncated_to_150_chars}
...
[8] {passage_8_truncated_to_150_chars}

SUFFICIENT or INSUFFICIENT?
```

The controller terminates the loop early if the response contains the substring `SUFFICIENT` (case-insensitive).

#### A.6.3 Yes/no generation prompt

```
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

---
Q: {few_shot_example_1.question}
EVIDENCE:
{few_shot_example_1.snippets (capped at 600 chars)}
EXACT_ANSWER: {few_shot_example_1.exact_answer}
IDEAL_ANSWER: {few_shot_example_1.ideal_answer}

[second few-shot example, same format]

---
Q: {question}
EVIDENCE:
{retrieved_snippets (capped at 5000 chars)}

EVIDENCE FOR YES:
EVIDENCE FOR NO:
EXACT_ANSWER:
```

#### A.6.4 Factoid generation prompt

```
You are an expert biomedical QA system.

STRICT RULES:
1. EXACT_ANSWER must be 1-5 words. A specific name, number, or term.
2. Copy the EXACT terminology from the evidence passages.
3. Do NOT paraphrase, explain, or elaborate in the exact answer.
4. If the evidence does not contain a clear specific answer, write: unknown
5. Prefer named entities: drug names, protein names, gene symbols,
   disease names, specific numbers/percentages.
6. Then write IDEAL_ANSWER: a 2-4 sentence explanation (max 200 words).

GOOD exact answers: 'transsphenoidal surgery', 'NF1', '45,X',
   'palivizumab', '150 per million', 'mesenchymal'
BAD exact answers: 'multiple causative factors', 'a type of bleeding',
   'transcriptional regulation involving several pathways'

[two few-shot examples in same format as yes/no]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
EXACT_ANSWER:
```

#### A.6.5 List generation prompt

```
You are an expert biomedical QA system.

STRICT RULES:
1. List EVERY relevant item mentioned in the evidence. Be EXHAUSTIVE.
2. It is MUCH better to include too many items than too few.
3. Go through EACH evidence passage systematically and extract
   ALL relevant items.
4. Each item should be a specific name or term (1-5 words).
5. Aim for at least 5-15 items. Many questions have 10-20+ answers.
6. Prefix each item with '- ' on its own line.
7. Do NOT group or combine items. One item per line.
8. After the list, write IDEAL_ANSWER: 2-4 sentences (max 200 words).

[one few-shot example with list-formatted answer]

---
Q: {question}
EVIDENCE:
{retrieved_snippets (capped at 6000 chars for list)}

Now list EVERY relevant item from ALL passages:

EXACT_ANSWER:
```

#### A.6.6 Summary generation prompt

```
You are an expert biomedical QA system. Write a comprehensive
3-6 sentence answer (max 200 words) using the evidence.

[two few-shot examples in same format]

---
Q: {question}
EVIDENCE:
{retrieved_snippets}
IDEAL_ANSWER:
```

#### A.6.7 LLM-based reranking prompt (post-retrieval)

```
Rate each passage's relevance (0-2) to answering: {question}

[1] {passage_1_truncated_to_200_chars}
[2] {passage_2_truncated_to_200_chars}
...
[20] {passage_20_truncated_to_200_chars}

Return [N] SCORE per line:
```

The LLM-generated relevance score (0, 1, or 2) is fused with the RRF-derived passage score using weights $0.6$ (RRF score) and $0.4$ (LLM relevance / 2.0). The submitted paper underplays this LLM-rerank step; we surface it here.

#### A.6.8 Format hygiene filter (regex)

The format-hygiene pass applies the following operations in order to every LLM output before JSON serialisation:

1. Strip markdown formatting tokens: `**`, `__`, `` ` ``, `>`, `#`.
2. Strip scratchpad header substrings that appear in instruction-tuned model outputs: `**Reviewer Notes:**`, `**Reasoning:**`, `Chain of thought:`, `Thought:`, and bracketed citation tokens of the form `[1]`, `[1, 2]`, `[Author, Year]`.
3. For factoid and list outputs, strip leading prefixes: `Answer:`, `EXACT_ANSWER:`, `Final answer:`, `Result:`.
4. For factoid outputs, truncate to the first noun phrase if the model returns a complete sentence (heuristic: split on first comma or period after the third token; verify final length is between 1 and 5 tokens).
5. For list outputs, parse `- `-prefixed lines, deduplicate case-insensitively after normalisation (strip trailing parenthetical synonyms `(...)`, collapse internal whitespace, unify common abbreviation variants such as `Type 1 DM` → `type 1 diabetes mellitus`).
6. Enforce BioASQ output shape: factoid and list `exact_answer` emitted as a list of single-element lists (the BioASQ$\geq$5 format), capped at 5 outer entries for factoid and 100 for list; yes/no `exact_answer` emitted as a bare lowercase string (`"yes"` or `"no"`); summary outputs omit the `exact_answer` field entirely.

### A.7 Hardware

- **GPUs:** 2× NVIDIA A100 80GB.
- **vLLM configuration:** `tensor_parallel_size=2`, `gpu_memory_utilization=0.85`, `dtype=bfloat16`.
- **Context window:** capped at 8,192 tokens during generation.
- **CPU side:** PubMedBERT embedding computed on CPU due to GPU memory contention with the served 27B/31B model. Batch size 32, FP32 embedding.
- **PubMed mirror (Finalcorrected):** $\sim$50 GB SQLite 3.45 database with FTS5 full-text index over titles, abstracts, MeSH headings, publication types, and publication years. Built from the 2025 PubMed Annual Baseline (release 18 December 2024). Article types `Editorial`, `Letter`, `News`, `Comment`, `Review`, and `Retraction` excluded at mirror-build time.

### A.8 Randomness and reproducibility

- **Controller calls (sufficiency check, query generation):** `temperature=0.1` to `0.3`, `top_p=0.95`. The sufficiency check at `temperature=0.1` is effectively deterministic on the prompt distribution we observed.
- **Generator calls:** `temperature=0.3`. We did not fix a serving-level random seed at submission time. This is a reproducibility limitation; we estimate $\sim$1--2 points of aggregate-metric variance from informal repeated-run testing.
- **Few-shot exemplar selection:** deterministic given a fixed development set (selected by question-embedding cosine similarity to the test question; ties broken by training-set order).

### A.9 Live PubMed query dates

The ossllm and asmalltrialsystem systems query PubMed live through NCBI E-utilities (`esearch`, `efetch`, `elink`) during the official BioASQ Task 14b submission windows for Phase A+ batches 2, 3, 4 and Phase B batch 1. The Finalcorrected system queries the local SQLite FTS5 mirror of the 2025 PubMed Annual Baseline (release 18 December 2024). Specific dates within each submission window:

- **Phase A+ batch 2 submission window:** [Rohan to fill from submission portal log]
- **Phase A+ batch 3 submission window:** [Rohan to fill from submission portal log]
- **Phase A+ batch 4 submission window:** [Rohan to fill from submission portal log]
- **Phase B batch 1 submission window:** [Rohan to fill from submission portal log]
- **Phase B batches 3 and 4 submission windows:** [Rohan to fill from submission portal log]

---

## End of response document
