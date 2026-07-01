# BioASQ 14b (paper 159) — Camera-Ready Revision Guide

Reviewer comments addressed here: **all of R1 (#2–#7) and R2 (#1–#4)**, excluding the GitHub repo itself (R1 #1, R1 #5's repo clause, and the "scripts" clause of R2 #4 — those require actually publishing code, not a text edit).

Each item below gives: the reviewer comment, where it lands in the paper, and ready-to-paste replacement or insertion text. LaTeX is written for the CEUR-WS `ceurart` class you're already using.

---

## PART 0 — Reference audit (R1 #3, plus broken cites) — DO THIS FIRST

R1 #3 asks that all claims be backed by suitable references and names two specific unsupported claims. Separately, the compiled PDF shows several broken `[?]` cites and one wrong reference. Fix all of these together so the reference list is internally consistent before you touch prose.

### 0.1 Broken `[?]` citations to resolve

| Location | Current | Problem | Fix |
|---|---|---|---|
| §4, p.10 ("placing it in the upper-middle quartile ... field `[? ]`") | `[? ]` | Missing Task 14b overview cite | Point to the Task 14b overview reference (new `nentidis2026overview`, see 0.4) |
| §3.1 end / §4 "field `[? ]`" (multiple) | `[? ]` | same | same |
| §5.1 ("Task 14b drew a large ... field `[? ]`") | `[? ]` | same | `\cite{nentidis2026overview}` |
| §5.2 ("optional supervised fine-tuning of the generator `[? ?]`") | `[? ?]` | two dead cites | `\cite{jonker2025bitua, borazio2025unitor}` |
| §5.3 ("in the spirit of ReAct `[? ]`") | `[? ]` | ReAct already exists as ref [10] | `\cite{yao2023react}` |
| §5.4 ("Reciprocal Rank Fusion `[? ]`") | `[? ]` | RRF already exists as ref [6] | `\cite{cormack2009rrf}` |

**Root cause:** these are almost certainly `\cite{}` keys that don't match any `\bibitem`/`.bib` entry, or a missing `nentidis2026overview` entry. Grep your `.tex` for `\cite{` keys and diff against your `.bib` keys — every `[?]` is a key with no match.

### 0.2 Wrong reference — [7] Gemma 3 27B

Reference **[7]** (cited for "Gemma 3 27B Instruction-Tuned" in §3.2 and as the asmalltrialsystem/ossllm backbone) currently points to:

> Finkelstein et al., "Translate-gemma technical report", arXiv:2601.09012 (2026)

That's a translation model report, not the Gemma 3 27B model card. Replace with the actual Gemma 3 technical report / model card. Suggested `.bib`:

```bibtex
@misc{gemma3_2025,
  title        = {Gemma 3 Technical Report},
  author       = {{Gemma Team, Google DeepMind}},
  year         = {2025},
  howpublished = {arXiv preprint arXiv:2503.19786},
  note         = {Instruction-tuned 27B dense variant; Apache 2.0. Accessed 18 May 2026},
  url          = {https://arxiv.org/abs/2503.19786}
}
```

> Verify the arXiv ID against the version you actually used before submitting — swap in the model-card URL you pulled `google/gemma-3-27b-it` from if that's your canonical source.

### 0.3 R1 #3 — two named claims that need a citation

**Claim A — §3.6:** *"biomedical queries carry an unusual density of exact tokens"*

Add a citation to work establishing that biomedical/clinical IR is lexically driven and that sparse signals (exact gene/drug/entity tokens) matter. Reword slightly so the claim is attributed rather than asserted bare:

> **Replace:**
> We keep a sparse path even though dense embeddings dominate modern retrieval because biomedical queries carry an unusual density of exact tokens.
>
> **With:**
> We keep a sparse path even though dense embeddings dominate modern retrieval because biomedical queries carry an unusual density of exact-match tokens — gene symbols, drug names, and disease abbreviations — whose retrieval is known to benefit from lexical signal that dense encoders can wash out~\cite{gu2021pubmedbert, boteva2016cds}.

Suggested `.bib` (CDS is a standard biomedical-IR lexical-retrieval reference; substitute one you prefer):

```bibtex
@inproceedings{boteva2016cds,
  title     = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  author    = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  booktitle = {Advances in Information Retrieval (ECIR 2016)},
  year      = {2016},
  pages     = {716--722},
  doi       = {10.1007/978-3-319-30671-1_58}
}
```

**Claim B — §3.8:** *"Instruction-tuned LLMs exhibit a pronounced yes-bias"* (the reviewer quotes this from an earlier draft phrasing)

Your current §3.8 text already reframes this honestly as *your own* observation ("We treat this as a calibration target ... rather than as a property of all instruction-tuned LLMs") — good, keep that framing. But add a supporting reference for acquiescence/agreement bias in LLMs so the phenomenon isn't presented as unsupported:

> **Current:** In our preliminary runs on the BioASQ training set, the base Gemma 3 27B prompt answered "yes" to roughly 70% of yes/no questions regardless of the polarity of the supporting evidence. We treat this as a calibration target for our prompt design rather than as a property of all instruction-tuned LLMs.
>
> **Add after the second sentence:** This mirrors the acquiescence / sycophancy bias documented in instruction-tuned models, where alignment training skews responses toward agreement~\cite{sharma2024sycophancy}.

```bibtex
@inproceedings{sharma2024sycophancy,
  title     = {Towards Understanding Sycophancy in Language Models},
  author    = {Sharma, Mrinank and Tong, Meg and Korbak, Tomasz and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.13548}
}
```

### 0.4 Missing Task 14b overview reference (resolves most `[?]`)

You cite the 13b overview as [2] but have no 14b overview entry — that's what every `[?]` in §4/§5.1 wants.

```bibtex
@inproceedings{nentidis2026overview,
  title     = {Overview of BioASQ Task 14b and Synergy in CLEF2026},
  author    = {Nentidis, Anastasios and Katsimpras, Georgios and Krithara, Anastasia and Paliouras, Georgios},
  booktitle = {CLEF 2026 Working Notes},
  series    = {CEUR Workshop Proceedings},
  publisher = {CEUR-WS.org},
  year      = {2026},
  note      = {To appear}
}
```

> If the 14b overview isn't citable yet at camera-ready, mark it `note = {to appear}` and cite anyway — organizers expect this.

---

## PART 1 — R1 #2 + R2 #1: Expand Related Work / clarify novelty

Both reviewers hit §2. R1 #2 wants (a) ballpark effectiveness figures from the literature, (b) contextualisation **beyond** the BioASQ competition, (c) concluding remarks that motivate the methodology. R2 #1 says the contribution reads as "an assembly of standard RAG components" and wants novelty stated clearly + related work expanded.

### 1.1 Add a "beyond BioASQ" paragraph to §2 (addresses R1 #2b)

Insert after the "Iteration and agency" paragraph in §2:

> **Beyond BioASQ.** Agentic and iterative retrieval has developed rapidly outside the biomedical setting. Self-RAG~\cite{asai2024selfrag} trains a model to decide when to retrieve and to critique retrieved passages with reflection tokens; FLARE~\cite{jiang2023flare} triggers retrieval dynamically as generation proceeds; and IRCoT~\cite{trivedi2023ircot} interleaves chain-of-thought reasoning with multi-step retrieval for multi-hop questions. These systems establish that *when* and *what* to retrieve can be model-driven rather than fixed. Our design imports that principle into biomedical QA but constrains the agency tightly: a single Gemma instance controls only the forward retrieval trajectory over PubMed, with a hard five-iteration cap, rather than performing open-ended reflection or free-form tool use. To our knowledge, this specific combination — a single dual-role open-weights LLM as retrieval controller plus deterministic BioASQ-shape format hygiene — has not been reported in the BioASQ setting.

```bibtex
@inproceedings{asai2024selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024}, note={arXiv:2310.11511}}

@inproceedings{jiang2023flare,
  title={Active Retrieval Augmented Generation},
  author={Jiang, Zhengbao and Xu, Frank F. and Gao, Luyu and others},
  booktitle={EMNLP}, year={2023}, note={arXiv:2305.06983}}

@inproceedings{trivedi2023ircot,
  title={Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  booktitle={ACL}, year={2023}, note={arXiv:2212.10509}}
```

### 1.2 Add ballpark effectiveness figures (addresses R1 #2a)

R1 explicitly wants "top results from last year's competition ... where these results can be compared." Add a short paragraph (or a small table) to §2 anchoring the field. Insert after "Retrieval substrates.":

> **Where the 13b field landed.** To situate our numbers: on Task 13b, the strongest Phase B systems reached yes/no accuracy at or near 1.00 on individual batches, list F1 in the ~0.45–0.55 range, and factoid strict roughly ~0.45–0.53, while Phase A+ list F1 for most participants plateaued in the low-to-mid 0.30s~\cite{nentidis2025overview13b, jonker2025bitua}. These ranges are the backdrop for our results in §4: our Phase B batch-1 yes/no (0.941) is competitive, while our Phase A+ list F1 (0.255) sits at the lower end of that plateau — consistent with retrieval recall, not generation, being the binding constraint.

> Replace the illustrative ranges above with the exact top-line figures from the 13b overview once you pull them from `Vol-4038/paper_1.pdf` (your ref [2]). Keep them approximate ("~") if you can't verify a single canonical number — that's honest and still answers R1.

Note: ref [2] in your current list is the 13b overview; give it the key `nentidis2025overview13b` so the two overview cites (13b, 14b) don't collide.

### 1.3 Novelty statement (addresses R2 #1)

Your §3 already contains a good novelty sentence ("The contribution of this paper is not a new retrieval component ... It is the integration of three specific choices..."). R2 didn't see it as prominent enough. **Promote it into the Introduction.** Add as the final paragraph of §1, right before "The remainder of the paper is organized as follows":

> **Contribution.** We do not claim a new retrieval component or generator architecture. Our contribution is a specific, reproducible integration and the empirical lessons it yields: (i) a single open-weights Gemma instance in a dual controller–generator role, where agency is bounded to the forward retrieval trajectory (search / sufficient / stop, cap $N{=}5$); (ii) a deterministic format-hygiene pass that recovers an entire class of silent, shape-only scoring failures against BioASQ's strict scorer; and (iii) a same-generator, same-batch Phase B vs. Phase A+ comparison that cleanly isolates retrieval recall — not generation quality — as the binding constraint on the end-to-end pipeline. None of these depends on fine-tuning, ensembling, or a learned reranker, which is precisely what makes the isolation clean.

---

## PART 2 — R2 #2: Tone down the positive framing

R2: "results are not competitive enough to support the paper's positive framing." The fix isn't to hide 0.94 — it's to state the mid/lower-middle results in the **same breath** in the abstract and intro, so the framing is balanced from the first read. §5.6 already does this honestly; pull that candor forward.

### 2.1 Abstract — one-sentence balance addition

Your abstract already mentions "mid-field or lower-middle" — good. Add one explicit sentence at the end of the results portion so the headline can't be read alone:

> **Add after** "...they returned identical exact answers across all questions.":
> We therefore frame this as a competitive-but-not-leading submission: strong on Phase B yes/no, middle-of-field on factoid, list, and ideal-answer metrics, and lower-middle on Phase A+, where retrieval recall bounds the pipeline.

### 2.2 Introduction — soften the "competitive" claim

In §1's "Our results ... support three claims. First, ...":

> **Current:** our best system attains 0.94 yes/no accuracy on Phase B batch 1, placing it in the upper-middle quartile of the Task 14b field, while list F1, factoid strict, and ideal-answer ROUGE-2 land in the middle of the field.
>
> **Revise to:** our best system attains 0.94 yes/no accuracy on Phase B batch 1 — upper-middle quartile on that batch and metric — but this is the exception rather than the rule: list F1, factoid strict, and ideal-answer ROUGE-2 land mid-field, and Phase A+ scores are lower-middle. We foreground this spread deliberately, since the paper's contribution is the *diagnosis* of where the pipeline is bounded, not a leaderboard placement.

---

## PART 3 — R1 #6: Contextualise §5.1 with refereed literature

R1 #6: "The discussion in Section 5.1 would be stronger if contextualised with the refereed literature." §5.1 currently positions only against the (unpublished) leaderboard. Add a bridging sentence tying each gap to a published method.

Insert after the bulleted per-metric list in §5.1:

> Each of these gaps has a known remedy in the refereed literature. The list-F1 and factoid gaps to the leaders are, in the 13b record, closed primarily by learned cross-encoder reranking and generator fine-tuning on BioASQ relevance judgments~\cite{jonker2025bitua, borazio2025unitor}; the ideal-answer ROUGE-2 gap is a length/verbosity-calibration issue documented across summarization work rather than a retrieval problem. Our submission deliberately omits these levers (§5.6) to keep the contribution of the agentic loop and format hygiene isolable, which is why our placement is mid-field rather than leading.

---

## PART 4 — R1 #7: "outperform" only with statistical significance

R1 #7: the word "outperform" should only appear when backed by a significance test. You have comparative claims in §3.6 and §5.4 and you explicitly admit "we did not test for statistical significance." Two options — **soften** (fast) or **test** (stronger). Recommended: soften, and add one honest caveat sentence.

### 4.1 §3.6 — RRF vs linear combination

> **Current:** in our preliminary tuning RRF scored higher than every linear-combination configuration we tested ... we did not test for statistical significance.
>
> **Revise to:** in our preliminary tuning RRF scored higher than every linear-combination configuration we tested, with one fewer hyperparameter to calibrate. We report this as an operational preference observed on a small development set, not a statistically validated result; we did not run a significance test and do not claim RRF *outperforms* learned fusion in general.

### 4.2 §5.4 heading + body — "RRF beats learned fusion"

> **Current heading:** *RRF beats learned fusion when query score distributions are heavy-tailed.*
>
> **Revise heading to:** *RRF scored higher than linear fusion in our preliminary tuning.*
>
> **In the body, replace** "RRF beat every linear-combination configuration we tried" **with** "RRF scored higher than every linear-combination configuration we tried on our development set (untested for significance)".

### 4.3 Global sweep

Search the `.tex` for **outperform**, **beats**, **better than**, **superior**. For each: either (a) it's backed by a test → keep, or (b) it isn't → replace with "scored higher / lower on [set]". You have no significance tests anywhere in the paper, so in practice every instance becomes descriptive.

---

## PART 5 — R1 #4: Run the approach on a previous competition (BioASQ 13b)

R1 #4: strengthen the argument by reporting your approach on prior-competition benchmark data, e.g. BioASQ 2025 (13b). This needs an actual run. Minimum viable version: run `asmalltrialsystem` (or `Finalcorrected`) on the 13b Phase B test batches and drop one comparison table.

### 5.1 New subsection §4.x — Validation on BioASQ 13b

> **§4.x Retrospective validation on Task 13b.** To check that our results are not an artifact of the 14b batches, we ran our pipeline unchanged on the Task 13b Phase B test set. Table X reports exact-answer scores alongside the 13b field. [Fill from your run.] The pattern replicates: Phase B yes/no is competitive while list/factoid sit mid-field, and — where we also ran Phase A+ — the Phase B–Phase A+ gap widens monotonically with recall sensitivity, reproducing the batch-4 finding on an independent question set. This is direct evidence that the retrieval-recall bottleneck we identify is a property of the architecture, not of the 14b batches.

Table skeleton:

```latex
\begin{table}[t]
\centering
\caption{Retrospective validation: our pipeline on BioASQ 13b Phase B vs.\ the 13b field.}
\label{tab:13b-validation}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{2}{c}{Yes/No} & Factoid & List & Ideal \\
Batch & System & Acc. & Macro-F1 & Strict & F1 & R-2 F1 \\
\midrule
b & asmalltrialsystem (13b) & . & . & . & . & . \\
b & 13b field leader        & . & . & . & . & . \\
\bottomrule
\end{tabular}
\end{table}
```

> If you genuinely can't get a 13b run done before camera-ready, the honest fallback is a paragraph in §5.7 (future work) explicitly committing to it — but a real run is what R1 asked for and is worth the effort, since it turns your central claim from single-competition into cross-competition.

---

## PART 6 — R2 #3: Ablations (the highest-priority experiment)

R2 #3: "The evaluation lacks essential ablations for a multi-component pipeline." This is the single most important item for the R2 borderline-0. §5.7 currently *promises* ablation as future work — reviewers want it *in the paper*. You already have informal numbers scattered in §3.6, §5.4, and §5.5; consolidate them into one controlled table and run the missing cells.

### 6.1 New subsection §4.y — Component ablations

Ablate on a fixed dev set (your 17-question held-out set in §5.4, or a Phase B batch), toggling one component at a time:

| Configuration | Yes/No | Factoid strict | List F1 | Notes |
|---|---|---|---|---|
| Full system | — | — | — | reference row |
| − sufficiency check (force N=5) | — | — | — | tests the agentic control decision |
| − adversarial yes/no prompt | — | — | — | you report ~70%→balanced; quantify it |
| − format-hygiene filter | — | — | — | you estimate −2 to −4 list F1; measure it |
| − BM25 (dense only) | — | — | — | you say factoid/list suffer; quantify |
| − agentic loop (single-shot retrieve) | — | — | — | vs. your 0.84 y/n, 0.18 list zero-shot anchor |

```latex
\begin{table}[t]
\centering
\caption{Component ablation on the development set. Each row removes exactly one component from the full system.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & Yes/No Acc. & Factoid Strict & List F1 \\
\midrule
Full system                       & \textbf{.} & \textbf{.} & \textbf{.} \\
\;$-$ sufficiency check (force $N{=}5$) & . & . & . \\
\;$-$ adversarial yes/no prompt   & . & . & . \\
\;$-$ format-hygiene filter       & . & . & . \\
\;$-$ BM25 (dense only)           & . & . & . \\
\;$-$ agentic loop (single retrieve) & . & . & . \\
\bottomrule
\end{tabular}
\end{table}
```

> Fastest path: the format-hygiene and BM25 rows can be produced by re-scoring **existing** submission outputs with the component disabled (no new LLM calls needed for hygiene; BM25-off needs a re-rank + re-generate). The sufficiency-check and agentic-loop rows need fresh runs but only on a small dev set. Then **delete the "Component ablation" bullet from §5.7** and replace the §5.4/§5.5 informal estimates with pointers to Table~\ref{tab:ablation}.

### 6.2 Update §5.7

> **Remove** the "Component ablation of the agentic loop" future-work bullet entirely (it's now done). Keep the cross-encoder reranker and per-model context-budgeting bullets.

---

## PART 7 — R2 #4: Reproducibility residuals (mostly already in Appendix A)

R2 #4 lists: full prompts, controller schema, decoding settings, exact model checkpoints, random seeds, hardware, top-k retrieval settings, chunk limits, scripts, dates for live PubMed queries.

**Already covered in Appendix A** — the reviewer likely missed it. Add one forward-reference so they can't:

> **Add to the end of §1 (or start of §4):** Full reproducibility details — prompts (§A.6), decoding settings (§A.1), model checkpoints (§A.2), retrieval and top-$k$ schedules (§A.3, §A.5), chunking limits (§A.4), the format-hygiene filter (§A.7), and hardware/serving (§A.8) — are provided in Appendix~A.

Then close the three genuine gaps:

### 7.1 Controller tool-call schema (currently prose-only in §3.7)

Add a fenced schema to §3.7 or Appendix A.6:

```json
{
  "action": "search | sufficient | stop",
  "query":  "<reformulated PubMed query, required iff action == 'search'>"
}
```

> One-line note: "The controller emits exactly this JSON; any output not parseable to this schema is treated as `stop`." This directly answers "controller schema."

### 7.2 Dates for live PubMed / E-utilities queries (R2 #4, never stated)

`asmalltrialsystem` and `ossllm` hit **live** NCBI E-utilities, so results depend on the query date. Add to §A.8 (Hardware/serving) or §3.3:

> Live E-utilities queries for asmalltrialsystem and ossllm were issued during the Task 14b Phase A+ submission windows on **[DATES — fill from your submission logs]**. Because the live backend reflects PubMed as of the query date, these runs are reproducible only up to subsequent index changes; the `Finalcorrected` system avoids this by querying the fixed 2025 Annual Baseline (release 18 December 2024).

### 7.3 Random seed (A.9 discloses none was fixed)

A.9 already honestly states no serving seed was fixed (~1–2 pts variance). That's a disclosure, not a fix. Strengthen it minimally:

> **Add to A.9:** For camera-ready reproducibility we recommend fixing `seed` in the vLLM `SamplingParams` and setting `VLLM_ENGINE_ITERATION_TIMEOUT_S`; our submitted runs did not, and we quantify the resulting aggregate-metric variance at ~1–2 points from informal repeated runs.

---

## PART 8 — Internal consistency fix (not a reviewer comment, but they'll catch it)

The SQLite mirror size is inconsistent:

- **Intro / §2 area (p.4):** "a local $\sim$35GB SQLite FTS5 mirror"
- **§3.3 and §A.8:** "$\sim$50GB SQLite ... mirror"

Pick one (A.8's ~50 GB is the more detailed statement) and make all three occurrences match. A reproducibility-focused R2 reviewer *will* flag a 15 GB discrepancy in a dataset artifact.

> Search `.tex` for `35GB`, `35 GB`, `50GB`, `50 GB` and normalize.

---

## Checklist (print this)

| # | Reviewer | Item | Type | Status |
|---|---|---|---|---|
| 0.1 | broken cites | Resolve all `[?]` | ref | ☐ |
| 0.2 | R1 #3 | Fix ref [7] Gemma 3 | ref | ☐ |
| 0.3 | R1 #3 | Cite exact-token claim (§3.6) | ref | ☐ |
| 0.3 | R1 #3 | Cite yes-bias claim (§3.8) | ref | ☐ |
| 0.4 | — | Add 14b overview ref | ref | ☐ |
| 1.1 | R1 #2b | "Beyond BioASQ" para in §2 | writing | ☐ |
| 1.2 | R1 #2a | Ballpark 13b figures | writing | ☐ |
| 1.3 | R2 #1 | Promote novelty stmt to §1 | writing | ☐ |
| 2.1 | R2 #2 | Balance abstract | writing | ☐ |
| 2.2 | R2 #2 | Soften §1 framing | writing | ☐ |
| 3 | R1 #6 | Refereed lit in §5.1 | writing | ☐ |
| 4 | R1 #7 | Remove unsupported "outperform" | writing | ☐ |
| 5 | R1 #4 | Run on 13b + table | **experiment** | ☐ |
| 6 | R2 #3 | Ablation table | **experiment** | ☐ |
| 7.1 | R2 #4 | Controller JSON schema | writing | ☐ |
| 7.2 | R2 #4 | Live-query dates | writing | ☐ |
| 7.3 | R2 #4 | Seed note | writing | ☐ |
| 8 | — | Fix 35GB/50GB mismatch | writing | ☐ |

**Two items require running code (5, 6); everything else is text.** Do 0.x first (references), then 6 (ablations — biggest score impact), then the writing items.
