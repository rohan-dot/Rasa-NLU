# BioASQ 14b (paper 159) — Revision Guide v2 (for the CURRENT draft)

This replaces the earlier guide, which was written against an older draft. **This version of your paper already resolves most of what the reviewers asked for** — Related Work is expanded with ballpark figures and beyond-BioASQ context, the novelty statement is promoted, the abstract is balanced, the Gemma 3 citation is corrected (arXiv:2503.19786), a Limitations section is added, and Appendix A now carries prompts, decoding, checkpoints, hardware, top-k, chunking, and randomness.

So what follows is the **residual** set only. It's dominated by **citation mismatches** (your priority), plus a few small easy items. Two big-ticket experiment requests (ablations, 13b run) are currently answered by *disclosure in Limitations* rather than by doing them — that's a judgment call flagged in Part E.

---

## PART A — Reference / citation fixes (PRIORITY)

Your reference *list* is mostly clean now, but several *in-text citations point at the wrong entry*. These are the kind of thing a reference-auditing reviewer (R1 #3) catches immediately. Each is verified below.

### A.1 — [2] BIT.UA points to the wrong year

**Problem:** §2 describes BIT.UA as a **Task 13b / 2025** system ("BioASQ Task 13b in 2025 saw... BIT.UA [2] combined dense retrieval with cross-encoder reranking and supervised fine-tuning"), but ref **[2]** is *"BIT.UA at bioasq **12**: From retrieval to answer generation"*, CLEF **2024**, Vol-3740. Wrong edition.

**Verified correct entry** (CEUR Vol-4038, paper_22 — the 13b/2025 working note you're actually describing):

```bibtex
@inproceedings{jonker2025bitua,
  title     = {{BIT.UA} at {BioASQ} 13B: Revisiting Evaluation, {DPRF}-Enhanced Retrieval and Fine-Tuned {LLMs}},
  author    = {Jonker, Richard A. A. and Almeida, Tiago and Almeida, Jo{\~a}o R. and Matos, S{\'e}rgio},
  booktitle = {CLEF 2025 Working Notes},
  editor    = {Faggioli, G. and Ferro, N. and Rosso, P. and Spina, D.},
  series    = {CEUR Workshop Proceedings},
  volume    = {4038},
  pages     = {328--342},
  publisher = {CEUR-WS.org},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4038/paper_22.pdf}
}
```

> Decide: if you genuinely meant the 2024 (12b) BIT.UA work, then fix the *text* to say 2024/12b instead. But your numeric claims (yes/no Macro-F1 ~0.96, list F ~0.55) describe the 13b results, so replacing the ref is the right move.

### A.2 — [7] points to a BioASQ overview but is cited for PubMedQA

**Problem:** §2 reads "Biomedical question answering has also been studied on adjacent benchmarks: PubMedQA **[7]** (research-question yes/no)". But ref **[7]** is *"Overview of BioASQ 2026: The fourteenth BioASQ Challenge"* — not PubMedQA. PubMedQA needs its own citation.

**Verified correct PubMedQA entry** (Jin et al., EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1259):

```bibtex
@inproceedings{jin2019pubmedqa,
  title     = {{PubMedQA}: A Dataset for Biomedical Research Question Answering},
  author    = {Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {2567--2577},
  year      = {2019},
  doi       = {10.18653/v1/D19-1259}
}
```

Then update the in-text cite: `PubMedQA~\cite{jin2019pubmedqa}`.

### A.3 — [1] is the 13th (2025) overview but cited for "Task 14b"

**Problem:** §1 opens "BioASQ Task 14b **[1]** is the biomedical question-answering shared task", but ref **[1]** is *"Overview of bioasq 2025: The **thirteenth** bioasq challenge"*. Task 14b is the **fourteenth** challenge (2026). The right target is your [7] or [10] (both 14b/2026 overviews).

**Fix:** change the intro cite to the 14b overview: `BioASQ Task 14b~\cite{nentidis2026overview}` (see A.4 on which entry that should be). Keep [1] (the 13th overview) only where you actually discuss the **2025 / 13b** field — e.g. in §2's "BioASQ Task 13b in 2025" sentence, where it's the correct anchor.

### A.4 — [7] and [10] are duplicate 14b overviews; collapse to one

**Problem:** both are "Overview of BioASQ ... fourteenth / Tasks 14b and Synergy14 ... CLEF2026":
- **[7]** "Overview of BioASQ 2026: The fourteenth BioASQ Challenge..." (LNCS / Springer proceedings)
- **[10]** "Overview of BioASQ Tasks 14b and Synergy14 in CLEF2026" (CLEF 2026 Working Notes)

These are the same event's two overview papers. Pick **one** canonical `nentidis2026overview` key and use it everywhere you refer to Task 14b (intro, §4.3, Table 4 leaderboard context). Keeping both invites "why are these cited interchangeably?" Unless you specifically need the Working-Notes version for the *task definition* and the LNCS version for the *challenge overview*, merge. Also note the year slips: [10]'s venue line says "CLEF **2025** Working Notes" while the title says CLEF2026 — fix to 2026.

### A.5 — [7,10] cited for a RAG-over-PubMed trend claim

**Problem:** §2 "RAG over PubMed has been a recurring approach in the broader biomedical NLP literature **[7, 10]**" — but [7] and [10] are BioASQ *overviews*, not RAG-over-PubMed method papers. The citation doesn't support the sentence.

**Fix (pick one):**
- Cite an actual RAG-over-biomedical-literature method/survey here (e.g. a MedRAG / biomedical-RAG reference you trust), **or**
- Soften to attribute the trend to the BioASQ field itself: "RAG over PubMed has been the dominant approach across recent BioASQ editions~\cite{nentidis2026overview, jonker2025bitua}." That keeps the claim supportable by the papers you actually have.

### A.6 — Uncited claim R1 #3 explicitly named (still open)

§3.3 still asserts, with no citation: *"biomedical queries carry an unusual density of exact tokens (gene symbols, drug names, disease abbreviations) that embedding similarity can soften."* R1 #3 quoted this exact claim. Add a citation and lightly attribute:

> **Replace:** We retain a sparse path because biomedical queries carry an unusual density of exact tokens (gene symbols, drug names, disease abbreviations) that embedding similarity can soften.
>
> **With:** We retain a sparse path because biomedical queries carry an unusual density of exact-match tokens — gene symbols, drug names, disease abbreviations — whose lexical signal dense encoders are known to soften, a well-documented motivation for hybrid sparse–dense biomedical retrieval~\cite{gu2021pubmedbert, boteva2016cds}.

```bibtex
@inproceedings{boteva2016cds,
  title     = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  author    = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  booktitle = {Advances in Information Retrieval (ECIR 2016)},
  pages     = {716--722},
  year      = {2016},
  doi       = {10.1007/978-3-319-30671-1_58}
}
```
(You already cite `gu2021pubmedbert` as [14], so reuse that key.)

### A.7 — Yes-bias claim (R1 #3, second named quote) — mostly OK, optional cite

§3.5 already reframes this as *your own* measured observation ("We treat this as a calibration target... rather than as a property of all instruction-tuned LLMs"). That satisfies R1 #3's core concern. **Optional** strengthening — add one reference for the general acquiescence/sycophancy phenomenon so it doesn't read as unsupported:

> After "...regardless of the polarity of the supporting evidence." add: This is consistent with the agreement/sycophancy bias documented in instruction-tuned models~\cite{sharma2024sycophancy}.

```bibtex
@inproceedings{sharma2024sycophancy,
  title     = {Towards Understanding Sycophancy in Language Models},
  author    = {Sharma, Mrinank and Tong, Meg and Korbak, Tomasz and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.13548}
}
```

---

## PART B — R1 #7: "outperform" without a significance test (still open)

R1 #7 is explicit: use "outperform" only with a supporting significance test. Two instances remain, and you never run a significance test anywhere, so soften both:

**B.1 — §3.3:**
> **Current:** in preliminary tuning RRF outperformed every linear-combination configuration we tried.
> **Revise to:** in preliminary tuning RRF scored higher than every linear-combination configuration we tried on our development set (we did not test this difference for statistical significance).

**B.2 — §5.1 (third design choice):**
> **Current:** RRF fusion of dense PubMedBERT and BM25 ranks outperformed every learned linear combination of normalized scores we tested during development.
> **Revise to:** RRF fusion of dense PubMedBERT and BM25 ranks scored higher than every learned linear combination of normalized scores we tested during development (untested for significance).

Then grep the `.tex` for `outperform` / `beats` / `superior` to be sure none slipped through.

---

## PART C — R1 #6: contextualise the discussion with refereed literature (partly open)

The reviewer wanted the discussion tied to published work. Your expanded §2 does a lot of this now, but §5.1 ("Where We Are Strong") and §5.2 ("Where We Stand") describe your own design choices and the leaderboard without citing the refereed methods that close each gap. One or two bridging cites would close R1 #6 cleanly.

Add to the end of §5.2 (after the field-leader comparison):

> Each gap to the field leaders maps onto a known, published remedy: the Phase A+ list-F and factoid gaps are, in the 13b record, closed primarily by learned cross-encoder reranking and generator fine-tuning on BioASQ relevance judgments~\cite{jonker2025bitua, borazio2025unitor}, while the ideal-answer ROUGE-2 gap is a verbosity-calibration effect rather than a retrieval one. We omit these levers deliberately (§5.3) to keep the agentic-loop and format-hygiene contribution isolable.

(Reuses `jonker2025bitua` from A.1 and your existing UNITOR ref [3].)

---

## PART D — R2 #4 residuals (Appendix A mostly closes this; two gaps left)

Appendix A now covers prompts (A.6), decoding (A.1), checkpoints (A.2), hardware (A.7), top-k (A.5), chunking (A.4), randomness/seeds (A.8). R2 #4 listed two more that are still missing:

**D.1 — Controller tool-call schema (prose-only in §3.4).** Add the explicit JSON schema, either in §3.4 or a new short A-subsection:

```json
{
  "action": "search | sufficient | stop",
  "query":  "<reformulated PubMed query; required iff action == 'search'>"
}
```
> Add one line: "Any controller output not parseable to this schema is treated as `stop`." That directly answers "controller schema."

**D.2 — Dates for live PubMed queries (never stated).** `ossllm` hits live NCBI E-utilities, so its results depend on the query date. Add to §3.3 or §A.7:

> Live E-utilities queries for the `ossllm` system were issued during the Task 14b Phase A+ submission windows on **[DATES — fill from your submission logs]**. Because the live backend reflects PubMed as of the query date, these runs reproduce only up to subsequent index changes; `Finalcorrected` avoids this by querying the fixed 2025 Annual Baseline (release 18 December 2024).

**D.3 — Seed (A.8 already discloses none was fixed).** Fine as disclosure. Optional one-line strengthening: recommend fixing `seed` in vLLM `SamplingParams` for camera-ready reproducibility, noting your submitted runs did not and this contributes the ~1–2 point variance you already report.

---

## PART E — The two experiment requests (currently disclosed, not done) — DECISION POINT

Both reviewers asked for experiments. Your current draft **acknowledges both as limitations in §5.3** rather than running them. That is a legitimate but *weaker* response — a borderline-0 reviewer (R2) may not be moved by "we acknowledge we didn't."

**E.1 — Ablations (R2 #3).** §3.7 + §5.3 explicitly say you cannot decompose the 0.941 headline into agentic-loop / format-hygiene / sufficiency-check contributions. If you can run even a *small* dev-set ablation before camera-ready, it converts R2 #3 from "acknowledged limitation" to "addressed," which is the single biggest lever on the R2 score. Minimum viable table:

```latex
\begin{table}[t]
\centering
\caption{Component ablation on the development set. Each row removes one component from the full system.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & Yes/No Acc. & Factoid Strict & List F1 \\
\midrule
Full system                          & \textbf{.} & \textbf{.} & \textbf{.} \\
\;$-$ sufficiency check (force $N{=}5$) & . & . & . \\
\;$-$ adversarial yes/no prompt      & . & . & . \\
\;$-$ format-hygiene filter          & . & . & . \\
\;$-$ BM25 (dense only)              & . & . & . \\
\;$-$ agentic loop (single retrieve) & . & . & . \\
\bottomrule
\end{tabular}
\end{table}
```
> The format-hygiene and BM25-off rows can often be produced by **re-scoring existing outputs** (hygiene off = re-score raw generations; no new LLM calls). Sufficiency-check and agentic-loop rows need fresh dev-set runs only. If you add this, delete the "no controlled ablations" sentences from §3.7 and §5.3 and point them at Table~\ref{tab:ablation}.

**E.2 — Run on BioASQ 13b (R1 #4).** §5.3 admits you didn't do cross-year validation. If you can run `ossllm` (or `Finalcorrected`) on the 13b Phase B test set and add one comparison table, it turns your central retrieval-bottleneck claim from single-competition into cross-competition evidence — exactly what R1 asked for.

> If neither experiment is feasible before the camera-ready deadline, keep the honest §5.3 disclosures you already have — but be aware that's the minimum, not the ask.

---

## PART F — Internal consistency (reviewers will notice these)

**F.1 — "two systems" vs "three systems".** The abstract/intro say **two** systems under **three** submission names; §3.7 reconciles this; but **§6 Conclusion** reverts to *"We submitted **three** open-weights agentic RAG systems."* Pick one framing and make the conclusion match the abstract (two configurations / three submission names).

**F.2 — Figure 3 labels use the retired name.** Figure 3's legend and caption say **"asmalltrialsystem b2 / b1"**, but §5.2 text refers to the same data as **"ossllm."** Since §3.7 folds asmalltrialsystem into ossllm, relabel Figure 3 to `ossllm` so the figure and prose agree.

**F.3 — Mirror size is now consistent (~50 GB).** Good — the earlier 35 GB/50 GB conflict is gone in this draft. No action.

**F.4 — Overview-entry venue years.** In [8] and [10], the "CLEF 2025 Working Notes" editor line conflicts with the "CLEF 2026" title. Normalize to 2026.

---

## Checklist

| # | Reviewer | Item | Type | Status |
|---|---|---|---|---|
| A.1 | R1 #3 | Fix [2] BIT.UA → 13b/2025 working note | ref | ☐ |
| A.2 | R1 #3 | Add real PubMedQA cite; fix [7] misuse | ref | ☐ |
| A.3 | R1 #3 | [1] (13th) miscited for Task 14b in §1 | ref | ☐ |
| A.4 | — | Merge duplicate 14b overviews [7]/[10] | ref | ☐ |
| A.5 | R1 #3 | [7,10] miscited for RAG-over-PubMed claim | ref | ☐ |
| A.6 | R1 #3 | Cite "exact tokens" claim (§3.3) | ref | ☐ |
| A.7 | R1 #3 | (opt.) cite yes-bias phenomenon (§3.5) | ref | ☐ |
| B.1 | R1 #7 | Soften "outperform" §3.3 | writing | ☐ |
| B.2 | R1 #7 | Soften "outperform" §5.1 | writing | ☐ |
| C | R1 #6 | Refereed-lit bridge in §5.2 | writing | ☐ |
| D.1 | R2 #4 | Controller JSON schema | writing | ☐ |
| D.2 | R2 #4 | Live-query dates | writing | ☐ |
| D.3 | R2 #4 | (opt.) seed note | writing | ☐ |
| E.1 | R2 #3 | Ablation table (or keep disclosure) | **experiment** | ☐ |
| E.2 | R1 #4 | 13b run (or keep disclosure) | **experiment** | ☐ |
| F.1 | — | "two" vs "three" systems in §6 | consistency | ☐ |
| F.2 | — | Figure 3 label asmalltrialsystem→ossllm | consistency | ☐ |
| F.4 | — | Overview-entry venue years | ref | ☐ |

**Do Part A first (all reference mismatches, ~30 min), then B/C/D/F (all text, ~1 hr). E is the only part needing runs and is where the R2 score actually lives.**
