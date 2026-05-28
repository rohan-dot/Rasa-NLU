# BioASQ Task 14b — Revision Checklist + Reviewer Readiness

Items 1–5 are accuracy/consistency fixes (do before camera-ready). 6–9 are quick polish. 10 is the optional contribution sentence. Then: an honest assessment of how strong the paper is and what would make it clearer to a reviewer.

---

# Part A — Copy-paste fixes

## 1. CRITICAL — Finalcorrected few-shot status is wrong

Your submission data says **Finalcorrected used NO few-shot.** Only `ossllm` used the three exemplars. The paper currently claims few-shot is shared and that Finalcorrected uses 3-shot. Fix in four places.

**1a. Table 1 — Finalcorrected few-shot cell:**
```latex
Few-shot            & yes (3-shot)       & no \\
```

**1b. Abstract — drop few-shot from the shared list, add it as a varying axis:**
```latex
Both share a common architecture: an LLM-controlled retrieval loop over PubMed, hybrid PubMedBERT and BM25 reranking fused via Reciprocal Rank Fusion (RRF), and a deterministic JSON format-hygiene filter on the model's output. The configurations vary the LLM backbone (Gemma 3 27B vs.\ Gemma 4 31B Dense), the PubMed access substrate (live NCBI E-utilities vs.\ a local SQLite FTS5 mirror of the Annual Baseline), and whether the answer-generation prompt is conditioned on training-set exemplars (the Gemma 3 system uses three; the Gemma 4 system uses none).
```

**1c. §3.2 — replace the "Both systems condition..." sentence:**
```latex
The ossllm system additionally conditions its answer-generation prompt on three training-set exemplars selected by question-embedding cosine similarity; the Finalcorrected system uses none.
```

**1d. §3.7 — three axes, not two:**
```latex
The submissions vary along three axes: generator scale, retrieval-substrate locality, and few-shot conditioning.
```

## 2. Conclusion (§6) contradicts the paper

Says "three systems" and "varying the answer-generation prompt." Replace the first sentence of §6:
```latex
We submitted two configurations to BioASQ Task 14b under three submission names, sharing an agentic retrieval loop, a hybrid PubMedBERT + BM25 + RRF reranker, and a format-hygiene filter, and varying the LLM backbone, the PubMed access substrate, and few-shot conditioning.
```

## 3. Figure 3 legend still says asmalltrialsystem

Prose says `ossllm`; the figure legend/caption still say `asmalltrialsystem`. In the Figure 3 code:
```latex
Phase A$^+$ (ossllm b2)     % was: asmalltrialsystem b2
Phase B (ossllm b1)         % was: asmalltrialsystem b1
```

## 4. Standardize the model name

§3.2 says "Gemma 4 31B"; abstract and Table 1 say "Gemma 4 31B Dense." Use "Gemma 4 31B Dense" everywhere:
```latex
the Finalcorrected system runs Gemma 4 31B Dense~\cite{gemma4...},
```

## 5. Reference [1] is the wrong BioASQ edition

[1] is the 2025 / Task 13b overview, cited in §1 for "Task 14b." A BioASQ organizer reviewing will catch this. Point the key at the actual 14b/2026 overview if it now exists, or annotate the bib note as the most recent prior overview. In-text it stays:
```latex
BioASQ Task 14b~\cite{nentidis2026bioasq} is the biomedical question-answering shared task of the CLEF lab series.
% verify this key resolves to the 14b overview, not the 13b one
```

## 6. Reconcile "two systems / three names / four submissions"

Add one clarifying sentence to §3.7:
```latex
We use ``submission'' to mean a per-batch system run; the four batch submissions came from two distinct configurations registered under three system names.
```

## 7. §3.4 — "two step process" is inaccurate

Withholding query history and the sufficiency checklist are independent design choices, not sequential steps. Restore:
```latex
Two choices in the loop do most of the work, and both are about restraint rather than machinery.
```

## 8. Quote / apostrophe rendering

Several places use straight quotes or lose them (stray closing quote after the NF1 example; "have I done enough?" renders unquoted). Use LaTeX conventions — `` `` `` opening, `` '' `` closing:
```latex
Consider ``In GBM, which cell state is associated with loss of NF1?''
```

## 9. Typos and table rendering

- §5.1: double period "during development.." → single.
- Table 3 batch 3 row: blank cells where dashes should be:
```latex
3 & \texttt{ossllm} & 0.455 & 0.450 & --- & --- & 0.201 \\
```
- Table 3 caption: write the legend as `` ``---'' = not scored ``.

## 10. OPTIONAL — contribution sentence in §1

Insert after the headline-numbers paragraph:
```latex
Our contribution is a demonstration that placing the LLM as a retrieval controller---deciding what to query and when to stop---rather than only as a post-retrieval generator, reaches competitive Phase B performance with no fine-tuning, and that the residual gap to the field is a retrieval-recall gap rather than a generation gap.
```

---

# Part B — Is the paper strong?

**For BioASQ working notes: yes, this clears the bar comfortably.** Working notes document participation and are accepted when the system is described coherently and results are reported honestly. Yours does both, with clear figures, clean tables, and a discussion that doesn't oversell. You will very likely be accepted regardless of the items below.

**As a piece of work, its real strengths are:**
- The "LLM as retrieval *controller*, not just post-retrieval generator" framing is a genuine angle and is what makes the paper worth citing.
- The format-hygiene contribution is practical, honest, and underreported in the BioASQ literature — reviewers who have submitted to BioASQ will recognize the pain.
- The retrieval-bottleneck diagnosis is well-argued and the honesty about where you sit in the field reads as credibility, not weakness.

**Its one real scientific weakness — the thing a critical reviewer will name:**

The paper makes three causal "what worked" claims (the agentic loop helps, format hygiene helps, RRF beats learned fusion) but **none are isolated by an ablation.** §5.5 already admits this. That's fine for working notes, but it means every "this helped" sentence is an *observation*, not a *measured result*. The fix is not new experiments — it's precision of language (Part C, item 3). Don't let an asserted claim read like a demonstrated one; a reviewer who spots one overclaim discounts the rest.

A secondary weakness: the central empirical claim (retrieval is the bottleneck) leans partly on a cross-phase, cross-batch comparison that is confounded by question difficulty. Your *cleanest* evidence is the single within-batch Finalcorrected batch-4 comparison. Lead with that (Part C, item 2).

---

# Part C — What to improve for clarity to a reviewer (no new experiments)

**1. Put the contribution sentence up front.** Item 10 above. Right now a reviewer reads two pages before learning what they're supposed to take away. One sentence in §1 fixes this and frames everything that follows.

**2. Lead §5.2 with the cleanest evidence, not the confounded view.** §5.2 currently opens with Figure 3 (asmalltrialsystem b2 vs b1 — *different batches*, confounded by question difficulty) and only later gives the within-batch Finalcorrected b4 comparison (same generator, same questions, gold-vs-retrieved). Flip the order: open with the within-batch b4 result (6/9/11-point gap on identical questions), call it your cleanest isolation of retrieval cost, then show Figure 3 as corroborating cross-batch evidence. Strongest argument first.

**3. Separate "measured" from "observed" in §5.1.** Three quick wording changes so a reviewer can't accuse you of overclaiming:
- Agentic loop: you have no with/without-loop number on the test set, so say "we observe that the loop converts under-supported answers into additional queries" — not "the loop carries this performance."
- Format hygiene: "~20–30 recovered list answers" is back-of-envelope — say so: "we estimate, without instrumenting it precisely, that the filter recovered roughly 20–30 list answers."
- RRF: this one you *did* test in development — keep it as a measured claim, and say "in development tuning" so the reviewer knows it's grounded.

**4. Make the agentic loop concrete with one real example.** The mechanism is your contribution but it's described abstractly. If you have even one logged controller trace (the NF1/GBM question decomposing into two sub-queries), put it in a small inline block — "iteration 1 query → pool covers NF1 but not cell-state taxonomy → iteration 2 query → sufficient." A reviewer believes a shown mechanism far more than a described one. If you don't have a logged trace, label the illustrative version "illustrative."

**5. Add a one-line reading guide for the system names early.** Even with the §3.7 fix, a reviewer hits three names and has to hold them in their head. One sentence at the end of §1 — "We report two configurations: a Gemma 3 system (E-utilities, few-shot) and a Gemma 4 system (local mirror, no few-shot)" — primes them before the names appear.

**6. Tighten the limitations from confession to roadmap.** §5.3 says "we did not benchmark / we did not measure" three times. Keep the honesty but frame one or two as forward-looking: "benchmarking newer encoders is the first item in §5.5" reads as a plan; "we did not benchmark them" reads as an apology. Same facts, stronger posture.

**Priority if time is short:** items 1, 2, and 3 are the highest-leverage clarity wins and take ~15 minutes combined. They make the paper read as a confident contribution with well-scoped claims rather than a modest report. Items 4–6 are polish.
