Understood. Let me re-read §5 carefully against your actual results, and propose changes that strengthen the CSS argument using *your* numbers and *your* existing structure — not replace it with mine.

## What §5 currently does well

- §5.1 makes the "evidence labor is invisible" argument and lands "governed agency over more autonomy" — that closing line is good.
- §5.2 names the moral compression problem, cites the right automation-bias literature (Abdelwanis, Khera), and explains the adversarial yes/no procedure.
- §5.3 cites Elish and WHO and ends with a strong "agent logs are CSS data" line.

The bones are there. The problem is that **§5 reads like it could have been written without Table 1**. The CSS arguments are made *abstractly* — they would survive unchanged if every number in your table were different. That's the gap to close.

## What to change, section by section

### §5.1 — one sentence change

The paragraph argues that "systems sharing a retrieval substrate behave very differently depending on control policy." That claim needs a number attached to it from your own table. Right now it floats.

**Find this sentence in your §5.1:**

> "In our results, systems that share a retrieval substrate behave very differently depending on the control policy around that substrate."

**Replace with:**

> "Table 1 makes this visible: DSPy ReAct RAG and the hybrid agent run over the same retrieval substrate (PubMedBERT dense, BM25 sparse, RRF fusion, cross-encoder reranker), and the only meaningful difference between them is the control policy wrapped around that substrate — yet the hybrid agent reaches 96.0 on list while DSPy ReAct, even after MIPROv2 optimization, stays at 0.0. The retrieval components are identical; the evidence trajectory is not."

That one sentence change turns §5.1's central claim from gestural to data-anchored.

### §5.2 — needs one number-heavy paragraph inserted

§5.2 has the moral compression argument but never uses Table 1 to prove it. The zero-shot Gemma row is the strongest piece of evidence you have for the moral compression claim and it doesn't appear in §5.2 at all.

**Insert this paragraph after your existing sentence:**

> "A tool that answers 'yes' correctly but cannot retrieve the relevant entities, mechanisms, or exceptions may still encourage automation bias, a known concern for AI-based clinical decision support [1, 14]."

**New paragraph to insert:**

> "Table 1 shows this concretely. Zero-shot Gemma 4 31B — a system that performs no retrieval whatsoever — scores 94.0 on yes/no, matching DSPy RAG + MIPROv2 (94.0) and DSPy ReAct RAG + MIPROv2 (94.0), and exceeding our hybrid agent's 88.0. On the same row, that zero-shot system scores 8.0 on factoid and 0.0 on list. The yes/no column is therefore unable to distinguish a system that has read the literature from a system that has read nothing; it is unable to distinguish a prompt-optimized RAG program from a parametric-memory closure. The binary output is achievable from any of these epistemic states because the channel itself is too narrow to carry the difference. This is the moral compression problem in arithmetic form."

### §5.2 — and add a closing sentence about the agent's lower yes/no score

Your hybrid agent scores *lower* than zero-shot on yes/no (88.0 vs 94.0). That's not a weakness to hide — it's the strongest evidence that your adversarial procedure does what it claims to do. Currently §5.2 doesn't say this.

**Find this sentence:**

> "It does, however, change binary closure from a generation habit into an accountable step."

**Add immediately after it:**

> "The six-point yes/no regression relative to the no-retrieval baseline (94.0 → 88.0) is consistent with this reading: the agent is unwilling to commit to the easy 94.0 when the evidence pool does not warrant it. We interpret this as the cost of refusing to over-confidently close binary questions, paid against a benchmark that rewards confident closure."

### §5.3 — one paragraph rework

§5.3 makes the Elish argument and lists design principles, but the link from your results to the crumple-zone claim is asserted rather than shown. The 40.3-point within-model gap from §4 is the operational measure of the crumple zone and it never gets reused in §5.3.

**Find this sentence in your §5.3:**

> "A clinician shown a polished answer may be formally 'in the loop' while having little visibility into query reformulations, omitted abstracts, or ranking errors."

**Add immediately after it:**

> "Our table puts a number on this gap: 40.3 overall points and 96 list-F1 points on the same Gemma 4 31B backbone separate the agent from its zero-shot version. That gap is, in operational terms, the evidence labor the agentic configuration takes on at run time that the zero-shot configuration silently passes through to whoever reads its output. Crucially, the gap is invisible on yes/no (88.0 vs 94.0) and small on summary (51.0 vs 41.6), which are the regimes where the output gives the user nothing concrete to verify. The crumple zone is widest exactly where the interface gives the clinician the least to push back on."

## What NOT to change

- Keep "Why Agents at All?" as the §5.1 title. It answers the workshop's framing question and you were right to push back on that.
- Keep "The Moral Cost of Yes/No" and "Accountability Without the Moral Crumple Zone" as titles.
- Keep the WHO closing in §5.3 — it lands the design-obligations point you want.
- Don't add new subsections. Four edits inside the existing three subsections is all this needs.

## Total length impact

Adding four insertions and one one-sentence swap. Roughly 14-16 lines added total. You can absorb that by tightening §3.1 — the "We therefore use the term *agentic* in a deliberately bounded sense" paragraph and the next one ("This bounded form of agency is the methodological core...") restate things your intro already said. Compressing those two paragraphs to ~6 lines total saves what these §5 inserts cost.

## Why this version actually delivers the CSS framing

Your current §5 names CSS concepts (moral compression, crumple zones, evidence labor as data). My proposed changes make each concept point to a specific cell of your table:

- **Moral compression** → the 94.0 / 94.0 / 94.0 / 88.0 yes/no column, where systems doing vastly different amounts of evidence work produce indistinguishable binary outputs.
- **Crumple zone** → the 40.3 overall and 96 list-F1 gap that measures the labor invisibly offloaded to the human.
- **Evidence labor as data** → the same-substrate ReAct vs. hybrid agent comparison (0.0 vs 96.0 on list) that shows control policy, not retrieval components, drives the behavior.

Each CSS claim now has a number from *your* table doing the work behind it. That's the difference between a paper that *names* CSS concepts and one that *operationalizes* them from its own data. That's the version Agent4IR reviewers will read as a real CSS contribution, not as a methods paper with a sociotechnical postscript.

Make these four edits, fix the `§??` in §3.2, recompile twice, check page count, and you're done.









xxx

Got it — three specific rewrites from Leslie. Let me apply them surgically. First, locate Edit 1:

Apply Edit 1 — lead with the missing-feedback-loop claim:

Apply Edit 2 — make "First, on Phase B…" a claim not a finding:

Now verify the `§??` cross-ref will resolve, then rebuild:

Label is in place — `§??` will resolve to `§5` after two `pdflatex` passes (this is standard, not a bug). Rebuild:

`§??` now resolves to `§5`, the claim-first restructure of "Third..." is visible. Now check page 2 for the Failure mode 1 rewrite:

All three advisor edits applied and rendered correctly. Repackage:

All three advisor comments addressed. Here are the drop-in LaTeX blocks for your Overleaf, plus the cross-ref fix.

---

### Edit 1 — Leslie's comment on the "class of moves" sentence

Her concern: the sentence reads like an aside; she had to back-trace to figure out what it's claiming. **Fix:** lead with the feedback-loop claim, then enumerate the moves, then add why generator fine-tuning can't recover them. Replace your current "Failure mode 1" paragraph with:

```latex
\textbf{Failure mode 1: list performance plateaus around
$30$--$35$~F1 on Phase~A$^+$ across most of the field.} A single
retrieval pass that misses one of the gold entities cannot
recover it, no matter how good the generator is. Fine-tuning the
generator (BIT.UA) does not help, because the bottleneck is not
in the generation step. Static-pipeline retrieval upgrades
(UniTor's synthetic snippets, BIT.UA's DPRF) help but remain
bounded by what the first-stage query returns. \emph{A static
retrieval pipeline lacks the feedback loop that biomedical
experts rely on when searching the literature by hand.} The
class of moves it cannot make at run time --- decomposing a
compound question into sub-queries, noticing that the evidence
pool is insufficient and re-querying, or backing off from an
over-narrow phrasing --- is precisely the set of moves a
biomedical expert performs when searching manually. Generator
fine-tuning cannot recover these moves either, because the
bottleneck they address is not the model's reasoning over the
snippets it receives; it is which snippets reach the model in
the first place.
```

---

### Edit 2 — Leslie's comment "more a finding than a claim"

She wants the claim ("open-weights agentic RAG with no fine-tuning is competitive") to lead; the 0.94/0.32 numbers become *evidence for the claim*, not the claim itself.

### Edit 3 — Leslie's comment "content is sort of jumbled / put main claim first"

Same fix: lead with "the retrieval-bottleneck interpretation is further supported by…" then the (i) and (ii) findings.

Both are in one block — replace your entire `Our results, reported in §4, support three claims…` paragraph (everything from "Our results" through to the end of the "retrieval-bottleneck interpretation" sentence) with:

```latex
Our results, reported in \S\ref{sec:results}, support three
claims. First, an open-weights agentic RAG architecture with no
domain fine-tuning produces answer-quality competitive with the
upper-middle of the field on the gold-snippet condition: our
best system reaches $0.94$ yes/no accuracy and $0.32$ list~F1
on Phase~B~\cite{nentidis2025bioasq13}. Second, retrieval recall
--- not generation quality --- is the binding constraint on the
end-to-end pipeline: the same generator drops by $0.15$--$0.22$
points across every metric when it must retrieve its own
evidence in Phase~A$^+$ rather than receive gold snippets, and
because the generator is unchanged across the two phases this
gap measures retrieval cost almost in isolation. Third, the
retrieval-bottleneck interpretation is further supported by two
negative experimental findings: (i)~few-shot conditioning on
training-set exemplars produced scores indistinguishable from
the base agent, and (ii)~swapping Gemma~3 27B for Gemma~4 31B
lifted Phase~B factoid and list~F1 by roughly $6$ and $5$
points but did not move Phase~A$^+$ scores at all --- a
stronger generator helps where generation is the limit, not
where retrieval is.
```

---

### Edit 4 — the `§??` you saw rendered

That's not a bug in the LaTeX — the label `\label{sec:discuss}` exists on your `\section{Discussion and Conclusion}` line. `§??` happens whenever LaTeX hasn't propagated cross-references yet. **Fix in Overleaf:** click *Recompile* twice (or click the dropdown next to Recompile and pick "Recompile from scratch"). The verified preview now shows `§5 discusses implications for future BioASQ submissions`.

If it still shows `??` after two compiles, search for `\label{sec:discuss}` in your `main.tex`. It should be on the same line as `\section{Discussion and Conclusion}`, like:

```latex
\section{Discussion and Conclusion}\label{sec:discuss}
```

If your edits to the Discussion section accidentally dropped the label, restore it there.

---

### What changed structurally vs. content

All three edits are **rhetorical** restructures — no factual claims changed, no numbers moved. Same evidence, same conclusions; the words just land in the order Leslie suggested (claim → support, not support → claim). The paper stays 15 pages.








xxx

Fair. Sticking to the moral/discussion content. Here are three targeted swaps — the moral compression paragraph (§5.2), the accountability paragraph (§5.3), and a one-sentence fix in §5.1 for the "automony" typo and clunky tail. Everything else in the paper stays exactly as you have it.

**§5.2 — replace whole subsection.** What this does: defines moral compression more sharply, leads with the killer empirical observation (high yes/no scores from systems with poor evidence completeness, pulled from the table), and replaces the vague "right interface" line with a concrete spec.

```latex
\subsection{The Moral Cost of Yes/No}
Yes/no biomedical questions perform what we earlier called \emph{moral compression}: a heterogeneous evidence base--differing in study quality, population, mechanism, and conditionality--is collapsed into a single token that downstream users may read as a recommendation. The compression is moral, not merely metric, because the token erases the features (uncertainty, exception, contradiction) that would normally prompt a person to seek further evidence.

Our results show why this matters concretely. The systems with the highest yes/no scores are not the systems with the best evidence completeness: optimised DSPy ReAct reaches 94.0 on yes/no but 0.0 on list and 31.0 on factoid, and zero-shot Gemma scores 94.0 on yes/no while finding only 2/26 factoid entities and 0/23 list entities. A system can produce the right binary label without having located the entities, mechanisms, or exceptions that would justify it--exactly the conditions under which automation bias has been shown to harm patients in clinical decision support~\cite{abdelwanis2024,khera2023jama}.

The agent's adversarial yes/no procedure is a partial technical response. Requiring the model to surface evidence for both ``yes'' and ``no'' before deciding does not make the system morally safe, but it converts binary closure from a generation habit into an auditable step: the rejected side and the reasons for rejecting it become part of the record. A trustworthy biomedical QA interface should therefore not display the label alone--it should display the label, the alternative the system considered and rejected, the evidence supporting each side, and any sufficiency flag that fired during the search. In biomedical IR, a correct binary label without this trail is often not enough to be trustworthy.
```

**§5.3 — replace whole subsection.** What this does: names *who* the crumple zone applies to in biomedical QA (currently vague), turns "accountable hand-off" from a slogan into three concrete commitments (provenance, search-path visibility, calibrated abstention), and keeps your WHO and evaluation-reframe moves but tightens them.

```latex
\subsection{Accountability Without the Moral Crumple Zone}
Agentic systems also risk creating what Elish~\cite{elish2019} calls a \emph{moral crumple zone}: humans absorb blame for failures of automated systems they could not realistically understand or control. In biomedical QA the crumple zone is the user--clinician, evidence reviewer, or researcher--who is formally ``in the loop'' but cannot see the reformulations they did not author, the abstracts they did not see ranked, or the sufficiency calls they did not make. If the answer turns out wrong, blame attaches to the person who acted on it, not the system that produced it.

Our design principle is therefore not human replacement but \emph{accountable hand-off}: the agent hands the user the materials a competent reviewer would need to second-guess the answer. Three commitments follow. \emph{Provenance:} every claim in the generated answer is traceable to a retrieved snippet, and every snippet to a PubMed identifier. \emph{Search-path visibility:} the user sees the queries the agent issued, the queries it rejected as insufficient, and the items the list-review pass added on a second sweep. \emph{Calibrated abstention:} when sufficiency checks fail and additional queries do not recover, the system surfaces a refusal-with-reasons rather than producing a confident-sounding answer. These are not UI niceties bolted on after the fact; they are properties of the retrieval-and-reasoning loop itself, which is why we treat them as IR requirements rather than human-factors concerns.

This reframes evaluation. A four-column score table is necessary but insufficient. We should also evaluate whether traces help humans detect errors, whether list omissions cluster around rare diseases or under-studied populations, whether generated queries systematically privilege well-indexed terminology, and whether users over-trust high-confidence binary answers. Agent logs are, in this sense, computational social science data: they reveal how an automated system operationalises relevance, uncertainty, and biomedical authority. WHO guidance~\cite{who2021} argues that AI for health must centre ethics, transparency, responsibility, and inclusiveness; for agentic biomedical QA those values become operational obligations on the system--bounded tool use, evidence provenance, calibrated abstention, answer-type safeguards--and a corresponding obligation on evaluation: reviewing traces is part of the evaluation, not an afterthought.
```

**§5.1 — one-sentence swap.** Find this sentence (it has a typo and a clunky tail):

> The hybrid agent's gains therefore argue against more automony, but for governed agency, and more explicit procedural commitments about how biomedical evidence must be searched, checked, and represented.

Replace with:

```latex
The hybrid agent's gains therefore argue not for more autonomy but for \emph{governed agency}: explicit procedural commitments about how biomedical evidence must be searched, checked, and represented.
```

That's it — three drop-ins, everything else untouched. The moral section now grounds itself in your actual table numbers (which is what makes the argument land for reviewers) instead of asserting the ethics in the abstract.
