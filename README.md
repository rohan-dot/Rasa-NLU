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
