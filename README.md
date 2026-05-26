\subsection{Accountability Beyond the Moral Crumple Zone}

Agentic systems also risk creating what Elish~\cite{elish2019moral}
calls a moral crumple zone: humans absorb blame for failures of
automated systems that they could not realistically understand or
control. A clinician shown a polished answer may be formally ``in
the loop'' while having little visibility into query
reformulations, omitted abstracts, or ranking errors. Our table
puts a number on this gap. Moving from zero-shot Gemma 4 31B to
the hybrid agent on the same backbone, the overall score rises by
40.3 points, list F1 rises by 96 points (from 0.0 to 96.0), and
factoid accuracy rises by 61 points (from 8.0 to 69.0). That is
the evidence labor the agentic configuration takes on at run time
that the zero-shot configuration silently passes through to
whoever reads its output. Crucially, the same comparison shows
almost no movement on yes/no (the agent loses 6 points, 94.0 to
88.0) and only modest movement on summary (the agent gains 9
points, 41.6 to 51.0). The regimes where the output gives the
reader nothing concrete to verify are exactly the regimes where
the gap between doing the evidence work and not doing it is
hardest to see. This asymmetry reshapes what accountability has
to mean in practice. The question is not whether to keep the
human ``in the loop,'' but how to hand evidence work back in a
form they can audit. Our design principle is therefore
accountable hand-off, not human replacement. The agent must
provide provenance for retrieved documents and snippets, record
its search path, expose sufficiency decisions, and abstain or
flag uncertainty when evidence is weak. These are IR requirements
for responsible biomedical agents, not user-interface features.
This also reframes evaluation. A four-column score table is
necessary but insufficient: we should evaluate whether traces
help humans detect errors, whether list omissions cluster around
rare diseases or under-studied populations, whether generated
queries systematically privilege well-indexed terminology, and
whether users over-trust high-confidence binary answers. Agent
logs become computational social science data: they reveal how
an automated system operationalizes relevance, uncertainty, and
biomedical authority. The WHO guidance on AI for
health~\cite{who2021ethics} calls for transparency,
responsibility, and inclusiveness; in agentic biomedical QA,
those values become concrete IR obligations: bounded tool use,
evidence provenance, calibrated abstention, answer-type-specific
safeguards, and meaningful human review.
