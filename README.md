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
