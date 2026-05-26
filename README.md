\section{Limitations and Conclusion}

The agentic system and the DSPy baselines run on different
backbones---Gemma 4 31B for the agent, gpt-oss-120b for the DSPy
programs---so Table~1 is not a model-controlled head-to-head. We
read it as a comparison of \emph{control-policy regimes} rather
than of language models: the within-model anchor for the agentic
loop is the Zero-shot Gemma 4 31B row, which isolates the
contribution of the control policy from the model. Other
limitations apply. The corpus is title/abstract-only, excluding
full-text evidence. DSPy optimization used question-answer pairs
from \texttt{rag-mini-bioasq}, not gold snippets, which may
disadvantage evidence-grounded behavior. Logged agent traces are
not automatically usable by clinicians or researchers; future work
must test whether traces actually improve human judgment.

We conclude that biomedical QA agents should be framed as
evidence mediators, not medical authorities. The technical
finding is that a domain-governed agentic loop improves
recall-sensitive and exact-answer regimes over static and generic
agentic baselines. The computational social science finding is
that the value of agents lies in making evidence labor visible
and governable. The central question is not only whether the
answer is correct, but whether the path to the answer can be
inspected, challenged, and responsibly handed back to human
experts.
