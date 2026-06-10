# Agentic CAITT — Speaker Script (Slides 3–14)

*Continuous spoken script with timing. First person, conversational register — read it as you'd say it, not word-for-word. Stage cues in [brackets]. Running clock assumes ~10 min across these slides, leaving room for your other slides + Q&A in a 15–20 min slot.*

**Through-line to hold across the whole talk:** *trust and traceability.* The Intro raises it, Motivation sharpens it into a gap, and everything after closes it.

---

## SLIDE 3 — Introduction
*Target: 1:15 · Cumulative: 1:15*

Let me start by setting the scene, because the problem we're solving lives in cyber incident response. And the key thing to understand is that incident response is fundamentally a coordination and reasoning task — not just a technical one. Teams aren't only running tools. They're sharing indicators and observations, forming and revising hypotheses, assigning actions, and converging on decisions. And almost all of that happens in real-time chat.

[advance through sub-bullets]

The challenge is *where* that reasoning lives. It's buried in long, unstructured conversations. Context gets fragmented across messages, key assumptions stay implicit, and the evidence is only weakly linked to the conclusions people draw. We run our method on Mattermost data, which is exactly this kind of fast-moving, sprawling chat.

So the result is that after the fact — or even mid-incident — it becomes genuinely difficult to reconstruct what was known, why it was believed, and how decisions were actually made.

**Transition:** That difficulty isn't just inconvenient. In our setting it's a real failure mode, and that's what the next slide is about.

---

## SLIDE 5 — Motivation
*Target: 1:15 · Cumulative: 2:30*

So chat-based workflows are relevant and necessary — but they introduce a critical failure mode: loss of context and traceability over time. And that loss leads to three concrete problems: incomplete or inconsistent understanding, competing interpretations of the same evidence, and increased reliance on speculation.

[advance to adversarial bullet]

Now in adversarial environments, this gets worse. Information is often noisy, delayed, or outright misleading. Analysts have to act under high uncertainty and time pressure. And without clear grounding in evidence, decision-making becomes less reliable and harder to trust. That word — *adversarial* — is what separates this from a generic chatbot problem.

[advance to Key Gap]

Which brings us to the key gap. Existing approaches can't reconstruct reasoning from conversational data, can't tie answers directly to supporting evidence, and can't evaluate answer quality in a consistent, repeatable way.

**Transition:** Those three gaps map directly onto the three pieces of our system — so let me show you the system.

---

## SLIDE 9 — Methodology (Overview)
*Target: 1:30 · Cumulative: 4:00*

Here's the whole system at a glance. The goal is reliable, evidence-grounded reasoning over unstructured conversational data, and it has three components — each one answering a gap from the previous slide.

Agentic Retrieval identifies the most relevant evidence. Grounded Answer Generation produces answers with explicit citations. And Judge-Based Evaluation systematically assesses answer quality.

[point to outcome]

The outcome is the part I want you to remember: answers become traceable to their source evidence, and the reasoning becomes transparent and verifiable. That's what closes the "hard to trust, hard to reconstruct" problem.

[walk the pipeline left to right]

And the pipeline runs like this: we collect chat logs, embed and store them in a FAISS vector store, run agentic retrieval, generate the answer, and then pass it to judge evaluation. The next few slides drill into each of these boxes.

**Transition:** Let's start at the front of the pipeline — ingestion and embedding.

---

## SLIDE 10 — Methodology: Ingestion and Embedding
*Target: 1:15 · Cumulative: 5:15*

This first stage turns raw, messy chat into something the agent can actually retrieve against. We collect from heterogeneous sources — chat logs, documents, whatever the operational context produces.

We then chunk that content into messages or small windows, and crucially we attach metadata to each chunk — so source and context travel with the content. This is semantic chunking, not naive fixed-size splitting.

[point to FAISS store]

Each snippet is embedded and indexed in FAISS, which gives us fast nearest-neighbor search at query time and runs locally with no external dependency. And the line I'd emphasize: we keep message IDs and permalinks for every chunk. That's what makes every retrieved piece of evidence traceable back to its source — and that's what makes citation possible downstream.

**Transition:** So that's the index. Now, what happens every time a user actually asks a question?

---

## SLIDE 11 — Methodology: Agentic Retrieval (Workflow)
*Target: 1:30 · Cumulative: 6:45*

Retrieval here isn't a single shot — it's an agentic pipeline that runs per query. [point to diagram, top]

First, we expand the query into multiple intent-preserving variants. A single phrasing is brittle, so we widen recall without drifting off-topic. Then we do hybrid retrieval — dense embeddings catch semantic matches, while lexical search catches the exact terms, names, and IDs that embeddings sometimes blur. Together they cover more than either alone.

[point to re-rank and top evidence]

Retrieval casts a wide net, so we re-rank the candidates with a higher-precision relevance model to push the best evidence to the top. And finally we select a compact evidence set — we don't dump everything into the answer step, which keeps the context clean and the citations tight.

In one line: expand for recall, re-rank for precision.

**Transition:** But how does the system know the evidence is actually good enough to answer? That's the next piece.

---

## SLIDE 12 — Methodology: Agentic Retrieval (Supervisor Loop)
*Target: 1:30 · Cumulative: 8:15*

This is the safety valve. Before answering, a supervisor checks whether the retrieved evidence is sufficient — and routes to one of three actions: answer, rewrite, or decompose.

[walk the flowchart]

So the retrieved evidence hits the "enough coverage?" decision. If yes, we generate the answer. If no, we either rewrite the query or decompose it into sub-questions, form a new query, and retrieve again — and we keep looping until support is strong.

[deliver this as your key line]

The point is that the supervisor stops the system from guessing. When recall or coverage is weak, it loops instead of producing a premature answer. That makes uncertainty transparent rather than hidden, which reduces hallucinations and yields more reliable, auditable outputs.

**Transition:** [If using slide 13, bridge here — e.g. "Once we have a grounded answer, the question becomes: how do we know it's any good?"] That's where evaluation comes in.

---

## SLIDE 14 — LLM-as-a-Judge (Tree of Thought)
*Target: 1:45 · Cumulative: 10:00*

For evaluation, we don't take a single score from a single model. We use three independent LLM judges, and each one reasons using Tree-of-Thought.

Tree-of-Thought matters for three reasons. It encourages multi-path reasoning — each judge explores multiple perspectives before committing, instead of snapping to a first impression. It breaks down a complex decision into structured, iterative steps — think, explore, evaluate, reflect. And it enhances critical analysis, giving a well-rounded, stepwise assessment rather than a shallow linear verdict.

[walk the visual, left to right]

Each judge receives the user query, the retrieved snippets, and the generated answer. It scores four metrics on a one-to-seven scale — directness, evidence consistency, coverage, and clarity. Those scores then aggregate across all four metrics and all three judges into a single, multi-dimensional quality score.

[land the takeaway]

So what this framework really does is turn a subjective question — "is this answer good?" — into a structured, evidence-based evaluation. And using three judges instead of one reduces single-model bias. It mimics human decision-making, which makes the evaluation more nuanced and reliable.

**Close / hand-off:** And that completes the loop — from messy chat, to grounded and cited answers, to a repeatable measure of how good those answers are. [transition to results / next section]

---

## Anticipated Q&A — one-liners to keep loaded

- **Why three judges, not one?** Variance reduction and bias control — one model's quirks don't decide the score.
- **What model backs the judges vs. the generator?** [fill in — and note whether judge ≠ generator to avoid self-grading bias.]
- **Did you validate the judge against humans?** [fill in — correlation with human ratings is the question they'll ask.]
- **How did you pick chunk/window size?** Trade-off between retrieval precision and keeping enough context per snippet.
- **Which embedding model, and why?** [fill in.]
- **Cost/latency of the supervisor loop?** Bounded by a max-iteration cap so it can't loop forever; worst case it answers with a flagged low-coverage note.
