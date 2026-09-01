This is a merged discver codebase — my earlier work plus a coworker's ("patchengineer") patch-generation attempt that is NOT working. READ-ONLY investigation, no edits.

FIRST read PATCHER_TODO.md — it's the coworker's own notes on the patching work; treat it as the primary clue for what he tried and what's broken. Then read AGENT.md, CHANGELOG.md, and README (1).md / README.md.

Then map src/ and answer:
1. Architecture — phases/orchestration, how crashes/PoVs are represented, module layout.
2. The build + PoV-replay mechanism — how does this repo compile the target and re-run a single crashing PoV input against it? Quote the actual functions/commands. This is what patch validation must hook into.
3. The patch-generation code — read it end to end. What does it try to do, what does its validation depend on, and WHY does it fail? Cross-reference against PATCHER_TODO.md. Name the specific failure point with evidence.

THEN propose (do NOT implement) how patching should work: reuse the build/replay functions you found; the oracle is apply-patch → rebuild → RE-RUN THE SPECIFIC PoV (not a fresh fuzz run — that's nondeterministic) → crash must be gone AND existing tests pass. Flag anything that BLOCKS this, especially if there's no programmatic build-and-replay path — that's the likely real reason patching fails.

Write findings + proposed design to PATCH_SURVEY.md. Do not edit AGENT.md/CHANGELOG/READMEs/PATCHER_TODO.md. Commit to agent/patch-survey.
