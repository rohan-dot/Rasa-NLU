export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1
export LITELLM_API_KEY=sk-...
export AGENT_MODEL=claude-opus-4-8
export AGENT_TLS_VERIFY=false
python microagent.py --repo $(pwd) --verify
python microagent.py --repo $(pwd)

This is an unfamiliar codebase — a coworker's version of the discver cyber reasoning system, with his own recent commits including a patch-generation attempt that is NOT working. Do a READ-ONLY investigation. Make NO code edits. Your job is to understand the whole system from the code and diagnose why patching fails.

Start by reading its own docs: AGENT.md, CHANGELOG, and both README files — take them as the author's intent. Then map src/:
1. Overall architecture — the phases/orchestration (likely main.py), how findings/crashes/PoVs are represented, and the module layout by area (LLM client, fuzzing, static analysis, crash handling, reporting).
2. The build + PoV-replay mechanism — how does this repo compile the target, and how does it re-run a single crashing input (PoV) against the built target? This is the critical piece for patching. Find exactly where and how it's invoked (fuzzer.py, run_health.py, harness code, or a builder class). Quote the actual functions/commands.
3. The existing patch-generation code — read it end to end. What is it trying to do, what does its validation step depend on, and WHY does it fail? Consider: is a build command / test command configured? Does it call a real rebuild-and-replay, or does it stub/skip? Does it hit LLM timeouts? Missing PoV wiring? Name the specific failure point with evidence from the code.

THEN, based only on what you found, propose (do NOT implement) how patch generation SHOULD work here: the design, which existing build/replay functions it must reuse, and the exact validation oracle — apply patch, rebuild, RE-RUN THE SPECIFIC PoV (a crash NOT firing on a fresh fuzz run is nondeterministic and proves nothing; replaying the exact PoV input is the real test), require the crash gone AND existing tests to pass. Flag anything that BLOCKS this — e.g. if there's no programmatic build-and-replay path, say so plainly, because that's likely the real reason his patching fails.

Write your findings and proposed design to a new file PATCH_SURVEY.md (do not touch his AGENT.md/CHANGELOG/READMEs). Report the module map, the build/replay location, the diagnosed failure cause, and the proposed design. Commit to this branch.
