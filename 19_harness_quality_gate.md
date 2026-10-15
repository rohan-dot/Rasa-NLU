# Task 19 — a quality gate for generated harnesses, ported from QuartetFuzz's method (reference root: quartetfuzz)

## Goal
discver's generated harnesses are unreliable: GenFuzzer0 ran 27M execs at coverage 10→10 (never reached the target),
and later findings came from harnesses aimed at third-party code. QuartetFuzz (refs/quartetfuzz) generates C/C++
harnesses, but its method transfers to Java: (1) choose entry points by ranking groups of related APIs by a danger
score; (2) write an API-protocol report (construction → parameters → call order → cleanup, each claim backed by source
lines) BEFORE generating; (3) a bounded compile-fix loop followed by a self-review; (4) an inline gate before a harness
gets any budget — a REACH probe (one valid input must demonstrably reach the target method) and a RUN probe (one
adversarial input must not crash the harness's own code); (5) a short empty-corpus coverage run as the harness's score.
Port the mechanism into discver's generator in Java/Jazzer terms. Do NOT copy code: that repo states no license. Read
it for the mechanism, then write discver's own implementation.

## Study first
- quartetfuzz: `harness/checker/pipeline.py` (stage order, the gate), `harness/checker/flow.py` (build-fix loop,
  self-review, the two probes, submit_harness / get_coverage), `harness/checker/lg_agent.py` (logic groups + danger
  ranking), `harness/checker/p2_agent.py` (protocol report), `harness/checker/tools/dynamic_analysis.py` and
  `harness/checker/dynamic.py` (how the probes are executed), `README.md` (principles P1–P4). Note file:line for each.
- repo: MAP.md "harness generation" (java_harness_agent.py, harness_agent.py :266 / :309–317, harness_templates.py,
  bug_swarm.py :191 / :201), "harness selection", and the task-16/17 code if present (harness_reach.py, the static check).
(skip any reference root that is not listed in ROOTS)

## Change (src/) — Java equivalents, each its own function, each logged
- Entry-point ranking: group candidate target methods by class / API family; rank groups by a static danger score
  (static-scan hits in the target such as CodeQL/CWE findings, parsing and index/array handling, string manipulation);
  take top-k. Log `harness-gen: <n> candidate groups, top-k=[<names>]`.
- Protocol report: before generating a harness for a group, ask the LLM for a short report (construction, parameters,
  call order, cleanup) where every claim cites file:line; reject a report without evidence; include the accepted report in
  the generator prompt. Log `harness-gen <name>: protocol report <k> claims`.
- Compile-fix loop: feed javac errors back up to N times; then a self-review against a checklist (fuzz input actually
  reaches the API; no broad catch around the target call; only declared exceptions caught; no network or server needed;
  the entry point is public API, not an internal helper).
- Gate before budget: REACH probe — run the harness once on a crafted valid input with `--coverage_report` (or the
  task-16 reach helper) and require the target method in the report; RUN probe — run once on an agent-written adversarial
  input and require that any exception's top frames are in the target code, not in the harness class. Log
  `harness-gen <name>: reach=<ok|fail> run=<ok|fail> → <admitted|rejected: reason>`.
- Score: a short empty-corpus run (e.g. 120 s) → coverage delta is the harness score; feed task 17's budget-by-reach.
- Safety: one call site, wrapped — any exception logs `[harness_gate] failed: <error>` and generation proceeds as today.

## Tests to add (tests/test_harness_quality_gate.py) — fakes only
- ranking: three fake groups with danger scores 5 / 2 / 9 → top-2 = [9, 5]; log line present.
- protocol report: a fake LLM reply with 3 claims each citing file:line is accepted; one without evidence is rejected.
- compile-fix loop: fake javac errors twice then success → the loop ends after 3 tries and logs each attempt.
- gate: a fake coverage report containing the target method + a run-probe exception whose top frame is in the target →
  admitted; a report without the target → rejected with the reason; an exception whose top frame is the harness class → rejected.
- the provided harness (name from MAP.md "harness selection") never goes through the gate.

## Acceptance
```
python -m pytest -q tests/test_harness_quality_gate.py
```

## Do not touch
The provided harness; sidecar; crash_dedup._get_crash_signature; anything outside src/ and tests/. No new pip dependencies.
