You have ~1 hour unattended. Work carefully and verify as you go; I will not be available to answer questions, so make reasonable decisions yourself and document them in CHANGELOG. If you hit an ambiguity, pick the safer option and note it.

EVIDENCE: run-evidence/ contains files from a real container run, copied FLAT (no subfolders). Start by listing it and reading everything relevant: orchestrator.log (the main log — ground truth for what happened), patch_index.json (per-crash patch outcomes), health.json (per-harness fuzzing health), summary.json (the deduped crash records the patcher consumed — check whether stack_trace/location/root_cause were actually populated), crash-* files (PoV inputs from all harnesses) and their .txt ASAN traces, and any bug-*.json (analyzer output, including replies that failed JSON parsing). Read all of it before diagnosing.

Key facts from the log: the sidecar build oracle was detected correctly ("build oracle path: SIDECAR (libCRS at /usr/local/bin/libCRS exposes apply-patch-build)"). Yet across 30 patch-loop iterations on two crashes there was only 1 test_patch call and zero validated patches. The PoV-backed crash X509Fuzzer-9bbf8868fd92 called test_patch 0 times in 15 iterations. The log contains NO per-turn record of what the agent did — no tool calls, no edits, no test results, and the single sidecar build produced no logged output. The analyzer logged "LLM returned invalid JSON — using raw report" four times, so the model is emitting non-parseable JSON.

YOUR JOB, in order:

(1) DIAGNOSE from the evidence and the code. Trace run_patch_agent and the patch loop's action parser in src/patch_generator.py. Determine why iterations pass with no tool calls: most likely the model's replies fail to parse into actions and are silently dropped, burning the iteration. Also check summary.json — if stack_trace/location were empty on the record, the patcher got a thin task despite the context fix; that is a separate bug. Confirm or refute each from the code paths and evidence, and state findings in CHANGELOG with evidence.

(2) PER-TURN OBSERVABILITY. Add INFO logging on the discver.reliability logger, one line per patch-loop iteration: the tool chosen (or "unparseable"/"no_action" with the first 300 chars of the raw reply), its key argument, and a short result. For test_patch log the full TestResult (build_ok, crash_gone, tests_pass) plus the first 500 chars of build/test output on failure. Log every sidecar invocation: exact command, return code, stderr tail. An iteration must never pass silently again.

(3) ACTION-PARSE ROBUSTNESS. When a reply fails to parse, do not drop it: log it, and feed the exact parse error back to the model as the observation with a one-line reminder of the required JSON action shape. Reuse the validate->repair machinery already in agent_tools.py rather than reimplementing. Ensure the patch loop's JSON extraction tolerates code fences, prose around the JSON, and the model emitting a diff instead of an action — handle those.

(4) FAIL FAST + NUDGE. If test_patch returns build_ok=False, log the build error and end that crash with cause build_failed — do not grind to the cap. If the agent has edited but not tested within 5 iterations, inject an observation telling it to call test_patch. If it has made no edit by the budget midpoint, log that explicitly.

(5) SELF-VERIFY. Write an offline simulation test that drives run_patch_agent with a stubbed LLM emitting (a) a well-formed action, (b) a code-fenced action, (c) prose with embedded JSON, (d) garbage — and assert each is handled: logged, repaired where possible, never silently dropped. Run py_compile on all touched files and the FULL test suite. If anything fails, fix it and re-run; do not stop at a red suite. Iterate until green.

(6) COMMIT when green, with a CHANGELOG entry containing: the diagnosis with evidence, exactly what you changed, and what the next container run should now show per turn. If you cannot make the suite green, commit work-in-progress on a separate branch and write clearly in CHANGELOG what is unresolved.

Do NOT touch crash_dedup._get_crash_signature or the all-PoV loop in validate_attempt.
