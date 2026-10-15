# Task 27 — a model refusal is a refusal: label it, retry once with the task's purpose stated plainly, then stop

## Goal
Latest run: `:1361 "outcome=timeout reply_chars=0 prompt_tokens=5463 completion_tokens=8 finish_reason=content_filter"`,
`:1360 "LLM returned empty content (finish_reason=content_filter, 2.7s)"`, and the finding ends as
"LLM timed out 3x (timeout_s=300/300/300 elapsed_s=3/2/2)". These were not timeouts (elapsed 2–3 s); the model
declined to answer. The mislabel comes from `src/patch_generator.py:402` (`outcome = "ok" if reply else "timeout"`)
and `:433 _reason_timeout`. The seed path is hit the same way: `:57 "LLM returned empty content
(finish_reason=content_filter, 2.0s)"` → `:64 "[pb_seeds] LLM never produced a usable spec — falling back"`.
Evidence of which prompts are affected: the index-out-of-bounds finding was answered and edited; the findings whose
prompts lead with injection/attack framing (ldap-injection, CWE-90, CWE-173) were refused. The purpose of every one
of these prompts is a fix for the project's own code or a test input for its own parser — say so.

## Study first
- results: `grep -nE "content_filter|empty content|timed out|outcome=timeout|refus" latest/orchestrator.log | head -40`;
  quote three refused calls and the one answered call (:1314 apply_edit ok) with their finding names.
- repo: `src/llm_client.py:245 chat()` (where finish_reason is available), `src/patch_generator.py:402` and `:433`,
  the prompt builders for the patch brief (MAP.md "patch loop") and for seeds (`pb_seeds.py` / seed_generator.py:501);
  read how each prompt opens and what it calls the task.

## Change (src/)
- `llm_client.chat()` returns the finish reason alongside the text; an empty reply with finish_reason in
  {content_filter, refusal} is outcome `refused`, never `timeout`. Log `[llm-call] … outcome=refused finish_reason=<r>`.
- Patch loop: on `refused`, retry ONCE with the same brief prefixed by an accurate purpose statement — that the request
  is a bug fix in the project's own source for a reproduced crash (attach the crash trace and the crash site) or a
  test input for the project's own parser — in plain engineering words. No other rewording. If the retry is also
  refused: `status=skipped, skip_cause=llm_refused, attempts=<n>, reason="model declined the patch request twice
  (finish_reason=<r>)"`, zero further iterations, listed in its own report section. Never counted as a timeout and
  never retried three times.
- Seed path: the same single purposeful retry; on a second refusal log `[pb_seeds] model declined the seed request
  (finish_reason=<r>) — using mined + format-aware seeds only` and continue (the other seed sources are unaffected).
- Prompt builders: the opening line of the patch brief and of the seed prompt states the purpose accurately (fix /
  test input for the project's own code, with the evidence). Nothing about this changes the brief's technical content.

## Tests to add (tests/test_llm_refusal_handling.py) — fake LLM client, no network
- a reply with empty content and finish_reason=content_filter → outcome `refused`, the log line, no timeout label.
- refused then answered → the retry carries the purpose prefix and the loop proceeds normally.
- refused twice → `skip_cause=llm_refused`, attempts=2, no third call, own report section.
- a real timeout (exception after timeout_s) still labels `timeout`.
- seed path refused twice → the fallback line is present and the mined/format seeds are still written.

## Acceptance
```
python -m pytest -q tests/test_llm_refusal_handling.py
```

## Do not touch
Sidecar invocations; crash_dedup._get_crash_signature; anything outside src/ and tests/.
