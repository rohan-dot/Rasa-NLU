# Task 26 — decide whether a crash site is in the project BEFORE spending patch iterations on it

## Goal
Several findings (GenFuzzer1-*, some static-*) burned their whole iteration budget with the agent looping on
`list_files ':**/LDAPConnection.java' -> (no files)`: the crash site is in a third-party library (unboundid LDAP) that is
not in the target's source tree, so no patch is possible there. Meanwhile the one in-scope crash (X509Utils.getCnFromDn)
was mislabelled "third-party dependency" by the diff bug (task 25). Fix the classification at the source: before the
first patch iteration, resolve the crash frames against the project's source tree; a finding whose crash site is not in
the project is recorded `out_of_scope_third_party` with the frame and the library name, gets ZERO iterations, and is
listed separately in the report. A finding whose site IS in the project is never given that label by anything else.

## Study first
- repo: MAP.md "patch loop" (where a finding enters run_patch_agent, the brief builder, how `static-*` findings are
  routed) and "sidecar/oracle" (read-only). `grep -nE "third.party|BUG_IN_DEPENDENCY|out_of_scope|unboundid|frames?\b|stack" src/patch_generator.py src/main.py src/crash_analyzer.py`.
- results: `grep -nE "LDAPConnection|unboundid|list_files|no files|third-party" latest/orchestrator.log prev/orchestrator.log | head -40`
  and the matching patch_index entries (attempts, test_calls, reason).
(skip any reference root that is not listed in ROOTS)

## Change (src/)
- `classify_crash_scope(frames, project_src_root) -> (in_project: bool, site: str, library: str|None)`: walk the stack
  from the top; the first frame whose class maps to a .java file under the project source tree is the patch site; if no
  frame maps to the project (only JDK / third-party / harness frames), the finding is out of scope and `library` is the
  package prefix of the top non-JDK frame.
- Wire it before the first iteration: out-of-scope → `status=skipped, skip_cause=out_of_scope_third_party, attempts=0,
  test_calls=0, reason="crash site <frame> is in <library>, not in the project source"`; log one line
  `[patch-scope] <finding>: out of scope (<library>) — 0 iterations`. In-scope → the brief names the resolved site
  (file:line) so the agent does not go hunting for files that do not exist.
- The `bug_in_dependency` label from the builder path (task 25) must never be applied to a finding this check marked in-project.

## Tests to add (tests/test_patch_scope_check.py) — fakes only
- frames [harness, com.unboundid.ldap.sdk.LDAPConnection.connect, …] with a fake project tree lacking that class →
  out of scope, library `com.unboundid.ldap.sdk`, 0 iterations, the log line and the patch_index fields as above.
- frames [harness, mil.disa.common.util.agentmanager.X509Utils.getCnFromDn:81, …] with that file in the fake tree →
  in project, site = X509Utils.java:81, patching proceeds, and the brief contains that file:line.
- a mixed stack (third-party frame on top, project frame below) → in project, site = the project frame.
- the report lists out-of-scope findings in their own section, not among failed patches.

## Acceptance
```
python -m pytest -q tests/test_patch_scope_check.py
```

## Do not touch
Sidecar invocations and crash_dedup._get_crash_signature; anything outside src/ and tests/.
