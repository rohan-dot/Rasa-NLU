# Task 19 — generated harnesses never reach the project's code: fix entry-point selection

## Goal
Latest run, every generated harness was rejected by the gate with the same reason:
`harness-gen GenFuzzer0: reach=fail (coverage_report) → rejected: coverage report lists no covered method in mil.disa.common.util.agentmanager; probe cov 15 over 192577 execs — does not reach target`
(GenFuzzer1: cov 6 over 5,190,559 execs; SinkFuzzer0: cov 6 over 5,273,257; SinkFuzzer1: cov 6 over 5,287,777; the
regenerated RegenFuzzer*_0 were rejected by the static check — `broad catch (Exception) around the target call`; the
RegenFuzzer*_1 again did not reach the target). The gate is correct; the generator is choosing the wrong entry points.
The provided harness's own comment says what they are: "discver's auto-picked entry points (LDAP connectors / getters,
which have flat coverage and need a network server)". The call graph confirms it — the most-called functions are
`getBaseSuffix`, `getHost`, `performLDAPModify` (LDAP plumbing) — while the one function with real parsing logic that
takes a String is `getCn` / `getCnFromDn` (X509Utils.java:76/81). Make the generator pick entry points the way a human
did: public methods in the project's own packages, taking fuzzable arguments (String / byte[] / InputStream / a parsed
object built from them), with branching logic, and NOT requiring a network connection, a live server, a file system
state or credentials. Rank them, generate for the top ones, and let the existing reach probe be the acceptance test.

## Study first
- results: `grep -nE "harness-gen|does not reach|rejected by static|Most-called|callers" latest/orchestrator.log | head -40`.
- repo: MAP.md "harness generation" — where candidate targets come from (`java_harness_agent.py` harness_targets around
  :530, `harness_agent.py`, `bug_swarm.py`:191/201), the call-graph data available at selection time (PHASE 0 output:
  functions, callers/callees, file paths), and the regeneration feedback from task 17.
- quartetfuzz (if present in ROOTS): `harness/checker/lg_agent.py` (logic-group discovery + danger ranking) and
  `README.md` P3/P4 — "entry functions should exercise the public API" and "reach a security-relevant surface".
  Read for the mechanism; write discver's own code (that repo states no license).
(skip any reference root that is not listed in ROOTS)

## Change (src/)
- `rank_entry_points(call_graph, sources) -> list[Candidate]`: candidates are public methods in the project's own
  packages (from discovery / the call graph, never a hardcoded package name). Score = + parameters that are directly
  fuzzable (String, byte[], char[], InputStream, Reader; or a type constructible from them) + branching density in the
  method body (if/switch/loops/indexOf/substring/split/parse) + static-scan hits at or below the method. Score = − for
  methods whose body or callees touch network / sockets / LDAP / JNDI / JDBC / HTTP clients, read credentials or
  config files, or require an initialised server object; − for trivial getters/setters (≤ 3 statements, no branching).
  Exclude anything with a negative score. Log `harness-gen: <n> candidates ranked; top-k=[<Class.method>(params) score=<s>, …]`.
- Generation targets the top-k ranked entry points, one harness per entry point; the generator prompt receives the
  method signature, its source, how it is called by existing callers (from the call graph), and the task-17 feedback
  on regeneration. The static check (broad catch) and the reach probe stay the acceptance gate.
- When every candidate is rejected or excluded, log `harness-gen: no admissible entry point — <reason summary>` instead
  of generating against connectors again.
- Safety: wrapped at the call site; `[harness_entry_points] failed: <error>` with traceback → fall back to today's
  selection.

## Tests to add (tests/test_harness_entry_points.py) — fakes only
- a fake call graph + sources with `getCnFromDn(String)` (branching, substring/indexOf), `getHost()` (trivial getter),
  `performLDAPModify(...)` (calls an LDAP connection) → ranking puts `getCnFromDn` first, excludes the other two with
  the reasons in the log line.
- a method taking `byte[]` with a parse loop ranks above a method taking a custom object with no fuzzable constructor.
- a static-scan hit on a method raises its score above an otherwise equal method.
- all candidates negative → the "no admissible entry point" line, no generation call.
- the generator is invoked once per top-k entry point with the signature and caller context in the prompt.

## Acceptance
```
python -m pytest -q tests/test_harness_entry_points.py
```

## Do not touch
The provided harness; the gate (task 16) and static check (task 17) logic; sidecar; crash_dedup._get_crash_signature; anything outside src/ and tests/.
