# Task 29 — provided harness sources the project's build did not build: discver compiles and runs them itself

## Goal
Latest run: `harness-discovery: scanned /opt/target/build: 18 entries, 1 candidates, 1 provided` →
`harnesses: provided=[X509Fuzzer] generated=[]`. The project's fuzz directory holds TWO harness sources
(`X509Fuzzer.java`, `GetPrettySignatureFuzzer.java`; both call `fuzzerTestOneInput(FuzzedDataProvider)`), and the
project's build script compiled only one, so the other was invisible to discver for every run so far. A CRS on an
unknown codebase must not depend on the project having built every harness: find harness SOURCES in the project
tree, compile the ones that have no wrapper in the build output using the same javac + wrapper path discver already
uses for its generated harnesses, mark them `provided (compiled by discver)`, and run them as primary harnesses
(probe → deep phase, their own seeds). Startup also prints `Target language: java (custom-harness compile: False)` —
find what that setting governs and make sure it does not block this; no operator flag is added.

## Study first
- repo: the discovery module from task 18 (`harness_discovery.py`), the generated-harness compile path — the javac
  invocation, how the classpath is derived from the jars in the build dir, the wrapper template and where it is written
  (MAP.md "harness generation": java_harness_agent.py / harness_agent.py :309–317 / bug_swarm.py), and the
  `custom-harness compile` setting (`grep -rn "custom-harness\|custom_harness" src/`).
- results: `grep -nE "harness-discovery|custom-harness|javac|compile" latest/orchestrator.log | head -30`;
  `list_files` on `latest/generated_harnesses/` — what the existing compile path produces for a generated harness
  (source, class/jar, wrapper, compile log) is exactly what a compiled-by-discver provided harness must look like.
- The project tree: the fuzz sources live under the project checkout (here `.../agentmanager-java8-master/fuzz/`);
  find them generically — any `*.java` under the project root containing `fuzzerTestOneInput` — never by path or name.

## Change (src/)
- `find_harness_sources(project_root) -> list[Path]`: `*.java` files containing `fuzzerTestOneInput`, excluding
  discver's own generated-harness directories.
- For each source whose class has no wrapper in the build output: compile with the existing javac path (classpath =
  every jar in the build dir plus the compiled class dir), write the wrapper with the existing template, and register
  it as provided. Log at startup:
  `harness-discovery: source harnesses found: [<names>]; built by project: [<names>]; compiled by discver: [<names>]`
  and the usual `harnesses: provided=[…] generated=[]` now listing all of them.
- A compile failure logs `harness-discovery: <name>: compile failed: <javac stderr tail>` and skips that harness —
  never aborts the run. A source whose class the project already built is not recompiled.
- Compiled-by-discver harnesses are primary (never gated like generated ones) and get their own seed set keyed by name.
- Safety: wrapped at its one call site; `[harness_sources] failed: <error>` with traceback → discovery proceeds as today.

## Tests to add (tests/test_harness_sources_compile.py) — fakes only (fake javac, fake wrapper writer)
- a fake project tree with two harness sources and a build dir containing a wrapper for one → the other is compiled
  (fake javac called once with the jars on the classpath), both are listed provided, the startup lines are correct.
- fake javac failure → the failure line with the stderr tail; the other harness still runs.
- a source already built → not recompiled.
- names come from scanning: the test uses harness names that appear nowhere in src/.

## Acceptance
```
python -m pytest -q tests/test_harness_sources_compile.py
```

## Do not touch
The project's build script or anything under the target checkout; sidecar; crash_dedup._get_crash_signature; anything outside src/ and tests/.
