# discver CRS — Java Gap Remediation Plan

Companion to the gap analysis. Phased by **dependency**, not just yield: Phase 0 unblocks everything downstream. Each item names the module it touches, the concrete technique, and either **drop-in code** (self-contained) or a **change-spec** (edit to your code — share the file and I'll write the patch).

Design principles respected throughout: swap *producers*, keep *consumers* (data-source swaps beat rewrites); add validation stages rather than replacing existing filters.

---

## Phase 0 — Foundations (do these first; everything else stands on them)

### 0.1 Real type-resolved call graph → `code_analysis.py`

**Why first:** your name-keyed `CallGraph.functions` corrupts ranking, coverage attribution, taint, harness targeting, and triage. Every later fix that reasons over reachability inherits this error.

**Approach — swap the graph *producer*, keep every consumer.** Your downstream code consumes `FunctionInfo` / callers / callees / `get_function_context()`. Keep all of that. Replace only *how the graph is built*: from name-matching to a bytecode-level, type-resolved graph keyed on `(declaring_class, method_name, jvm_descriptor)`.

**Tooling choice:**
- **WALA** — most stable Java call-graph API; RTA/0-CFA/1-CFA options; battle-tested. Best if you want reliability.
- **SootUp** — modern Soot rewrite, cleaner API, CHA/RTA; API still churns, pin the version.
- **Reuse CodeQL** — if you already build the Java DB, its call-graph is free; query it and normalize.

Pick RTA (rapid type analysis) as the default — sound enough, cheap, resolves virtual dispatch far better than name matching. Entry points = your harness target methods + public API surface.

**Change-spec for `code_analysis.py`:**
- `FunctionInfo` key becomes `(class, method, descriptor)`; keep a `simple_name` field for display only.
- `callers()/callees()` read from the resolved graph, not name matching.
- `covered_functions()` maps JVM coverage (see 3.1) onto the same keys.
- Emit the *same JSON shape* your consumers already expect, so ranking/triage/harness code is untouched.

**Scaffold (WALA-based extractor, structural — pin versions, adapt entry-point scope):**
```java
// CallGraphExtractor.java — emits FunctionInfo/edges JSON your code_analysis.py ingests.
// Deps: com.ibm.wala:com.ibm.wala.core (pin a release).
AnalysisScope scope = AnalysisScopeReader.instance.makeJavaBinaryAnalysisScope(appJarsCp, exclusionsFile);
ClassHierarchy cha = ClassHierarchyFactory.make(scope);
// Entry points: your resolved harness targets + public API, NOT just "main".
Iterable<Entrypoint> entries = makeEntrypoints(cha, targetMethodsFromHarnessTargetsJson);
AnalysisOptions opts = new AnalysisOptions(scope, entries);
CallGraphBuilder<?> builder = Util.makeRTABuilder(opts, new AnalysisCacheImpl(), cha); // RTA
CallGraph cg = builder.makeCallGraph(opts, null);
// For each node: emit {class, method, descriptor, callers[], callees[], is_static}.
// Key = declaringClass + "#" + selector (name+descriptor) — never the bare name.
```
Run this as a producer step; your Python side reads the JSON. Surgical: one new data source, zero consumer changes.

---

### 0.2 Interprocedural taint → reuse CodeQL in `external_analyzers.py` / `taint_tracker.py`

**Why:** your `taint_tracker.py` is intra-procedural; real Java flows cross methods, classes, and framework boundaries. You already run CodeQL for findings — you're leaving its dataflow engine on the table.

**Approach:** stop using CodeQL only for canned findings; run **custom dataflow/taint queries** whose sources are fuzzer-reachable inputs and whose sinks are *your* sink categories (`sinks.py`). Normalize the returned source→step→sink paths into target ranking and into harness entry-point selection (so you generate a harness that actually drives a real path, not a random public method).

**Change-spec:**
- Add a CodeQL query pack: `TaintToSink.ql` using `semmle.code.java.dataflow.TaintTracking`, with a `sink` predicate populated from your `sinks.py` categories (deserialization, SSRF, injection, path traversal, etc.).
- Parse the SARIF `codeFlows` into `(source, [steps], sink)` records.
- Feed those into "Target ranking and promotion" — a function on a *proven* source→sink path outranks a danger-scored-but-unreachable one.
- Feed the sink's enclosing method into harness entry-point selection.

This is query + glue work, not core surgery. Highest reachability ROI after the call graph.

---

## Phase 1 — Realign to the Java bug class (highest yield, low disruption)

### 1.1 Timeout/OOM/ReDoS/infinite-loop become first-class vulnerabilities → crash path + `crash_analyzer.py`

**Why:** these dominate Java benchmarks (CWE-400/834/835/1333/407) and you currently route them to `NonCrashInfo`, "never submitted as PoV."

**Approach:** add a Java non-crash → vulnerability classifier that runs on timeout/OOM artifacts + stuck traces, maps to a CWE, and promotes qualifying cases through the PoV path (still fail-closed via N-of-M replay from 3.3, since timeouts are flaky).

**Drop-in classifier (self-contained; wire into your harvest loop for Java targets):**
```python
# java_noncrash_classifier.py — promotes Java resource-exhaustion artifacts to vuln candidates.
import re

# Signals extracted from the stuck trace / repeated top-of-stack sampling you already collect.
REGEX_FRAMES   = ("java.util.regex.Pattern", "java.util.regex.Matcher")
LOOP_HINTS     = ("while", "for", "iterate", "next(")
ALLOC_FRAMES   = ("java.util.Arrays.copyOf", "ArrayList.grow", "AbstractStringBuilder")

def classify_java_noncrash(kind, stuck_trace, top_frames, alloc_bytes=None):
    """
    kind: 'timeout' | 'oom'
    stuck_trace: text backtrace captured at the hang (you already cap 5/harness)
    top_frames: list[str] most-sampled frames
    returns (cwe, title, promote: bool) or (None, None, False)
    """
    joined = "\n".join(top_frames) + "\n" + (stuck_trace or "")

    if any(f in joined for f in REGEX_FRAMES):
        return ("CWE-1333", "Inefficient Regular Expression Complexity (ReDoS)", True)

    if kind == "timeout":
        # Same frame dominating the sample window => tight/infinite loop.
        if top_frames and top_frames.count(top_frames[0]) / max(len(top_frames), 1) > 0.6:
            if any(h in joined.lower() for h in LOOP_HINTS):
                return ("CWE-835", "Loop with Unreachable Exit Condition (Infinite Loop)", True)
            return ("CWE-834", "Excessive Iteration", True)
        return ("CWE-400", "Uncontrolled Resource Consumption", True)

    if kind == "oom":
        if any(f in joined for f in ALLOC_FRAMES) or (alloc_bytes and alloc_bytes > (256 << 20)):
            return ("CWE-789", "Memory Allocation with Excessive Size Value", True)
        return ("CWE-400", "Uncontrolled Resource Consumption", True)

    return (None, None, False)
```

**Change-spec for `crash_analyzer.py`:** when the artifact is a Java timeout/OOM, call the above; if `promote` is true, send it through the normal CrashAnalyzer LLM root-cause pass (with a ReDoS/complexity-specific prompt) and into the PoV path — *not* the hang-report side channel. Keep the hang report as an artifact, but stop letting it be the terminal state.

---

### 1.2 Stop the harness from swallowing Jazzer's bug signals → `java_harness_agent.py` / `harness_templates.py`

**Why:** your fixed exception skeleton "rethrows RuntimeException/Error except IAE" — that risks intercepting the `FuzzerSecurityIssue*` throwables Jazzer uses to *report* SSRF/injection/deserialization/path-traversal.

**Rule:** never place a catch around the target call that can absorb, reclassify, or normalize a Jazzer security issue or Jazzer-internal error. Catch only *narrow, declared, expected* exceptions of the target API, and only when you specifically want to treat them as uninteresting.

**Drop-in corrected template (replace your fixed exception skeleton):**
```java
// Correct Jazzer harness skeleton — surfaces security issues instead of eating them.
import com.code_intelligence.jazzer.api.FuzzedDataProvider;

public final class GenFuzzer_TARGET {
    public static void fuzzerTestOneInput(FuzzedDataProvider data) {
        // --- argument derivation (generated per target) ---
        // e.g. String s = data.consumeRemainingAsString();
        try {
            Target.vulnerableMethod(/* derived args */);
        } catch (IllegalArgumentException | java.io.IOException expected) {
            // ONLY exceptions the API legitimately declares for bad input.
            // Everything else — including FuzzerSecurityIssue* and all unchecked
            // exceptions — MUST propagate to the driver.
        }
    }
}
```
**Anti-pattern to delete anywhere it appears:**
```java
try { Target.m(...); }
catch (RuntimeException | Error e) { throw e; }   // <-- can wrap/mask FuzzerSecurityIssue*; do not do this
// or worse:
catch (Throwable t) { /* swallow */ }              // <-- blinds every Jazzer sanitizer
```

**Change-spec:** in `java_harness_agent.py` generation modes, (a) drop the blanket rethrow skeleton, (b) enable the relevant Jazzer bug detectors explicitly per harness (`--sanitizer`/autofuzz flags for the sink class you're targeting), and (c) add a generation-time lint that rejects any produced harness containing `catch (Throwable`, `catch (RuntimeException`, or a rethrow wrapper around the target call.

---

## Phase 2 — Reach the format-gated bugs (unlocks XXE / SSRF-via-format / deserialization)

### 2.1 Structure-aware input generation → `java_harness_agent.py` + corpus stage

**Why:** byte-level `FuzzedDataProvider` almost never yields a valid XML/XLSX/PDF/serialized stream, so the format-gated sinks are never reached. Two complementary mechanisms:

**Mechanism A — model writes a *generator*, not an input** (the trick every AIxCC team converged on). For a target whose input is a known format, have the LLM emit a small deterministic function that turns `FuzzedDataProvider` bytes into a *structurally valid, mutated* document. Contract:
```
GENERATOR CONTRACT (prompt spec):
  input:  FuzzedDataProvider
  output: a syntactically valid <FORMAT> byte[]/stream whose *content* is byte-driven
  rules:  structure always valid; only leaf values / attributes / entity refs are fuzzed;
          expose the security-relevant knobs (XML: DOCTYPE/ENTITY/SYSTEM; XLSX: external refs;
          archive: entry names/paths) as fuzzable so XXE/SSRF/traversal become reachable.
```

**Drop-in example — harness-side structured XML builder (drives XXE/SSRF reachability):**
```java
// Byte-driven but structurally valid XML: keeps the doc parseable while fuzzing the
// exact features that gate XXE/SSRF (DOCTYPE, external ENTITY, SYSTEM ids).
static String buildXml(FuzzedDataProvider d) {
    boolean withDoctype = d.consumeBoolean();
    boolean externalEntity = d.consumeBoolean();
    String sysId = d.consumeAsciiString(64);          // becomes SYSTEM "..." — SSRF/file surface
    String elem  = sanitizeName(d.consumeAsciiString(16));
    String body  = d.consumeRemainingAsString();
    StringBuilder x = new StringBuilder("<?xml version=\"1.0\"?>");
    if (withDoctype) {
        x.append("<!DOCTYPE r [");
        if (externalEntity) x.append("<!ENTITY x SYSTEM \"").append(sysId).append("\">");
        x.append("]>");
    }
    x.append("<").append(elem).append(">")
     .append(externalEntity ? "&x;" : escape(body))
     .append("</").append(elem).append(">");
    return x.toString();
}
```

**Mechanism B — format-keyed seed corpus.** In your corpus stage, harvest real valid samples (OSS-Fuzz corpora, project test resources) keyed by harness input format, and seed the shared corpus with them. Valid seeds + structural mutation reach depths random bytes never will.

**Change-spec:** in entry-point selection, detect the target's input format (parameter types, parser class names, imports). If it's a known structured format, switch generation mode to "generator" and attach the format-keyed seeds.

---

### 2.2 Deserialization awareness → `sinks.py` + ranking (detection/reachability only)

**Scope note:** this is discovery — detect the sink, reach it, prove the bug exists. I'm giving detection + reachability, not weaponized gadget chains.

**Approach:**
- Add a **deserialization sink category** to `sinks.py`: `ObjectInputStream.readObject`, `XMLDecoder.readObject`, framework deserializers (Jackson polymorphic, XStream, SnakeYAML `load`, etc.).
- **Rank up** targets where a tainted source (from 0.2) reaches such a sink.
- For PoV construction, generate a *valid serialized stream of an allowed class* to reach the sink and let Jazzer's deserialization sanitizer fire — reachability + detector, not a turnkey RCE payload.
- As a ranking signal only, note whether known gadget libraries are present on the target classpath (elevates severity/priority); don't synthesize the chain.

---

## Phase 3 — Precision and closed loops

### 3.1 Method-targeted coverage/refinement loop → `java_harness_agent.py` + `main.py`

**Why:** you generate Java harnesses but never check whether they reached the intended target method; refinement is blind (your own flagged gap).

**Approach:** capture per-method JVM coverage (Jazzer's coverage output, or JaCoCo on the harnessed run), diff against `harness_targets.json`'s intended target, and drive refinement when the target wasn't reached:
```
after smoke/first campaign window:
  covered = jvm_covered_methods(harness)          # keyed (class, method, descriptor) — same keys as 0.1
  target  = harness_targets[harness].target_method
  if target not in covered:
      # re-derive argument construction toward `target`; synthesize method-directed seeds
      refine_harness(harness, goal=target)         # analog of your plateau seed synth, but method-directed
```
Feeds `covered_functions()` in `code_analysis.py` too — closes the C-side `covered_funcs` gap with the same plumbing.

---

### 3.2 Hypothesis-validation triage → upgrade `triage.py` (keep grep rounds as pre-filter)

**Why:** "3 grep-backed rounds VALID/INVALID/UNCERTAIN" can't reason about whether a guard dominates the path to the sink — which is the whole question for Java logical bugs. VulAgent's structured validation cut FPR ~49%→~20% at equal accuracy.

**Approach — add a stage after your grep rounds (surgical, keeps existing filter as cheap pre-pass):**
```
1. HYPOTHESIS: for each surviving candidate, LLM emits
     { cwe, trigger_conditions:[...], trigger_path:[source -> ... -> sink] }
2. ASSUMPTION PRUNING: check each condition against real context (0.1 call graph + 0.2 taint).
     A condition contradicted by a structural invariant  -> drop the hypothesis.
3. GUARD DOMINANCE: does a defensive check (bounds/null/sanitizer/early-return) dominate
     ALL feasible paths from source to sink under the surviving assumptions?
       dominated  -> INVALID (false positive)
       not dominated -> VALID
4. UNCERTAIN only when context is genuinely missing (flag for execution grounding, 4.1).
```
Because 0.1/0.2 give you a real graph and real flows, step 3 is now answerable instead of guessed. **[token-budget lever]** run the full chain on every candidate — this is your cheapest large precision win.

---

### 3.3 Fail-closed evidence lifecycle → `fuzzer.py` / finalization (`main.py` Phase 3)

**Why (your own flagged risk):** the watcher copies crash/leak into the PoV dir *before* standalone replay, so replay timeouts/errors can be promoted as reproduced. Java replay is *more* flaky (JVM warmup, classpath, `<clinit>`, timeout variance).

**Change-spec:** reorder so promotion happens *only after* an independent replay succeeds. Replay timeout/error = **unreproduced**, never reproduced. For timeout-class Java findings specifically, require **N-of-M** successful replays before promotion (kills warmup-flake false positives). Stage artifacts in a quarantine dir; move to `/output/povs` only on confirmed replay.

---

## Phase 4 — Depth (do after 0–3 are paying off)

### 4.1 Execution-grounded exploit loop → wrap `ExploiterAgent`
Replace "single short libFuzzer campaign" with a **plan → execute → evaluate → refine** loop for the logical bugs fuzzing can't stumble into: model proposes a concrete trigger, you run it, feed the *exact* exception/stdout back, model revises. Cap ~15–20 iterations (Co-RedTeam plateaued there; removing execution feedback was its single biggest ablation, 59%→17.5%). **[token-budget lever]** token-heavy per bug, affordable for you.

### 4.2 Persistent layered memory → new `memory.py`, retrieved by all agents
Cross-run, embedding-retrieved, three layers: **vulnerability patterns** (symptom→hypothesis→confirming test), **strategies**, **technical actions** (a working XLSX-SSRF trigger, a serialized-stream recipe). Warm-start new targets from it. Co-RedTeam's 2nd-biggest ablation; drove *sustained* CyberGym gains. Your `strategies.add_crash()` is the seed of this — promote it from within-run to persistent + retrievable.

### 4.3 Single-model debate approximation → strengthen `ensemble.py`
You can't get true cross-model disagreement on one open-source model, but **[token-budget lever]** approximate MDASH's debate FP-filter with heavy multi-sample self-consistency + an explicit **auditor-vs-debater self-play scoped to reachability and guard-dominance** — that's where disagreement is the useful signal. Your adversarial judge is the hook; make the debate specifically about "can the debater refute reachability?"

---

## Suggested execution order (dependency-correct)

1. **0.1 call graph** + **0.2 CodeQL taint** — foundational; unblock 3.2, ranking, harness targeting.
2. **1.1 timeouts-as-bugs** + **1.2 exception policy** — biggest Java yield, low disruption, independent of Phase 0.
3. **2.1 structure-aware generation** — reachability for the format-gated classes.
4. **3.1 coverage loop** + **3.2 hypothesis triage** + **3.3 fail-closed** — precision and directed refinement (3.2 wants 0.1/0.2 done).
5. **2.2 deserialization**, then **Phase 4** depth items.

You can run track 2 in parallel with track 1 — the exception-policy and timeout-classifier fixes don't depend on the call graph.

---

*Drop-in code above (`java_noncrash_classifier.py`, the corrected Jazzer template, the XML builder, the WALA scaffold) is self-contained. The change-specs (`code_analysis.py`, `taint_tracker.py`, `triage.py`, `java_harness_agent.py`, `fuzzer.py`) are edits to your modules — share the module and I'll turn its spec into an actual patch against your code.*
