# CRS Full Code Audit — All Steps 1–7

## Step 0 — Files to DELETE immediately

These are stale stubs that will cause import shadowing or conflict:

| File | Why |
|---|---|
| `types.py` | Defines a completely different schema for CyberGymTask, CodeContext, LLMRouter, PoCResult, BuildResult — conflicts with all real implementations. The fuzzer was wrongly coded against this. |
| `config-1.py` | Exact duplicate of `config.py` with different field names — confusing and unused |
| `config-2.py` | Another stub config with different field names |
| `data_loader-1.py` | Old stub data loader with different CyberGymTask (has `description` and `source_files` fields) — this caused bugs in code_intelligence.py |

---

## The Root Problem: Two Conflicting Schemas

Your AI coder generated two different definitions of the same dataclasses and never reconciled them.

**Schema A** (the real one — `data_loader.py` + `code_intelligence.py`):
- `CyberGymTask.vulnerability_description` (str)
- `CyberGymTask.project_language` (str)
- `CyberGymTask.project_name` (str)
- `CodeContext.ranked_files` (list of (Path, float) tuples)
- `CodeContext.top_snippets` (str — formatted code text)
- `LLMRouter.chat(system_prompt, user_prompt, ...)`

**Schema B** (the wrong one — `types.py`, now deleted):
- `CyberGymTask.description` (str)
- `CyberGymTask.vuln_type` (str)
- `CyberGymTask.metadata` (dict)
- `CodeContext.top_snippets` (list of CodeSnippet objects)
- `LLMRouter.query(prompt, ...)` (single positional arg)

Every bug below is caused by one module using Schema A and another using Schema B.

---

## Bug #1 — config.py: All attributes are on `cfg` instance, not module-level

**Root cause:** `config.py` exports `cfg = CRSConfig()`. All fields like `LLM_MODEL`, `WORK_DIR`, `BUILD_TIMEOUT` etc. are attributes on that instance. Multiple modules try to import them as bare module-level names, which fails.

**Fix — Add these module-level aliases to the BOTTOM of `config.py`** (after the `cfg = CRSConfig()` line):

```python
# Module-level aliases so other modules can do:
#   from crs.config import WORK_DIR, BUILD_TIMEOUT, etc.
WORK_DIR          = cfg.WORK_DIR
BUILD_TIMEOUT     = cfg.BUILD_TIMEOUT
RUN_TIMEOUT       = cfg.RUN_TIMEOUT
USE_SANITIZERS    = cfg.USE_SANITIZERS
FUZZING_ENABLED   = cfg.FUZZING_ENABLED
FUZZING_TIMEOUT   = 120  # seconds (used by fuzzer.py)
LLM_MODEL         = cfg.LLM_MODEL
LLM_BASE_URL      = cfg.LLM_BASE_URL
LLM_API_KEY       = cfg.LLM_API_KEY
MAX_TOKENS        = cfg.MAX_TOKENS
MAX_RETRIES       = cfg.MAX_RETRIES

# Aliases main.py looks for
DEFAULT_MODEL     = cfg.LLM_MODEL
DEFAULT_BASE_URL  = cfg.LLM_BASE_URL
DEFAULT_API_KEY   = cfg.LLM_API_KEY
POC_RUN_TIMEOUT   = cfg.RUN_TIMEOUT
VERBOSE           = False
```

---

## Bug #2 — `llm_router.py`: Wrong import + config attribute access crashes on import

**Line 19:** `from crs import config`

Then lines 79–83 use `config.LLM_MODEL`, `config.LLM_BASE_URL`, `config.LLM_API_KEY`, `config.MAX_TOKENS` as default argument values. These don't exist at module level — they're on `cfg`.

**Fix:**
```python
# CHANGE line 19 from:
from crs import config

# TO:
from crs import config
from crs.config import cfg as _cfg
```

Then change all default argument references in `LLMRouter.__init__` and `.chat()`:
```python
# CHANGE (lines 79-83):
primary_model: str = config.LLM_MODEL,
primary_base_url: str = config.LLM_BASE_URL,
primary_api_key: str = config.LLM_API_KEY,

# TO:
primary_model: str = _cfg.LLM_MODEL,
primary_base_url: str = _cfg.LLM_BASE_URL,
primary_api_key: str = _cfg.LLM_API_KEY,

# CHANGE (chat() line):
max_tokens: int = config.MAX_TOKENS,
# TO:
max_tokens: int = _cfg.MAX_TOKENS,
```

Also fix lines 91 and 103 — `config.TIMEOUT` does not exist:
```python
# CHANGE:
timeout=config.TIMEOUT if hasattr(config, "TIMEOUT") else 120.0,
# TO:
timeout=float(_cfg.RUN_TIMEOUT),
```

---

## Bug #3 — `code_intelligence.py`: Three bugs from stub schema

### Bug 3a — `task.description` does not exist (line 498)
```python
# CHANGE:
description = task.description
# TO:
description = task.vulnerability_description
```

### Bug 3b — `task.source_files` and `task.gather_source_files()` do not exist (lines 501–505)
These methods only existed on the deleted `data_loader-1.py` stub. The real `CyberGymTask` doesn't have them.
```python
# CHANGE (lines 501–505):
if not task.source_files:
    task.gather_source_files()
ranked = rank_files_by_relevance(task, task.source_files, description)

# TO:
from crs.data_loader import get_source_files
source_files = get_source_files(task)
ranked = rank_files_by_relevance(task, source_files, description)
```

### Bug 3c — `load_task` in `__main__` block (line 543)
```python
# CHANGE:
from crs.data_loader import load_task
task = load_task(task_dir)
# TO:
from crs.data_loader import load_task_from_local
task = load_task_from_local(task_dir)
```

---

## Bug #4 — `poc_strategies.py`: Four bugs from schema mismatch

### Bug 4a — `router.query()` does not exist — LLMRouter uses `.chat()`
Every strategy calls `router.query(system_prompt=..., user_prompt=...)`. This method doesn't exist — it's `router.chat()`. 

Search for every occurrence of `router.query(` in `poc_strategies.py` and replace:
```python
# CHANGE every instance of:
raw = router.query(
    system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
    user_prompt=user_prompt,
)
# TO:
raw = router.chat(
    system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
    user_prompt=user_prompt,
)
```
Do this for all 5 strategy classes (roughly lines 278, 328, 372, 435, 578, 657).

### Bug 4b — `context.ranked_snippets` does not exist
`CodeContext` has `ranked_files`, not `ranked_snippets`.

In `_top_snippets_text()` function and `CallPathTargetedPoC._find_target_functions()`:
```python
# CHANGE every occurrence of:
context.ranked_snippets
# TO:
context.ranked_files
```

Then fix `_top_snippets_text` to work with (Path, float) tuples instead of snippet objects:
```python
def _top_snippets_text(context: CodeContext, max_files: int = 5) -> str:
    """Format the highest-ranked code snippets for inclusion in a prompt."""
    # context.ranked_files is list of (Path, float) tuples
    parts: list[str] = []
    for i, (fpath, score) in enumerate(context.ranked_files[:max_files], 1):
        try:
            code = fpath.read_text(errors="replace")[:3000]
        except OSError:
            continue
        parts.append(f"### Snippet {i} — {fpath.name} (score={score:.2f})\n```\n{code}\n```")
    return "\n\n".join(parts) if parts else "(no snippets available)"
```

And fix `CallPathTargetedPoC._find_target_functions` which also uses `ranked_snippets`:
```python
# CHANGE (in _find_target_functions):
ranked_files = context.ranked_snippets[:3]
all_source = ""
file_contents: list[tuple[str, str]] = []
for snip in ranked_files:
    code = snip.code if hasattr(snip, "code") else str(snip)
    file_contents.append((getattr(snip, "file_path", "unknown"), code))
    all_source += code + "\n"

# TO:
all_source = ""
file_contents: list[tuple[str, str]] = []
for fpath, score in context.ranked_files[:3]:
    try:
        code = fpath.read_text(errors="replace")[:5000]
    except OSError:
        code = ""
    file_contents.append((str(fpath), code))
    all_source += code + "\n"
```

### Bug 4c — `context.task_id` does not exist
```python
# CHANGE every occurrence of:
context.task_id
# TO:
context.task.task_id
```
(Appears in `_save_poc` calls inside strategy classes.)

### Bug 4d — `context.build_type` does not exist (DirectDescriptionPoC, line 272)
```python
# CHANGE:
f"## Build System\n{context.build_type}\n\n"
# TO:
f"## Build System\n{context.build_info.get('type', 'unknown')}\n\n"
```

---

## Bug #5 — `build_executor.py`: Three bugs

### Bug 5a — Wrong config import
```python
# CHANGE (lines 19-23):
from crs.config import (
    BUILD_TIMEOUT,
    RUN_TIMEOUT,
    WORK_DIR,
)
# TO (after applying Bug #1 fix to config.py, these will work — but safer to use cfg):
from crs.config import cfg as _cfg
BUILD_TIMEOUT = _cfg.BUILD_TIMEOUT
RUN_TIMEOUT   = _cfg.RUN_TIMEOUT
WORK_DIR      = _cfg.WORK_DIR
```
OR just apply Bug #1 fix to config.py first and the import will work as-is.

### Bug 5b — `poc_result.source_path` does not exist (line 425)
`PoCResult` has `poc_path`, not `source_path`.
```python
# CHANGE (line 425):
poc_source = Path(poc_result.source_path)
# TO:
poc_source = Path(poc_result.poc_path)
```

### Bug 5c — `poc_result.strategy` does not exist (lines 418 and 620)
`PoCResult` has `strategy_name`, not `strategy`.
```python
# CHANGE (line 418):
strategy_name = getattr(poc_result, "strategy", "poc")
# TO:
strategy_name = getattr(poc_result, "strategy_name", "poc")

# CHANGE (line 620 in execute_pipeline):
print(f"--- PoC {idx}/{len(poc_results)}: {getattr(poc, 'strategy', '?')} ---")
# TO:
print(f"--- PoC {idx}/{len(poc_results)}: {getattr(poc, 'strategy_name', '?')} ---")
```

---

## Bug #6 — `evaluator.py`: Four bugs

### Bug 6a — `task.vulnerability_type` doesn't exist (line 97)
```python
# CHANGE:
vuln_type = getattr(task, "vulnerability_type", "unknown")
# TO:
vuln_type = getattr(task, "vuln_type", "unknown")  # from CodeContext, not task
```
Wait — `vuln_type` is on `CodeContext`, not on `CyberGymTask`. The evaluator doesn't have `context`. Best fix: accept `vuln_type` as a parameter, or derive it from the description.
```python
# Simplest fix — derive from description like code_intelligence does:
from crs.code_intelligence import classify_vulnerability
vuln_type = classify_vulnerability(
    getattr(task, "vulnerability_description", "")
)
```

### Bug 6b — `getattr(best, "strategy", "none")` — RunResult has no `strategy` field (line 110)
`best` is a `RunResult`. Strategy is nested at `best.poc_build.poc_result.strategy_name`.
```python
# CHANGE (line 110):
strategy_used=getattr(best, "strategy", "none") if best else "none",
# TO:
strategy_used=(
    best.poc_build.poc_result.strategy_name
    if best and best.poc_build and best.poc_build.poc_result
    else "none"
),
```

### Bug 6c — `getattr(best, "confidence", 0.0)` — RunResult has no `confidence` (line 113)
```python
# CHANGE (line 113):
confidence=getattr(best, "confidence", 0.0) if best else 0.0,
# TO:
confidence=(
    best.poc_build.poc_result.confidence
    if best and best.poc_build and best.poc_build.poc_result
    else 0.0
),
```

### Bug 6d — `getattr(best, "poc_code", "")` — RunResult has no `poc_code` (line 116)
```python
# CHANGE (line 116):
poc_code=getattr(best, "poc_code", "") if best else "",
# TO:
poc_code=(
    best.poc_build.poc_result.poc_code
    if best and best.poc_build and best.poc_build.poc_result
    else ""
),
```

Also fix `_build_notes()` which has the same problem:
```python
# CHANGE (line 250):
strat = getattr(r, "strategy", "?")
stderr_snip = getattr(r, "stderr", "")[:120].replace("\n", " ")
# TO:
strat = (r.poc_build.poc_result.strategy_name
         if r.poc_build and r.poc_build.poc_result else "?")
stderr_snip = r.run_log[:120].replace("\n", " ")
```

---

## Bug #7 — `main.py`: Six bugs

### Bug 7a — `config.DEFAULT_MODEL`, `config.DEFAULT_BASE_URL`, `config.DEFAULT_API_KEY`, `config.POC_RUN_TIMEOUT` don't exist (lines 244–285)
**Fixed by applying Bug #1** (adding aliases to config.py). No change needed in main.py after that.

### Bug 7b — `build_project(task, context)` passes CodeContext not dict (line 81)
`build_executor.build_project` signature is `build_project(task, build_info: dict)`.
```python
# CHANGE (line 81):
build_result: BuildResult = build_project(task, context)
# TO:
build_result: BuildResult = build_project(task, context.build_info)
```

### Bug 7c — Stub `RunResult` constructor uses wrong field names (lines 94–103)
`RunResult` from `build_executor.py` takes: `poc_build, triggered, crash_type, sanitizer_output, return_code, run_log`.

The stub in `main.py` passes `returncode, stdout, stderr, confidence, poc_code, strategy` — none of which exist on `RunResult`.

```python
# CHANGE (lines 94–103):
stub = RunResult(
    triggered=False,
    returncode=-1,
    stdout="",
    stderr=getattr(compiled, "error", "compile failed"),
    crash_type="",
    confidence=0.0,
    poc_code=getattr(poc, "code", ""),
    strategy=label,
)

# TO:
stub = RunResult(
    poc_build=compiled,       # PoCBuildResult object
    triggered=False,
    crash_type="compile_failed",
    sanitizer_output="",
    return_code=-1,
    run_log=compiled.compile_log,
)
```

### Bug 7d — `result.strategy = label` — RunResult has no `strategy` attribute (lines 110–111)
```python
# REMOVE these two lines entirely:
if not getattr(result, "strategy", ""):
    result.strategy = label
```
The strategy is accessible via `result.poc_build.poc_result.strategy_name` when needed.

### Bug 7e — `result.returncode` does not exist (line 115)
RunResult uses `return_code` not `returncode`.
```python
# CHANGE (line 115):
print(f"  [PoC {idx}] {tag}  (exit={result.returncode})")
# TO:
print(f"  [PoC {idx}] {tag}  (exit={result.return_code})")
```

### Bug 7f — `orchestrator.run(context, strategies=requested)` — `.run()` only takes `context` (line 156)
`PoCOrchestrator.run()` signature is `run(self, context: CodeContext)` — no `strategies` kwarg.

```python
# CHANGE (line 156):
poc_results = orchestrator.run(context, strategies=requested)
# TO:
poc_results = orchestrator.run(context)
# (strategy filtering is not implemented in the orchestrator — remove for now
#  or implement it separately by filtering DEFAULT_ORDER before passing to orchestrator)
```

### Bug 7g — `try_fuzzing(context, router=router, task=task)` — missing `build_result` arg (line 173)
`fuzzer.try_fuzzing` signature is `try_fuzzing(context, build_result, router, task)`.

```python
# CHANGE (line 173):
fuzz_poc = try_fuzzing(context, router=router, task=task)
# TO:
fuzz_build_result: BuildResult = build_project(task, context.build_info)
fuzz_poc = try_fuzzing(context, fuzz_build_result, router, task)
```

---

## Bug #8 — `fuzzer.py`: Four bugs

### Bug 8a — Imports from `crs.types` (lines 25–31)
`types.py` is being deleted. Fix all imports:
```python
# CHANGE (lines 25–31):
from crs.types import (
    BuildResult,
    CodeContext,
    CyberGymTask,
    LLMRouter,
    PoCResult,
)

# TO:
from crs.build_executor import BuildResult
from crs.code_intelligence import CodeContext
from crs.data_loader import CyberGymTask
from crs.llm_router import LLMRouter
from crs.poc_strategies import PoCResult
```

### Bug 8b — `context.task.description` does not exist (line 152)
```python
# CHANGE (line 152):
description=context.task.description,
# TO:
description=context.task.vulnerability_description,
```

### Bug 8c — `context.top_snippets` is a string, not a list of CodeSnippet objects (line 146–149)
```python
# CHANGE (lines 146–149):
snippets_text = "\n---\n".join(
    f"// {s.filepath}:{s.start_line}-{s.end_line}\n{s.content}"
    for s in context.top_snippets[:6]
) or "(no snippets available)"

# TO:
snippets_text = context.top_snippets[:8000] or "(no snippets available)"
```

### Bug 8d — `router.query(prompt)` does not exist (line 156)
`LLMRouter` uses `.chat(system_prompt, user_prompt)` not `.query(prompt)`.
```python
# CHANGE (line 156):
harness_code = router.query(prompt, max_tokens=2048, temperature=0.3)

# TO:
harness_code = router.chat(
    system_prompt="You are a security researcher writing a libFuzzer harness. Output only C code.",
    user_prompt=prompt,
    max_tokens=2048,
    temperature=0.3,
)
```
Do the same for any other `router.query()` calls in `fuzzer.py` (search for all occurrences).

---

## Summary Table

| File | Bugs | Action |
|---|---|---|
| `types.py` | Wrong schema for everything | **DELETE** |
| `config-1.py` | Duplicate / conflicting stub | **DELETE** |
| `config-2.py` | Duplicate / conflicting stub | **DELETE** |
| `data_loader-1.py` | Wrong CyberGymTask definition | **DELETE** |
| `config.py` | No module-level aliases | Add ~15 alias lines at bottom |
| `llm_router.py` | Wrong config import, bad defaults, missing TIMEOUT | 3 small fixes |
| `code_intelligence.py` | `task.description`, `task.source_files`, `load_task` | 3 fixes |
| `poc_strategies.py` | `router.query`, `ranked_snippets`, `task_id`, `build_type` | 4 fixes |
| `build_executor.py` | `poc_result.source_path`, `poc_result.strategy` | 2 fixes |
| `evaluator.py` | RunResult field navigation, `vulnerability_type` | 5 fixes |
| `main.py` | 6 separate bugs across pipeline | 6 fixes |
| `fuzzer.py` | Wrong imports, Schema B fields, `router.query` | 4 fixes |

---

## Recommended Fix Order

1. Delete the 4 stale files first
2. Fix `config.py` (add aliases) — everything depends on this
3. Fix `llm_router.py` — everything calls it
4. Fix `code_intelligence.py` — Step 2 of the pipeline
5. Fix `poc_strategies.py` — Step 4
6. Fix `build_executor.py` — Step 5
7. Fix `evaluator.py` — Step 7
8. Fix `fuzzer.py` — optional step
9. Fix `main.py` last — it ties everything together

After each fix, run the integration test for that module before moving to the next one.
