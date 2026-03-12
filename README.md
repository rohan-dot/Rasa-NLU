Sure:

1. **Load** — reads the vulnerable repo tarball and vulnerability description from the task directory, extracts source files, detects build system, and classifies the vuln type

2. **Context** — ranks source files by relevance to the description, extracts key code snippets, and packages everything into a `CodeContext` object for the LLM

3. **Generate** — runs 5 strategies in parallel (analyze-first, call-path-targeted, pattern-replay, iterative-refine, direct-description), each prompting Gemma to write a C PoC that triggers the bug

4. **Build & Run** — tries to build the vulnerable project with ASAN, compiles each PoC against it, executes each binary, and checks output for sanitizer crash signatures

5. **Evaluate** — picks the best triggered PoC (or best attempt if none triggered), saves the result to `results.json` and `results.md` with strategy used, confidence, crash type, and time elapsed
