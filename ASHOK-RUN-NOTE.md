# Running the Nov 5 discver drop — your workflow does not change

Same overlay, same compile check, same `oss-crs run` command, same output folder, same files back.
One new thing: a knowledge-base folder the container must be able to see. One optional thing you can skip.

## 1. Exactly what you did last time (unchanged)

1. Unzip `discver-src-tests-FULL-nov05.zip` into `example/discver/` so it overwrites `src/` and `tests/`.
   (New subfolders `src/lang/`, `src/workshop/`, `src/tools/` arrive with it — nothing to configure.)
2. `python -m compileall -q example/discver/src` (same purpose as the `py_compile` you ran before; covers the subfolders).
3. `uv run oss-crs run --compose-file example/discver/compose.yaml --fuzz-proj-path example/discver/test-targets/java-ant-byo --target-harness auto`

Output is where it always is: `runs/<run-id>/crs/discver/java-ant-byo_<hash>/LOG_DIR/auto/discver/`.

## 2. The one new thing: the knowledge base folder

`kb-nov05.zip` unzips to a folder `kb/` (inside: `recipes.jsonl`, `cwe_playbooks/`, `index.json`). discver looks for it **inside the
container** at `/opt/discver/kb`. Put the folder at `example/discver/kb/`, then make it visible in the container — pick one:

- **Volume mount (recommended, no image rebuild):** in `example/discver/compose.yaml`, under the discver service's `volumes:`, add
  `- ./kb:/opt/discver/kb:ro`
- **Or bake it in:** `COPY kb /opt/discver/kb` in the discver Dockerfile and rebuild the image.
- Different path? Add `DISCVER_KB=/that/path` to the service's `environment:`.

If the folder is not found, **the run still works exactly as before** — the log says `[strategy] no KB … default plan`. Nothing breaks.

## 3. Skip for now

`src/tools/Dockerfile.snippet` (CodeQL / Semgrep in the image). Without it discver logs `scanner <name> unavailable — skipped` and continues.
Add it in a later round if we want the scanners.

## 4. Two grep lines that tell us it worked (first ~200 lines of orchestrator.log)

```
grep -n "Target language\|\[strategy\]" orchestrator.log | head
grep -n "\] failed:" orchestrator.log
```
Expect `Target language: java …`, then `[strategy] KB /opt/discver/kb: 4100 recipes …` and numbered `[strategy] 1. …` plan lines after Phase 0.
`[strategy] no KB` means the mount did not take (run still valid). Any `[<module>] failed:` line — send it first.
On java-ant-byo the finding should be the same as last time: `X509Utils.getCnFromDn`, X509Utils.java:81.

## 5. Then CRSBench atlanta-imaging, twice

Exactly as it was run on Oct 29, same settings and budget, two runs:
- **B1 — without KB:** comment out the `./kb:/opt/discver/kb` volume line (or set `DISCVER_KB=/nonexistent`). Log shows `no KB … default plan`.
- **B2 — with KB:** volume line back in.
Send back both run folders and the two CSVs (`*_benchmark_summary.csv`, `*_trial_metrics.csv`).

## 6. What to send back per run (same as before)

The `LOG_DIR/auto/discver/` folder: `orchestrator.log`, `health.json`, `REPORT.md`, `bugs/`, `crashes/`, `unique_bugs/`,
plus the new `strategy_plan.json` and `strategy_outcomes.json`.
