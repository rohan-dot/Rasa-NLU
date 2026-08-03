# discver agent runbook

Everything runs on the lair box (`ro31337@g52lambda02`), in the Jupyter
terminal. Repo lives at:
`/panfs/g52-panfs/exp/FY26/AIxCC/ro31337/discver-java-dev`

---

## 0. Set up LiteLLM access (the "key stuff")

These three env vars point the agent at the MIT LL gateway. Paste each line,
Enter after each. Nothing prints — that's normal.

```bash
export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1
export LITELLM_API_KEY=sk-GhojGlZcfXdeDOG7syHKEQ
export AGENT_MODEL=claude-opus-4-8
export AGENT_TLS_VERIFY=false
```

(If the key ever stops working, regenerate one in the LiteLLM UI →
Virtual Keys, and paste the new `sk-...` here.)

Confirm they took:

```bash
echo $LITELLM_BASE_URL
echo $AGENT_MODEL
```

### Cheap connection test (one small call — do this before real work)

```bash
cd /panfs/g52-panfs/exp/FY26/AIxCC/ro31337/discver-java-dev
python microagent.py --repo $(pwd) --verify
```

Want to see: `[verify] OK — model=claude-opus-4-8 replied: 'ok'`
- If OK → gateway + key + TLS all good, continue.
- If it fails → check the four env vars above (most often the key line got
  skipped in a fresh terminal).

---

## 1. Get on the right branch (home base)

Your real work + instrumentation live on `agent/scaffold-improvements`.
The baseline commit is `9f5bd21`.

```bash
cd /panfs/g52-panfs/exp/FY26/AIxCC/ro31337/discver-java-dev
git checkout agent/scaffold-improvements
git log --oneline | head -3
```

Should show the 7-commit history ending in the telemetry commit.

---

## 2. Start the agent

```bash
python microagent.py --repo $(pwd)
```

Drops you to a `task>` prompt. Paste each task below at `task>`.
- `→ name(...)` lines = the agent using tools (reading/editing/running). Normal.
- When it finishes it prints a summary and returns to `task>`.
- To leave the agent: type `exit` (back to the normal shell for git commands).

---

## TASK 1 — Fix telemetry emission (do first: makes everything measurable)

```
The reliability telemetry in src/agent_tools.py (reliability_metrics as _rel) is confirmed present but its discver.reliability summary never appears in run logs. Find and fix why: (1) is _rel.log_summary() in a finally block so it runs on every exit path including early returns/exceptions? (2) how is the discver.reliability logger configured — level, handlers, propagation — would discver's logging setup suppress its INFO line? (3) is log_summary gated behind a condition that's false in normal runs? Fix whatever prevents emission (always-on, no flag — this is observability). Add a test that asserts the summary is emitted. Verify with py_compile and tests, commit.
```

After it lands, exit and check:
```bash
git show --stat HEAD
```

---

## TASK 2 — Coverage (put crash log in place first)

From a normal shell, copy your crash log into the repo so the agent can read it:
```bash
cp /path/to/your/crashlog.md /panfs/g52-panfs/exp/FY26/AIxCC/ro31337/discver-java-dev/crashlog.md
```
(Replace `/path/to/your/crashlog.md` with wherever your crash log actually is.)

Then in the agent at `task>`:
```
Priority #2 is fuzzer coverage. Last run came back WEAK: fuzzer barely ran (523 execs on X509Fuzzer), coverage never increased on any harness, no new corpus units. Diagnose the root cause from the code and the crash log at ./crashlog.md. discver generates harnesses and seeds at runtime via harness_agent.py, java_harness_agent.py, seed_generator.py, harness_templates.py. Read those plus fuzzer.py and run_health.py. Determine WHY coverage stays flat — too few harnesses, shallow/wrong entry points, or seeds not reaching interesting paths — citing evidence from the log. Then implement ONE bounded fix targeting the biggest lever, behind a flag (env DISCVER_*) defaulting to current behavior so it can't make the WEAK problem worse. Add tests. Verify with py_compile and tests. Update CHANGELOG/README. Report the diagnosis and what you changed. Commit.
```

After it lands: `git show --stat HEAD` and READ its diagnosis — that's the
most valuable output (tells you if WEAK was code or environment).

---

## TASK 3 — Triage precision

At `task>`:
```
Priority #3 is triage precision: too many low-confidence LLM candidates reach the final report mixed with confirmed crashes. In triage.py, crash_analyzer.py, crash_dedup.py, strengthen separation so unverified model leads are clearly demoted below fuzzer-confirmed reproduced crashes — behind a flag defaulting to current behavior. Never touch the confirmed-crash/PoV path. Add tests, verify, update docs, commit.
```

After it lands: `git show --stat HEAD`

---

## 3. Package for Ashok (only changed .py files — no patches)

Exit the agent, then:
```bash
git diff --name-only 9f5bd21 -- '*.py'
mkdir -p for-ashok
git diff --name-only 9f5bd21 -- '*.py' | while read f; do mkdir -p "for-ashok/$(dirname "$f")"; cp "$f" "for-ashok/$f"; done
ls -R for-ashok
```

Download the `for-ashok` folder from the Jupyter file browser and send it.

Also list any new flags your tasks added, so Ashok knows what to turn on:
```bash
grep -rho "DISCVER_[A-Z_]*" src/*.py | sort -u
```

---

## 4. Note to send Ashok

> Drop these .py files into example/discver/src/ (match the paths), overwriting.
> Before running, confirm you're on my version:
>     grep -n reliability_metrics src/agent_tools.py    # must print a line
>     python -m py_compile src/*.py                      # must be silent
> In compose.yaml under the discver service's environment:, set:
>     - DISCVER_GUIDED_JSON=1
>     - DISCVER_SINGLE_ACTION=1
>     (plus any new DISCVER_* flags I list)
> Run your usual: uv run oss-crs run --compose-file example/discver/compose.yaml ...
> Then: grep -ri reliability /var/log/discver/   — the summary line should appear.

---

## Time budget (2 hours)

- Setup + verify: ~5 min
- Task 1 (emission fix): ~10 min  ← non-negotiable, makes everything measurable
- Task 2 (coverage diagnosis + fix): ~15 min  ← diagnosis is the high-value part
- Task 3 (triage): ~10 min  ← most droppable if short on time
- Package + note: ~5 min

If short on time, priority order is: Task 1 → Task 2 diagnosis → package.

---

## Quick troubleshooting

- **`--verify` fails** → re-check the 4 env vars in step 0 (usually the key line).
- **agent hangs >30s on a task** → Ctrl-C, check the error; often TLS/network.
- **git says wrong branch** → `git checkout agent/scaffold-improvements`.
- **want to stop the agent cleanly** → type `exit` (not Ctrl-Z, which just parks it).
- **a task edits the wrong thing** → the per-task commits are revertible:
  `git show HEAD` to review, `git revert HEAD` to undo one.
