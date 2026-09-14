# Kickoff for Claude Code on kumar-dev

## Install (Ashok, 2 minutes)
1. Unzip this kit into the discver project directory — the one that contains `src/` and `tests/` (was `example/discver/`). It adds `CLAUDE.md`, `RUNLOG.md`, and `.claude/agents/`, `.claude/commands/`. Nothing in `src/` is touched.
2. Clone the reference repos next to it as `reference/atlantis` and `reference/buttercup` (or fix the two paths in CLAUDE.md).
3. `git checkout -b claude/discver-fix` in that directory so every cycle is a commit Rohan can pull back.
4. Start Claude Code from that directory. It must be allowed to run Bash (tests, docker compose, the run command, grep over run dirs). Without execution access this is the edit-only setup again and will not converge.
5. Paste the message below as the first prompt.

## First message (paste verbatim)

Read CLAUDE.md and RUNLOG.md fully before doing anything.

Then, in this order, reporting after each step with the actual command output:

1. ORIENT. Confirm the layout with `ls`: `src/`, `tests/`, the compose file, the newest run dir under `.oss-crs-workdir/`, the reference repos, and the runtime LLM endpoint from the compose file or env. Correct any wrong path in CLAUDE.md and record the endpoint in RUNLOG.md.

2. TRIAGE run 8 (2026-09-10 20:57, run id starting `1789065189`): `/triage <that run dir>`. I want the two `patch_index.json` reason strings, the cause of `LLM-generated seeds: 0`, and whether `[llm-call]` lines exist in orchestrator.log.

3. MEASURE the runtime LLM directly: `curl` the endpoint's `/v1/models`, then time one tiny completion and one ~4k-token prompt, with and without thinking disabled if the model supports it. Put the numbers in RUNLOG.md. This replaces guessing about timeouts.

4. FAST ORACLE. If `src/patch_replay.py` does not exist, build it as described in CLAUDE.md ("Build the fast oracle first") and prove it two ways on run 8's X509Utils:81 crash: (a) `--scenario` with a scripted `read_source → apply_edit(bounds check on the index at X509Utils.java:81) → test_patch` sequence against the real sidecar — this must come back `validated`, or the pipeline is broken independent of the model; (b) the real runtime LLM. Record both results in RUNLOG.md.

5. Only then start cycles with `/cycle`, one hypothesis each, in the order the triage evidence suggests. Use `/reference` when you want to know how Atlantis or Buttercup solved the same problem; port mechanisms, not code.

Rules that override anything else: one behavioural change per cycle; every claim backed by pasted output; no flags; fail loud; never call a patch validated without PoV replay plus tests. If you need Ashok to do something you cannot (permissions, docker access), ask for exactly one command and wait.

Definition of done is in CLAUDE.md. Report progress as RUNLOG.md entries, not prose.
