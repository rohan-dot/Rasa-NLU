# Task 28 — the candidate patch carries discver's work-copy path prefix, so the sidecar cannot apply it

## Goal
Latest run, the reproduced crash's finding: the diff now reaches the builder (`patch.diff (1125 bytes)`), and libCRS
fails to apply it three ways:
`[sidecar] cmd=['/usr/local/bin/libCRS', 'apply-patch-build', '/tmp/discver_patch_cache/validation/<id>/candidate.patch', '.../build_response'] exit=1`
`[patch-turn] X509Fuzzer-9bbf8868fd92 iter=5 SIDECAR FAILURE class=patch_apply_failed … stderr_tail='patch.diff (1125 bytes)
WARN: strict git apply failed, retrying with -C1: Checking patch output/patch_work/X509Fuzzer-9bbf8868fd92/java-project/agentmanager-java8-master/src/mil/disa/common/util/agentmanager/X509Utils.java...
error: output/patch_work/X509Fuzzer-9bbf8868fd92/java-project/agentmanager-java8-master/src/mil/disa/common/util/agentmanager/X509Utils.java: No such file or directory
WARN: git apply -C1 also failed, trying patch binary: … ERROR: all patch methods failed.'`
The file paths inside the patch start with discver's work copy (`output/patch_work/<finding>/`). libCRS applies the
patch inside the project checkout, where the file is `java-project/agentmanager-java8-master/src/mil/.../X509Utils.java`
(the call graph lists project files with exactly that prefix, relative to the project root). Make every path in the
candidate patch relative to the project root, git-style (`--- a/<rel>` / `+++ b/<rel>`, `diff --git a/<rel> b/<rel>`),
and refuse to send a patch whose paths do not exist under the project root. Only the diff producer changes; the three
sidecar invocations stay byte-for-byte untouched.

## Study first
- repo: `_diff_from_copy` (main.py ~:379–398, as changed by task 25) and the code that writes `candidate.patch`;
  how the work copy under `output/patch_work/<finding>/` is created and from which root (the project checkout —
  `/opt/target/src` in the container — or a subdirectory of it); the diff argv (which two directories, from which cwd —
  `diff -ruN` puts the argument paths verbatim into the `---`/`+++` headers, which is where the prefix comes from).
- results: `grep -nE "apply-patch-build|patch_apply_failed|Checking patch|No such file|candidate.patch|patch_work" latest/orchestrator.log | head -20`
  — quote the apply error and the work-copy path; note the project-relative path the call graph uses for the same file.
  `list_files` on `latest/patches/` — if an attempted patch/diff file is there, read its `---` / `+++` headers: that is
  the exact text libCRS rejected, and the fixed producer must turn it into `a/<project-relative>` / `b/<project-relative>`.

## Change (src/ — the diff producer only)
- Produce the diff with paths relative to the project root: run diff from the common parent with relative arguments,
  or post-process every `---` / `+++` / `diff` header: strip the original-root and work-copy-root prefixes, emit
  `--- a/<rel>` and `+++ b/<rel>` (`/dev/null` for added/deleted files), and a `diff --git a/<rel> b/<rel>` header per
  file, so both `git apply` (-p1) and `patch -p1` accept it.
- Before writing `candidate.patch`: every `<rel>` must exist under the ORIGINAL project root (the tree the sidecar
  applies to); otherwise log `[patch] candidate patch path not under project root: <path> (root=<root>)`, classify the
  attempt `diff_mismatch`, and do not call the sidecar with it.
- Log once per candidate: `[patch] candidate patch: <n> file(s), paths relative to <project root>, first=<rel>`.

## Tests to add (tests/test_patch_paths_relative.py) — temp trees, real `diff`, `git apply --check` or `patch -p1 --dry-run`
- original tree `root/java-project/x/src/A.java` + work copy with one changed line → the patch headers are
  `--- a/java-project/x/src/A.java` / `+++ b/java-project/x/src/A.java`; no `output/patch_work` anywhere in the patch.
- `git apply --check` in a fresh copy of the original tree succeeds (fallback `patch -p1 --dry-run` if git is absent).
- a changed file outside the project root → `diff_mismatch`, no sidecar call, the log line names path and root.
- an added file → `--- /dev/null` / `+++ b/<rel>` and the apply check passes.
- the sidecar invocation strings are byte-identical before and after (snapshot from the current code).

## Acceptance
```
python -m pytest -q tests/test_patch_paths_relative.py
```

## Do not touch
`build()`'s sidecar calls, `_build_sidecar`, the three sidecar invocations, crash_dedup._get_crash_signature; anything outside src/ and tests/.
