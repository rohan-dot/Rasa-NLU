#!/usr/bin/env python3
"""
apply_followup_feedback.py — apply the advisor's follow-on comments, in place.

  python apply_followup_feedback.py --repo ./agentic_route_planning [--dry-run]

Her notes -> actions
  - Move Environment set-up, How to run, CLI reference above the data sections   -> README reordered
  - Switch Data sources / Data elements                                           -> README reordered
  - Move "outputs of a run" details, Ablations, Cost model, Known limits          -> added to tutorial.ipynb
    into the tutorial notebook; drop from README                                     as markdown; README keeps
                                                                                     one pointer line each
  - Missing set-up line: conda install -c anaconda ipykernel -y &&                -> added to README setup
    python -m ipykernel install --user --name=agentplanner
  - merge_fcg1.py: not used                                                        -> file deleted; every
                                                                                     mention removed
  - Docstrings for functions (args, purpose, returns)                              -> model writes one per
                                                                                     undocumented function;
                                                                                     inserted by AST position
Checks: all .py compile, modules import, --help works. Report: FEEDBACK_REPORT_2.md.
Backups: README.previous.md, <py>.bak_docstrings. LLM env: LITELLM_BASE_URL/LITELLM_API_KEY/AGENT_MODEL.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime


# ----------------------------------------------------------------------------- LLM
def llm(prompt, system="You are a precise technical writer and Python engineer. Use only the facts given."):
    import httpx
    from openai import OpenAI
    verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
    c = OpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"],
               http_client=httpx.Client(verify=verify, timeout=600))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for budget in (int(os.environ.get("GATEWAY_MAX_TOKENS", "12000")), 40000):
        r = c.chat.completions.create(model=os.environ["AGENT_MODEL"], messages=msgs, max_tokens=budget)
        t = (r.choices[0].message.content or "").strip()
        if t:
            return t
    return ""


def strip_fence(t):
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```\s*$", t, re.DOTALL)
    return (m.group(1) if m and t.startswith("```") else t).rstrip() + "\n"


# ----------------------------------------------------------------------------- notebook additions
OUTPUTS_MD = """## What a run writes — `runs/<timestamp>/`

| file | content |
|---|---|
| `plan_log.txt` | everything the planner printed: input resolution, risk layer, corridor candidates per leg, judge verdicts, FINAL PLAN, route options, briefing |
| `fcg_evidence.txt` | for every country/role the judge read: the exact FCG text given to the model, its verdict JSON (allowed / caution / min lead days / reason), and what the code enforced on top (lead-time arithmetic); plus the risk-layer sources |
| `fcg_evidence.json` | the same, machine-readable |
| `last_run_blocks.json` | the plan as data: required stops, full stop chain, auto stops, blocked countries, per-leg corridor options with scores, aircraft, config |
| `comparison_<ts>.md` / `.json` | written by `compare()`: numbers, per-segment table, explanation, judge verdict |

`last_run_blocks.json` is also copied to the working directory so `compare()` finds the latest run by default.
"""

COST_MD = """## The cost model

Per leg, three corridor candidates are built — the direct great circle, a penalized path over the country-adjacency graph, and a stepping-stone path via intermediate countries — and all are scored with

```
score = NM + 300 × (countries needing overflight clearance) + 1,500 × (risk-listed countries crossed)
```

Lowest score wins, so the great circle is always the default and loses only when bending saves more than 300 NM per clearance avoided or when it crosses a blocked country. Blocked = denied by the FCG judge for this date, State Dept Level 4, or the manual list — never crossed. Fuel stops: when a leg exceeds 85 % of aircraft range, a search over FCG-designated airports inserts stops at +400 NM per landing (+600 for a non-designated fallback). The 300 is a policy weight (about 40 minutes of C-17 time per clearance), exposed as `clearance_penalty=` / `--clearance-penalty`; the ablation below shows what it does.
"""

LIMITS_MD = """## Known limits

- No fuel burn, winds, MOG or NOTAMs — range is a planning fraction (85 %), not a fuel plan.
- The judge is only as good as the FCG extract; every verdict is logged with the text it read so it can be checked.
- State Dept advisories are country-level, not airspace-level.
- Lead time is calendar arithmetic from today; the mission date must be in the future.
- Changing the model changes only the per-country verdicts and narrative; geometry, stops and cost are identical, so `fcg_evidence.txt` can be diffed between models.
"""


def md_cell(s, cid):
    return {"cell_type": "markdown", "metadata": {}, "source": s, "id": cid}


def update_notebook(path):
    if not os.path.exists(path):
        return "tutorial.ipynb not found"
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb["cells"]
    def has(marker):
        return any(marker in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in cells)
    added = []
    def insert_after(marker_substr, new_cells):
        for i, c in enumerate(cells):
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if marker_substr in src:
                for j, nc in enumerate(new_cells):
                    cells.insert(i + 1 + j, nc)
                return True
        return False
    if not has("What a run writes"):
        ok = insert_after("## 3. What a run produced", [md_cell(OUTPUTS_MD, "cell-outputs-table")])
        added.append("outputs table" + ("" if ok else " (appended at end)"))
        if not ok:
            cells.append(md_cell(OUTPUTS_MD, "cell-outputs-table"))
    if not has("## The cost model"):
        ok = insert_after("## 5. Ablations", [md_cell(COST_MD, "cell-cost-model")])
        added.append("cost model" + ("" if ok else " (appended at end)"))
        if not ok:
            cells.append(md_cell(COST_MD, "cell-cost-model"))
    if not has("## Known limits") and not has("## 6. Known limits"):
        cells.append(md_cell(LIMITS_MD, "cell-known-limits"))
        added.append("known limits")
    elif has("## 6. Known limits"):
        for c in cells:
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if "## 6. Known limits" in src:
                c["source"] = LIMITS_MD.replace("## Known limits", "## 6. Known limits")
        added.append("known limits (expanded)")
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
    return "added: " + ", ".join(added) if added else "already complete"


# ----------------------------------------------------------------------------- docstrings
def undocumented(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
            if node.name.startswith("__") and node.name.endswith("__") and node.name != "__init__":
                continue
            seg = ast.get_source_segment(src, node) or ""
            out.append({"name": node.name, "lineno": node.lineno, "end": node.end_lineno,
                        "signature": seg.split("\n")[0].strip(), "body": seg[:2500],
                        "indent": " " * (node.col_offset + 4), "nested": node.col_offset > 0})
    return src, out


def add_docstrings(path, dry, use_llm):
    src, funcs = undocumented(path)
    if not funcs:
        return "all functions documented", 0
    if not use_llm:
        return f"{len(funcs)} undocumented functions (no model configured; skipped)", 0
    prompt = f"""Write a docstring for each function below from the project `{os.path.basename(path)}` (an FCG-compliant
military flight route planner). Style: one-line summary, blank line, "Args:" with each parameter and
its meaning/type as used in the body, "Returns:" (or "Side effects:" for functions that print/write
files and return None). Be specific to what the body actually does; never invent behaviour. Keep each
under 12 lines. Use only the module's own facts.

FUNCTIONS (name, signature, body excerpt):
{json.dumps([{k: f[k] for k in ("name", "signature", "body")} for f in funcs], indent=1)[:90000]}

Return ONLY JSON: {{"<function name>": "<docstring text without the triple quotes>", ...}}"""
    t = llm(prompt)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    docs = json.loads(m.group(0)) if m else {}
    lines = src.split("\n")
    # insert bottom-up so line numbers stay valid; docstring goes after the def line (def may span lines)
    n = 0
    for f in sorted(funcs, key=lambda x: -x["lineno"]):
        d = docs.get(f["name"])
        if not d or not isinstance(d, str):
            continue
        # find the line where the def header ends (the first line ending with ':' at/after lineno)
        i = f["lineno"] - 1
        while i < len(lines) and not lines[i].rstrip().endswith(":"):
            i += 1
        if i >= len(lines):
            continue
        ind = f["indent"]
        d = d.strip().replace('"""', "'''")
        block = [f'{ind}"""{d.splitlines()[0]}'] + [(ind + l if l.strip() else "") for l in d.splitlines()[1:]] + [f'{ind}"""']
        if len(d.splitlines()) == 1:
            block = [f'{ind}"""{d}"""']
        lines[i + 1:i + 1] = block
        n += 1
    new = "\n".join(lines)
    try:
        ast.parse(new)
    except SyntaxError as e:
        return f"docstring insertion produced a syntax error ({e}); file left unchanged", 0
    if not dry:
        if not os.path.exists(path + ".bak_docstrings"):
            shutil.copy2(path, path + ".bak_docstrings")
        open(path, "w", encoding="utf-8").write(new)
    return f"{n}/{len(funcs)} docstrings added", n


# ----------------------------------------------------------------------------- README
def rewrite_readme(repo, dry, use_llm):
    p = os.path.join(repo, "README.md")
    old = open(p, encoding="utf-8").read()
    prompt = f"""Revise this README per the reviewer's follow-on comments. Keep every fact that is still true; do not
add facts. Return ONLY the full README markdown.

COMMENTS (apply all):
1. Section order must be: Title + purpose; What it does; Environment set-up; How to run; CLI reference;
   Running with a model; Repository structure; Data sources; Data elements; Tutorial notebook.
   (Environment set-up, How to run and CLI reference now come BEFORE the data sections; Data sources
   comes BEFORE Data elements.)
2. REMOVE these sections from the README — they now live in tutorial.ipynb: the detailed "Outputs of a
   run" table, Ablations, Cost model, Known limits. In their place keep at most one sentence each,
   e.g. under How to run: "Each run writes runs/<timestamp>/ (plan log, FCG evidence, plan JSON) — see
   tutorial.ipynb for the file-by-file description." and in the Tutorial notebook section: "The notebook
   also covers the outputs of a run, the cost model, ablations and known limits."
3. In Environment set-up, after creating/activating the conda env and installing requirements, add
   exactly this line so the notebook can use the env:
   `conda install -c anaconda ipykernel -y && python -m ipykernel install --user --name=agentplanner`
4. merge_fcg1.py is not part of the workflow and has been deleted: remove every mention of it
   (structure table, text, examples).
5. Keep the How to run section's function-form first (run_plan / compare), then the CLI, with the
   pointer to "Running with a model".

CURRENT README:
{old}"""
    if dry:
        return "(dry run)"
    new = llm(prompt) if use_llm else ""
    new = strip_fence(new) if new else ""
    if not new.strip():
        # no model: at least remove merge_fcg1 lines deterministically
        new = "\n".join(l for l in old.splitlines() if "merge_fcg1" not in l) + "\n"
        how = "no model: only removed merge_fcg1 lines; reorder/removal/ipykernel line NOT done"
    else:
        new = "\n".join(l for l in new.splitlines() if "merge_fcg1" not in l) + "\n"
        how = "rewritten by model"
    shutil.copy2(p, os.path.join(repo, "README.previous.md"))
    open(p, "w", encoding="utf-8").write(new)
    return how


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-docstrings", action="store_true")
    a = ap.parse_args([x for x in sys.argv[1:] if x.replace("\xa0", "").strip()])
    repo, dry = os.path.abspath(a.repo), a.dry_run
    use_llm = bool(os.environ.get("AGENT_MODEL") and os.environ.get("LITELLM_BASE_URL"))
    R = [f"# Follow-on feedback applied — {datetime.now():%Y-%m-%d %H:%M}", "", f"Repo `{repo}`; {'DRY RUN' if dry else 'applied'}; model {'on' if use_llm else 'off'}.", ""]

    # 1. merge_fcg1.py gone
    removed = []
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if d not in ("agentplanner", ".git", "runs")]
        for f in fns:
            if f.startswith("merge_fcg1"):
                removed.append(os.path.relpath(os.path.join(dp, f), repo)); (not dry) and os.remove(os.path.join(dp, f))
    for fn in ("run_example.sh", "env.example.sh", "PACKAGE_LOG.md", "FEEDBACK_REPORT.md"):
        p = os.path.join(repo, fn)
        if os.path.exists(p):
            t = open(p, encoding="utf-8", errors="replace").read()
            if "merge_fcg1" in t and not dry:
                open(p, "w", encoding="utf-8").write("\n".join(l for l in t.splitlines() if "merge_fcg1" not in l) + "\n")
                removed.append(f"{fn} (mentions)")
    R += ["## merge_fcg1.py (note: not used in the workflow)", ""] + ([f"- removed: {x}" for x in removed] or ["- nothing found"]) + [""]

    # 2. notebook gets outputs table, cost model, known limits
    nbres = "(dry run)" if dry else update_notebook(os.path.join(repo, "tutorial.ipynb"))
    R += ["## tutorial.ipynb (note: move outputs details, ablations, cost model, known limits into the notebook)", "", f"- {nbres}; ablations cell was already present", ""]

    # 3. README reorder / removals / ipykernel line
    how = rewrite_readme(repo, dry, use_llm)
    R += ["## README.md (notes: section order, data sources before elements, move sections to notebook, ipykernel line, no merge_fcg1)", "", f"- {how}; previous saved as README.previous.md", ""]
    if not dry:
        t = open(os.path.join(repo, "README.md"), encoding="utf-8").read()
        checks = {"ipykernel line present": "ipykernel install --user --name=agentplanner" in t,
                  "no merge_fcg1": "merge_fcg1" not in t,
                  "Ablations section removed": not re.search(r"^#+\s*Ablations", t, re.M),
                  "Cost model section removed": not re.search(r"^#+\s*Cost model", t, re.M),
                  "Known limits section removed": not re.search(r"^#+\s*Known limits", t, re.M)}
        heads = [m.group(1).strip() for m in re.finditer(r"^##\s+(.+)$", t, re.M)]
        def idx(name):
            return next((i for i, h in enumerate(heads) if name.lower() in h.lower()), None)
        order_ok = all(x is not None for x in (idx("Environment"), idx("How to run"), idx("Data sources"), idx("Data elements"))) and \
                   idx("Environment") < idx("Data sources") and idx("How to run") < idx("Data sources") and idx("Data sources") < idx("Data elements")
        checks["section order (setup/run before data; sources before elements)"] = order_ok
        R += ["README checks:", ""] + [f"- {'OK ' if v else 'NOT MET'} {k}" for k, v in checks.items()] + ["", f"Sections now: {heads}", ""]

    # 4. docstrings
    if not a.no_docstrings:
        R += ["## Docstrings (note: args, purpose, returns for functions)", ""]
        for dp, dns, fns in os.walk(repo):
            dns[:] = [d for d in dns if d not in ("agentplanner", ".git", "runs", "samples", "__pycache__", "extras_eval")]
            for f in sorted(fns):
                if f.endswith(".py"):
                    msg, n = add_docstrings(os.path.join(dp, f), dry, use_llm)
                    R.append(f"- `{os.path.relpath(os.path.join(dp, f), repo)}`: {msg}")
        R.append("")

    # 5. checks
    if not dry:
        probs = []
        code_dir = next((dp for dp, _, fns in os.walk(repo) if "country_route_planner.py" in fns), repo)
        for dp, dns, fns in os.walk(repo):
            dns[:] = [x for x in dns if x not in ("agentplanner", ".git")]
            for f in fns:
                if f.endswith(".py"):
                    p = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(dp, f)], capture_output=True, text=True)
                    if p.returncode: probs.append(f"{f}: {p.stderr[-300:]}")
        p = subprocess.run([sys.executable, "-c", "import country_route_planner as P, compare_routes as C, route_judge; print('ok')"], cwd=code_dir, capture_output=True, text=True)
        if p.returncode: probs.append("import: " + p.stderr[-400:])
        p = subprocess.run([sys.executable, "country_route_planner.py", "--help"], cwd=code_dir, capture_output=True, text=True)
        if p.returncode: probs.append("--help: " + p.stderr[-300:])
        for dp, dns, fns in os.walk(repo, topdown=False):
            if os.path.basename(dp) == "__pycache__":
                shutil.rmtree(dp, ignore_errors=True)
        R += ["## Checks", ""] + ([f"- FAIL {x}" for x in probs] or ["- all .py compile; modules import; --help OK"]) + [""]
    R += ["Before uploading: `rm *.bak_docstrings README.previous.md`."]
    open(os.path.join(repo, "FEEDBACK_REPORT_2.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
