#!/usr/bin/env python3
"""
transition_agent.py — read a "transition package" (a template repo that shows how
delivered code must be structured and documented), learn its conventions, and
re-package the Agentic Route Planning code in that same shape.

  python transition_agent.py --template ../transition_package --source ./agentic_route_planning
  python transition_agent.py --template ../transition_package --source ./agentic_route_planning --dry-run
  python transition_agent.py ... --out ./agentic_route_planning_transitioned --rounds 2

Guarantees
  * READ-ONLY on --template and --source. Nothing there is modified or deleted.
  * Everything is written to --out (default <source>_transitioned). If --out
    exists it is NOT emptied; files are only added/overwritten there.
  * Every LLM-written file gets a header/footer comment marking it as generated,
    and anything the model could not fill from evidence is left as a TODO line
    and listed in TRANSITION_REPORT.md for a human.
  * Code is copied byte-for-byte unless the template requires a change the
    model can justify (e.g. a package __init__, a CLI entry point, a logging
    convention); each such change is diffed in the report.

LLM backend: same env as the planner — LITELLM_BASE_URL / LITELLM_API_KEY /
AGENT_MODEL (gateway), or VLLM_URL / VLLM_MODEL (local server).

Steps (each saved under <out>/_transition/ so you can inspect or resume):
  1 inventory   walk both trees -> file lists + text contents (size-capped)
  2 analyze     model: what the template requires (layout, docs, setup, env,
                tests, CI, naming, headers) -> conventions.json + conventions.md
  3 plan        model: source file -> destination path; files to create; files
                to adapt; with a reason for each -> plan.json
  4 execute     copy / create / adapt; model writes each new/adapted file with
                the template's matching file as the style example
  5 verify      py_compile all .py; required files present; rounds of
                model-driven fixes for anything missing or broken
  6 report      TRANSITION_REPORT.md: mapping table, generated files, diffs,
                TODOs for a human
"""

import argparse
import difflib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from datetime import datetime

# ----------------------------------------------------------------------------- LLM
_GW = {"client": None, "model": None}
LLM_TIMEOUT = 300
MAX_TOKENS = int(os.environ.get("GATEWAY_MAX_TOKENS", "8000"))


def llm_backend():
    if os.environ.get("LLM_BACKEND"):
        return os.environ["LLM_BACKEND"]
    return "gateway" if os.environ.get("AGENT_MODEL") else "vllm"


def llm_chat(prompt, system="You are a meticulous software packaging engineer. Follow output "
                            "format instructions exactly. Never invent facts about the code; "
                            "when unsure, write a TODO line instead."):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if llm_backend() == "gateway":
        if _GW["client"] is None:
            import httpx
            from openai import OpenAI
            base, key, model = (os.environ.get("LITELLM_BASE_URL"), os.environ.get("LITELLM_API_KEY"),
                                os.environ.get("AGENT_MODEL"))
            if not (base and key and model):
                sys.exit("[FATAL] set LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL")
            verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
            _GW["client"] = OpenAI(base_url=base, api_key=key,
                                   http_client=httpx.Client(verify=verify, timeout=LLM_TIMEOUT))
            _GW["model"] = model
            print(f"[llm] gateway model {model}")
        for budget in (MAX_TOKENS, MAX_TOKENS * 4):
            r = _GW["client"].chat.completions.create(model=_GW["model"], messages=messages,
                                                      max_tokens=budget)
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
            print(f"[llm] empty reply at budget {budget}; retrying larger")
        return ""
    import requests
    url = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
    model = os.environ.get("VLLM_MODEL", "default")
    r = requests.post(url, json={"model": model, "messages": messages, "max_tokens": 4000},
                      timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_json(prompt, system=None):
    txt = llm_chat(prompt, system) if system else llm_chat(prompt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in reply: {txt[:300]!r}")
    return json.loads(m.group(0))


def llm_file(prompt):
    """Ask for a whole file; accept fenced or bare content."""
    txt = llm_chat(prompt)
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```\s*$", txt, re.DOTALL)
    return (m.group(1) if m else txt).rstrip() + "\n"


# ----------------------------------------------------------------------------- inventory
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv", "venv",
             "runs", "eval_results", ".idea", ".vscode", "dist", "build", ".tox", ".eggs"}
TEXT_EXT = {".py", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".sh",
            ".bash", ".env", ".example", ".in", ".csv", ".gitignore", ".dockerignore", ".license",
            "", ".conf", ".mk", ".bat", ".ps1", ".ipynb", ".html", ".j2", ".tmpl", ".template"}
PRIORITY = ("readme", "contributing", "setup", "pyproject", "requirements", "environment", "makefile",
            "dockerfile", "docker-compose", "license", ".env", "config", "changelog", "install",
            "usage", "docs/", "doc/", "tests/", "test_", "ci", ".github/", ".gitlab-ci", "tox",
            "__init__", "cli", "main", "conftest", "manifest", "version")
BIG_DATA_EXT = {".csv", ".geojson", ".json", ".parquet", ".pkl", ".npy", ".zip"}


def walk(root, data_cap_mb=5):
    """List every file (relative path, size, is_text) and load text under caps."""
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in sorted(dns) if d not in SKIP_DIRS]
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1].lower()
            is_text = ext in TEXT_EXT or fn.lower() in ("makefile", "dockerfile", "license", "notice")
            files.append({"path": rel, "size": size, "ext": ext, "is_text": is_text,
                          "is_data": ext in BIG_DATA_EXT and size > 200_000})
    return files


def read_text(root, rel, cap):
    try:
        with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as f:
            t = f.read(cap + 1)
        return t[:cap] + ("\n... [truncated]" if len(t) > cap else "")
    except Exception as e:
        return f"[unreadable: {e}]"


def tree_text(files):
    return "\n".join(f"{f['path']}  ({f['size']:,} B)" for f in files)


def pick_context(root, files, total_cap, per_file_cap):
    """Highest-value text files first (README, setup, config, docs, tests, small .py)."""
    def score(f):
        p = f["path"].lower()
        s = 0
        for i, key in enumerate(PRIORITY):
            if key in p:
                s += 100 - i
        if f["ext"] == ".py":
            s += 20 - min(f["size"] // 5000, 15)
        if f["is_data"]:
            s -= 1000
        return -s
    out, used = [], 0
    for f in sorted([f for f in files if f["is_text"] and not f["is_data"]], key=score):
        if used >= total_cap:
            break
        cap = min(per_file_cap, total_cap - used)
        t = read_text(root, f["path"], cap)
        out.append(f"===== {f['path']} =====\n{t}")
        used += len(t)
    return "\n\n".join(out)


# ----------------------------------------------------------------------------- exercise the code
def default_matrix(missions, aircraft, date, human):
    """Variations the agent runs to learn and verify how the project is used.
    Each row: name, argv (relative to the source dir), what it demonstrates."""
    m1 = missions[0]
    rows = [
        {"name": "help_planner", "argv": ["country_route_planner.py", "--help"], "shows": "CLI reference"},
        {"name": "help_compare", "argv": ["compare_routes.py", "--help"], "shows": "comparison CLI"},
        {"name": "list_airports", "argv": ["country_route_planner.py", "USA", "--list", "--date", date, "--no-llm"],
         "shows": "designated-airport lookup"},
        {"name": "plan_basic", "argv": ["country_route_planner.py", m1, "--date", date, "--aircraft", aircraft],
         "shows": "minimal plan, LLM judge"},
        {"name": "plan_full", "argv": ["country_route_planner.py", m1, "--date", date, "--aircraft", aircraft,
                                       "--auto-stops", "--risk", "--alternatives", "3"],
         "shows": "auto fuel stops, risk feed, route options"},
        {"name": "plan_no_llm", "argv": ["country_route_planner.py", m1, "--date", date, "--aircraft", aircraft,
                                         "--auto-stops", "--no-llm"], "shows": "geometry only, no model"},
        {"name": "plan_penalty_800", "argv": ["country_route_planner.py", m1, "--date", date, "--aircraft", aircraft,
                                              "--auto-stops", "--clearance-penalty", "800", "--alternatives", "3"],
         "shows": "policy knob: clearance penalty"},
    ]
    for i, m in enumerate(missions[1:], 2):
        rows.append({"name": f"plan_mission{i}", "argv": ["country_route_planner.py", m, "--date", date,
                                                        "--aircraft", aircraft, "--auto-stops", "--risk"],
                     "shows": "plain-English itinerary parsing" if re.search(r"\b(via|stop|round|back)\b", m, re.I)
                              else "another mission"})
    if human:
        rows.append({"name": "compare_human", "argv": ["compare_routes.py", "--human", human, "--runs", "1"],
                     "shows": "human vs planner comparison + LLM judge"})
    return rows


def run_matrix(src_root, rows, work, timeout):
    """Run each variation in the SOURCE dir (it only writes under runs/), capture
    everything, copy the run folder into work/runs/<name>/."""
    results = []
    rdir = os.path.join(work, "runs")
    os.makedirs(rdir, exist_ok=True)
    before = set(os.listdir(os.path.join(src_root, "runs"))) if os.path.isdir(os.path.join(src_root, "runs")) else set()
    for row in rows:
        t0 = datetime.now()
        try:
            p = subprocess.run([sys.executable] + row["argv"], cwd=src_root, capture_output=True,
                               text=True, timeout=timeout)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -1, (e.stdout or ""), f"TIMEOUT after {timeout}s"
        secs = (datetime.now() - t0).total_seconds()
        dest = os.path.join(rdir, row["name"])
        os.makedirs(dest, exist_ok=True)
        open(os.path.join(dest, "stdout.txt"), "w").write(out)
        open(os.path.join(dest, "stderr.txt"), "w").write(err)
        # new run folder(s) produced by the planner -> copy next to the stdout
        produced = []
        rp = os.path.join(src_root, "runs")
        if os.path.isdir(rp):
            new = sorted(set(os.listdir(rp)) - before)
            for d in new:
                shutil.copytree(os.path.join(rp, d), os.path.join(dest, d), dirs_exist_ok=True)
                produced.append(d)
            before |= set(new)
        files = []
        for d in produced:
            for dp, _, fns in os.walk(os.path.join(dest, d)):
                files += [os.path.relpath(os.path.join(dp, f), dest) for f in fns]
        tail = "\n".join(out.strip().splitlines()[-40:])
        head = "\n".join(out.strip().splitlines()[:25])
        res = {**row, "cmd": "python " + " ".join(sh_quote(a) for a in row["argv"]), "returncode": rc,
               "seconds": round(secs, 1), "ok": rc == 0, "stdout_head": head, "stdout_tail": tail,
               "stderr_tail": "\n".join(err.strip().splitlines()[-15:]), "produced_files": files[:40]}
        results.append(res)
        print(f"[run] {row['name']:18} rc={rc} {secs:6.1f}s  {row['shows']}")
    json.dump(results, open(os.path.join(work, "run_results.json"), "w"), indent=1)
    return results


def sh_quote(a):
    return f'"{a}"' if (" " in a or not a) else a


def relevance(src_root, S):
    """Static view of what the delivered project actually needs: entry points,
    local modules reachable from them, data files referenced by reachable code."""
    pyfiles = {f["path"]: read_text(src_root, f["path"], 400_000) for f in S if f["ext"] == ".py"}
    mods = {os.path.splitext(os.path.basename(p))[0]: p for p in pyfiles}
    def imports(text):
        out = set()
        for m in re.finditer(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", text, re.M):
            name = (m.group(1) or m.group(2)).split(".")[0]
            if name in mods:
                out.add(name)
        return out
    entry = [p for p, t in pyfiles.items() if "__main__" in t and not p.startswith("extras")]
    reach, stack = set(), [os.path.splitext(os.path.basename(p))[0] for p in entry]
    while stack:
        m = stack.pop()
        if m in reach:
            continue
        reach.add(m)
        stack += list(imports(pyfiles[mods[m]]) - reach)
    reachable_files = sorted(mods[m] for m in reach)
    data_in, data_out = set(), set()
    WRITE = re.compile(r"""(["']w["']|json\.dump\(|to_csv\(|\.write\(|shutil\.copy|open\([^)]*,\s*["']a["'])""")
    for m in reach:
        text = pyfiles[mods[m]]
        for line in text.splitlines():
            for lit in re.findall(r"[\"']([\w./-]+\.(?:csv|geojson|json|txt|md|yaml|yml))[\"']", line):
                (data_out if WRITE.search(line) else data_in).add(lit)
    data_in -= data_out
    present_data = sorted(f["path"] for f in S if f["path"] in data_in or os.path.basename(f["path"]) in data_in)
    missing_data = sorted(d for d in data_in if not any(f["path"] == d or os.path.basename(f["path"]) == d for f in S))
    unreachable_py = sorted(p for p in pyfiles if p not in reachable_files)
    return {"entry_points": entry, "reachable_modules": reachable_files, "unreachable_python": unreachable_py,
            "data_files_shipped": present_data, "data_files_user_supplies": missing_data,
            "files_the_code_writes": sorted(data_out)}


# ----------------------------------------------------------------------------- steps
def step_analyze(T, tctx, work):
    prompt = f"""You are given a TRANSITION PACKAGE: a template repository that defines how code
must be delivered to a receiving team (layout, documentation, environment setup, tests, CI,
naming, headers, licensing, versioning, run instructions).

FILE TREE:
{tree_text(T)}

FILE CONTENTS (highest-value files, possibly truncated):
{tctx}

IMPORTANT: the template is a DIFFERENT project. We will deliver an unrelated project (a flight
route planner) in the SAME FORM. Extract the NORMS only — how things are organised, documented,
installed, configured, run, tested, versioned — never the template's own subject matter, names,
or content. Start from the README: its section order, tone, how "how to run" is written (commands,
Makefile targets, scripts, notebooks?), how environment setup is described, how inputs/outputs and
configuration are documented. Be concrete and cite the template file that shows each norm.
Return ONLY JSON:
{{"summary": "<3-5 sentences: what form this package expects a delivered project to take>",
  "readme_outline": [{{"section": "<heading exactly as the template styles it>", "what_goes_here": "...", "template_example_excerpt": "<1-3 lines copied from the template README showing the style>"}}],
  "how_to_run_style": "<how the template tells the reader to run things: e.g. numbered shell blocks, make targets, a run.sh, a CLI table, notebooks>",
  "layout": [{{"path": "<dir or file pattern>", "purpose": "...", "required": true/false, "example": "<template file>"}}],
  "documentation": {{"files": ["README.md", ...], "readme_sections": ["...", ...],
                     "docstring_style": "...", "other_docs": ["..."]}},
  "environment_setup": {{"mechanism": "<requirements.txt | pyproject | conda env.yml | setup.py | Makefile | Dockerfile | ...>",
                         "files": ["..."], "python_version": "...", "install_command": "...", "config_mechanism": "<env vars | .env | config.yaml | ...>"}},
  "entry_points": {{"style": "<argparse scripts | console_scripts | Makefile targets | ...>", "example": "..."}},
  "testing": {{"framework": "...", "location": "...", "required": true/false}},
  "ci_or_automation": ["..."],
  "code_conventions": {{"file_header": "<required header text or null>", "logging": "...", "naming": "...", "package_structure": "<flat scripts | src/ package | ...>"}},
  "data_handling": "<where data files/inputs live, what is excluded from the repo>",
  "required_files_checklist": ["<every file/dir a conforming project MUST have>"],
  "unclear": ["<anything the template does not make clear; a human should confirm>"]}}"""
    conv = llm_json(prompt)
    json.dump(conv, open(os.path.join(work, "conventions.json"), "w"), indent=1)
    md = ["# Transition package conventions (model's reading)", "", conv.get("summary", ""), "",
          "## Required files", ""] + [f"- `{x}`" for x in conv.get("required_files_checklist", [])] + [
          "", "## Layout", ""] + [f"- `{l.get('path')}` — {l.get('purpose')} "
                                   f"({'required' if l.get('required') else 'optional'}; e.g. `{l.get('example')}`)"
                                   for l in conv.get("layout", [])] + [
          "", "## README outline the delivered project must follow", ""] + [
              f"- **{s.get('section')}** — {s.get('what_goes_here')}  \n  _e.g._ `{s.get('template_example_excerpt','')}`"
              for s in conv.get("readme_outline", [])] + [
          "", f"How-to-run style: {conv.get('how_to_run_style')}",
          "", "## Documentation", "", f"- files: {conv.get('documentation', {}).get('files')}",
          f"- README sections: {conv.get('documentation', {}).get('readme_sections')}",
          f"- docstrings: {conv.get('documentation', {}).get('docstring_style')}",
          "", "## Environment / setup", "", json.dumps(conv.get("environment_setup", {}), indent=1),
          "", "## Entry points", "", json.dumps(conv.get("entry_points", {}), indent=1),
          "", "## Testing", "", json.dumps(conv.get("testing", {}), indent=1),
          "", "## CI / automation", ""] + [f"- {x}" for x in conv.get("ci_or_automation", [])] + [
          "", "## Code conventions", "", json.dumps(conv.get("code_conventions", {}), indent=1),
          "", "## Data handling", "", str(conv.get("data_handling", "")),
          "", "## Unclear — confirm with a human", ""] + [f"- {x}" for x in conv.get("unclear", [])]
    open(os.path.join(work, "conventions.md"), "w").write("\n".join(md) + "\n")
    print(f"[analyze] {conv.get('summary', '')[:200]}")
    return conv


def evidence_text(results, rel):
    if not results:
        return "(the code was not executed; no evidence)"
    L = ["RUNS PERFORMED (only commands with ok=true may appear in documentation as working examples):"]
    for r in results:
        L.append(f"- {r['name']}: ok={r['ok']} rc={r['returncode']} {r['seconds']}s  cmd: {r['cmd']}  shows: {r['shows']}")
        if r["ok"]:
            L.append(f"    output tail:\n      " + r["stdout_tail"][-1500:].replace("\n", "\n      "))
            if r["produced_files"]:
                L.append(f"    files produced: {r['produced_files'][:12]}")
        else:
            L.append(f"    stderr: {r['stderr_tail'][-600:]}")
    L += ["", "RELEVANCE (static analysis of the source):", json.dumps(rel, indent=1)]
    return "\n".join(L)


def step_plan(conv, S, sctx, T, work, project_name, results=None, rel=None):
    prompt = f"""CONVENTIONS the delivered project must follow (from the transition package):
{json.dumps(conv, indent=1)}

EXECUTION EVIDENCE — the agent ran the source project; this is what actually works and what
each file is for. Use it to decide relevance: ship what the entry points need (reachable modules,
shipped data files), document the user-supplied data files, and treat unreachable code as
optional/extras or drop it unless the template wants such material (e.g. tests, notebooks).
{evidence_text(results, rel)}

TEMPLATE FILE TREE (for reference paths):
{tree_text(T)}

SOURCE PROJECT to convert — name: {project_name}
FILE TREE:
{tree_text(S)}

SOURCE FILE CONTENTS (truncated):
{sctx}

Produce a conversion plan. The template is an unrelated project: mirror its FORM (layout, file
names, doc structure, setup/run mechanism), never its content. Rules:
- Every source file must appear exactly once in "copy", "adapt" or "drop" (drop only for
  caches, scratch, run outputs, or duplicates; give the reason).
- "copy": byte-identical copy to a destination path that fits the template layout.
- "adapt": the file is needed but must change to fit conventions (e.g. README rewritten in the
  template's section order, a script gaining the required header, a config file in the template's
  mechanism). Say exactly what changes and which template file is the style example.
- "create": files the template requires that the source lacks (setup/pyproject, env file, docs,
  tests skeleton, CI, LICENSE/NOTICE, CHANGELOG, Makefile...). Name the template file to imitate.
- Data inputs the user supplies (FCG extract, OurAirports, Natural Earth, mapping) are NOT in the
  source; plan where they go and how the docs/ignore files refer to them.
- Keep the project runnable: the three main scripts (country_route_planner.py, compare_routes.py,
  route_judge.py) must still import each other; if you move them into a package, say how.
Return ONLY JSON:
{{"destination_layout_note": "...",
  "copy":   [{{"src": "...", "dst": "...", "reason": "..."}}],
  "adapt":  [{{"src": "...", "dst": "...", "changes": "...", "style_example": "<template path or null>"}}],
  "create": [{{"dst": "...", "purpose": "...", "style_example": "<template path or null>", "needs_from_human": ["..."]}}],
  "drop":   [{{"src": "...", "reason": "..."}}],
  "examples_dir": "<where sample runs/outputs belong in the template's layout, e.g. examples/ or docs/examples/; null if the template has no such place>",
  "post_steps": ["<commands or checks a human should run after>"]}}"""
    plan = llm_json(prompt)
    json.dump(plan, open(os.path.join(work, "plan.json"), "w"), indent=1)
    # completeness check: every source file accounted for
    planned = {x["src"] for k in ("copy", "adapt", "drop") for x in plan.get(k, [])}
    missing = [f["path"] for f in S if f["path"] not in planned]
    for p in missing:                      # default: copy at same relative path
        plan.setdefault("copy", []).append({"src": p, "dst": p, "reason": "not in plan; copied as-is"})
    print(f"[plan] copy {len(plan.get('copy', []))}, adapt {len(plan.get('adapt', []))}, "
          f"create {len(plan.get('create', []))}, drop {len(plan.get('drop', []))}"
          + (f"; {len(missing)} unplanned files copied as-is" if missing else ""))
    return plan


GEN_MARK = "generated by transition_agent.py"


def mark(content, dst):
    """Header comment in the right syntax so generated files are identifiable."""
    ext = os.path.splitext(dst)[1].lower()
    stamp = f"{GEN_MARK} on {datetime.now():%Y-%m-%d} — review before release"
    if ext == ".py" or ext in (".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".txt", ".in") \
            or os.path.basename(dst).lower() in ("makefile", "dockerfile", ".gitignore"):
        if content.startswith("#!"):
            first, rest = content.split("\n", 1)
            return f"{first}\n# {stamp}\n{rest}"
        return f"# {stamp}\n{content}"
    if ext in (".md", ".rst", ".html"):
        return content.rstrip() + f"\n\n<!-- {stamp} -->\n"
    return content


def step_execute(plan, conv, src_root, tpl_root, out_root, S, T, work, dry, results=None, rel=None):
    log = {"copied": [], "adapted": [], "created": [], "dropped": [], "diffs": {}, "todos": [], "examples": []}
    ev = evidence_text(results, rel)[:20000]
    # sample runs -> examples dir (trimmed), so docs can point at real outputs
    ex_dir = plan.get("examples_dir")
    if ex_dir and results and not dry:
        for r in results:
            if not r["ok"]:
                continue
            d = os.path.join(out_root, ex_dir, r["name"])
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "command.txt"), "w").write(r["cmd"] + "\n")
            srcd = os.path.join(work, "runs", r["name"])
            for dp, _, fns in os.walk(srcd):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    if os.path.getsize(p) <= 2_000_000:
                        rp = os.path.relpath(p, srcd)
                        os.makedirs(os.path.dirname(os.path.join(d, rp)) or d, exist_ok=True)
                        shutil.copy2(p, os.path.join(d, rp))
            log["examples"].append(os.path.join(ex_dir, r["name"]))
        print(f"[examples] {len(log['examples'])} sample runs -> {ex_dir}/")
    def ensure(dst):
        os.makedirs(os.path.dirname(os.path.join(out_root, dst)) or out_root, exist_ok=True)
    for it in plan.get("copy", []):
        if not dry:
            ensure(it["dst"])
            shutil.copy2(os.path.join(src_root, it["src"]), os.path.join(out_root, it["dst"]))
        log["copied"].append(it)
    for it in plan.get("drop", []):
        log["dropped"].append(it)

    def example(path):
        if not path:
            return ""
        hit = next((f for f in T if f["path"] == path), None)
        return f"\nTEMPLATE STYLE EXAMPLE ({path}):\n{read_text(tpl_root, path, 12000)}\n" if hit else ""

    for it in plan.get("adapt", []):
        src_text = read_text(src_root, it["src"], 60000)
        prompt = f"""Rewrite this file so it follows the transition package conventions.
CONVENTIONS: {json.dumps(conv.get('code_conventions', {}))}; documentation: {json.dumps(conv.get('documentation', {}))}
REQUIRED CHANGES: {it.get('changes')}
{example(it.get('style_example'))}
SOURCE FILE ({it['src']}):
{src_text}

EXECUTION EVIDENCE (every command you document must be one that ran ok; quote real output
snippets and real produced file names; sample runs live under {plan.get('examples_dir')}):
{ev}

Rules: preserve ALL functionality and ALL factual content; change only what the conventions
require; do not shorten documentation — restructure it into the template's outline
({json.dumps(conv.get('readme_outline', []))[:3000]}) and how-to-run style
({conv.get('how_to_run_style')}); the style example shows FORM only — never copy its subject
matter, names, commands or examples; for anything you cannot know, write a line starting with
'TODO(transition):'. Return ONLY the complete new file content in one fenced code block."""
        new = llm_file(prompt) if not dry else src_text
        new = mark(new, it["dst"])
        if not dry:
            ensure(it["dst"])
            open(os.path.join(out_root, it["dst"]), "w", encoding="utf-8").write(new)
        diff = "".join(difflib.unified_diff(src_text.splitlines(True), new.splitlines(True),
                                            it["src"], it["dst"]))
        log["diffs"][it["dst"]] = diff[:20000]
        log["adapted"].append(it)
        log["todos"] += [f"{it['dst']}: {l.strip()}" for l in new.splitlines() if "TODO(transition)" in l]
        print(f"[adapt] {it['src']} -> {it['dst']}")

    src_readme = next((f["path"] for f in S if f["path"].lower().startswith("readme")), None)
    src_summary = read_text(src_root, src_readme, 15000) if src_readme else "(no README)"
    for it in plan.get("create", []):
        prompt = f"""Create the file `{it['dst']}` for the delivered project, following the transition
package conventions.
PURPOSE: {it.get('purpose')}
CONVENTIONS: {json.dumps(conv, indent=0)[:6000]}
{example(it.get('style_example'))}
PROJECT FACTS (from its README — use only these, do not invent):
{src_summary}
PROJECT FILE LIST: {[f['path'] for f in S]}
EXECUTION EVIDENCE (document only commands that ran ok; cite real outputs; sample runs are under
{plan.get('examples_dir')}):
{ev}
Things the human must supply (leave as 'TODO(transition): ...' lines): {it.get('needs_from_human')}

If this is the README or another document: follow the template's outline exactly
({json.dumps(conv.get('readme_outline', []))[:3000]}) and its how-to-run style
({conv.get('how_to_run_style')}), but every sentence must be about THIS project; never carry over
the template's subject matter, names, commands or examples.
Return ONLY the complete file content in one fenced code block."""
        new = llm_file(prompt) if not dry else f"# dry run placeholder for {it['dst']}\n"
        new = mark(new, it["dst"])
        if not dry:
            ensure(it["dst"])
            open(os.path.join(out_root, it["dst"]), "w", encoding="utf-8").write(new)
        log["created"].append(it)
        log["todos"] += [f"{it['dst']}: {l.strip()}" for l in new.splitlines() if "TODO(transition)" in l]
        print(f"[create] {it['dst']}")
    json.dump(log, open(os.path.join(work, "execute_log.json"), "w"), indent=1)
    return log


def step_verify(conv, out_root, work, rounds, dry):
    problems_all = []
    for rnd in range(1, rounds + 1):
        problems = []
        for dp, dns, fns in os.walk(out_root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and d != "_transition"]
            for fn in fns:
                if fn.endswith(".py"):
                    p = os.path.join(dp, fn)
                    try:
                        py_compile.compile(p, doraise=True)
                    except py_compile.PyCompileError as e:
                        problems.append({"file": os.path.relpath(p, out_root), "problem": str(e)[:800]})
        have = set()
        for dp, dns, fns in os.walk(out_root):
            dns[:] = [d for d in dns if d != "_transition"]
            for fn in fns:
                have.add(os.path.relpath(os.path.join(dp, fn), out_root).replace(os.sep, "/"))
        for req in conv.get("required_files_checklist", []):
            r = req.strip("/").rstrip("*")
            if not any(h == r or h.startswith(r + "/") or h.startswith(r) for h in have):
                problems.append({"file": req, "problem": "required by template but missing"})
        # main scripts still import each other?
        try:
            r = subprocess.run([sys.executable, "-c",
                                "import importlib,sys; sys.path.insert(0,'.'); "
                                "[importlib.import_module(m) for m in ('route_judge',) if __import__('os').path.exists(m+'.py')]"],
                               cwd=out_root, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                problems.append({"file": "route_judge.py", "problem": r.stderr[-800:]})
        except Exception as e:
            problems.append({"file": "(import check)", "problem": str(e)[:300]})
        print(f"[verify] round {rnd}: {len(problems)} problem(s)")
        problems_all = problems
        if not problems or dry:
            break
        # ask the model to fix each problem it can (missing files / syntax), never delete
        for pr in problems:
            if pr["problem"].startswith("required by template"):
                dst = pr["file"].strip("/")
                if dst.endswith("/") or "." not in os.path.basename(dst):
                    os.makedirs(os.path.join(out_root, dst), exist_ok=True)
                    open(os.path.join(out_root, dst, ".gitkeep"), "a").close()
                    continue
                new = llm_file(f"Create `{dst}` required by the transition package conventions "
                               f"{json.dumps(conv, indent=0)[:4000]}. Use TODO(transition): lines for "
                               f"anything unknown. Return ONLY the file content in one fenced block.")
                os.makedirs(os.path.dirname(os.path.join(out_root, dst)) or out_root, exist_ok=True)
                open(os.path.join(out_root, dst), "w").write(mark(new, dst))
            else:
                p = os.path.join(out_root, pr["file"])
                if not os.path.exists(p):
                    continue
                cur = open(p, encoding="utf-8", errors="replace").read()
                new = llm_file(f"This file fails to compile/import:\n{pr['problem']}\n\nFILE ({pr['file']}):\n{cur}\n\n"
                               f"Fix the problem with the smallest change. Return ONLY the full file in one fenced block.")
                open(p, "w", encoding="utf-8").write(new)
    json.dump(problems_all, open(os.path.join(work, "verify.json"), "w"), indent=1)
    return problems_all


def step_report(conv, plan, log, problems, out_root, src_root, tpl_root, work, results=None):
    L = [f"# Transition report — {os.path.basename(os.path.abspath(src_root))} -> "
         f"{os.path.basename(os.path.abspath(out_root))}", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Template: `{tpl_root}`. Source untouched.", "",
         "## What the template requires", "", conv.get("summary", ""), "",
         f"Required files: {', '.join('`%s`' % x for x in conv.get('required_files_checklist', []))}", "",
         f"Layout note: {plan.get('destination_layout_note', '')}", "",
         "## File mapping", "", "| action | source | destination | note |", "|---|---|---|---|"]
    for it in log["copied"]:
        L.append(f"| copy | `{it['src']}` | `{it['dst']}` | {it.get('reason', '')} |")
    for it in log["adapted"]:
        L.append(f"| adapt | `{it['src']}` | `{it['dst']}` | {it.get('changes', '')} |")
    for it in log["created"]:
        L.append(f"| create | — | `{it['dst']}` | {it.get('purpose', '')} |")
    for it in log["dropped"]:
        L.append(f"| drop | `{it['src']}` | — | {it.get('reason', '')} |")
    if results:
        L += ["", "## Runs performed to learn and verify usage", "",
              "| run | ok | s | command | shows |", "|---|---|---|---|---|"]
        L += [f"| {r['name']} | {'yes' if r['ok'] else 'NO'} | {r['seconds']} | `{r['cmd']}` | {r['shows']} |"
              for r in results]
        if log.get("examples"):
            L += ["", f"Sample outputs copied to: " + ", ".join(f"`{e}`" for e in log["examples"])]
    L += ["", "## Verification", ""]
    L += ([f"- {p['file']}: {p['problem'][:200]}" for p in problems] or ["- all .py compile; required files present"])
    L += ["", "## TODO(transition) — needs a human", ""]
    L += ([f"- {t}" for t in log["todos"]] or ["- none"])
    L += ["", "## Unclear in the template (confirm with the receiving team)", ""]
    L += ([f"- {x}" for x in conv.get("unclear", [])] or ["- none"])
    L += ["", "## Post-steps suggested by the plan", ""]
    L += ([f"- {x}" for x in plan.get("post_steps", [])] or ["- none"])
    L += ["", "## Diffs for adapted files", ""]
    for dst, d in log["diffs"].items():
        L += [f"### {dst}", "", "```diff", d or "(no textual change)", "```", ""]
    L += ["", f"_Working files: `{os.path.relpath(work, out_root)}/` (conventions.md, plan.json, "
              f"execute_log.json, verify.json)._"]
    path = os.path.join(out_root, "TRANSITION_REPORT.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[report] {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="path to the transition package (read-only)")
    ap.add_argument("--source", required=True, help="path to agentic_route_planning (read-only)")
    ap.add_argument("--out", default=None, help="destination (default <source>_transitioned)")
    ap.add_argument("--name", default="agentic_route_planning", help="project name to use")
    ap.add_argument("--rounds", type=int, default=2, help="verify/fix rounds")
    ap.add_argument("--dry-run", action="store_true", help="analyze + plan only; write nothing but the plan")
    ap.add_argument("--ctx-chars", type=int, default=120_000, help="max template/source text sent to the model")
    ap.add_argument("--run", action="store_true",
                    help="execute the source project in several variations first (planner, flags, compare) "
                         "and use the real outputs as evidence for relevance and documentation")
    ap.add_argument("--mission", action="append", default=None,
                    help='mission text for the runs (repeatable). Default: "KSUU to FKKD" and '
                         '"KSUU to FKKD via stops in DGAA"')
    ap.add_argument("--aircraft", default="C-17")
    ap.add_argument("--date", default=None, help="mission date for the runs (default: today + 60 days)")
    ap.add_argument("--human", default=None, help='human route for a compare run, e.g. "KSUU,TJSJ,DGAA,FKKD"')
    ap.add_argument("--run-timeout", type=int, default=1800, help="seconds per run")
    ap.add_argument("--matrix", default=None, help="JSON file with a custom list of runs [{name, argv, shows}]")
    args = ap.parse_args()

    tpl, src = os.path.abspath(args.template), os.path.abspath(args.source)
    out = os.path.abspath(args.out or (src.rstrip("/\\") + "_transitioned"))
    for p in (tpl, src):
        if not os.path.isdir(p):
            sys.exit(f"[FATAL] not a directory: {p}")
    if os.path.commonpath([out, src]) == src or os.path.commonpath([out, tpl]) == tpl:
        sys.exit("[FATAL] --out must not be inside --source or --template")
    work = os.path.join(out, "_transition")
    os.makedirs(work, exist_ok=True)
    print(f"[llm] backend: {llm_backend()}\n[io] template={tpl}\n[io] source={src}\n[io] out={out}")

    T, S = walk(tpl), walk(src)
    json.dump({"template": T, "source": S}, open(os.path.join(work, "inventory.json"), "w"), indent=1)
    print(f"[inventory] template {len(T)} files, source {len(S)} files")
    # the template's README(s)/docs are the norms: read them in full, before anything else
    readmes = [f for f in T if re.match(r"(readme|contributing|install|usage|setup|getting[_ -]?started)", 
                                        os.path.basename(f["path"]).lower()) or f["path"].lower().startswith(("docs/", "doc/"))]
    full = "\n\n".join(f"===== {f['path']} (FULL) =====\n{read_text(tpl, f['path'], 60_000)}" for f in readmes)
    tctx = full + "\n\n" + pick_context(tpl, T, args.ctx_chars, 15_000)
    sctx = pick_context(src, S, args.ctx_chars // 2, 12_000)

    results, rel = None, relevance(src, S)
    json.dump(rel, open(os.path.join(work, "relevance.json"), "w"), indent=1)
    print(f"[relevance] entry points {rel['entry_points']}; unreachable python {rel['unreachable_python']}")
    if args.run:
        date = args.date or (datetime.now() + __import__('datetime').timedelta(days=60)).strftime("%Y-%m-%d")
        missions = args.mission or ["KSUU to FKKD", "KSUU to FKKD via stops in DGAA"]
        rows = json.load(open(args.matrix)) if args.matrix else default_matrix(missions, args.aircraft, date, args.human)
        print(f"[run] executing {len(rows)} variations in {src} (writes only under its runs/)")
        results = run_matrix(src, rows, work, args.run_timeout)
        ok = sum(r["ok"] for r in results)
        print(f"[run] {ok}/{len(results)} succeeded; details in {work}/run_results.json")
    conv = step_analyze(T, tctx, work)
    plan = step_plan(conv, S, sctx, T, work, args.name, results, rel)
    if args.dry_run:
        print(f"[dry-run] see {work}/conventions.md and plan.json; nothing else written")
        return
    log = step_execute(plan, conv, src, tpl, out, S, T, work, dry=False, results=results, rel=rel)
    problems = step_verify(conv, out, work, args.rounds, dry=False)
    step_report(conv, plan, log, problems, out, src, tpl, work, results)
    print(f"\n[done] {out}\n       read TRANSITION_REPORT.md first; every generated file is marked "
          f"'{GEN_MARK}'; TODO(transition) lines need you.")


if __name__ == "__main__":
    main()
