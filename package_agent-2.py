#!/usr/bin/env python3
"""
package_agent.py — make the repo self-explaining: a verified environment, a sample run,
automatic ablations, and a README that only documents what actually ran.

  python package_agent.py --repo ./agentic_route_planning --requirements ./approved_requirements.txt \
      --mission "KSUU to FKKD via stops in DGAA" --aircraft C-17 --date 2026-12-15 \
      --human "KSUU,TJSJ,DGAA,FKKD"

Steps
  1 venv      create  <repo>/agentplanner  (or --venv-dir) with the base python; install ONLY
              what the code imports, at EXACTLY the versions in --requirements (the file is
              also used as a pip constraints file so transitive deps obey it). Anything the code
              needs that is not in the file is reported and NOT installed.
  2 sample    run the planner on --mission in the venv (LLM judge if the gateway env is set,
              else --no-llm) and, if --human is given, compare_routes.py against it.
              -> <repo>/samples/sample_run/
  3 ablations run the same mission under variations (clearance penalty 100/300/800, with/without
              risk feed, with/without auto-stops, --no-llm) plus 2-3 similar missions proposed
              by the model (or a built-in fallback), and tabulate NM / landings / clearances /
              stops from each run's last_run_blocks.json.   -> <repo>/samples/ablations/ + ABLATIONS.md
  4 readme    write <repo>/README.md (old one kept as README.previous.md) from: file list,
              --help text, requirements, the sample run, the ablation table. Every command in
              it is one that ran successfully here. Model-written if the gateway is set,
              otherwise a deterministic template.

Read-only on your code. Writes: agentplanner/, samples/, README.md, requirements.pinned.txt,
PACKAGE_LOG.md. LLM env (optional): LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------- LLM (optional)
_GW = {"c": None, "m": None}


def llm_on():
    return bool(os.environ.get("AGENT_MODEL") and os.environ.get("LITELLM_BASE_URL"))


def llm(prompt, system="You are a precise technical writer and engineer. Use only the facts given."):
    if _GW["c"] is None:
        import httpx
        from openai import OpenAI
        verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
        _GW["c"] = OpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"],
                          http_client=httpx.Client(verify=verify, timeout=300))
        _GW["m"] = os.environ["AGENT_MODEL"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for budget in (int(os.environ.get("GATEWAY_MAX_TOKENS", "8000")), 32000):
        r = _GW["c"].chat.completions.create(model=_GW["m"], messages=msgs, max_tokens=budget)
        t = (r.choices[0].message.content or "").strip()
        if t:
            return t
    return ""


def llm_json(prompt):
    t = llm(prompt)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


# ----------------------------------------------------------------------------- helpers
def sh(cmd, cwd=None, timeout=3600, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def vkey(v):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v)]


ALIAS = {"sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "Pillow", "pptx": "python-pptx",
         "docx": "python-docx", "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "dotenv": "python-dotenv",
         "shapefile": "pyshp", "cv2": "opencv-python", "fitz": "PyMuPDF"}
STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {"__future__"}
OPTIONAL = {"torch", "transformers", "matplotlib", "python-pptx", "seaborn"}


def load_requirements(path):
    """-> {norm_name: version} using the highest exact pin per package."""
    out = {}
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:==|===|:|,|\s)\s*([0-9][A-Za-z0-9.+!*-]*)\s*$", line)
        if m:
            k, v = norm(m.group(1)), m.group(2)
            if k not in out or vkey(v) > vkey(out[k]):
                out[k] = v
    return out


def discover_imports(repo):
    needed, local = {}, set()
    files = []
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if not d.startswith(".") and d not in ("__pycache__", "runs", "samples", "agentplanner")]
        for fn in fns:
            if fn.endswith(".py"):
                local.add(fn[:-3])
                if "extras_eval" not in dp:
                    files.append(os.path.join(dp, fn))
    for p in files:
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))", txt, re.M):
            for n in ([m.group(1)] if m.group(1) else [x.strip() for x in m.group(2).split(",")]):
                top = n.split(".")[0]
                if top and top not in STDLIB and top not in local and not top.startswith("_"):
                    d = ALIAS.get(top, top)
                    needed.setdefault(norm(d), {"dist": d, "files": set()})["files"].add(os.path.relpath(p, repo))
    return needed


def find_code_dir(repo):
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if not d.startswith(".") and d not in ("agentplanner", "samples")]
        if "country_route_planner.py" in fns:
            return dp
    sys.exit("[FATAL] country_route_planner.py not found under --repo")


def copy_run(code_dir, before, dest):
    """Copy the newest runs/<ts>/ folder created since `before` into dest; return its blocks json."""
    rp = os.path.join(code_dir, "runs")
    if not os.path.isdir(rp):
        return None
    new = sorted(set(os.listdir(rp)) - before)
    blocks = None
    for d in new:
        shutil.copytree(os.path.join(rp, d), os.path.join(dest, d), dirs_exist_ok=True)
        bj = os.path.join(rp, d, "last_run_blocks.json")
        if os.path.exists(bj):
            blocks = json.load(open(bj))
    return blocks


def summarize(blocks):
    if not blocks:
        return {}
    stops = blocks.get("stops", [])
    corr = blocks.get("corridors", [])
    clear = set()
    for c in corr:
        clear |= set(c.get("chosen", {}).get("clearances", []))
    return {"stops": " -> ".join(stops), "n_stops": max(0, len(stops) - 2), "total_nm": blocks.get("total_nm"),
            "clearance_countries": len(clear), "blocked": len(blocks.get("blocked_countries", [])),
            "auto_stops": " ".join(blocks.get("auto_stops", []))}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--requirements", required=True, help="pinned/approved versions (pkg==ver lines; other simple formats ok)")
    ap.add_argument("--mission", default="KSUU to FKKD via stops in DGAA")
    ap.add_argument("--aircraft", default="C-17")
    ap.add_argument("--date", default=None, help="default today+60d")
    ap.add_argument("--human", default=None, help='route humans flew for the comparison, e.g. "KSUU,TJSJ,DGAA,FKKD"')
    ap.add_argument("--venv-dir", default=None, help="default <repo>/agentplanner")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no-llm", action="store_true", help="never call the model (planner runs with --no-llm, README from template)")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per planner run")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    code_dir = find_code_dir(repo)
    venv = a.venv_dir or os.path.join(repo, "agentplanner")
    py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
    date = a.date or (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
    use_llm = llm_on() and not a.no_llm
    samples = os.path.join(repo, "samples")
    os.makedirs(samples, exist_ok=True)
    LOG = [f"# package_agent log — {datetime.now():%Y-%m-%d %H:%M}", "", f"repo `{repo}`; code dir `{code_dir}`; "
           f"requirements `{a.requirements}`; venv `{venv}`; LLM {'on (' + os.environ.get('AGENT_MODEL', '') + ')' if use_llm else 'off'}", ""]
    print("\n".join(LOG))

    # ---------------- 1. venv from the requirements file only
    reqs = load_requirements(a.requirements)
    needed = discover_imports(repo)
    from importlib import metadata as _md
    def current_version(dist):
        try:
            return _md.version(dist)
        except Exception:
            return None
    pins, source, optional, unpinned = {}, {}, [], []
    for k, info in sorted(needed.items()):
        d = info["dist"]
        if k in reqs:
            pins[d], source[d] = reqs[k], "requirements file"
        elif k in OPTIONAL and not current_version(d):
            optional.append(d)                                   # not in file, not installed: feature off
        elif current_version(d):
            pins[d], source[d] = current_version(d), "current environment (not in requirements file)"
        else:
            unpinned.append(d)                                   # will install whatever pip resolves; recorded after
            source[d] = "latest from pip (not in file, not installed here)"
    LOG += ["## 1. Environment `agentplanner`", "", f"{len(reqs)} packages in the requirements file; code imports {len(needed)} third-party packages.", ""]
    LOG += ["| package | version | source |", "|---|---|---|"] + [f"| {d} | {pins[d]} | {source[d]} |" for d in pins] + \
           [f"| {d} | (latest) | {source[d]} |" for d in unpinned]
    if optional:
        LOG += ["", f"optional, not in file and not installed (features off): {', '.join(optional)}"]
    fallback = [d for d in pins if source[d] != "requirements file"] + unpinned
    if fallback:
        LOG += ["", f"**Not in the requirements file, fell back to the current machine's versions:** {', '.join(fallback)} — "
                    f"add these to the approved list or accept the versions in requirements.pinned.txt."]
    print("\n".join(LOG[-(len(pins) + len(unpinned) + 6):]))

    work = os.path.join(samples, "_env")
    os.makedirs(work, exist_ok=True)
    need_file = os.path.join(work, "requirements.needed.txt")
    cons_file = os.path.join(work, "constraints.txt")
    open(need_file, "w").write("".join(f"{d}=={v}\n" for d, v in pins.items()) + "".join(f"{d}\n" for d in unpinned))
    open(cons_file, "w").write("".join(f"{k}=={v}\n" for k, v in reqs.items()))
    if os.path.exists(venv):
        shutil.rmtree(venv)
    rc, out, err = sh([a.python, "-m", "venv", venv])
    if rc:
        sys.exit(f"[FATAL] venv creation failed: {err}")
    sh([py, "-m", "pip", "install", "--upgrade", "pip"], timeout=900)
    rc, out, err = sh([py, "-m", "pip", "install", "-c", cons_file, "-r", need_file], timeout=3600)
    open(os.path.join(work, "pip_install.log"), "w").write(out + "\n" + err)
    if rc:
        LOG += ["", "pip install FAILED:", "```", err[-3000:], "```"]
        open(os.path.join(repo, "PACKAGE_LOG.md"), "w").write("\n".join(LOG))
        sys.exit("[FATAL] install failed on the pinned versions — see samples/_env/pip_install.log")
    rc, frozen, _ = sh([py, "-m", "pip", "freeze"])
    hdr = [f"# exact environment verified by package_agent.py on {datetime.now():%Y-%m-%d} (venv agentplanner)",
           f"# direct dependencies and where each version came from:"] + \
          [f"#   {d}=={pins.get(d, '?')}  <- {source[d]}" for d in list(pins) + unpinned]
    open(os.path.join(repo, "requirements.pinned.txt"), "w").write("\n".join(hdr) + "\n" + frozen)
    for d in unpinned:                                   # record what pip actually chose
        m = re.search(rf"^{re.escape(d)}==(.+)$", frozen, re.M | re.I)
        if m:
            pins[d] = m.group(1)
    print(f"[env] agentplanner ready: {len(frozen.splitlines())} packages")
    LOG += ["", f"Installed OK. `pip freeze` saved to `requirements.pinned.txt`.", ""]

    # ---------------- 2. sample run
    env = os.environ.copy()
    if not use_llm:
        for k in ("AGENT_MODEL", "LITELLM_BASE_URL", "LITELLM_API_KEY", "LOCAL_MODEL_PATH"):
            env.pop(k, None)
    base = ["country_route_planner.py", a.mission, "--date", date, "--aircraft", a.aircraft, "--auto-stops", "--risk", "--alternatives", "3"]
    if not use_llm:
        base.append("--no-llm")
    sample_dir = os.path.join(samples, "sample_run")
    os.makedirs(sample_dir, exist_ok=True)
    runs_before = set(os.listdir(os.path.join(code_dir, "runs"))) if os.path.isdir(os.path.join(code_dir, "runs")) else set()
    print(f"[sample] python {' '.join(base)}")
    rc, out, err = sh([py] + base, cwd=code_dir, timeout=a.timeout, env=env)
    open(os.path.join(sample_dir, "command.txt"), "w").write("python " + " ".join(f'"{x}"' if " " in x else x for x in base) + "\n")
    open(os.path.join(sample_dir, "stdout.txt"), "w").write(out)
    open(os.path.join(sample_dir, "stderr.txt"), "w").write(err)
    blocks = copy_run(code_dir, runs_before, sample_dir)
    sample = {"cmd": "python " + " ".join(f'"{x}"' if " " in x else x for x in base), "ok": rc == 0, "summary": summarize(blocks),
              "final_plan": out[out.find("=== FINAL PLAN"):][:6000] if "=== FINAL PLAN" in out else out[-4000:]}
    print(f"[sample] rc={rc} {sample['summary']}")
    LOG += ["## 2. Sample run", "", f"`{sample['cmd']}` -> {'OK' if rc == 0 else 'FAILED'}", "", "```", sample["final_plan"][:3000], "```", ""]
    if rc != 0:
        LOG += ["stderr:", "```", err[-2000:], "```"]
    compare = None
    if a.human and rc == 0:
        cmp_cmd = ["compare_routes.py", "--human", a.human] + (["--runs", "1"] if use_llm else ["--no-judge"])
        rc2, out2, err2 = sh([py] + cmp_cmd, cwd=code_dir, timeout=a.timeout, env=env)
        open(os.path.join(sample_dir, "compare_stdout.txt"), "w").write(out2 + "\n" + err2)
        # the comparison .md lands in the run folder; copy it up for visibility
        for d in os.listdir(sample_dir):
            dd = os.path.join(sample_dir, d)
            if os.path.isdir(dd):
                for f in os.listdir(dd):
                    if f.startswith("comparison_") and f.endswith(".md"):
                        shutil.copy2(os.path.join(dd, f), os.path.join(sample_dir, "comparison.md"))
        compare = {"cmd": "python " + " ".join(f'"{x}"' if " " in x else x for x in cmp_cmd), "ok": rc2 == 0,
                   "tail": "\n".join(out2.splitlines()[-25:])}
        print(f"[compare] rc={rc2}")
        LOG += ["### Comparison with the human route", "", f"`{compare['cmd']}` -> {'OK' if rc2 == 0 else 'FAILED'}", "", "```", compare["tail"], "```", ""]

    # ---------------- 3. ablations
    abl_rows = []
    if not a.skip_ablations and rc == 0:
        abl_dir = os.path.join(samples, "ablations")
        os.makedirs(abl_dir, exist_ok=True)
        common = ["--date", date, "--aircraft", a.aircraft] + ([] if use_llm else ["--no-llm"])
        variants = [
            ("baseline_full", [a.mission, "--auto-stops", "--risk"], "same mission, default penalty 300"),
            ("penalty_100", [a.mission, "--auto-stops", "--risk", "--clearance-penalty", "100"], "distance matters more"),
            ("penalty_800", [a.mission, "--auto-stops", "--risk", "--clearance-penalty", "800"], "avoid clearances"),
            ("no_risk_feed", [a.mission, "--auto-stops"], "manual avoid list only, no State Dept feed"),
            ("no_auto_stops", [a.mission, "--risk"], "no fuel-stop insertion"),
        ]
        if use_llm:
            variants.append(("no_llm", [a.mission, "--auto-stops", "--risk", "--no-llm"], "geometry only, no FCG judge"))
        if len(re.split(r"\s+to\s+|\bvia\b|\bstop", a.mission, flags=re.I)) > 2:
            variants.append(("optimize_order", [a.mission, "--auto-stops", "--risk", "--optimize-order"], "reorder interior stops"))
        # similar missions
        similar = []
        if use_llm:
            try:
                js = llm_json(f"Mission: '{a.mission}' with aircraft {a.aircraft}. Propose 3 SIMILAR missions for an "
                              f"ablation: same region/direction and comparable distance, using real ICAO codes of "
                              f"military-usable airports, varied so the routing problem differs a little (a nearby "
                              f"origin, a different destination in the same region, one with an extra required "
                              f"stop). Return ONLY JSON {{\"missions\": [\"A to B\", \"A to B via C\", ...]}}")
                similar = [m for m in js.get("missions", []) if isinstance(m, str) and " to " in m][:3]
            except Exception as e:
                print(f"[ablation] similar-mission proposal failed: {e}")
        if not similar:
            similar = ["KSUU to DGAA", "KLRF to FKKD via stops in TJSJ", "KSUU to GVAC"]
        for i, m in enumerate(similar, 1):
            variants.append((f"similar_{i}", [m, "--auto-stops", "--risk"], f"similar mission: {m}"))

        for name, argv, why in variants:
            full = ["country_route_planner.py"] + argv + common
            before = set(os.listdir(os.path.join(code_dir, "runs")))
            t0 = datetime.now()
            rc3, out3, err3 = sh([py] + full, cwd=code_dir, timeout=a.timeout, env=env)
            d = os.path.join(abl_dir, name)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "command.txt"), "w").write("python " + " ".join(f'"{x}"' if " " in x else x for x in full) + "\n")
            open(os.path.join(d, "stdout.txt"), "w").write(out3 + "\n" + err3)
            b = copy_run(code_dir, before, d)
            row = {"name": name, "why": why, "ok": rc3 == 0, "seconds": round((datetime.now() - t0).total_seconds()),
                   "cmd": "python " + " ".join(f'"{x}"' if " " in x else x for x in full), **summarize(b)}
            if rc3 != 0:
                row["error"] = (err3.strip().splitlines() or [out3.strip().splitlines()[-1] if out3.strip() else ""])[-1][:160]
            abl_rows.append(row)
            print(f"[ablation] {name:16} rc={rc3} {row.get('total_nm')} NM, {row.get('clearance_countries')} clearances, "
                  f"{row.get('n_stops')} stops  ({row['seconds']}s)")
        tbl = ["| run | what it tests | ok | total NM | en-route stops | clearance countries | blocked | stops |", "|---|---|---|---|---|---|---|---|"]
        for r in abl_rows:
            tbl.append(f"| {r['name']} | {r['why']} | {'yes' if r['ok'] else 'NO: ' + r.get('error', '')} | {r.get('total_nm', '')} | "
                       f"{r.get('n_stops', '')} | {r.get('clearance_countries', '')} | {r.get('blocked', '')} | {r.get('stops', '')} |")
        abl_md = [f"# Ablations — {a.mission} ({a.aircraft}, {date})", "",
                  "Same planner, same data; one thing changed per row. Score = NM + penalty x clearances (+400/landing).", ""] + tbl + [
                  "", "Reading it: penalty_100 vs penalty_800 shows how many miles the planner trades per clearance avoided; "
                  "no_risk_feed shows what the State Dept feed removes; no_auto_stops shows legs that exceed range; "
                  "similar_* show whether the behaviour generalises to neighbouring missions.", "",
                  "Each row's folder has `command.txt`, `stdout.txt` and the full run output (plan_log, fcg_evidence, last_run_blocks)."]
        open(os.path.join(abl_dir, "ABLATIONS.md"), "w").write("\n".join(abl_md) + "\n")
        LOG += ["## 3. Ablations", ""] + tbl + [""]

    # ---------------- 4. README
    helps = {}
    for s in ("country_route_planner.py", "compare_routes.py"):
        rc4, out4, _ = sh([py, s, "--help"], cwd=code_dir, timeout=120, env=env)
        helps[s] = out4 if rc4 == 0 else ""
    files = sorted(os.path.relpath(os.path.join(dp, f), repo) for dp, dns, fns in os.walk(repo)
                   for f in fns if not any(x in dp for x in ("agentplanner", "runs", "__pycache__", "samples", ".git")) and f.endswith((".py", ".csv", ".sh", ".txt", ".md")))
    old_readme = os.path.join(repo, "README.md")
    old_text = open(old_readme).read() if os.path.exists(old_readme) else ""
    if old_text:
        shutil.copy2(old_readme, os.path.join(repo, "README.previous.md"))
    local_llm = {
        "no model": "add --no-llm to any planner command (geometry, fuel, risk, lead-time still run; no FCG verdicts); "
                    "compare_routes.py --no-judge",
        "gateway (what this package was verified with)": "export LITELLM_BASE_URL=https://<gateway>/v1 LITELLM_API_KEY=<key> "
                    f"AGENT_MODEL={os.environ.get('AGENT_MODEL', '<model>')} ; optional AGENT_TLS_VERIFY=false GATEWAY_MAX_TOKENS=6000",
        "local OpenAI-compatible server (vLLM / Ollama / llama.cpp)": "start the server, then "
                    "export LLM_BACKEND=vllm VLLM_URL=http://localhost:8000/v1/chat/completions VLLM_MODEL=<served model name>  "
                    "(vLLM: python -m vllm.entrypoints.openai.api_server --model <hf id or local dir> --port 8000 ; "
                    "Ollama: ollama serve, VLLM_URL=http://localhost:11434/v1/chat/completions, VLLM_MODEL=<ollama model>)",
        "in-process model from a folder (no server)": "pip install torch transformers (not needed otherwise); "
                    "export LLM_BACKEND=hf LOCAL_MODEL_PATH=/path/to/saved_model_dir LOCAL_MAX_NEW_TOKENS=1200 ; "
                    "needs a GPU with enough memory for the model; slower than a server",
        "selection rule": "LLM_BACKEND overrides; else AGENT_MODEL set -> gateway; else LOCAL_MODEL_PATH set -> hf; else vllm. "
                          "Every script prints '[llm] backend: ...' at start. No temperature is sent; reasoning models need "
                          "a large token budget (GATEWAY_MAX_TOKENS / LOCAL_MAX_NEW_TOKENS).",
        "what changes with the model": "only the per-country FCG verdicts and the narrative; geometry, fuel stops and cost "
                                       "are identical across backends, so runs/<ts>/fcg_evidence.txt can be diffed between models.",
    }
    facts = {"repo_files": files, "python": sys.version.split()[0], "venv": "agentplanner",
             "dependency_sources": source, "fell_back_to_local_versions": fallback, "llm_backends": local_llm,
             "install": f"python -m venv agentplanner && agentplanner/bin/pip install -c {os.path.basename(a.requirements)} -r requirements.pinned.txt",
             "pinned": pins, "optional_missing": optional, "sample": sample, "compare": compare, "ablations": abl_rows,
             "help": helps, "data_files_user_supplies": ["fcg_extract.csv (with overflight_raw)", "mapping.csv",
                                                          "countries_codes_and_coordinates.csv", "airports.csv (OurAirports)",
                                                          "ne_110m_admin_0_countries.geojson (Natural Earth)", "fcg1.csv (optional supplement)"]}
    readme = None
    if use_llm:
        try:
            readme = llm(f"""Write the README.md for this repository. Audience: an engineer receiving the code who has
never seen it. Use ONLY the facts below; every command you show must be one listed as having run OK.
Structure: title + one-paragraph purpose; what it does (planner, comparison, judge; math decides geometry,
model decides feasibility — keep this short); repository layout (table from repo_files, one line each);
data files the user must supply; environment setup (venv named agentplanner, the exact install command,
a note on dependency_sources — which pins came from the requirements file and which fell back to the
build machine's versions); a full section "Running with a local model" built from llm_backends: one
subsection per backend with the exact export lines and server start commands, the selection rule, and
what does/does not change with the model — the receiving team may have no gateway at all); how to run
(the sample command and what it printed — quote the FINAL PLAN excerpt; the comparison command and its
tail); outputs of a run (runs/<ts>/plan_log.txt, fcg_evidence.txt, last_run_blocks.json, comparison_*.md);
ablations (reproduce the table and 2-3 sentences on how to read it; point to samples/ablations/);
the cost model in one paragraph (score = NM + 300 per clearance + 1500 per risk country + 400 per landing,
all tunable); CLI reference (condensed from the --help text); known limits. Plain Markdown, no emoji,
no marketing language. Prior README (facts only, keep anything still true and not contradicted):
{old_text[:12000]}

FACTS (json):
{json.dumps(facts, indent=1, default=str)[:60000]}

Return ONLY the README content.""")
        except Exception as e:
            print(f"[readme] model failed: {e}; using template")
    if not readme:
        R = ["# Agentic Route Planning", "", "FCG-compliant military flight route planning with an evaluation against routes humans flew. "
             "Geometry, fuel and routing are deterministic code; an LLM reads the raw FCG text per country and returns a strict verdict.", "",
             "## Setup", "", "```bash", "python -m venv agentplanner", "source agentplanner/bin/activate",
             f"pip install -c {os.path.basename(a.requirements)} -r requirements.pinned.txt", "```", "",
             f"Dependency versions: {', '.join(f'{d}=={v} ({source[d]})' for d, v in pins.items())}.", "",
             "## Running with a local model (or none)", ""] + [f"**{k}** — {v}" + "\n" for k, v in local_llm.items()] + [
             "## Data you supply (next to the scripts)", ""] + [f"- {d}" for d in facts["data_files_user_supplies"]] + [
             "", "## Run", "", "```bash", sample["cmd"], "```", "", "Output (excerpt):", "", "```", sample["final_plan"][:2500], "```", ""]
        if compare:
            R += ["Compare with the human route:", "", "```bash", compare["cmd"], "```", "", "```", compare["tail"], "```", ""]
        R += ["Each run writes `runs/<timestamp>/` with `plan_log.txt`, `fcg_evidence.txt` (every FCG passage the judge read and its verdict), "
              "`last_run_blocks.json` (the plan), and `comparison_*.md` when compared.", ""]
        if abl_rows:
            R += ["## Ablations", "", "See `samples/ablations/ABLATIONS.md`.", ""] + tbl + [""]
        R += ["## Cost model", "", "score = NM + 300 per country needing overflight clearance + 1,500 per risk-listed country + 400 per en-route landing. "
              "Great circle, penalized country graph and stepping-stone paths are scored per leg; lowest wins. All weights are flags.", "",
              "## CLI", "", "```", helps.get("country_route_planner.py", ""), "```", "", "```", helps.get("compare_routes.py", ""), "```", "",
              "## Layout", ""] + [f"- `{f}`" for f in files] + ["", "## Limits", "",
              "No fuel burn/winds/MOG/NOTAMs; FCG extract quality bounds the judge; advisories are country-level; lead time is calendar arithmetic from today."]
        readme = "\n".join(R) + "\n"
    open(old_readme, "w", encoding="utf-8").write(readme)
    LOG += ["## 4. README", "", f"README.md written ({'model' if use_llm and readme else 'template'}); previous saved as README.previous.md.", ""]
    open(os.path.join(repo, "PACKAGE_LOG.md"), "w").write("\n".join(LOG) + "\n")
    print(f"\n[done] venv agentplanner | samples/sample_run | samples/ablations | README.md | requirements.pinned.txt | PACKAGE_LOG.md")


if __name__ == "__main__":
    main()
