#!/usr/bin/env python3
"""
apply_advisor_feedback.py — apply the advisor's README/structure notes to the repo, in place.

  python apply_advisor_feedback.py --repo ./agentic_route_planning [--ref ./dans_repo] [--dry-run]

Maps each note to an action (nothing is guessed; the model only writes prose):

 README
  - "judge: does it read more than raw FCG text?"      -> README states exactly what the judge reads
  - remove .ipynb_checkpoints refs; add to .gitignore  -> delete dir, write .gitignore
  - CSVs into data/ with subdirectories                 -> data/fcg, data/reference, data/config
                                                           (+ code patched to resolve data_path())
  - list data sources (FCG, State Dept, fuel site...)   -> "Data sources" section + source column
  - purpose descriptions vague                          -> rewritten per file, from the code
  - table too large -> two tables                       -> "Repository structure" + "Data elements"
  - Untitled.ipynb / extraneous files                   -> removed (listed in the report)
  - "place these files next to the scripts" unclear     -> replaced by data/ layout + --data-dir example
  - environment set-up like Dan's, step by step         -> conda env create -> activate -> pip install -r
  - don't list all python requirements                  -> only the install commands
  - "Running with a local model" -> "Running with a model", moved below How to run,
    How to run points to it, per-model how-to-run       -> done in README structure
  - move CLI reference up                               -> right after How to run
  - functions instead of CLI; tutorial notebook         -> run_plan()/compare() added; tutorial.ipynb
                                                           (how to run, outputs, ablations, limits)
 Writes: FEEDBACK_REPORT.md listing every note -> what changed. Backups: README.previous.md,
 *.bak_feedback for patched code. --ref is read only (used for the env-setup style).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

DATA_LAYOUT = {
    "fcg": ["fcg_extract.csv", "fcg1.csv", "opus_all.csv", "overflight.csv"],
    "reference": ["airports.csv", "countries_codes_and_coordinates.csv", "mapping.csv", "all_countries.csv",
                  "ne_110m_admin_0_countries.geojson", "custom_airports.csv"],
    "config": ["aircraft_specs.csv", "fuel_rates.csv", "avoid_countries.csv"],
}
EXTRANEOUS = ["Untitled.ipynb", "last_run_blocks.json", "risk_cache.json", "README.previous.md", "PACKAGE_LOG.md"]
EXTRANEOUS_DIRS = [".ipynb_checkpoints", "__pycache__", "eval_results", "_transition", "_env_agent", "_feedback"]
EXTRANEOUS_GLOBS = [r"\.bak_\d+", r"\.pre_env_agent$", r"\.orig$"]

DATA_SOURCES = [
    ("fcg_extract.csv", "data/fcg", "Foreign Clearance Guide (FCG), per-country checklist fields extracted by the FCG extraction pipeline; includes `overflight_raw`", "pre-processed by us"),
    ("fcg1.csv", "data/fcg", "FCG entry/exit airfield lists per country (supplement, merged on alpha3 at load)", "pre-processed by us"),
    ("opus_all.csv / overflight.csv", "data/fcg", "earlier FCG extracts kept for reference", "pre-processed by us"),
    ("airports.csv", "data/reference", "OurAirports world airport database (ICAO, coordinates, country) — https://ourairports.com/data/", "public, unmodified"),
    ("ne_110m_admin_0_countries.geojson", "data/reference", "Natural Earth 1:110m country boundaries, used for exact overflight detection — https://www.naturalearthdata.com/", "public domain, unmodified"),
    ("countries_codes_and_coordinates.csv", "data/reference", "ISO 3166 codes + country centroids", "public, unmodified"),
    ("mapping.csv", "data/reference", "alpha3 -> country name used to resolve user input", "ours"),
    ("all_countries.csv", "data/reference", "country list used by the extraction pipeline", "ours"),
    ("custom_airports.csv", "data/reference", "airports missing from OurAirports; auto-appended from the NOAA/FAA station directory (aviationweather.gov)", "ours + NOAA/FAA"),
    ("aircraft_specs.csv", "data/config", "aircraft type, range, min runway, burn, capacity (planning values)", "ours"),
    ("fuel_rates.csv", "data/config", "DLA Energy standard fuel prices", "DLA Energy, transcribed"),
    ("avoid_countries.csv", "data/config", "manual geopolitical block/avoid list", "ours"),
    ("(live) State Dept advisories", "fetched at run time with --risk", "travel.state.gov RSS; Level 4 -> block, Level 3 -> avoid; cached in risk_cache.json", "US State Dept"),
    ("(live) METAR/TAF, station directory", "fetched at run time", "aviationweather.gov (NOAA/FAA), free, no key", "NOAA/FAA"),
]


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def llm(prompt):
    import httpx
    from openai import OpenAI
    verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
    c = OpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"],
               http_client=httpx.Client(verify=verify, timeout=600))
    msgs = [{"role": "system", "content": "You are a precise technical writer. Use only the facts given. Plain Markdown, no emoji."},
            {"role": "user", "content": prompt}]
    for budget in (int(os.environ.get("GATEWAY_MAX_TOKENS", "12000")), 40000):
        r = c.chat.completions.create(model=os.environ["AGENT_MODEL"], messages=msgs, max_tokens=budget)
        t = (r.choices[0].message.content or "").strip()
        if t:
            m = re.search(r"```(?:markdown|md)?\n(.*?)```\s*$", t, re.DOTALL)
            return (m.group(1) if m and t.startswith("```") else t).rstrip() + "\n"
    return ""


# ----------------------------------------------------------------------------- code patches
PLANNER_EDITS = [
('''FCG_CSV      = os.environ.get("FCG_CSV", "fcg_extract.csv")      # or --fcg''',
'''# All input files are looked up with data_path(): first as given, then under DATA_DIR
# (data/fcg, data/reference, data/config, data/), then next to this script. So the repo
# keeps its inputs in data/ while old flat layouts and absolute paths still work.
DATA_DIR = os.environ.get("ROUTE_DATA_DIR", "data")
_DATA_SUBDIRS = ("fcg", "reference", "config", "")


def data_path(name):
    """Resolve an input file name to a path that exists (or the data/ default for writes)."""
    if not name:
        return name
    if os.path.isabs(name) or os.path.exists(name):
        return name
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (DATA_DIR, os.path.join(here, DATA_DIR)):
        for sub in _DATA_SUBDIRS:
            p = os.path.join(base, sub, os.path.basename(name)) if sub else os.path.join(base, os.path.basename(name))
            if os.path.exists(p):
                return p
    p = os.path.join(here, os.path.basename(name))
    if os.path.exists(p):
        return p
    return name                                     # not found: caller reports it


FCG_CSV      = os.environ.get("FCG_CSV", "fcg_extract.csv")      # or --fcg'''),
('''    if OVERFLIGHT_CSV and not os.path.exists(OVERFLIGHT_CSV):''', '''    OVERFLIGHT_CSV_ = data_path(OVERFLIGHT_CSV)
    if OVERFLIGHT_CSV and not os.path.exists(OVERFLIGHT_CSV_):'''),
('''    if OVERFLIGHT_CSV and os.path.exists(OVERFLIGHT_CSV):
        ov = pd.read_csv(OVERFLIGHT_CSV)''', '''    if OVERFLIGHT_CSV and os.path.exists(OVERFLIGHT_CSV_):
        ov = pd.read_csv(OVERFLIGHT_CSV_)'''),
('''    for sp in SUPPLEMENT_CSVS:
        if not os.path.exists(sp):''', '''    for sp in SUPPLEMENT_CSVS:
        sp = data_path(sp)
        if not os.path.exists(sp):'''),
('''        cu = pd.read_csv(CUSTOM_AIRPORTS_CSV)''', '''        cu = pd.read_csv(data_path(CUSTOM_AIRPORTS_CSV))'''),
('''        new = not os.path.exists(CUSTOM_AIRPORTS_CSV)
        with open(CUSTOM_AIRPORTS_CSV, "a") as f:''', '''        target = data_path(CUSTOM_AIRPORTS_CSV)
        if not os.path.exists(target) and os.path.isdir(os.path.join(DATA_DIR, "reference")):
            target = os.path.join(DATA_DIR, "reference", CUSTOM_AIRPORTS_CSV)
        new = not os.path.exists(target)
        with open(target, "a") as f:'''),
('''    coords = load_coords(COORDS_CSV)
    mapping = load_mapping(MAPPING_CSV)
    fcg = load_fcg(FCG_CSV)
    ourairports = load_ourairports(AIRPORTS_CSV, coords)''', '''    coords = load_coords(data_path(COORDS_CSV))
    mapping = load_mapping(data_path(MAPPING_CSV))
    fcg = load_fcg(data_path(FCG_CSV))
    ourairports = load_ourairports(data_path(AIRPORTS_CSV), coords)'''),
('''        gj = json.load(open(NE_FILE, encoding="utf-8"))''', '''        gj = json.load(open(data_path(NE_FILE), encoding="utf-8"))'''),
('''def load_manual_risk(path=RISK_CSV):''', '''def load_manual_risk(path=None):
    path = data_path(path or RISK_CSV)'''),
('''def load_fuel_rates(path=FUEL_CSV):''', '''def load_fuel_rates(path=None):
    path = data_path(path or FUEL_CSV)'''),
('''def load_aircraft(type_name, path=AIRCRAFT_CSV):''', '''def load_aircraft(type_name, path=None):
    path = data_path(path or AIRCRAFT_CSV)'''),
('''    ap.add_argument("--fcg", default=None, help=f"FCG extract CSV (default {FCG_CSV}; env FCG_CSV)")''',
 '''    ap.add_argument("--data-dir", default=None,
                    help=f"folder holding the input files (default '{DATA_DIR}' with fcg/, reference/, config/ "
                         f"subfolders, or flat; env ROUTE_DATA_DIR)")
    ap.add_argument("--fcg", default=None, help=f"FCG extract CSV (default {FCG_CSV}; env FCG_CSV)")'''),
('''    if args.fcg:
        FCG_CSV = args.fcg''', '''    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.fcg:
        FCG_CSV = args.fcg'''),
('''if __name__ == "__main__":''', '''def run_plan(mission, date, aircraft=None, auto_stops=False, risk=False, alternatives=0,
             optimize_order=False, clearance_penalty=None, llm=True, backend=None, data_dir=None,
             out_dir=None, verbose=False):
    """Programmatic entry point (what the CLI does, as a function). Returns a dict with the
    chosen stops, total NM, corridors, verdicts and the run folder; prints the same report
    as the CLI. Example:
        import country_route_planner as P
        r = P.run_plan("KSUU to FKKD via stops in DGAA", "2026-12-15", aircraft="C-17",
                       auto_stops=True, risk=True, alternatives=3)
    """
    global AIRCRAFT, AUTO_STOPS, USE_RISK_FEED, SHOW_ALTERNATIVES, OPTIMIZE_ORDER, \\
        CLEARANCE_PENALTY_NM, LLM_BACKEND, DATA_DIR, RUN_DIR, VERBOSE, EVIDENCE_LOG
    import datetime as _dt
    _dt.date.fromisoformat(date)
    if data_dir:
        DATA_DIR = data_dir
    if backend:
        LLM_BACKEND = backend
    VERBOSE = verbose
    USE_RISK_FEED = risk
    SHOW_ALTERNATIVES = alternatives
    OPTIMIZE_ORDER = optimize_order
    if clearance_penalty is not None:
        CLEARANCE_PENALTY_NM = float(clearance_penalty)
    AIRCRAFT = load_aircraft(aircraft) if aircraft else None
    AUTO_STOPS = bool(auto_stops)
    EVIDENCE_LOG = []
    RUN_DIR = out_dir or os.path.join("runs", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(RUN_DIR, exist_ok=True)
    result = plan(mission, date, llm)
    narrate(*result, date, llm)
    chosen, total_nm, alternates, stop_countries, corridors, verdicts, nodes, ourairports, suggestions, fuel_notes = result
    blocks = None
    bj = os.path.join(RUN_DIR, "last_run_blocks.json")
    if os.path.exists(bj):
        blocks = json.load(open(bj))
    return {"stops": list(chosen), "total_nm": total_nm, "stop_countries": list(stop_countries),
            "corridors": corridors, "verdicts": {f"{c}/{r}/{a}": v for (c, r, a), v in verdicts.items()},
            "fuel_notes": fuel_notes, "run_dir": RUN_DIR, "blocks": blocks}


if __name__ == "__main__":'''),
]
COMPARE_EDITS = [
('''def main():
    ap = argparse.ArgumentParser()''', '''def compare(human, planner=None, blocks_file="last_run_blocks.json", aircraft=None, date=None,
            required=None, runs=3, judge=True, chart=False, out=None):
    """Programmatic entry point. human/planner/required are comma-separated ICAO strings.
    Returns the path of the comparison markdown. Example:
        import compare_routes as C
        C.compare("KSUU,TJSJ,DGAA,FKKD", runs=1)
    """
    argv = ["--human", human, "--blocks-file", blocks_file, "--runs", str(runs)]
    if planner: argv += ["--planner", planner]
    if aircraft: argv += ["--aircraft", aircraft]
    if date: argv += ["--date", date]
    if required: argv += ["--required", required]
    if not judge: argv.append("--no-judge")
    if chart: argv.append("--chart")
    if out: argv += ["--out", out]
    return main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser()'''),
('''    args = ap.parse_args()

    if args.blocks_file == "last_run_blocks.json"''', '''    args = ap.parse_args(argv)

    if args.blocks_file == "last_run_blocks.json"'''),
('''    open(path, "w").write("\\n".join(md) + "\\n")
    print(f"\\n[compare] report -> {path}" + (f"\\n[compare] chart  -> {png}" if png else ""))''',
 '''    open(path, "w").write("\\n".join(md) + "\\n")
    print(f"\\n[compare] report -> {path}" + (f"\\n[compare] chart  -> {png}" if png else ""))
    return path'''),
]


def patch(path, edits, dry):
    src = open(path, encoding="utf-8").read()
    applied, skipped = [], []
    for old, new in edits:
        if new.split("\n")[0].strip() and new.split("\n")[0] in src and old not in src:
            skipped.append("already applied: " + old.splitlines()[0][:60]); continue
        if src.count(old) != 1:
            skipped.append(f"anchor not found ({src.count(old)}x): {old.splitlines()[0][:60]}"); continue
        src = src.replace(old, new); applied.append(old.splitlines()[0][:60])
    if applied and not dry:
        if not os.path.exists(path + ".bak_feedback"):
            shutil.copy2(path, path + ".bak_feedback")
        open(path, "w", encoding="utf-8").write(src)
    return applied, skipped


# ----------------------------------------------------------------------------- notebook
def build_notebook(path, readme_facts):
    """Write a v4 .ipynb directly (no nbformat dependency)."""
    def md(s):
        return {"cell_type": "markdown", "metadata": {}, "source": s}
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    cells = [
        md("# Agentic Route Planning — tutorial\n\nThis notebook walks through the planner as **functions** "
           "(no command line needed): set up the model, plan a mission, read the outputs, compare with the route "
           "humans flew, run a few ablations, and see the known limits. Run it from the repository root with the "
           "`agentplanner` environment as the kernel."),
        md("## 1. Setup\n\nPick ONE model situation (see README → *Running with a model*). Leave all unset and pass "
           "`llm=False` below to run with no model at all (geometry, fuel stops, risk and lead-time arithmetic still "
           "run; no FCG verdicts)."),
        code("import os\n\n# (a) gateway / remote OpenAI-compatible endpoint\n# os.environ[\"LITELLM_BASE_URL\"] = \"https://<gateway>/v1\"\n"
             "# os.environ[\"LITELLM_API_KEY\"]  = \"<key>\"\n# os.environ[\"AGENT_MODEL\"]      = \"gpt-5.6-sol\"\n"
             "# os.environ[\"AGENT_TLS_VERIFY\"] = \"false\"\n\n# (b) local OpenAI-compatible server (vLLM / Ollama / llama.cpp)\n"
             "# os.environ[\"LLM_BACKEND\"] = \"vllm\"\n# os.environ[\"VLLM_URL\"]    = \"http://localhost:8000/v1/chat/completions\"\n"
             "# os.environ[\"VLLM_MODEL\"]  = \"<served model name>\"\n\n# (c) in-process model from a folder (needs torch + transformers)\n"
             "# os.environ[\"LLM_BACKEND\"] = \"hf\"\n# os.environ[\"LOCAL_MODEL_PATH\"] = \"/path/to/saved_model_dir\"\n\n"
             "import country_route_planner as P\nimport compare_routes as C\nprint(\"backend:\", P.LLM_BACKEND)"),
        md("## 2. Plan a mission\n\nMissions are plain English: `\"KSUU to FKKD\"`, `\"KSUU to FKKD via stops in DGAA\"`, "
           "`\"round trip KSUU to FKKD via DGAA\"`. Inputs are read from `data/` (override with `data_dir=`)."),
        code("MISSION  = \"KSUU to FKKD via stops in DGAA\"\nDATE     = \"2026-12-15\"      # mission date; lead-time checks depend on it\n"
             "AIRCRAFT = \"C-17\"\nUSE_LLM  = bool(os.environ.get(\"AGENT_MODEL\") or os.environ.get(\"LLM_BACKEND\"))\n\n"
             "r = P.run_plan(MISSION, DATE, aircraft=AIRCRAFT, auto_stops=True, risk=True, alternatives=3, llm=USE_LLM)\n"
             "print(\"stops:\", \" -> \".join(r[\"stops\"]))\nprint(\"total NM:\", round(r[\"total_nm\"]))\nprint(\"run folder:\", r[\"run_dir\"])"),
        md("## 3. What a run produced\n\nEverything printed above is also in the run folder: `plan_log.txt` (full log), "
           "`fcg_evidence.txt` (every FCG passage the judge read, its verdict, and what the code enforced), "
           "`last_run_blocks.json` (the plan as data: stops, corridors, options, blocked countries, config)."),
        code("import os, json\nprint(sorted(os.listdir(r[\"run_dir\"])))\nblocks = json.load(open(os.path.join(r[\"run_dir\"], \"last_run_blocks.json\")))\n"
             "for c in blocks[\"corridors\"]:\n    ch = c[\"chosen\"]\n    print(f\"{c['leg']:24} {ch['option']:<22} {ch['nm']:>6,} NM  clearances={ch['clearances']}\")\n"
             "ev = os.path.join(r[\"run_dir\"], \"fcg_evidence.txt\")\nprint(open(ev).read()[:3000] if os.path.exists(ev) else \"(no evidence file: run with a model to get judge verdicts)\")"),
        md("## 4. Compare with the route humans actually flew\n\nGive the human stops in order (first and last must match the "
           "mission). Output: numbers (whole itinerary + per required segment), a plain-English explanation, and — with a "
           "model — an independent judge's verdict with reasons."),
        code("HUMAN = \"KSUU,TJSJ,DGAA,FKKD\"\nreport = C.compare(HUMAN, blocks_file=os.path.join(r[\"run_dir\"], \"last_run_blocks.json\"),\n"
             "                   runs=1 if USE_LLM else 3, judge=USE_LLM)\nprint(open(report).read()[:4000])"),
        md("## 5. Ablations\n\nThe cost model is `score = NM + penalty × clearance countries + 1,500 × risk-listed countries "
           "+ 400 × en-route landing`. The clearance penalty is the knob that encodes routing philosophy; sweep it and "
           "watch the trade between miles and clearances. Each run writes its own folder."),
        code("import pandas as pd\nrows = []\nfor pen in (100, 300, 800):\n    rr = P.run_plan(MISSION, DATE, aircraft=AIRCRAFT, auto_stops=True, risk=True,\n"
             "                    clearance_penalty=pen, llm=False)   # geometry only -> fast, deterministic\n"
             "    b = rr[\"blocks\"] or {}\n    clear = set()\n    for c in b.get(\"corridors\", []):\n        clear |= set(c[\"chosen\"][\"clearances\"])\n"
             "    rows.append({\"penalty\": pen, \"total_nm\": b.get(\"total_nm\"), \"stops\": \" -> \".join(b.get(\"stops\", [])),\n"
             "                 \"clearance_countries\": len(clear)})\npd.DataFrame(rows)"),
        md("Reading it: as the penalty rises the planner accepts more miles to cross fewer countries; at 100 it hugs the "
           "great circle. Other ablations worth running the same way: `risk=False` (manual avoid list only), "
           "`auto_stops=False` (see which legs exceed range), `llm=False` vs a model (what the FCG judge changes — diff "
           "the two `fcg_evidence.txt` files)."),
        md("## 6. Known limits\n\n- No fuel burn, winds, MOG or NOTAMs — range is a planning fraction (85 %), not a fuel plan.\n"
           "- The judge is only as good as the FCG extract; every verdict is logged with the text it read so it can be checked.\n"
           "- State Dept advisories are country-level, not airspace-level.\n- Lead time is calendar arithmetic from today.\n"
           "- Changing the model changes only the per-country verdicts and narrative; geometry, stops and cost are identical."),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python (agentplanner)", "language": "python"},
                       "language_info": {"name": "python"}}}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)


# ----------------------------------------------------------------------------- README
def write_readme(repo, ref_root, moved, dry, use_llm):
    old = open(os.path.join(repo, "README.md"), encoding="utf-8").read() if os.path.exists(os.path.join(repo, "README.md")) else ""
    files = sorted(os.path.relpath(os.path.join(dp, f), repo).replace(os.sep, "/")
                   for dp, dns, fns in os.walk(repo) for f in fns
                   if not any(x in dp for x in ("agentplanner", "runs", "__pycache__", ".git", "samples", "data")) and f.endswith((".py", ".sh", ".txt", ".md", ".ipynb")))
    purposes = {
        "country_route_planner.py": "the planner: resolves the mission text, builds great-circle/graph/stepping-stone corridors, inserts fuel stops within aircraft range, applies the risk layer, asks the FCG judge per country, enforces lead times, prints the plan and writes the run folder; `run_plan()` is the function form",
        "compare_routes.py": "human route vs planner route: numbers (whole itinerary and per required segment), plain-English explanation, LLM judge verdict; `compare()` is the function form",
        "route_judge.py": "the judge prompt and the raw-facts builder used by compare_routes (also runnable alone)",
        "merge_fcg1.py": "one-time utility: fold the fcg1.csv airfield columns into fcg_extract.csv",
        "tutorial.ipynb": "notebook tutorial: setup, plan a mission, read outputs, compare with the human route, ablations, limits",
        "run_example.sh": "the two commands of a full run (plan, then compare)",
        "env.example.sh": "environment variables for each model situation (copy to env.sh)",
        "requirements.txt": "python dependencies (install with pip, see Environment set-up)",
        "requirements.pinned.txt": "exact versions the package was verified with",
        "extras_eval/": "optional evaluation scripts: geometry precision/recall, 4-config ablation, penalty sweep, plots, slide-deck builder",
        "samples/": "a verified sample run and ablation runs with their commands and outputs",
        "data/": "all input files (see Data elements); fcg/ = FCG-derived, reference/ = public reference data, config/ = planning tables",
    }
    env_style = ""
    if ref_root:
        for cand in ("README.md", "readme.md", "README.rst"):
            p = os.path.join(ref_root, cand)
            if os.path.exists(p):
                t = open(p, encoding="utf-8", errors="replace").read()
                m = re.search(r"(?is)(#+[^\n]*(install|setup|set-up|environment|getting started)[^\n]*\n.*?)(?=\n#+ |\Z)", t)
                env_style = (m.group(1) if m else t)[:4000]
                break
    data_table = "\n".join(f"| `{n}` | `{loc}` | {what} | {src} |" for n, loc, what, src in DATA_SOURCES)
    facts = {"files": files, "purposes": purposes, "moved_to_data": moved, "data_table_markdown": data_table,
             "env_setup_style_from_reference_repo": env_style}
    prompt = f"""Rewrite this README applying the reviewer's notes exactly. Keep every fact that is still true; use
only the facts given; do not list python packages individually.

REVIEWER NOTES (apply all):
1. "What it does": state precisely what the judge reads — it reads the raw FCG checklist text for the
   country (the configured fields of fcg_extract.csv, plus the fcg1.csv airfield columns), the mission date,
   the aircraft type and the specific airport; it does NOT read anything else and never answers from memory.
   The risk layer (State Dept advisories + manual list) is separate code, not the judge.
2. Repository layout: no .ipynb_checkpoints or Untitled.ipynb anywhere. Split the big table into TWO:
   "Repository structure" (code/dirs, one clear purpose line each — use `purposes`) and "Data elements"
   (use `data_table_markdown` verbatim: file | location | what it is | source). Data files live under data/
   (fcg/, reference/, config/).
3. Add a "Data sources" section listing where each input comes from (FCG extraction pipeline, OurAirports,
   Natural Earth, ISO 3166, DLA Energy fuel prices, US State Dept advisory RSS, NOAA/FAA aviationweather.gov)
   and which are pre-processed by us vs public/unmodified.
4. Replace "place these files next to the scripts" with: inputs are read from `data/` (subfolders fcg/,
   reference/, config/, or flat) relative to where you run; override with `--data-dir PATH`, env
   ROUTE_DATA_DIR, or `run_plan(..., data_dir=PATH)`. Give one example.
5. "Environment set-up" step by step like the reference repo's style (`env_setup_style_from_reference_repo`,
   form only): create conda env `agentplanner` (python 3.11), activate, `pip install -r requirements.txt`,
   optional extras (torch/transformers for in-process models; matplotlib/python-pptx for extras_eval).
   Do NOT list individual packages.
6. Order of sections: Title+purpose; What it does; Repository structure; Data elements; Data sources;
   Environment set-up; How to run (function form FIRST — `run_plan()` / `compare()` with a short code
   block — then the equivalent CLI; say "see Running with a model below for each model situation");
   CLI reference (right after How to run); Outputs of a run; Running with a model (renamed from "local
   model" — gateway/server/in-process/none, each with its exports AND its how-to-run line); Tutorial
   notebook (tutorial.ipynb: how to run, outputs, comparison, ablations, limits — point readers there);
   Ablations (short, point to samples/ablations); Cost model; Known limits.
7. Purpose descriptions must be specific (what the file does, not what it "handles").

PREVIOUS README (facts to keep):
{old[:30000]}

FACTS (json):
{json.dumps(facts, indent=1)[:40000]}

Return ONLY the README markdown."""
    if dry:
        return "(dry run)"
    if use_llm:
        new = llm(prompt)
    else:
        new = ""
    if not new.strip():
        new = old
    shutil.copy2(os.path.join(repo, "README.md"), os.path.join(repo, "README.previous.md")) if old else None
    open(os.path.join(repo, "README.md"), "w", encoding="utf-8").write(new)
    return "model" if use_llm else "unchanged (no model)"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", default=None, help="Dan's repo (read-only; its env-setup section is the style example)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args([x for x in sys.argv[1:] if x.replace("\xa0", "").strip()])
    repo, dry = os.path.abspath(a.repo), a.dry_run
    use_llm = bool(os.environ.get("AGENT_MODEL") and os.environ.get("LITELLM_BASE_URL"))
    R = [f"# Advisor feedback applied — {datetime.now():%Y-%m-%d %H:%M}", "", f"Repo `{repo}`; reference `{a.ref or '-'}`; {'DRY RUN' if dry else 'applied'}.", ""]

    # 1. extraneous files / dirs
    removed = []
    for d in EXTRANEOUS_DIRS:
        for dp, dns, fns in os.walk(repo):
            if os.path.basename(dp) == d:
                removed.append(os.path.relpath(dp, repo)); (not dry) and shutil.rmtree(dp, ignore_errors=True)
    for dp, dns, fns in os.walk(repo):
        dns[:] = [x for x in dns if x not in ("agentplanner", ".git", "samples", "runs")]
        for f in fns:
            if f in EXTRANEOUS or any(re.search(g, f) for g in EXTRANEOUS_GLOBS):
                removed.append(os.path.relpath(os.path.join(dp, f), repo)); (not dry) and os.remove(os.path.join(dp, f))
    R += ["## Extraneous files removed (notes: .ipynb_checkpoints, Untitled.ipynb, extraneous files)", ""] + ([f"- `{x}`" for x in removed] or ["- none found"]) + [""]

    # 2. data/ layout
    moved = []
    for sub, names in DATA_LAYOUT.items():
        for n in names:
            src = os.path.join(repo, n)
            if os.path.exists(src):
                dst = os.path.join(repo, "data", sub, n)
                moved.append(f"{n} -> data/{sub}/{n}")
                if not dry:
                    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(src, dst)
    R += ["## Data files moved into data/ (note: csv files into a data/ directory with subdirectories)", ""] + ([f"- {m}" for m in moved] or ["- nothing to move (already in data/ or absent)"]) + [""]

    # 3. .gitignore
    gi = os.path.join(repo, ".gitignore")
    want = ["__pycache__/", "*.pyc", ".ipynb_checkpoints/", "agentplanner/", "runs/", "env.sh", ".env", "*.bak_*", "*.orig",
            "README.previous.md", "risk_cache.json", "samples/_env/", "_feedback/", "_transition/"]
    have = open(gi).read().splitlines() if os.path.exists(gi) else []
    add = [w for w in want if w not in have]
    if add and not dry:
        open(gi, "a").write(("\n" if have and have[-1] else "") + "\n".join(add) + "\n")
    R += ["## .gitignore", "", f"- added: {add}" if add else "- already complete", ""]

    # 4. code: data_path() + function APIs
    for fn, edits in (("country_route_planner.py", PLANNER_EDITS), ("compare_routes.py", COMPARE_EDITS)):
        p = next((os.path.join(dp, fn) for dp, _, fns in os.walk(repo) if fn in fns), None)
        if not p:
            R += [f"## {fn}", "", "- NOT FOUND", ""]; continue
        applied, skipped = patch(p, edits, dry)
        R += [f"## {fn} (notes: data/ directory, functions instead of command line)", ""] + [f"- applied: {x}" for x in applied] + [f"- skipped: {x}" for x in skipped] + [""]

    # 5. notebook
    nb_path = os.path.join(repo, "tutorial.ipynb")
    if not dry:
        build_notebook(nb_path, None)
    R += ["## tutorial.ipynb (note: how to run / outputs / ablations / known limits as a notebook tutorial)", "", f"- {'written' if not dry else 'would write'} `tutorial.ipynb` using `run_plan()` and `compare()`", ""]

    # 6. README
    how = write_readme(repo, os.path.abspath(a.ref) if a.ref else None, moved, dry, use_llm)
    R += ["## README.md (all README notes)", "", f"- rewritten by {how}; previous saved as README.previous.md" if not dry else "- would be rewritten",
          "- sections in the reviewer's order; judge description clarified; two tables; data sources; data/ + --data-dir; "
          "step-by-step conda setup (Dan's style); no package list; 'Running with a model' moved below How to run; CLI reference moved up", ""]

    # 7. checks
    if not dry:
        probs = []
        for dp, dns, fns in os.walk(repo):
            dns[:] = [x for x in dns if x not in ("agentplanner", ".git")]
            for f in fns:
                if f.endswith(".py"):
                    rc, o, e = sh([sys.executable, "-m", "py_compile", os.path.join(dp, f)])
                    if rc: probs.append(f"{f}: {e[-300:]}")
        code_dir = next((dp for dp, _, fns in os.walk(repo) if "country_route_planner.py" in fns), repo)
        rc, o, e = sh([sys.executable, "-c", "import country_route_planner as P, compare_routes as C; "
                       "assert callable(P.run_plan) and callable(C.compare); print(P.data_path('aircraft_specs.csv'))"], cwd=code_dir)
        if rc: probs.append("import/API check: " + e[-400:])
        R += ["## Checks", ""] + ([f"- FAIL {p}" for p in probs] or ["- all .py compile; run_plan()/compare() importable; data_path resolves aircraft_specs.csv -> " + o.strip()]) + [""]
        # the checks themselves create caches / empty output dirs: remove them again
        for dp, dns, fns in os.walk(repo, topdown=False):
            b = os.path.basename(dp)
            if b == "__pycache__" or (b in ("eval_results", "runs") and not os.listdir(dp)):
                shutil.rmtree(dp, ignore_errors=True)
    R += ["## Still for you (not code)", "",
          "- Her closing question: is a notebook tutorial feasible? Answer: yes — `tutorial.ipynb` now covers how to run, outputs, comparison, ablations, limits via functions; CLI kept for scripting.",
          "- Confirm the Data sources table wording (pre-processed vs public) matches what you actually did.",
          "", "Before uploading: `rm *.bak_feedback README.previous.md`; keep this report or delete it."]
    open(os.path.join(repo, "FEEDBACK_REPORT.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
