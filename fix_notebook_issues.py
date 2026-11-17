#!/usr/bin/env python3
"""
fix_notebook_issues.py — fixes from the first external run of tutorial.ipynb.

  python fix_notebook_issues.py --repo ./agentic_route_planning

1. country_route_planner.py
   - load_fcg: drop rows whose alpha3 code is empty/NaN (newer extracts have them) and
     stringify the rest -> no float codes anywhere downstream
   - build_dataset: sort unmatched codes with key=str (belt and braces)
2. tutorial.ipynb setup cell: the gateway option is explicit (LLM_BACKEND="gateway") and
   the cell prints which backend is active so a reader can tell it took effect
3. README.md Environment set-up: add `pip install notebook ipykernel` before the kernel
   registration line; mention that the notebook server is not in requirements.txt
4. requirements.txt: nothing changed (notebook/ipykernel are dev tools, kept in setup text)
Idempotent; backups: *.bak_nbfix. Report printed and written to FEEDBACK_REPORT_3.md.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime


def patch_text(path, edits, label, report):
    src = open(path, encoding="utf-8").read()
    changed = False
    for old, new, marker in edits:
        if marker in src:
            report.append(f"- {label}: already applied — {marker.strip()[:50]}")
            continue
        if src.count(old) != 1:
            report.append(f"- {label}: ANCHOR NOT FOUND ({src.count(old)}x) — {old.strip().splitlines()[0][:60]}")
            continue
        src = src.replace(old, new)
        changed = True
        report.append(f"- {label}: applied — {marker.strip()[:50]}")
    if changed:
        if not os.path.exists(path + ".bak_nbfix"):
            shutil.copy2(path, path + ".bak_nbfix")
        open(path, "w", encoding="utf-8").write(src)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args([x for x in sys.argv[1:] if x.replace("\xa0", "").strip()])
    repo = os.path.abspath(a.repo)
    R = [f"# Notebook-run fixes — {datetime.now():%Y-%m-%d %H:%M}", ""]

    # ---------------- 1. planner
    planner = next((os.path.join(dp, "country_route_planner.py") for dp, _, fns in os.walk(repo)
                    if "country_route_planner.py" in fns and "agentplanner" not in dp), None)
    if not planner:
        sys.exit("country_route_planner.py not found")
    src = open(planner, encoding="utf-8").read()
    edits = []
    # (a) load_fcg: find the code-column normalisation, whatever its exact form, and add a NaN drop before it
    m = re.search(r"^(\s*)df\[FCG_CODE_COL\] = \(df\[FCG_CODE_COL\]\.astype\(str\)", src, re.M)
    if m:
        ind = m.group(1)
        old = m.group(0)
        new = (f"{ind}df = df[df[FCG_CODE_COL].notna() & (df[FCG_CODE_COL].astype(str).str.strip() != \"\")]  # drop rows with no country code\n"
               + old)
        edits.append((old, new, "drop rows with no country code"))
    else:
        # older form: df[FCG_CODE_COL] = df[FCG_CODE_COL].str.upper() or similar — insert after read_csv in load_fcg
        m2 = re.search(r"(def load_fcg\(.*?\n(?:.*\n)*?)(\s*)(df = pd\.read_csv\(path\)\n)", src)
        if m2:
            ind = m2.group(2)
            old = m2.group(2) + m2.group(3)
            new = old + f"{ind}df = df[df[FCG_CODE_COL].notna() & (df[FCG_CODE_COL].astype(str).str.strip() != \"\")]  # drop rows with no country code\n{ind}df[FCG_CODE_COL] = df[FCG_CODE_COL].astype(str)\n"
            edits.append((old, new, "drop rows with no country code"))
        else:
            R.append("- planner: could not locate load_fcg's read_csv; add the NaN drop by hand")
    # (b) sorted(...) with mixed types
    old_sort = 'unmatched = sorted(fcg_codes - set(coords["iso3"]))'
    if old_sort in src:
        edits.append((old_sort, 'unmatched = sorted((c for c in fcg_codes - set(coords["iso3"]) if isinstance(c, str)), key=str)',
                      'if isinstance(c, str)), key=str)'))
    patch_text(planner, edits, "country_route_planner.py", R)
    # compile check
    import py_compile
    try:
        py_compile.compile(planner, doraise=True)
        R.append("- country_route_planner.py compiles")
    except py_compile.PyCompileError as e:
        R.append(f"- COMPILE ERROR after patch: {e}; restoring backup")
        shutil.copy2(planner + ".bak_nbfix", planner)

    # ---------------- 2. notebook setup cell
    nb_path = os.path.join(repo, "tutorial.ipynb")
    if os.path.exists(nb_path):
        nb = json.load(open(nb_path, encoding="utf-8"))
        done = False
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if "import country_route_planner as P" in s and "LITELLM_BASE_URL" in s:
                if "PICK ONE MODEL SITUATION" in s:
                    R.append("- tutorial.ipynb: setup cell already explicit")
                else:
                    new = (
                        "import os\n\n"
                        "# ======= PICK ONE MODEL SITUATION by un-commenting its block (or none -> llm=False later) =======\n\n"
                        "# (a) GATEWAY / remote OpenAI-compatible endpoint (what this package was verified with)\n"
                        "# os.environ[\"LLM_BACKEND\"]     = \"gateway\"\n"
                        "# os.environ[\"LITELLM_BASE_URL\"] = \"https://<gateway>/v1\"\n"
                        "# os.environ[\"LITELLM_API_KEY\"]  = \"<key>\"\n"
                        "# os.environ[\"AGENT_MODEL\"]      = \"<model name on the gateway>\"\n"
                        "# os.environ[\"AGENT_TLS_VERIFY\"] = \"false\"      # only if the gateway cert is self-signed\n\n"
                        "# (b) LOCAL OpenAI-compatible server (vLLM / Ollama / llama.cpp), already running\n"
                        "# os.environ[\"LLM_BACKEND\"] = \"vllm\"\n"
                        "# os.environ[\"VLLM_URL\"]    = \"http://localhost:8000/v1/chat/completions\"\n"
                        "# os.environ[\"VLLM_MODEL\"]  = \"<served model name>\"\n\n"
                        "# (c) IN-PROCESS model from a folder (needs torch + transformers, a GPU)\n"
                        "# os.environ[\"LLM_BACKEND\"]      = \"hf\"\n"
                        "# os.environ[\"LOCAL_MODEL_PATH\"] = \"/path/to/saved_model_dir\"\n\n"
                        "import country_route_planner as P\n"
                        "import compare_routes as C\n\n"
                        "USE_LLM = P.LLM_BACKEND == \"gateway\" and bool(os.environ.get(\"AGENT_MODEL\")) \\\n"
                        "          or P.LLM_BACKEND == \"vllm\" and bool(os.environ.get(\"VLLM_MODEL\")) \\\n"
                        "          or P.LLM_BACKEND == \"hf\" and bool(os.environ.get(\"LOCAL_MODEL_PATH\"))\n"
                        "print(\"backend:\", P.LLM_BACKEND, \"| model configured:\", USE_LLM)\n"
                        "if not USE_LLM:\n"
                        "    print(\"No model configured -> cells below run with llm=False (no FCG verdicts). \"\n"
                        "          \"Un-comment one block above and re-run this cell to use a model.\")\n"
                    )
                    c["source"] = new
                    R.append("- tutorial.ipynb: setup cell rewritten (explicit LLM_BACKEND per option, prints active backend)")
                done = True
                break
        # the plan cell should use USE_LLM from the setup cell, not recompute it
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                if 'USE_LLM  = bool(os.environ.get("AGENT_MODEL")' in s:
                    c["source"] = s.replace('USE_LLM  = bool(os.environ.get("AGENT_MODEL") or os.environ.get("LLM_BACKEND"))\n', "")
                    R.append("- tutorial.ipynb: plan cell uses USE_LLM from the setup cell")
        if not done:
            R.append("- tutorial.ipynb: setup cell not recognised; edit by hand")
        json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1)
    else:
        R.append("- tutorial.ipynb not found")

    # ---------------- 3. README setup
    rd = os.path.join(repo, "README.md")
    if os.path.exists(rd):
        t = open(rd, encoding="utf-8").read()
        if "pip install notebook ipykernel" in t:
            R.append("- README: notebook install line already present")
        else:
            kern = "conda install -c anaconda ipykernel -y && python -m ipykernel install --user --name=agentplanner"
            if kern in t:
                t = t.replace(kern, "pip install notebook ipykernel   # notebook server + kernel are not in requirements.txt\n"
                                    + kern, 1)
                R.append("- README: added `pip install notebook ipykernel` before the kernel registration line")
            else:
                m = re.search(r"pip install -r requirements\.txt[^\n]*\n", t)
                if m:
                    t = t[:m.end()] + "pip install notebook ipykernel   # notebook server + kernel are not in requirements.txt\n" + \
                        "conda install -c anaconda ipykernel -y && python -m ipykernel install --user --name=agentplanner\n" + t[m.end():]
                    R.append("- README: added notebook/ipykernel lines after pip install -r requirements.txt")
                else:
                    R.append("- README: could not find the install lines; add `pip install notebook ipykernel` by hand")
            shutil.copy2(rd, rd + ".bak_nbfix")
            open(rd, "w", encoding="utf-8").write(t)

    R += ["", "Reply to the tester: thanks — (1) `pip install notebook ipykernel` is now in the setup steps; (2) the setup cell "
              "now shows the gateway option explicitly with LLM_BACKEND; (3) the TypeError came from rows with an empty "
              "country code in the newer FCG extract — fixed in load_fcg.", "",
          "Before uploading: `rm *.bak_nbfix`."]
    open(os.path.join(repo, "FEEDBACK_REPORT_3.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
