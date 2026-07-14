"""
Chat layer: talk to your FCG + route data conversationally.

Same trust contract as agent.py - the model can only call verified tools and
must ground every claim in a tool result. Adds one extra action:
  {"thought": "...", "action": "respond", "message": "..."}   -> shown to you.

If you ask it to plan a route, it still cannot finalize one without passing
the independent validate_route check (it calls the tool and must report the
verdict, blockers and all).

Run:  python chat.py
Then: > plan a route from KIAD to VIDP departing 2026-08-20, no hazmat
      > what are the customs rules at ETAR?
      > can I overfly between ETAR and OTBH?
"""

import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from tools import Toolbox, TOOL_SPECS

# ------------------------- CONFIG (match agent.py) ------------------
MODEL         = "gemma-4-31B-it"
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

AIRPORTS_CSV  = "./data/airports.csv"
SCHEMA        = "ourairports"
RUNWAYS_CSV   = "./data/runways.csv"
COUNTRIES_CSV = "./data/countries.csv"
FCG_CSV       = "./fcg_extract.csv"
MAX_TOOL_TURNS_PER_REPLY = 12
# --------------------------------------------------------------------

ACTION_SCHEMA = {
    "type": "object",
    "properties": {"thought": {"type": "string"}, "action": {"type": "string"},
                   "args": {"type": "object"}, "message": {"type": "string"}},
    "required": ["thought", "action"],
}

SYSTEM = """You are a foreign-clearance and route-planning assistant on VERIFIED tools.

Rules:
1. Every fact (airports, distances, clearance rules, lead times, overflight)
   must come from a tool result in this conversation. If a tool returns NA or
   an error, say so plainly - never fill gaps from memory.
2. One JSON action per turn. Use {"action":"respond","message":"..."} to answer
   the user; any other action name calls a tool with "args".
3. Route requests: gather candidates and requirements with tools, then run
   validate_route and report its verdict verbatim, including blockers,
   warnings, and filing deadlines. A route is only "good" if validation says ok.
4. Always surface uncertainties (unknown runway lengths, unparsed lead times,
   missing FCG rows, approximate overflight results).

Tools and args:
""" + json.dumps({k: v["args"] for k, v in TOOL_SPECS.items()}, indent=1)


def parse_action(raw):
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[4:] if txt[:4].lower() == "json" else txt
    start = txt.find("{")
    return json.loads(txt[start:]) if start >= 0 else json.loads(txt)


def main():
    tb = Toolbox(AIRPORTS_CSV, FCG_CSV, schema=SCHEMA,
                 runways_csv=RUNWAYS_CSV, countries_csv=COUNTRIES_CSV)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
    print(f"Loaded {tb.ingest_report['accepted']} verified airports, "
          f"{len(tb.fcg)} FCG records. Ask away (ctrl-c to quit).")
    messages = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        for _ in range(MAX_TOOL_TURNS_PER_REPLY):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, max_tokens=1500,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "action", "schema": ACTION_SCHEMA}})
            try:
                act = parse_action(resp.choices[0].message.content or "")
            except Exception as e:
                messages.append({"role": "user",
                                 "content": f"Unparseable JSON ({e}); send one valid action."})
                continue
            messages.append({"role": "assistant", "content": json.dumps(act)})

            if act["action"] == "respond":
                print("\n" + act.get("message", "(empty)"))
                break
            if act["action"] not in TOOL_SPECS:
                messages.append({"role": "user",
                                 "content": f"Unknown tool. Allowed: {list(TOOL_SPECS)} or 'respond'."})
                continue
            args = {k: v for k, v in (act.get("args") or {}).items()
                    if k in TOOL_SPECS[act["action"]]["args"]}
            try:
                result = getattr(tb, act["action"])(**args)
            except TypeError as e:
                result = {"error": f"bad args: {e}"}
            print(f"  [tool] {act['action']}({', '.join(f'{k}={v}' for k, v in args.items())})")
            messages.append({"role": "user",
                             "content": "TOOL RESULT:\n" + json.dumps(result, default=str)[:8000]})
        else:
            print("(tool budget hit for this reply)")


if __name__ == "__main__":
    main()
