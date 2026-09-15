#!/usr/bin/env python3
"""
build_eval_deck.py — build a detailed PowerPoint from the planner run + eval outputs.

It does NOT look at plot pixels. It reads the actual data the planner and the
evaluation wrote, so every sentence on a slide is backed by a number or a
verdict from your run:

  last_run_blocks.json (or --blocks-file)  planner decisions: stops, corridors,
                                           alternatives considered, judge
                                           verdicts with reasons, risk blocks
  eval_results/endpoints_compare.csv       human vs planner (endpoints-only)
  eval_results/endpoints_legs.csv          per-leg detail for both routes
  eval_results/reference_compare.csv       fixed-stops comparison (optional)
  eval_results/routing_runs.csv,
              penalty_sweep.csv,
              geometry_per_route.csv       benchmark stats (optional)
  eval_results/judge_scores.json           judge vs human (optional)
  eval_results/plots/*.png                 figures (optional, embedded if present)
  airports.csv + ne_110m_admin_0_countries.geojson   for the route map

Usage:
  python build_eval_deck.py --blocks-file blocks_m1.json \\
      --reference "KSUU,TKPK,DGAA,FKKD,FKKL" --aircraft C-17 \\
      -o fcg_planner_eval.pptx [--llm]

--llm adds a model-written narrative to the speaker notes of the decision
slide (uses the planner's vLLM endpoint). Everything else is deterministic.

pip install python-pptx matplotlib pandas
"""

import argparse
import json
import os
import sys
from datetime import date

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

OUT = "eval_results"
PLOTS = f"{OUT}/plots"
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x5F, 0x66, 0x70)
ACCENT = RGBColor(0x18, 0x5F, 0xA5)
W, H = Inches(13.333), Inches(7.5)

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def load_csv(name):
    p = f"{OUT}/{name}"
    try:
        return pd.read_csv(p) if os.path.exists(p) else None
    except pd.errors.EmptyDataError:
        return None


def text(slide, x, y, w, h, paras, size=16, bold=False, color=INK,
         bullets=False, space=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(paras):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(o.get("space", space))
        r = p.add_run()
        r.text = ("• " if o.get("bullet", bullets) else "") + t
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", False)
        r.font.color.rgb = o.get("color", color)
    return box


def title(slide, t, sub=None):
    text(slide, 0.6, 0.35, 12.1, 0.9, [t], size=28, bold=True)
    if sub:
        text(slide, 0.6, 1.1, 12.1, 0.5, [sub], size=14, color=MUTED)


def picture(slide, path, x, y, w, h):
    if not os.path.exists(path):
        text(slide, x, y, w, h, [f"(figure not found: {path})"], size=12,
             color=MUTED)
        return
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))
    if pic.width > Inches(w):
        s = Inches(w) / pic.width
        pic.width, pic.height = int(pic.width * s), int(pic.height * s)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def table(slide, x, y, w, rows, col_w=None, size=11, header=True):
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w),
                                 Inches(0.3 * nrows))
    tb = shp.table
    if col_w:
        for i, cw in enumerate(col_w):
            tb.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tb.cell(r, c)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = (r == 0 and header)
                    run.font.color.rgb = INK
    return shp

# ----------------------------------------------------------------------------
# optional LLM narration (gateway = LiteLLM/Opus via env vars; else planner vLLM)
# ----------------------------------------------------------------------------

_NARR = {"mode": None, "client": None, "model": None}

def narrator_setup(mode):
    _NARR["mode"] = mode
    if mode == "gateway":
        try:
            import httpx
            from openai import OpenAI
            base = os.environ.get("LITELLM_BASE_URL")
            key = os.environ.get("LITELLM_API_KEY")
            model = os.environ.get("AGENT_MODEL")
            if not (base and key and model):
                print("[llm] LITELLM_BASE_URL / LITELLM_API_KEY / AGENT_MODEL not set; "
                      "narration disabled")
                _NARR["mode"] = None
                return
            verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
            _NARR["client"] = OpenAI(base_url=base, api_key=key,
                                     http_client=httpx.Client(verify=verify, timeout=180))
            _NARR["model"] = model
            print(f"[llm] narration via gateway model {model}")
        except Exception as e:
            print(f"[llm] gateway unavailable ({e}); narration disabled")
            _NARR["mode"] = None
    elif mode == "vllm":
        print("[llm] narration via planner vLLM endpoint")


def narrate(prompt, max_words=120):
    """Grounded narration: the prompt must contain all facts; returns '' on failure."""
    sysmsg = ("You write plain-English text for slides. Use ONLY the facts in the "
              "prompt; never add numbers, names, or claims that are not there. "
              f"At most {max_words} words. No markdown.")
    try:
        if _NARR["mode"] == "gateway":
            r = _NARR["client"].chat.completions.create(
                model=_NARR["model"], max_tokens=600,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip()
        if _NARR["mode"] == "vllm":
            import country_route_planner as P
            return P.llm_chat(prompt, system=sysmsg).strip()
    except Exception as e:
        print(f"[llm] narration failed: {e}")
    return ""

# ----------------------------------------------------------------------------
# route map
# ----------------------------------------------------------------------------

def make_map(ref_stops, planner_stops, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        ap = pd.read_csv("airports.csv", usecols=["ident", "latitude_deg",
                                                  "longitude_deg"])
        ap = ap.set_index("ident")
    except Exception as e:
        print(f"[map] airports.csv unavailable: {e}")
        return False
    import math

    def pt(ident):
        r = ap.loc[ident]
        return float(r["latitude_deg"]), float(r["longitude_deg"])

    def gc(p1, p2, n=60):
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        v1 = (math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1))
        v2 = (math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2))
        ang = math.acos(max(-1, min(1, sum(a * b for a, b in zip(v1, v2)))))
        pts = []
        for i in range(n + 1):
            f = i / n
            if ang < 1e-9:
                w1, w2 = 1 - f, f
            else:
                w1 = math.sin((1 - f) * ang) / math.sin(ang)
                w2 = math.sin(f * ang) / math.sin(ang)
            x, y, z = (w1 * a + w2 * b for a, b in zip(v1, v2))
            pts.append((math.degrees(math.atan2(y, x)), math.degrees(math.asin(z))))
        return pts

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # borders
    try:
        gj = json.load(open("ne_110m_admin_0_countries.geojson", encoding="utf-8"))
        for f in gj["features"]:
            g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for poly in polys:
                xs = [c[0] for c in poly[0]]
                ys = [c[1] for c in poly[0]]
                ax.fill(xs, ys, color="#E5E3DC", linewidth=0.3, edgecolor="#B4B2A9")
    except Exception as e:
        print(f"[map] borders unavailable ({e}); drawing routes only")
    all_lons, all_lats = [], []
    for stops, color, label, ls in [(ref_stops, "#D85A30", "human route", "--"),
                                    (planner_stops, "#1D9E75", "planner route", "-")]:
        first = True
        for a, b in zip(stops, stops[1:]):
            try:
                pts = gc(pt(a), pt(b))
            except KeyError as e:
                print(f"[map] unknown airport {e}")
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                    linestyle=ls, linewidth=2, label=label if first else None)
            first = False
        for s in stops:
            try:
                la, lo = pt(s)
            except KeyError:
                continue
            all_lons.append(lo); all_lats.append(la)
            ax.plot(lo, la, "o", color=color, markersize=6)
            ax.annotate(s, (lo, la), textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=color)
    if all_lons:
        ax.set_xlim(min(all_lons) - 12, max(all_lons) + 12)
        ax.set_ylim(min(all_lats) - 12, max(all_lats) + 12)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return True

# ----------------------------------------------------------------------------
# deck
# ----------------------------------------------------------------------------

def build(args):
    blocks = json.load(open(args.blocks_file))
    ep = load_csv("endpoints_compare.csv")
    legs = load_csv("endpoints_legs.csv")
    refc = load_csv("reference_compare.csv")
    runs = load_csv("routing_runs.csv")
    sweep = load_csv("penalty_sweep.csv")
    geom = load_csv("geometry_per_route.csv")
    judge = None
    if os.path.exists(f"{OUT}/judge_scores.json"):
        judge = json.load(open(f"{OUT}/judge_scores.json"))
    cfg = blocks.get("config", {})
    pen = cfg.get("clearance_penalty_nm", 300)
    land_pen = cfg.get("landing_penalty_nm", 400)
    ref_stops = [s.strip().upper() for s in args.reference.split(",")]
    pl_stops = blocks.get("stops", [])
    verdicts = blocks.get("verdicts", {})
    ac = blocks.get("aircraft") or {}

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]

    # 1 title
    s = prs.slides.add_slide(blank)
    text(s, 0.8, 2.3, 11.7, 1.4, ["Agentic foreign-clearance route planning"],
         size=38, bold=True)
    text(s, 0.8, 3.7, 11.7, 1.2,
         [f"How the planner chose a route for {pl_stops[0] if pl_stops else '?'} -> "
          f"{pl_stops[-1] if pl_stops else '?'}, and how it compares to a "
          f"human-planned mission", "Method, decisions, and quantitative evaluation"],
         size=18, color=MUTED)
    text(s, 0.8, 6.5, 11.7, 0.5, [date.today().strftime("%B %d, %Y")], size=12,
         color=MUTED)
    notes(s, "Deck generated from the planner's own run outputs and the "
             "evaluation harness. Every number is traceable to a CSV or JSON "
             "file produced by the run.")

    # 2 problem
    s = prs.slides.add_slide(blank)
    title(s, "The problem", "Turning hundreds of pages of clearance prose into a compliant route")
    text(s, 0.6, 1.8, 12.1, 5, [
        "The Foreign Clearance Guide describes, per country, overflight permission, "
        "diplomatic lead times, designated entry airports, customs, HazMat and "
        "operating rules — as prose.",
        "Planning one mission means cross-referencing every country the route "
        "touches, and the answer changes with the mission date, the aircraft, and "
        "the state of the world.",
        "We built a pipeline that takes an origin, a destination, a date and an "
        "aircraft, and returns a route with every clearance requirement judged "
        "from the FCG text — or an honest statement that no compliant route exists.",
    ], size=17, bullets=True, space=12)
    notes(s, "Frame the pain: manual cross-referencing across dozens of countries, "
             "date-dependent lead times, no tool that reasons over the FCG text.")

    # 3 architecture
    s = prs.slides.add_slide(blank)
    title(s, "How it works: math decides geometry, the model decides feasibility")
    text(s, 0.6, 1.7, 6.0, 5.3, [
        ("Deterministic code", {"bold": True, "size": 18}),
        "Great-circle geometry, exact overflight detection against country borders",
        "Dijkstra over countries as stepping stones (any path length)",
        "Dijkstra over designated airports to insert fuel stops within aircraft range",
        "Dynamic program to pick the best entry airport per stop",
        "Lead-time arithmetic: days available vs days required",
    ], size=14, bullets=True, space=8)
    text(s, 6.9, 1.7, 6.0, 5.3, [
        ("Local LLM (vLLM, air-gapped)", {"bold": True, "size": 18}),
        "Resolves user input against a closed world: codes must exist in our data",
        "Reads each country's raw FCG text and returns a strict-JSON verdict "
        "(allowed / caution / minimum lead days) for its role: overflight or landing",
        "Writes the plain-language briefing",
        "It never supplies facts from memory; every verdict is auditable to the "
        "text it read",
    ], size=14, bullets=True, space=8)
    notes(s, "Key design principle: anything factual or arithmetic is code; the "
             "model only does judgment over text, with structured output that "
             "code validates. The plan-check-replan loop converges because each "
             "iteration permanently removes an option.")

    # 4 cost model
    s = prs.slides.add_slide(blank)
    title(s, "The cost model: miles plus operational burden")
    text(s, 0.6, 1.7, 12.1, 5.3, [
        f"Route score = distance (NM) + {pen:,.0f} NM per country needing overflight "
        f"clearance + {land_pen:,.0f} NM per extra landing" +
        (f" + {cfg.get('risk_avoid_penalty_nm', 1500):,.0f} NM per risk-listed country"
         if cfg.get('risk_avoid_penalty_nm') else ""),
        f"Interpretation: one diplomatic clearance is worth a {pen:,.0f} NM detour. "
        f"International waters cost nothing — nobody to ask.",
        "Hard constraints (never crossed): countries the FCG judge denies for this "
        "date, and countries on the geopolitical avoid list.",
        "Three corridor candidates per leg — direct great circle, penalized land "
        "graph, stepping-stone search via other countries — lowest score wins.",
        f"Fuel stops: any leg over {cfg.get('range_fraction', 0.85)*100:.0f}% of "
        f"aircraft range is split by searching designated airports; each landing "
        f"costs {land_pen:,.0f} NM so the planner does not hop needlessly.",
    ], size=15, bullets=True, space=12)
    notes(s, "The penalty is the single knob that encodes routing philosophy. "
             "The Pareto sweep later shows what happens when it changes.")

    # 5 test setup
    s = prs.slides.add_slide(blank)
    title(s, "The test: same inputs a human planner gets")
    text(s, 0.6, 1.7, 12.1, 5.3, [
        f"Origin {ref_stops[0]}, destination {ref_stops[-1]}, mission date "
        f"{blocks.get('date')}, aircraft {ac.get('type', 'n/a')} "
        f"(range {ac.get('range_nm', 0):,.0f} NM).",
        f"Human-planned reference route: {' -> '.join(ref_stops)}",
        f"Planner route (chose its own stops): {' -> '.join(pl_stops)}",
        "Both scored with the same rule; the FCG judge's denials from the planner "
        "run were applied to both.",
        f"Query given to the planner: \"{blocks.get('query')}\"",
    ], size=16, bullets=True, space=12)
    notes(s, "Endpoints-only test: the planner was not told the human's stops. "
             "A second test (later slide) fixes the stops and compares only the "
             "corridors between them.")

    # 6 map
    map_png = f"{PLOTS}/route_map.png"
    os.makedirs(PLOTS, exist_ok=True)
    if make_map(ref_stops, pl_stops, map_png):
        s = prs.slides.add_slide(blank)
        title(s, "The two routes on a map")
        picture(s, map_png, 0.6, 1.5, 12.1, 5.6)
        notes(s, "Dashed = human route, solid = planner route. Dots are landings.")

    # 7 planner route leg by leg
    s = prs.slides.add_slide(blank)
    title(s, "The planner's route, leg by leg")
    rows = [["Leg", "NM", "Flies over", "Lands in", "Stop type"]]
    auto = set(blocks.get("auto_stops", []))
    if legs is not None:
        for _, r in legs[legs.who == "planner"].iterrows():
            dest = r.leg.split("->")[1]
            rows.append([r.leg, f"{r.nm:,.0f}", r.overflights if isinstance(r.overflights, str) and r.overflights else "open water",
                         r.lands_in, "auto fuel/staging stop" if dest in auto else "requested"])
    else:
        for c in blocks.get("corridors", []):
            ch = c["chosen"]
            rows.append([c["leg"], f"{ch['nm']:,}", " ".join(ch["clearances"]) or "open water",
                         c["countries"].split("->")[1].strip(), ""])
    table(s, 0.6, 1.6, 12.1, rows, col_w=[2.4, 1.2, 4.0, 1.5, 3.0], size=12)
    text(s, 0.6, 1.6 + 0.35 * len(rows) + 0.3, 12.1, 1.5, [
        f"Total {blocks.get('total_nm', 0):,} NM direct. Auto-inserted stops: "
        f"{', '.join(sorted(auto)) or 'none'}."], size=14, color=MUTED)
    notes(s, "Each leg is under the aircraft's range cap. 'Flies over' lists the "
             "countries whose borders the great-circle track actually crosses, "
             "detected against Natural Earth polygons.")

    # 8 why each stop: judge verdicts
    s = prs.slides.add_slide(blank)
    title(s, "Why each stop was accepted: the FCG judge's verdicts")
    paras = []
    for key, v in verdicts.items():
        code, role, ap = key.split("/")
        if role != "stop":
            continue
        tag = "OK" if v.get("allowed") and v.get("airport_ok", True) else "DENIED"
        paras.append((f"{ap} ({code}) — {tag}" + (", caution" if v.get("caution") else "") +
                      (f", min lead {v.get('min_lead_days')} d" if v.get("min_lead_days") else ""),
                      {"bold": True, "size": 13, "space": 2}))
        paras.append((str(v.get("reason", ""))[:260], {"size": 12, "space": 8}))
    text(s, 0.6, 1.5, 12.1, 5.7, paras or ["(no stop verdicts in blocks file)"], size=12)
    notes(s, "These are the model's one-sentence verdicts, produced from each "
             "country's raw FCG text for the landing role. Lead times were "
             "extracted as numbers and checked in code against the days "
             "available before the mission date.")

    # 9 why this corridor: options considered
    s = prs.slides.add_slide(blank)
    title(s, "Why this path between stops: options the search considered")
    paras = []
    for c in blocks.get("corridors", []):
        ch = c["chosen"]
        paras.append((f"{c['leg']}: chose {ch['option']} — {ch['nm']:,} NM, "
                      f"{len(ch['clearances'])} clearances "
                      f"({' '.join(ch['clearances']) or 'open water'}), score {ch['score']:,}",
                      {"bold": True, "size": 13, "space": 2}))
        for a in c.get("alternatives", [])[:3]:
            paras.append((f"    alt {a['option']}: {a['nm']:,} NM, "
                          f"{len(a['clearances'])} clearances, score {a['score']:,}",
                          {"size": 12, "space": 1, "color": MUTED}))
        if c.get("direct_rejected_blocked"):
            paras.append((f"    direct rejected: crosses blocked "
                          f"{', '.join(c['direct_rejected_blocked'])}",
                          {"size": 12, "space": 6, "color": MUTED}))
        else:
            paras.append(("", {"size": 6, "space": 4}))
    text(s, 0.6, 1.5, 12.1, 5.7, paras or ["(no corridor log in blocks file — rerun the planner with the latest version)"], size=12)
    notes(s, "For every leg the planner scores the direct great circle, the "
             "penalized land route and stepping-stone chains through other "
             "countries; the lowest score wins. Alternatives are shown so the "
             "choice is auditable.")

    # 9b FCG evidence: what the judge actually read, per country on the route
    try:
        import country_route_planner as P
        fcg = P.load_fcg(P.FCG_CSV)
        fcg = fcg.drop_duplicates(P.FCG_CODE_COL).set_index(P.FCG_CODE_COL)
        evid_cols = [c for c in P.FEASIBILITY_COLS if c in fcg.columns][:4]
    except Exception as e:
        print(f"[deck] FCG evidence slides skipped: {e}")
        fcg, evid_cols = None, []
    route_countries = []
    for c in blocks.get("stop_countries", []):
        if c not in route_countries:
            route_countries.append(c)
    for c in blocks.get("corridors", []):
        for x in c["chosen"]["clearances"]:
            if x not in route_countries:
                route_countries.append(x)
    if fcg is not None and route_countries:
        s = prs.slides.add_slide(blank)
        title(s, "The evidence: what the FCG says about each country on the route",
              "Next slides quote the source text the judge read, beside its verdict")
        text(s, 0.6, 1.8, 12.1, 5, [
            f"Countries on the planner's route, in order: {' -> '.join(route_countries)}",
            "For each: the raw FCG fields for overflight, lead time, entry/exit and "
            "customs, truncated for the slide; the full text is what the model read.",
            "The verdict line is the model's structured output for that country's "
            "role (landing or overflight); lead times were compared in code against "
            f"the {blocks.get('date')} mission date.",
        ], size=15, bullets=True, space=12)
        for code in route_countries[:8]:
            s = prs.slides.add_slide(blank)
            v_stop = [(k, v) for k, v in verdicts.items()
                      if k.split("/")[0] == code and k.split("/")[1] == "stop"]
            v_ov = [(k, v) for k, v in verdicts.items()
                    if k.split("/")[0] == code and k.split("/")[1] == "overflight"]
            role = "landing" if v_stop else ("overflight" if v_ov else "on route")
            title(s, f"{code}: FCG evidence ({role})")
            vparas = []
            for k, v in v_stop + v_ov:
                tag = "OK" if v.get("allowed") and v.get("airport_ok", True) else "DENIED"
                vparas.append((f"Verdict ({k.split('/')[1]}): {tag}"
                               + (f", min lead {v.get('min_lead_days')} days" if v.get("min_lead_days") else "")
                               + f" — {str(v.get('reason',''))[:220]}",
                               {"bold": True, "size": 12, "space": 6, "color": ACCENT}))
            eparas = []
            if code in fcg.index:
                row = fcg.loc[code]
                for col in evid_cols:
                    val = row.get(col)
                    if isinstance(val, str) and val.strip():
                        eparas.append((col.replace("_", " "), {"bold": True, "size": 11, "space": 1}))
                        eparas.append((val.strip()[:380] + ("…" if len(val.strip()) > 380 else ""),
                                       {"size": 10, "space": 6}))
            else:
                eparas.append(("(no FCG row for this code)", {"size": 11, "color": MUTED}))
            summary = ""
            if _NARR["mode"] and eparas:
                summary = narrate(
                    f"Country {code}, role {role}. Summarize in 2-3 plain sentences what "
                    f"these FCG excerpts require of a US military flight, for a "
                    f"non-expert audience:\n" + "\n".join(
                        p[0] for p in eparas if isinstance(p, tuple)), max_words=80)
            text(s, 0.6, 1.3, 12.1, 1.2, vparas or [("No verdict recorded for this country", {"size": 12, "color": MUTED})], size=12)
            if summary:
                text(s, 0.6, 2.4, 12.1, 0.9, [(summary, {"italic": True, "size": 12})], size=12)
            text(s, 0.6, 3.3 if summary else 2.5, 12.1, 4.0, eparas, size=10)
            notes(s, "Source: fcg_extract.csv row for " + code + ". " +
                  (summary or "Raw fields shown as extracted; verdict from the judge."))

    # 10 constraints applied
    s = prs.slides.add_slide(blank)
    title(s, "Constraints applied on this date")
    denied = [(k, v) for k, v in verdicts.items() if v and not v.get("allowed", True)
              and k.split("/")[1] == "overflight"]
    lead = [k for k, v in denied if "LEAD TIME" in str(v.get("reason", ""))]
    risk = [(k, v) for k, v in verdicts.items() if k.split("/")[1] == "risk"]
    text(s, 0.6, 1.6, 6.0, 5.5, [
        ("FCG overflight denials", {"bold": True, "size": 16}),
        f"{len(denied)} countries denied for this mission date "
        f"({len(lead)} because the required lead time exceeds the days available).",
        ", ".join(k.split("/")[0] for k, _ in denied) or "none",
    ], size=13, space=8)
    text(s, 6.9, 1.6, 6.0, 5.5, [
        ("Geopolitical avoid list", {"bold": True, "size": 16}),
        f"{sum(1 for _, v in risk if not v.get('allowed', True))} blocked, "
        f"{sum(1 for _, v in risk if v.get('allowed', True))} penalized "
        f"(manual list + State Dept advisories).",
        "; ".join(f"{k.split('/')[0]}: {str(v.get('reason',''))[:40]}"
                  for k, v in risk[:12]) + (" …" if len(risk) > 12 else ""),
    ], size=13, space=8)
    notes(s, "Blocked countries are removed from the graph before routing; "
             "penalized ones are routed around when an alternative exists.")

    # 10b how we evaluate (plain language, before any results)
    s = prs.slides.add_slide(blank)
    title(s, "How we evaluate, in plain language")
    text(s, 0.6, 1.6, 12.1, 5.6, [
        ("Test 2 — endpoints only (the main test)", {"bold": True, "size": 16, "space": 4}),
        "We give the planner exactly what a human planner gets: where to start, where "
        "to end, the date, the aircraft. It must choose its own fuel stops and paths. "
        "We then score its route and the human's route with one rule: miles flown, "
        f"plus {pen:.0f} NM for every country whose permission is needed, plus "
        f"{land_pen:.0f} NM for every landing. Lower is better. Think of the penalties "
        "as the paperwork and delay a clearance or a landing really costs.",
        ("Test 1 — same stops", {"bold": True, "size": 16, "space": 4}),
        "We hand the planner the human's stops and only let it choose how to fly each "
        "leg. This isolates one skill: picking the path with the fewest permissions.",
        ("Benchmarks (optional slides)", {"bold": True, "size": 16, "space": 4}),
        "Many random routes: switching parts of the method off to show what each part "
        "buys; sweeping the clearance penalty to show the miles-vs-permissions trade; "
        "checking our 'which countries do we fly over' detector against true borders; "
        "and comparing the model's clearance verdicts with a human's.",
    ], size=13, space=8)
    notes(s, "Say this before showing any number so the audience knows what a "
             "score means and why lower is better.")

    s = prs.slides.add_slide(blank)
    title(s, "What a fair comparison needs")
    text(s, 0.6, 1.6, 12.1, 5.6, [
        "Same inputs: origin, destination, date, aircraft, and the same clearance "
        "denials from the FCG judge apply to both routes.",
        "Same scoring rule, computed by code — the model never scores itself.",
        "The human route is scored as straight lines between its stops (its exact "
        "filed track is not known); if the real overflights are supplied the "
        "comparison becomes exact.",
        "Every overflight is detected against country borders, so 'flies over "
        "Nigeria' means the track enters Nigerian territory, not merely passes near it.",
        "The decision is arithmetic: whichever score is lower wins, and the slide "
        "shows where the difference came from.",
    ], size=15, bullets=True, space=12)
    notes(s, "Pre-empt the 'you rigged the comparison' question.")

    # 11 human vs planner numbers
    s = prs.slides.add_slide(blank)
    title(s, "Human route vs planner route: the numbers")
    if ep is not None and not ep.empty:
        r1 = ep[ep.who == "reference"].iloc[0]
        r2 = ep[ep.who == "planner"].iloc[0]
        rows = [["", "Human route", "Planner route"],
                ["Stops", r1.stops, r2.stops],
                ["Total distance", f"{r1.total_nm:,.0f} NM", f"{r2.total_nm:,.0f} NM"],
                ["Landings en route", str(r1.landings), str(r2.landings)],
                ["Countries needing clearance", str(r1.clearances), str(r2.clearances)],
                ["Legs beyond aircraft range", str(r1.legs_over_range), str(r2.legs_over_range)],
                [f"Score (NM + {pen:.0f}/clearance + {land_pen:.0f}/landing)",
                 f"{r1.score:,.0f}", f"{r2.score:,.0f}"]]
        table(s, 0.6, 1.6, 12.1, rows, col_w=[3.6, 4.25, 4.25], size=12)
        picture(s, f"{PLOTS}/endpoints_compare.png", 0.6, 4.2, 12.1, 3.0)
    else:
        text(s, 0.6, 1.6, 12.1, 2, ["(run eval_plots.py --endpoints-only first)"])
    notes(s, "Same scoring rule for both. The figure below the table shows the "
             "same numbers plus each route's legs.")

    # 12 decision
    s = prs.slides.add_slide(blank)
    title(s, "Decision and why")
    paras = []
    if ep is not None and not ep.empty:
        d = r1.score - r2.score
        if d > 0:
            paras.append((f"The planner's route is better by {d:,.0f} points.",
                          {"bold": True, "size": 20, "color": ACCENT}))
            paras.append(f"It flies {r2.total_nm - r1.total_nm:+,.0f} NM and needs "
                         f"{r2.clearances - r1.clearances:+d} clearances with "
                         f"{r2.landings - r1.landings:+d} landings. At {pen:.0f} NM per "
                         f"clearance the extra distance pays for itself.")
        elif d < 0:
            paras.append((f"The human's route is better by {-d:,.0f} points.",
                          {"bold": True, "size": 20, "color": ACCENT}))
            paras.append("The human route likely encodes knowledge the planner does "
                         "not have yet (fuel contracts, MOG, crew rest). The delta "
                         "measures how much that knowledge is worth.")
        else:
            paras.append(("The routes tie.", {"bold": True, "size": 20}))
        if legs is not None:
            hr = legs[legs.who == "reference"]
            pr = legs[legs.who == "planner"]
            paras.append(("Where the difference comes from", {"bold": True, "size": 15}))
            paras.append("Human: " + "; ".join(
                f"{r.leg} over {r.overflights if isinstance(r.overflights, str) and r.overflights else 'water'}"
                for _, r in hr.iterrows()))
            paras.append("Planner: " + "; ".join(
                f"{r.leg} over {r.overflights if isinstance(r.overflights, str) and r.overflights else 'water'}"
                for _, r in pr.iterrows()))
        if int(r1.legs_over_range) > 0:
            paras.append((f"Note: {int(r1.legs_over_range)} human leg(s) exceed 85% of "
                          f"the {ac.get('type','')}'s range — the reference may assume a "
                          f"different aircraft or air refueling.", {"italic": True, "color": MUTED}))
    text(s, 0.6, 1.6, 12.1, 5.5, paras or ["(no comparison data)"], size=14, space=10)
    nt = ("The decision is arithmetic on the cost model, not a judgment call. "
          "The trade-off is explicit: miles versus diplomatic clearances.")
    if _NARR["mode"] and ep is not None:
        extra = narrate(
            "Write speaker notes explaining, for a technical advisor, why the "
            "planner's route differs from the human route and what the trade-off "
            "is. Facts:\n" + ep.to_string()
            + "\n" + (legs.to_string() if legs is not None else ""), max_words=150)
        if extra:
            nt += "\n\n" + extra
    notes(s, nt)

    # 13 fixed-stops test
    if refc is not None and not refc.empty and os.path.exists(f"{PLOTS}/reference_compare.png"):
        s = prs.slides.add_slide(blank)
        title(s, "Second test: same stops, who flies each leg better?",
              "Stops fixed to the human route; only the corridor between stops is chosen")
        picture(s, f"{PLOTS}/reference_compare.png", 0.6, 1.7, 8.0, 5.3)
        tr, tp = refc.ref_gc_nm.sum(), refc.planner_nm.sum()
        cr, cp = refc.ref_clearances_detected.sum(), refc.planner_clearances.sum()
        text(s, 8.9, 1.7, 4.0, 5.3, [
            f"Reference straight lines: {tr:,.0f} NM, {cr} clearances",
            f"Planner corridors: {tp:,.0f} NM, {cp} clearances",
            f"Score: reference {tr + pen*cr:,.0f} vs planner {tp + pen*cp:,.0f}",
            ("Legs that differ: " + ", ".join(
                r.leg for _, r in refc.iterrows()
                if r.planner_nm != r.ref_gc_nm or r.planner_clearances != r.ref_clearances_detected) or "none",
             {"color": MUTED, "size": 12}),
        ], size=13, bullets=True, space=10)
        notes(s, "This isolates the corridor logic from stop selection.")

    # 14 ablation
    if runs is not None and os.path.exists(f"{PLOTS}/ablation.png"):
        s = prs.slides.add_slide(blank)
        title(s, "Routing ablation: what each part of the method buys",
              f"{runs.origin.nunique()} routes, four configurations, penalty {pen:.0f} NM")
        picture(s, f"{PLOTS}/ablation.png", 0.6, 1.7, 8.0, 5.3)
        d = runs[(runs.penalty == pen) & (runs.found == True)]
        g = d.groupby("config").agg(nm=("nm", "mean"), cl=("clearances", "mean"))
        lab = {"A_direct": "Direct great circle", "B_land_distance": "Shortest land route",
               "C_penalized_nostones": "Penalized, no ocean hops", "D_full": "Full system"}
        text(s, 8.9, 1.7, 4.0, 5.3,
             [f"{lab.get(k, k)}: {v.nm:,.0f} NM, {v.cl:.1f} clearances"
              for k, v in g.iterrows()] +
             [("Read: lowest clearance bar at an acceptable distance cost.",
               {"color": MUTED, "size": 12})], size=13, bullets=True, space=10)
        notes(s, "Each configuration removes one capability. The full system "
                 "trades a small distance increase for the largest clearance reduction.")

    # 15 pareto
    if sweep is not None and os.path.exists(f"{PLOTS}/pareto.png"):
        s = prs.slides.add_slide(blank)
        title(s, "Miles versus clearances: the exchange rate made explicit")
        picture(s, f"{PLOTS}/pareto.png", 0.6, 1.7, 7.0, 5.3)
        sw = sweep.sort_values("penalty")
        text(s, 7.9, 1.7, 5.0, 5.3,
             [f"penalty {r.penalty:.0f}: {r.nm_mean:,.0f} NM, {r.clearances_mean:.2f} clearances"
              for _, r in sw.iterrows()] +
             [("Each step left avoids clearances at increasing cost in miles. "
               "The planner operates at 300; the curve lets an operator move the dial.",
               {"color": MUTED, "size": 12})], size=13, bullets=True, space=8)
        notes(s, "This is the routing philosophy quantified.")

    # 16 geometry
    if geom is not None and os.path.exists(f"{PLOTS}/geometry.png"):
        s = prs.slides.add_slide(blank)
        title(s, "Overflight detection accuracy",
              "Our detector vs true country borders (Natural Earth)")
        picture(s, f"{PLOTS}/geometry.png", 0.6, 1.7, 8.0, 5.3)
        text(s, 8.9, 1.7, 4.0, 5.3, [
            f"Recall {geom.recall.mean():.2f} — fraction of truly overflown countries flagged",
            f"Precision {geom.precision.mean():.2f} — fraction of flags that were real",
            f"F1 {geom.f1.mean():.2f} over {len(geom)} routes",
            ("A missed country would be an unrequested clearance; an extra flag is "
             "only paperwork.", {"color": MUTED, "size": 12}),
        ], size=13, bullets=True, space=10)
        notes(s, "With polygon detection enabled, these should be ~1.0.")

    # 17 judge
    s = prs.slides.add_slide(blank)
    title(s, "Is the FCG judge trustworthy? Agreement with human labels")
    if judge:
        picture(s, f"{PLOTS}/judge.png", 0.6, 1.7, 7.0, 5.3)
        text(s, 7.9, 1.7, 5.0, 5.3, [
            f"n = {judge.get('n')} country/role cases",
            f"Accuracy {judge.get('accuracy')}, Cohen's kappa {judge.get('cohen_kappa')}",
            f"Deny precision {judge.get('deny_precision')}, recall {judge.get('deny_recall')}",
            f"Lead-time exact match {judge.get('lead_time_exact_match')}",
            f"Self-consistency (3 runs) {judge.get('self_consistency_3x')}",
        ], size=14, bullets=True, space=10)
    else:
        text(s, 0.6, 1.7, 12.1, 5, [
            "In progress: a 120-case labeling sheet (country x role x FCG text) has "
            "been generated; the model's verdict and extracted lead time are recorded "
            "three times per case.",
            "A human labels allow/deny and the minimum lead days blind to the model's "
            "answer; we then report accuracy, Cohen's kappa, deny precision/recall, "
            "lead-time exact match, and self-consistency.",
        ], size=15, bullets=True, space=12)
    notes(s, "The judge is the only non-deterministic component; this is the "
             "measurement that bounds its error.")

    # 18 limitations
    s = prs.slides.add_slide(blank)
    title(s, "Limitations and next steps")
    text(s, 0.6, 1.6, 12.1, 5.5, [
        "The reference route is the human's stop sequence with straight great circles "
        "between stops; the real filed routing may differ (feed actual overflights for "
        "an exact comparison).",
        "The human route may encode fuel availability, MOG, crew rest and relationships "
        "the planner does not model yet — candidates for new cost terms.",
        "Designated-airport extraction depends on ICAO codes being present in the FCG "
        "text; this extract yielded few, so auto-stops used large-airport fallbacks "
        "(flagged for review).",
        "Live weather and traffic are snapshots, not forecasts for the mission date.",
        "Next: label the judge sheet; add runway/weight-bearing hard filters; encode "
        "fuel-contract availability per airport; validate corridors against OpenSky "
        "historical tracks.",
    ], size=15, bullets=True, space=12)
    notes(s, "Be explicit about what is and isn't measured.")

    prs.save(args.out)
    print(f"[deck] wrote {args.out} with {len(prs.slides)} slides")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-file", default="last_run_blocks.json")
    ap.add_argument("--reference", required=True,
                    help='human route stops, e.g. "KSUU,TKPK,DGAA,FKKD,FKKL"')
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("-o", "--out", default="fcg_planner_eval.pptx")
    ap.add_argument("--llm", action="store_true",
                    help="narrate with the planner's local vLLM model")
    ap.add_argument("--gateway", action="store_true",
                    help="narrate with the LiteLLM gateway model (LITELLM_BASE_URL, "
                         "LITELLM_API_KEY, AGENT_MODEL env vars, e.g. Opus)")
    args = ap.parse_args()
    narrator_setup("gateway" if args.gateway else ("vllm" if args.llm else None))
    build(args)
