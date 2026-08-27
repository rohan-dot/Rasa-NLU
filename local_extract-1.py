#!/usr/bin/env python3
"""
local_extract.py — STAGE 2: run the SAME extraction with a LOCAL model loaded
directly in this process via transformers. No vLLM, no server, no ports, no
HTTP timeouts. Consumes LEARNINGS.md (written by opus_learn.py) as expert hints
injected into every prompt, so the weak model benefits from the strong model's
strategies.

Requires: torch, transformers, accelerate (pip install --user transformers accelerate)
Optional: bitsandbytes for --load-4bit if the model doesn't fit in GPU memory.

Usage (one line):
  python local_extract.py --model /panfs/g52-panfs/exp/FY26/models/gemma-4-31B-it \
    --checklist checklist_enriched.json --fcg-dir data/fcg \
    --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json \
    --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv \
    --countries ITA --learnings LEARNINGS.md --out local_answers.csv

Memory: bf16 31B wants ~62GB across your GPUs (device_map=auto spreads it).
If you OOM, add --load-4bit (needs bitsandbytes; ~17GB, fits one 24GB card).
"""

import argparse
import csv
import json
from pathlib import Path

import fcg_extract as fx  # loaders, chunker, merge, JSON repair

BATCH_FIELDS = 8          # smaller batches: kinder to a local model
MAX_NEW_TOKENS = 1200


def build_prompt(hints: str, dossier: str, fields):
    lines = "\n".join(f"- {it['col']}: {it['question']}" for it in fields)
    return (
        "You are a strict extraction engine for aviation mission planning.\n"
        + (f"EXPERT HINTS (follow these strategies):\n{hints}\n\n" if hints else "")
        + f"SOURCE DOSSIER:\n{dossier}\n\n"
        f"FIELDS:\n{lines}\n\n"
        "For each field, answer ONLY with text quoted verbatim from the dossier "
        "(join multiple snippets with ' | '). The dossier may use different "
        "wording than the field name — quote the passage that addresses the "
        "topic. Use \"NA\" only when nothing relevant exists. Respond ONLY with "
        "a single JSON object mapping each field name exactly to its answer "
        "string. Escape newlines as \\n inside strings. No markdown fences."
    )


class LocalModel:
    def __init__(self, path: str, load_4bit: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        fx.log(f"Loading tokenizer from {path} ...")
        self.tok = AutoTokenizer.from_pretrained(path)
        fx.log("Loading model (this takes a few minutes) ...")
        kw = dict(device_map="auto", torch_dtype=torch.bfloat16)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw = dict(device_map="auto",
                      quantization_config=BitsAndBytesConfig(
                          load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw)
        self.model.eval()
        fx.log("Model loaded.")

    def generate(self, prompt: str) -> str:
        import torch
        msgs = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        # apply_chat_template may return a BatchEncoding dict — take pieces explicitly
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc.get("attention_mask")
        attn = attn.to(self.model.device) if attn is not None else None
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text


def extract_over_chunks(lm, hints, text, fields, label):
    chunks = fx.chunk_text(text, fx.CHUNK_SIZE, fx.CHUNK_OVERLAP)
    fx.log(f"  {label}: {len(chunks)} chunk(s)")
    merged = {it["col"]: "NA" for it in fields}
    batches = [fields[i:i + BATCH_FIELDS] for i in range(0, len(fields), BATCH_FIELDS)]
    for ci, ch in enumerate(chunks):
        for batch in batches:
            prompt = build_prompt(hints, ch, batch)
            try:
                raw = lm.generate(prompt)
                ans = fx.parse_json_answer(raw, [it["col"] for it in batch])
            except Exception as e:
                fx.log(f"    chunk {ci+1} batch failed ({e}); continuing")
                continue
            fx.merge_answers(merged, ans)
        fx.log(f"    chunk {ci+1}/{len(chunks)} done")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local model weights")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit quantized load (fits one 24GB GPU; needs bitsandbytes)")
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--fcg-dir", required=True)
    ap.add_argument("--svc")
    ap.add_argument("--meis", nargs="*", default=[])
    ap.add_argument("--airports")
    ap.add_argument("--codes")
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--learnings", default="LEARNINGS.md")
    ap.add_argument("--out", default="local_answers.csv")
    args = ap.parse_args()

    items = fx.load_checklist(Path(args.checklist))
    extract_items = [it for it in items if it["type"] == "extract"]
    fx.log(f"Checklist: {len(items)} items ({len(extract_items)} extract)")

    hints = ""
    lp = Path(args.learnings)
    if lp.exists():
        hints = fx.cap(lp.read_text(encoding="utf-8", errors="replace"), 6000, "HINTS")
        fx.log(f"Loaded learnings hints: {len(hints)} chars")
    else:
        fx.log("No LEARNINGS.md found — running without hints (still works)")

    codes_map = fx.load_codes_csv(Path(args.codes)) if args.codes else {}
    airports_by_a2 = fx.load_airports_csv(Path(args.airports)) if args.airports else {}
    svc_by_icao = fx.load_svc(Path(args.svc)) if args.svc else {}
    meis_by_icao, meis_by_country = fx.load_meis(args.meis) if args.meis else ({}, {})

    fcg_files = sorted(Path(args.fcg_dir).glob("*.json"))
    if args.countries:
        want = {c.lower() for c in args.countries}
        fcg_files = [f for f in fcg_files if f.stem.lower() in want]
    if not fcg_files:
        fx.die("No FCG files matched")

    lm = LocalModel(args.model, args.load_4bit)

    rows = []
    for f in fcg_files:
        country, a2, a3, disp = fx.resolve_country(f, codes_map)
        fx.log(f"== {country} ({disp}) ==")
        row = {"country": country, "country_name": disp, "alpha2": a2 or "", "alpha3": a3 or ""}
        src = {}
        for it in items:
            if it["type"] != "extract":
                row[it["col"]] = "NA"
                src[it["col"]] = it["type"].upper()   # WORKFLOW / REFERENCE / etc.

        fcg_txt = fx.cap(fx.fcg_text_for(f), fx.BUDGET_FCG, "FCG")
        p1 = extract_over_chunks(lm, hints, fcg_txt, extract_items, f"{country} FCG")
        for it in extract_items:
            v = p1.get(it["col"], "NA")
            row[it["col"]] = v
            src[it["col"]] = "FCG" if v != "NA" else "NA"

        na = [it for it in extract_items if row[it["col"]] == "NA"]
        if na:
            icaos = airports_by_a2.get(a2, []) if a2 else []
            body = fx.build_ancillary_dossier(icaos, svc_by_icao, meis_by_icao,
                                              meis_by_country, disp)
            if body.strip():
                p2 = extract_over_chunks(lm, hints, body, na, f"{country} ANCILLARY")
                for it in na:
                    v = p2.get(it["col"], "NA")
                    if v != "NA":
                        row[it["col"]] = v
                        src[it["col"]] = "ANCILLARY"

        filled = sum(1 for it in extract_items if row[it["col"]] != "NA")
        fx.log(f"  {country}: FILLED {filled}/{len(extract_items)}")
        for it in items:
            row[f"{it['col']}__src"] = src[it["col"]]
        rows.append(row)

    cols = (["country", "country_name", "alpha2", "alpha3"]
            + [it["col"] for it in items] + [f"{it['col']}__src" for it in items])
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fx.log(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
