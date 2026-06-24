Nice — glad it's working now. Here's a clean deck outline you can copy-paste and fill. I kept the wording plain so it reads as yours.

**Title slide**
Automated Extraction of Aircraft Clearance Data from Foreign Clearance Guide Pages
Subtitle: An offline LLM pipeline for structured, verifiable checklist extraction
(Your name · team · date)

**Slide 1 — Introduction / Problem**
- Foreign Clearance Guide pages hold the entry/exit rules crews need, but the answers are scattered across long, inconsistently formatted documents.
- Manually pulling a fixed checklist per country is slow and error-prone.
- Goal: turn a folder of country pages into one structured CSV — a row per country, named columns per checklist item — with nothing invented.

**Slide 2 — Approach**
- Clean each page to plain text, then slice to the relevant sections (I and II) before the model sees it.
- A locally served Gemma model fills a constrained JSON schema: one verbatim "raw" answer and one short summary per checklist item.
- Per-item handling: list-type items (e.g. authorized airports) return the full roster; others return the minimal supporting span.
- A strict second pass verifies each answer against the source and flags anything corrected — runs fully offline.

**Slide 3 — Dataset / Document Style**
- Source: per-country Foreign Clearance Guide pages (HTML/.cfm), one file per country, named by code.
- Structure: numbered sections and subsections (e.g. Section II.B "Lead-Time and Validity") with deeply nested lists and long enumerated airport rosters.
- Challenges: identical headings worded slightly differently across countries, cross-references, and the same field appearing in multiple contexts.
- Checklist fields extracted: diplomatic lead time, HAZMAT, entry/exit airports, airfield restrictions, customs/immigration, operating hours, country-specific forms, Air Card/cash.

**Slide 4 — Results** *(to fill)*
- Coverage: __ countries processed, __ fields per country.
- Accuracy: __% fields correct vs. a hand-checked sample; common error types: __.
- List fields: airport roster completeness __.
- Throughput: __ per country / full set in __.
- Example row + a screenshot of the CSV.

Want me to build this into an actual .pptx like last time, or keep it as copy-paste text? Either works — just say which.


Got it — here's a methodology-focused set, walking through how the code actually works. Copy-paste and trim as you like.

**Slide 1 — Pipeline Overview**
- Input: a folder of country pages (HTML/.cfm); output: one CSV, a row per country.
- Four stages per file: clean → slice → LLM extract → verify → write row.
- Deterministic steps do the heavy lifting; the model is used only where judgment is needed.
- Runs fully offline against a locally served model.

**Slide 2 — Text Preparation (Clean + Slice)**
- Clean: strip scripts, styles, nav, and markup with BeautifulSoup; keep visible body text and tables.
- Slice: cut the text to Sections I–II using case-sensitive heading markers (`SECTION I:` → `SECTION III:`), anchored on the colon so inline cross-references like "see Section III" don't trigger the cut.
- Guard: if a slice comes out suspiciously small, fall back to the full text instead of feeding the model almost nothing.
- Result: the model receives only the relevant, low-noise region.

**Slide 3 — Constrained Extraction**
- The checklist is defined as a list of items; each item's ID becomes two CSV columns: `<item>_raw` and `<item>_summary`.
- A JSON schema is built from that list and passed to the model via constrained decoding, so output is always valid, schema-shaped JSON.
- Per-item prompting: default items return the minimal verbatim span; items flagged as lists (e.g. airports) return the complete roster.
- Rules baked into the prompt: copy verbatim, summary may only use facts from raw, never invent, use "NA" when absent, and don't merge conflicting values — keep each with its qualifier.

**Slide 4 — Verification & Reliability**
- Strict second pass: the model re-checks every field against the source text and corrects unsupported answers, with a safeguard so it can't blank a field the first pass already answered.
- Deterministic verbatim check: each raw span is confirmed to exist in the source; mismatches are flagged.
- A `flags` column records what happened per row (verified, corrected, repaired, etc.) so output is self-documenting.
- Precedence rule resolves duplicates — the authoritative section wins over incidental mentions.

**Slide 5 — Robustness & Scale**
- Async concurrency processes many files at once to saturate the local server.
- Retry with token-budget escalation; malformed or truncated JSON is auto-repaired and salvaged rather than lost.
- Resume logic re-runs only the rows that failed, so large jobs restart cleanly.
- Tunable knobs: concurrency, max tokens, verify on/off, section markers, and the checklist itself.

Want this as a built .pptx, or leave it as copy-paste text?
