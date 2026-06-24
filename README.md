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
