Here's the AI methodology as copy-paste text.

**AI Methodology — The Extraction Agent**

The core of the system is an agent built on a locally served Gemma model that runs two constrained passes over each document — one to extract, one to check its own work against the source.

**Two-pass agent loop**
- Section text: the cleaned, sliced source for one country is handed to the agent.
- Extract pass (LLM): the model fills a constrained JSON schema, producing two values per checklist item — a verbatim "raw" span copied from the source and a short grounded summary.
- Draft answers: schema-valid JSON; list-type items (e.g. the airport roster) are returned in full rather than sampled.
- Verify pass (LLM): a second, strict pass re-reads every field against the source and corrects unsupported answers, with a safeguard that prevents it from blanking a value the first pass already found.
- Final row: the verified answers, plus a QA flags column recording what was corrected or repaired, are written to the CSV.

**Principles that keep it honest**
- Constrained decoding: output is forced to valid, schema-shaped JSON, so the model can't drift into free-form text.
- Grounded answers: each summary may use only facts already present in its verbatim raw span — no new numbers or claims.
- Precedence and no-merge: when a value appears in more than one place, the authoritative section wins; genuinely conflicting values are kept separately with their qualifiers, never averaged into one wrong number.
- Never invents: fields absent from the text are marked "NA," and a deterministic check confirms every raw span actually exists in the source, flagging anything that doesn't.

**Why two passes**
- Extraction alone is fast but occasionally paraphrases or misses a value; verification alone can't generate answers. Running them in sequence lets the model catch and correct its own errors while a deterministic verbatim check backstops the parts the model can't judge — giving reliable output without human review of every row.
