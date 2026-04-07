You're absolutely right — the gold snippets from Phase A won't cover everything, and test questions can reference articles that don't appear anywhere in training. You need the actual PubMed baseline indexed locally so you can retrieve relevant passages for any question. Let me build that full pipeline.

Now let me update the Phase B solver to actually pull from the baseline index:

Now update the runner to pass through the baseline arg:

Here's how it all connects now:

**Step 1 — Download & index the PubMed baseline:**
```bash
pip install lxml requests vllm transformers torch rouge-score

# Quick test with 10 files (~300K articles):
python pubmed_baseline.py download \
    --output-dir ./pubmed_xml \
    --db pubmed_index.db \
    --start 1 --end 10

# Full baseline (~36M articles, takes hours):
python pubmed_baseline.py download \
    --output-dir ./pubmed_xml \
    --db pubmed_index.db
```

This connects to NCBI's FTP, downloads the gzipped XML files, parses every article (PMID, title, abstract, MeSH terms), and loads them into a SQLite database with FTS5 full-text search. It deletes the XML after indexing to save disk — the DB itself is much smaller.

**Step 2 — Run the solver with baseline augmentation:**
```bash
python run_pipeline.py submit \
    --test-input BioASQ-task14bPhaseB-testset1.json \
    --training training13b.json \
    --model google/gemma-2-9b-it \
    --pubmed-db pubmed_index.db \
    -o submission_phaseB.json
```

What happens per question now:
1. Takes gold snippets from Phase A
2. Looks up the gold document PMIDs in the local baseline to pull their full abstracts and extract additional relevant sentences
3. Searches the entire baseline for more articles matching the question keywords (BM25 via FTS5)
4. Feeds all of that combined evidence to Gemma through the agentic generate → verify → refine loop
5. Consensus across 3 passes

**Step 3 — Eval first on training data:**
```bash
python run_pipeline.py eval \
    --training training13b.json \
    --model google/gemma-2-9b-it \
    --pubmed-db pubmed_index.db \
    -n 30
```

This tells you your accuracy/F1/ROUGE before you submit anything.
