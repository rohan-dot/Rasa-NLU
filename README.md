# Phase A+ with hybrid (PubMed + FAISS) — best for submission
python main.py --phase A+ --test testset.json --train trainining14b.json

# Phase A+ offline (FAISS only, if PubMed is unreachable)
python main.py --phase A+ --test testset.json --train trainining14b.json --no-pubmed

# Phase B (gold snippets, same as before)
python main.py --phase B --test BioASQ-task14bPhaseB-testset1.json --train trainining14b.json
