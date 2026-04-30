# Your results
python eval_any.py --pred batch4_local_phaseA.json --gold 13B1_golden.json

# His results
python eval_any.py --pred 13B1_golden.multihop.optimized.json --gold 13B1_golden.json

# His other version
python eval_any.py --pred 13B1_golden.rag.optimized.json --gold 13B1_golden.json
