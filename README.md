export FCG_BUDGET_MEIS=20000
export FCG_BUDGET_SVC=10000
python fcg_extract_simple.py data/checklist/items.json \
    --countries data/fcg_countries --sources data/sources
