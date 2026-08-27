grep -c "CK_LOCK" fcg_extract.py && grep -c "Resume:" opus_learn.py


nohup env LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=sk-YOUR-KEY AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false FCG_GUIDED=0 python opus_learn.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out opus_all.csv > opus_all.log 2>&1 &
