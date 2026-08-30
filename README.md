python -c "import json; d=json.load(open('checklist_enriched.json')); ov=[{'id':'Overflight Clearance Requirements','question':'Overflight clearance/permission requirements: lead times, validity, who approves, blanket/annual availability, and any overflight prohibitions or restricted routes/FIRs','type':'extract'},{'id':'Overflight Fees','question':'Overflight fees, air navigation charges, and how they are paid','type':'extract'}]; json.dump(ov,open('checklist_overflight.json','w'),indent=1); d2=d+ov; json.dump(d2,open('checklist_enriched.json','w'),indent=1); print('overflight-only checklist written; also appended to main checklist for future runs')"


nohup env LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=sk-YOUR-KEY AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false FCG_GUIDED=0 python opus_learn.py --checklist checklist_overflight.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out overflight.csv > overflight.log 2>&1 &


python -c "
import csv
main = list(csv.DictReader(open('opus_all.csv')))
ov = list(csv.DictReader(open('overflight.csv')))
cols = list(main[0].keys()) + [c for c in ov[0].keys() if c not in main[0] and c!='country' and not c.startswith(('country_name','alpha'))]
by = {r['country']: r for r in ov}
w = csv.DictWriter(open('opus_all_v2.csv','w',newline=''), fieldnames=cols, extrasaction='ignore')
w.writeheader()
for r in main:
    r.update({k:v for k,v in by.get(r['country'],{}).items() if k in cols and k not in r or (k in cols and r.get(k) in ('','NA'))})
    w.writerow(r)
print('merged ->', 'opus_all_v2.csv', len(main), 'countries with overflight columns')"
