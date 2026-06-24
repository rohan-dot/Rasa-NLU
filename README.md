
python -c "import airport_extract as a; t=a.slice_sections(a.clean_html('airports/FRA.cfm')); print('LEN',len(t)); print('HAS_LEADTIME','LEAD-TIME' in t.upper()); import re; m=re.search(r'LEAD.TIME AND VALIDITY.{0,600}', t, re.S|re.I); print(m.group(0) if m else 'NOT IN SLICED TEXT')"
