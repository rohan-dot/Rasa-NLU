
cat > _t.py <<'EOF'
import asyncio, airport_extract_new1 as a
t = a.slice_sections(a.clean_html('airports/FRA.cfm.html'))
print(asyncio.run(a.call_model(t))[0].get('diplomatic_lead_time_raw'))
EOF
python _t.py
