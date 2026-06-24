
import asyncio, importlib, airport_extract_new1 as a
importlib.reload(a)
t = a.slice_sections(a.clean_html('airports/FRA.cfm.html'))
data, note = await a.call_model(t)
print("LEAD:", data.get('diplomatic_lead_time_raw'))
