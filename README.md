import json
def keys(o, p="", d=0):
    if d > 4: return
    if isinstance(o, dict):
        for k, v in o.items(): print("  "*d + p + k); keys(v, "", d+1)
    elif isinstance(o, list) and o:
        print("  "*d + "[0]"); keys(o[0], "", d+1)
keys(json.load(open("airports/XXX.json")))
