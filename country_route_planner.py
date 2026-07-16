def load_mapping(path: str) -> dict:
    """mapping.csv: code,name — parsed manually so names containing
    commas (e.g. 'kor,Korea, Republic of') don't break the load."""
    mapping = {}
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                print(f"[load] mapping line {i+1} skipped (no comma): {line!r}")
                continue
            code, name = line.split(",", 1)
            code = code.strip().strip('"').upper()
            name = name.strip().strip('"')
            if i == 0 and (code.lower() in ("code", "airport", "abbr")
                           or name.lower() in ("name", "country")):
                continue  # header row
            mapping[code] = name
    if not mapping:
        sys.exit(f"[FATAL] mapping.csv parsed to 0 entries — check the file.")
    print(f"[load] mapping: {len(mapping)} codes "
          f"(e.g. {list(mapping.items())[:3]})")
    return mapping
