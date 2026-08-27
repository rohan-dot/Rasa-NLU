#!/usr/bin/env python3
"""
split_compounds.py — Split compound checklist items ("X / Y") into separate
single-question fields. Weak models return NA on compound questions when they
can only answer one half; separate fields fix that.

Run once:  python split_compounds.py
Idempotent: items already split (or missing) are skipped.
Then: rm -f data/work/checkpoint.json and regenerate BOTH CSVs so the
comparison uses the same columns.
"""

import json
from pathlib import Path

F = Path("checklist_enriched.json")

# old_id -> list of new items (inherit scope/route from the old item)
SPLITS = {
    "Diplomatic Clearance Lead Time / HazMat Requirements": [
        ("Diplomatic Clearance Lead Time",
         "Diplomatic/aircraft clearance lead time(s) and validity — every value "
         "with its category (blanket, annual, DV, military, unmanned)"),
        ("HazMat Clearance Requirements",
         "HAZMAT/dangerous goods clearance requirements, lead times, approvals, "
         "and restrictions"),
    ],
    "AIR Card / Cash Payment Requirement": [
        ("AIR Card Acceptance",
         "AIR Card acceptance for fuel/services purchase, vendors that take it, "
         "and any conditions"),
        ("Cash Payment Requirement",
         "Cash-only or advance-payment requirements for fuel, services, or fees"),
    ],
    "Customs / Ag / Immigration": [
        ("Customs Procedures",
         "Customs clearance requirements, procedures, locations and availability "
         "for arriving/departing aircraft"),
        ("Agriculture Quarantine",
         "Agriculture/quarantine inspection requirements (food, plants, animals, "
         "disinsection)"),
        ("Immigration Requirements",
         "Immigration/passport/visa requirements and procedures for aircrew"),
    ],
    "Ops Hrs / Holiday Closures": [
        ("Operating Hours",
         "Airfield/service operating hours (times in Z or local)"),
        ("Holiday Closures",
         "Holiday, weekend, or seasonal closures and reduced-operation dates"),
    ],
    "MOG and HazMat Requirements": [
        ("Maximum On Ground MOG",
         "Maximum aircraft on ground (MOG) limits by aircraft type or parking"),
        ("HazMat Ground Handling",
         "HAZMAT handling requirements/restrictions on the ground (storage, "
         "loading, notification)"),
    ],
    "Paying Agent (Purser) / Landing Fee Requirements (as required)": [
        ("Paying Agent Requirement",
         "Paying agent/purser requirements for settling charges"),
        ("Landing and Handling Fees",
         "Landing fees, handling fees, parking fees and how they are paid"),
    ],
}


def main():
    d = json.loads(F.read_text(encoding="utf-8"))
    existing_ids = {i.get("id") for i in d}
    out = []
    split_count = 0
    for it in d:
        iid = it.get("id")
        if iid in SPLITS and it.get("type") == "extract":
            new_items = []
            for nid, q in SPLITS[iid]:
                if nid in existing_ids:      # already split earlier — keep as is
                    continue
                new_items.append({"id": nid, "question": q, "type": "extract",
                                  "scope": it.get("scope", []),
                                  "route": it.get("route", [])})
            if new_items:
                out.extend(new_items)
                split_count += 1
                print(f"split: {iid}  ->  {[n['id'] for n in new_items]}")
                continue
        out.append(it)
    F.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_ex = sum(1 for i in out if i.get("type") == "extract")
    print(f"\n{split_count} compound items split. "
          f"Checklist now {len(out)} items ({n_ex} extract).")
    print("Next: rm -f data/work/checkpoint.json, then re-run opus_learn.py "
          "and local_extract.py so both CSVs share the new columns.")


if __name__ == "__main__":
    main()
