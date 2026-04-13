No worries. Run this quick eval on your machine instead — it'll print a summary I can work with from a screenshot:

Run this on your machine:

```bash
python quick_eval.py \
    --pred submission_phase13.json \
    --gold 13B4_golden_bioasq_taskb_format.json
```

It'll print something like:

```
==========================================================
BIOASQ QUICK EVAL
==========================================================
  factoid   :  12/ 25 correct (48%)
  yesno     :   8/ 20 correct (40%)
  list      :   5/ 15 correct (33%)
  summary   :  25/ 25 correct (100%)
  OVERALL   :  50/ 85 (59%)

ERRORS (35 total):
  YESNO WRONG: Is Zotiraciclib effective...  gold=no pred=yes
  FACTOID WRONG: What is the treatment...    gold=['tss'] pred=surgical resection
  ...
```

Screenshot that output and send it to me. I'll see exactly which types are failing and which questions, then I'll fix the prompts and retrieval depth for the specific failure patterns. That's way more useful than scrolling through 85 questions in photos.
